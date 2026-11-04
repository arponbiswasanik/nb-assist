"""Hybrid retrieval: dense (bge-m3) + lexical (char n-gram TF-IDF), fused.

Two build paths:
  - build_index(faq_rows, paraphrase_rows):
      Embeds everything via Ollama at startup. Used when no DB is
      available (or for tests with mocked embeddings).
  - build_index_from_storage(faq_rows, paraphrase_rows):
      Uses embeddings already stored in Postgres + pgvector. Skips
      the embedding step; vectors come from the DB. Used in production
      and in eval runs when the DB is up.

At query time: embed the query, cosine vs all FAQ dense vectors;
transform the query through TF-IDF, cosine vs all FAQ lexical rows;
fuse by weighted sum (default) or reciprocal rank fusion. Return
top-k with per-signal scores and the gap between top-1 and top-2.

The dense matrix lives in memory either way; for 30 entries this is
fine. For tens of thousands, swap this class for a QdrantRetriever
or PgvectorRetriever implementation that satisfies the same
interface - callers in pipeline.py do not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .llm import OllamaClient


@dataclass
class RetrievalResult:
    faq_id: int
    fused_score: float
    dense_score: float
    lexical_score: float
    rank: int


@dataclass
class RetrievalOutput:
    results: list[RetrievalResult]
    margin: float          # top-1 minus top-2 fused score (top-1 if only 1)


def _norm_cosine(scores: np.ndarray) -> np.ndarray:
    """Cosine is in [-1, 1]; rescale to [0, 1] for clean fusion."""
    return (scores + 1.0) / 2.0


class HybridRetriever:
    """Dense + lexical hybrid retriever over FAQ + paraphrase index."""

    def __init__(self, config: dict, client: OllamaClient):
        self.config = config
        self.client = client
        self.embedding_model = config["models"]["embedding"]

        r = config["retrieval"]
        self.top_k = r["top_k"]
        self.dense_weight = r["dense_weight"]
        self.lexical_weight = r["lexical_weight"]
        self.fusion = r.get("fusion", "weighted_sum")
        ngram = r.get("char_ngram_range", [3, 5])

        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(ngram[0], ngram[1]),
            sublinear_tf=True,
        )

        self.faq_ids: list[int] = []
        self.dense_matrix: np.ndarray | None = None     # (N, D), L2-normalized
        self.lexical_matrix: np.ndarray | None = None    # (N, V), sparse

    def build_index(self, faq_rows: list[dict],
                    paraphrase_rows: list[dict]) -> "HybridRetriever":
        """Embed and TF-IDF-index every FAQ entry via Ollama at startup.

        faq_rows: list of dicts with at least {id, question}
        paraphrase_rows: list of dicts with at least {faq_id, text}
        """
        # Group canonical + paraphrase texts per FAQ id.
        texts_per_faq: dict[int, list[str]] = {}
        for row in faq_rows:
            texts_per_faq[int(row["id"])] = [row["question"]]
        for row in paraphrase_rows:
            fid = int(row["faq_id"])
            if fid in texts_per_faq:
                texts_per_faq[fid].append(row["text"])

        self.faq_ids = sorted(texts_per_faq.keys())

        avg_vectors = []
        lexical_docs = []
        for fid in self.faq_ids:
            docs = texts_per_faq[fid]
            vectors = self.client.embed(self.embedding_model, docs)
            stacked = np.asarray(vectors, dtype=np.float32)
            avg = stacked.mean(axis=0)
            avg_vectors.append(avg)
            lexical_docs.append(" ".join(docs))

        # L2-normalize dense vectors so cosine = dot product.
        self.dense_matrix = np.vstack(avg_vectors).astype(np.float32)
        norms = np.linalg.norm(self.dense_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense_matrix = self.dense_matrix / norms

        self.lexical_matrix = self.tfidf.fit_transform(lexical_docs)
        return self

    def build_index_from_storage(self, faq_rows: list[dict],
                                 paraphrase_rows: list[dict]) -> "HybridRetriever":
        """Build the index from Postgres-loaded rows that already include
        embeddings. Skips the embedding step (vectors come from the DB).

        faq_rows: dicts with at least {id, question, embedding}
        paraphrase_rows: dicts with at least {faq_id, text, embedding}
        """
        # Group paraphrases per FAQ id (same as build_index, but skip embed).
        texts_per_faq: dict[int, list[str]] = {}
        for row in faq_rows:
            texts_per_faq[int(row["id"])] = [row["question"]]
        for row in paraphrase_rows:
            fid = int(row["faq_id"])
            if fid in texts_per_faq:
                texts_per_faq[fid].append(row["text"])

        self.faq_ids = sorted(texts_per_faq.keys())

        # Use stored embeddings (FAQ: average of canonical + paraphrases).
        avg_vectors = []
        lexical_docs = []
        for fid in self.faq_ids:
            docs = texts_per_faq[fid]
            # Collect embeddings for this FAQ: one from the FAQ row,
            # plus any from paraphrase rows.
            vectors = []
            for row in faq_rows:
                if int(row["id"]) == fid and row.get("embedding") is not None:
                    vectors.append(np.asarray(row["embedding"],
                                              dtype=np.float32))
            for row in paraphrase_rows:
                if int(row["faq_id"]) == fid and row.get("embedding") is not None:
                    vectors.append(np.asarray(row["embedding"],
                                              dtype=np.float32))
            if vectors:
                stacked = np.vstack(vectors)
                avg = stacked.mean(axis=0)
            else:
                # Fallback: zero vector (will get zero similarity).
                avg = np.zeros(1024, dtype=np.float32)
            avg_vectors.append(avg)
            lexical_docs.append(" ".join(docs))

        self.dense_matrix = np.vstack(avg_vectors).astype(np.float32)
        norms = np.linalg.norm(self.dense_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense_matrix = self.dense_matrix / norms

        self.lexical_matrix = self.tfidf.fit_transform(lexical_docs)
        return self

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalOutput:
        if self.dense_matrix is None or self.lexical_matrix is None:
            raise RuntimeError("index not built; call build_index first")
        k = top_k or self.top_k

        # Dense signal.
        q_vec = np.asarray(
            self.client.embed(self.embedding_model, [query])[0],
            dtype=np.float32,
        )
        n = np.linalg.norm(q_vec)
        if n > 0:
            q_vec = q_vec / n
        dense_cos = self.dense_matrix @ q_vec                  # (N,)
        dense_scores = _norm_cosine(dense_cos)

        # Lexical signal.
        q_tfidf = self.tfidf.transform([query])
        lex_cos = cosine_similarity(q_tfidf, self.lexical_matrix).ravel()
        lex_scores = _norm_cosine(lex_cos)

        # Fusion.
        if self.fusion == "rrf":
            fused = self._rrf_fusion(dense_scores, lex_scores)
        else:
            fused = (self.dense_weight * dense_scores
                     + self.lexical_weight * lex_scores)

        # Top-k with per-signal scores.
        order = (-fused).argsort()
        results = []
        for rank, idx in enumerate(order[:k]):
            results.append(RetrievalResult(
                faq_id=self.faq_ids[idx],
                fused_score=float(fused[idx]),
                dense_score=float(dense_scores[idx]),
                lexical_score=float(lex_scores[idx]),
                rank=rank,
            ))
        margin = (results[0].fused_score - results[1].fused_score
                  if len(results) >= 2 else float(results[0].fused_score))
        return RetrievalOutput(results=results, margin=float(margin))

    def _rrf_fusion(self, dense_scores: np.ndarray,
                    lex_scores: np.ndarray) -> np.ndarray:
        """Reciprocal rank fusion. k=60 is the common default."""
        K = 60
        dense_rank = (-dense_scores).argsort()
        lex_rank = (-lex_scores).argsort()
        rrf_dense = np.zeros_like(dense_scores)
        rrf_lex = np.zeros_like(lex_scores)
        for r, i in enumerate(dense_rank):
            rrf_dense[i] = 1.0 / (K + r + 1)
        for r, i in enumerate(lex_rank):
            rrf_lex[i] = 1.0 / (K + r + 1)
        return self.dense_weight * rrf_dense + self.lexical_weight * rrf_lex