"""Unit tests for the hybrid retriever. Mocked embeddings.

The fake embedder maps each text to a topic one-hot vector based on
keyword presence. This makes dense scores deterministic and lets us
assert rankings without depending on bge-m3.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from nb_assist.retrieve import HybridRetriever


CFG = {
    "models": {"embedding": "bge-m3-mock"},
    "retrieval": {
        "top_k": 3,
        "dense_weight": 0.7,
        "lexical_weight": 0.3,
        "fusion": "weighted_sum",
        "char_ngram_range": [3, 5],
    },
}


def _fake_client():
    client = MagicMock()

    def fake_embed(model, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            tl = t.lower()
            if "pin" in tl:
                v = np.array([1.0, 0.0, 0.0, 0.0])
            elif "card" in tl and ("fee" in tl or "annual" in tl):
                v = np.array([0.0, 1.0, 0.0, 0.0])
            elif "loan" in tl:
                v = np.array([0.0, 0.0, 1.0, 0.0])
            else:
                v = np.array([0.5, 0.5, 0.5, 0.0])
            out.append(v.astype(np.float32))
        return out

    client.embed.side_effect = fake_embed
    return client


def _faq():
    return [
        {"id": 1, "topic": "PIN", "question": "How do I reset my card PIN?"},
        {"id": 2, "topic": "Debit card",
         "question": "What is the annual fee for the debit card?"},
        {"id": 3, "topic": "Loan",
         "question": "What is the personal loan interest rate?"},
    ]


def _paraphrases():
    return [
        {"faq_id": 1, "language": "banglish",
         "text": "amar pin kivabe reset korbo"},
        {"faq_id": 2, "language": "banglish",
         "text": "debit card er annual fee koto"},
        {"faq_id": 3, "language": "banglish",
         "text": "personal loan interest rate koto"},
    ]


def test_build_index_shape():
    r = HybridRetriever(CFG, _fake_client())
    r.build_index(_faq(), _paraphrases())
    assert len(r.faq_ids) == 3
    assert r.dense_matrix.shape == (3, 4)
    assert r.lexical_matrix.shape[0] == 3


def test_retrieve_returns_top_k():
    r = HybridRetriever(CFG, _fake_client())
    r.build_index(_faq(), _paraphrases())
    out = r.retrieve("card pin reset")
    assert len(out.results) == 3
    assert out.results[0].rank == 0
    assert out.results[1].rank == 1
    assert out.results[2].rank == 2


def test_retrieve_top1_matches_intent():
    r = HybridRetriever(CFG, _fake_client())
    r.build_index(_faq(), _paraphrases())
    out = r.retrieve("card pin reset")
    assert out.results[0].faq_id == 1


def test_retrieve_loan_query_returns_loan_top():
    r = HybridRetriever(CFG, _fake_client())
    r.build_index(_faq(), _paraphrases())
    out = r.retrieve("what is the personal loan interest rate")
    assert out.results[0].faq_id == 3


def test_retrieve_margin_nonnegative():
    r = HybridRetriever(CFG, _fake_client())
    r.build_index(_faq(), _paraphrases())
    out = r.retrieve("personal loan")
    assert out.margin >= 0.0
    assert out.results[0].fused_score >= out.results[1].fused_score


def test_retrieve_before_build_raises():
    r = HybridRetriever(CFG, _fake_client())
    with pytest.raises(RuntimeError):
        r.retrieve("test")


def test_retrieve_respects_custom_top_k():
    r = HybridRetriever(CFG, _fake_client())
    r.build_index(_faq(), _paraphrases())
    out = r.retrieve("card pin reset", top_k=1)
    assert len(out.results) == 1


def test_rrf_fusion_still_ranks():
    cfg = {
        "models": {"embedding": "bge-m3-mock"},
        "retrieval": {
            "top_k": 3,
            "dense_weight": 0.5,
            "lexical_weight": 0.5,
            "fusion": "rrf",
            "char_ngram_range": [3, 5],
        },
    }
    r = HybridRetriever(cfg, _fake_client())
    r.build_index(_faq(), _paraphrases())
    out = r.retrieve("card pin reset")
    assert len(out.results) == 3
    assert out.results[0].faq_id == 1