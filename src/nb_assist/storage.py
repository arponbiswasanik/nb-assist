"""Postgres + pgvector storage layer.

Single source of truth for FAQ content, paraphrases, and the
conversation audit log. Embeddings live alongside the text they
describe; retrieval loads them into memory at startup.

Schema:
  faq_entries      (id, topic, question, answer, embedding vector(1024))
  paraphrases      (id, faq_id, language, text, embedding vector(1024))
  conversation_log (id, ts, query, language, predicted_path,
                    predicted_faq_id, gate_zone, verifier_said,
                    latency_s, reply_excerpt)

The retriever interface in retrieve.py stays unchanged; only the
source of the dense matrix changes (CSV-built at startup -> DB-loaded
at startup). A future swap to Qdrant or hosted pgvector would only
touch this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import psycopg2
from psycopg2.extras import execute_values


@dataclass
class StorageConfig:
    host: str
    port: int
    name: str
    user: str
    password: str


class Storage:
    """Thin wrapper around a Postgres + pgvector connection."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self._conn = None

    @classmethod
    def from_config(cls, config: dict) -> "Storage":
        db = config["database"]
        return cls(StorageConfig(
            host=db["host"], port=db["port"], name=db["name"],
            user=db["user"], password=db["password"],
        ))

    def connect(self) -> "Storage":
        self._conn = psycopg2.connect(
            host=self.config.host, port=self.config.port,
            dbname=self.config.name, user=self.config.user,
            password=self.config.password,
        )
        self._conn.autocommit = False
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # --- schema ---------------------------------------------------------

    def init_schema(self) -> None:
        """Create tables and the pgvector extension if missing."""
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS faq_entries (
                    id          INTEGER PRIMARY KEY,
                    topic       TEXT NOT NULL,
                    question    TEXT NOT NULL,
                    answer      TEXT NOT NULL,
                    embedding   vector(1024) NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS paraphrases (
                    id          SERIAL PRIMARY KEY,
                    faq_id      INTEGER NOT NULL
                                REFERENCES faq_entries(id) ON DELETE CASCADE,
                    language    TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    embedding   vector(1024) NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_log (
                    id              SERIAL PRIMARY KEY,
                    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    query           TEXT NOT NULL,
                    language        TEXT NOT NULL,
                    predicted_path  TEXT NOT NULL,
                    predicted_faq_id INTEGER,
                    gate_zone       TEXT,
                    verifier_said   TEXT,
                    latency_s       REAL NOT NULL,
                    reply_excerpt   TEXT
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_log_ts
                    ON conversation_log(ts DESC);
            """)
        self._conn.commit()

    # --- upsert ---------------------------------------------------------

    def upsert_faq(self, faq_id: int, topic: str, question: str,
                   answer: str, embedding: list[float] | None) -> None:
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO faq_entries (id, topic, question, answer, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    topic     = EXCLUDED.topic,
                    question  = EXCLUDED.question,
                    answer    = EXCLUDED.answer,
                    embedding = EXCLUDED.embedding
            """, (faq_id, topic, question, answer,
                  _vector_param(embedding)))

    def upsert_paraphrase(self, faq_id: int, language: str, text: str,
                          embedding: list[float] | None) -> None:
        # Delete existing paraphrases for this faq_id, then insert fresh.
        # Simpler than upserting per row since paraphrases have no stable id.
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM paraphrases WHERE faq_id = %s;",
                        (faq_id,))
        if text:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO paraphrases (faq_id, language, text, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (faq_id, language, text, _vector_param(embedding)))

    def upsert_paraphrases_batch(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        # Group by faq_id, delete existing, then bulk insert.
        faq_ids = sorted(set(int(r["faq_id"]) for r in rows))
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM paraphrases WHERE faq_id = ANY(%s);",
                        (faq_ids,))
            values = [
                (int(r["faq_id"]), r["language"], r["text"],
                 _vector_param(r.get("embedding")))
                for r in rows
            ]
            execute_values(cur, """
                INSERT INTO paraphrases (faq_id, language, text, embedding)
                VALUES %s
            """, values)
        self._conn.commit()

    def commit(self) -> None:
        self._conn.commit()

    # --- read -----------------------------------------------------------

    def load_faq(self) -> list[dict]:
        """Return all FAQ rows with their embeddings."""
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT id, topic, question, answer, embedding
                FROM faq_entries
                ORDER BY id
            """)
            rows = cur.fetchall()
        return [
            {"id": r[0], "topic": r[1], "question": r[2],
             "answer": r[3], "embedding": _pgvector_to_list(r[4])}
            for r in rows
        ]

    def load_paraphrases(self) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT id, faq_id, language, text, embedding
                FROM paraphrases
                ORDER BY faq_id, id
            """)
            rows = cur.fetchall()
        return [
            {"id": r[0], "faq_id": r[1], "language": r[2],
             "text": r[3], "embedding": _pgvector_to_list(r[4])}
            for r in rows
        ]

    # --- audit log ------------------------------------------------------

    def log_conversation(self, query: str, language: str,
                         predicted_path: str, predicted_faq_id: int | None,
                         gate_zone: str | None, verifier_said: str | None,
                         latency_s: float, reply_excerpt: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_log
                    (query, language, predicted_path, predicted_faq_id,
                     gate_zone, verifier_said, latency_s, reply_excerpt)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (query, language, predicted_path, predicted_faq_id,
                  gate_zone, verifier_said, latency_s,
                  reply_excerpt[:500] if reply_excerpt else None))
        self._conn.commit()


def _vector_param(embedding: list[float] | None) -> str | None:
    """Format a Python list as a pgvector literal for SQL insertion."""
    if embedding is None:
        return None
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def _pgvector_to_list(v) -> list[float] | None:
    """Convert a pgvector value returned by psycopg2 to a Python list."""
    if v is None:
        return None
    if isinstance(v, list):
        return v
    # Some psycopg2 versions return the vector as a string like "[0.1,0.2,...]"
    if isinstance(v, str):
        return [float(x) for x in v.strip("[]").split(",") if x]
    return list(v)