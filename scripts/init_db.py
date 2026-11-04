"""One-shot: create the schema and ingest the FAQ from CSV into Postgres.

Embeds each FAQ entry's canonical question plus all its paraphrases
through the bge-m3 Ollama model and stores the vectors in pgvector.

Usage:
  - Start Postgres first: docker compose up -d
  - Then run this script: python scripts/init_db.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from nb_assist.llm import OllamaClient, LLMUnavailable
from nb_assist.storage import Storage


def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    faq_rows = _load_csv(ROOT / config["paths"]["faq"])
    paraphrase_rows = _load_csv(ROOT / config["paths"]["paraphrase_index"])
    print(f"FAQ entries to ingest: {len(faq_rows)}")
    print(f"paraphrases to ingest: {len(paraphrase_rows)}")

    # Connect to Postgres.
    print("connecting to Postgres...")
    try:
        storage = Storage.from_config(config).connect()
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  is `docker compose up -d` running?")
        return 1

    # Wait for Postgres to be ready (up to 30s).
    for attempt in range(6):
        try:
            with storage._conn.cursor() as cur:
                cur.execute("SELECT 1")
            print("  postgres ready")
            break
        except Exception:
            print(f"  waiting for postgres... ({attempt + 1}/6)")
            time.sleep(5)
    else:
        print("  ERROR: postgres not ready after 30s")
        return 1

    # Create schema.
    print("creating schema...")
    storage.init_schema()
    print("  schema ready")

    # Set up Ollama client for embeddings.
    client = OllamaClient(config)
    embedding_model = config["models"]["embedding"]
    print(f"embedding model: {embedding_model}")

    # Embed and upsert each FAQ entry.
    print("embedding and upserting FAQ entries...")
    t0 = time.perf_counter()
    for i, row in enumerate(faq_rows, 1):
        faq_id = int(row["id"])
        text_to_embed = row["question"]
        try:
            vectors = client.embed(embedding_model, [text_to_embed])
            embedding = vectors[0]
        except LLMUnavailable as e:
            print(f"  [{i:>3}/{len(faq_rows)}] id={faq_id} embed ERROR {e}")
            embedding = None
        storage.upsert_faq(faq_id, row["topic"], row["question"],
                           row["answer"], embedding)
        storage.commit()
        if i % 10 == 0 or i == len(faq_rows):
            elapsed = time.perf_counter() - t0
            print(f"  [{i:>3}/{len(faq_rows)}] faq_id={faq_id} "
                  f"({elapsed:.1f}s elapsed)")
    print(f"  FAQ done in {time.perf_counter() - t0:.1f}s")

    # Group paraphrases by faq_id, embed in batches, upsert.
    print("embedding and upserting paraphrases...")
    t0 = time.perf_counter()

    # Embed all paraphrases in one batch per faq_id (faster).
    by_faq = {}
    for r in paraphrase_rows:
        by_faq.setdefault(int(r["faq_id"]), []).append(r)

    total_done = 0
    for faq_id, paras in by_faq.items():
        texts = [p["text"] for p in paras]
        try:
            vectors = client.embed(embedding_model, texts)
        except LLMUnavailable as e:
            print(f"  faq_id={faq_id} embed ERROR {e}")
            vectors = [None] * len(texts)
        rows_with_emb = []
        for p, v in zip(paras, vectors):
            rows_with_emb.append({
                "faq_id": faq_id,
                "language": p["language"],
                "text": p["text"],
                "embedding": v,
            })
        storage.upsert_paraphrases_batch(rows_with_emb)
        total_done += len(rows_with_emb)
        if total_done % 30 == 0 or faq_id == max(by_faq.keys()):
            print(f"  {total_done}/{len(paraphrase_rows)} paraphrases done "
                  f"({time.perf_counter() - t0:.1f}s)")
    print(f"  paraphrases done in {time.perf_counter() - t0:.1f}s")

    storage.close()
    print("\ninit_db complete. You can now run:")
    print("  python eval/run_eval.py --dataset test --run-name final")
    return 0


if __name__ == "__main__":
    sys.exit(main())