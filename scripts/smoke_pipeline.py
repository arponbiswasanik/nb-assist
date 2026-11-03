"""End-to-end smoke test on real Ollama.

Runs 4 sample questions (EN, BN, Banglish, out-of-scope) through the
full pipeline and prints the trace per question. No eval, no metrics
— just verify the pipeline wires up and behaves.

Usage:
    python scripts/smoke_pipeline.py

Prerequisites:
    - ollama serve running on the configured host:port
    - gemma3:4b pulled
    - bge-m3 pulled
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import yaml

# Make src importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nb_assist.llm import OllamaClient, LLMUnavailable
from nb_assist.retrieve import HybridRetriever
from nb_assist.pipeline import handle_query
from nb_assist.trace import TraceSink


# 4 sample questions covering each path through the pipeline.
SAMPLES = [
    ("EN — in scope, PIN reset",
     "How do I reset my card PIN?"),
    ("BN — in scope, lost card",
     "আমার কার্ড হারিয়ে গেছে, এখন কী করব?"),
    ("Banglish — in scope, debit card fee",
     "debit card er annual fee koto?"),
    ("Out of scope — student loan (age 17, near-miss)",
     "Amar boyosh 17, ami ki student loan pabo?"),
]


def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_faq_lookup(rows: list[dict]) -> dict:
    return {int(r["id"]): {"question": r["question"], "answer": r["answer"]}
            for r in rows}


def _load_answer_bank(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        out[int(r["faq_id"])] = {
            "en": r["answer_en"],
            "bn": r["answer_bn"],
            "banglish": r["answer_banglish"],
        }
    return out


def _load_idk(rows: list[dict]) -> dict:
    return {r["language"]: r["text"] for r in rows}


def _print_separator(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def _print_trace(reply) -> None:
    t = reply.trace
    print(f"language        : {t.language}")
    print(f"normalized      : {t.normalized!r}")
    print(f"policy          : handoff={t.policy_handoff} rule={t.policy_rule}")
    if t.retrieval_top:
        print(f"retrieval top3  :")
        for r in t.retrieval_top:
            print(f"                  faq_id={r['faq_id']} fused={r['fused']:.3f}")
    print(f"gate            : zone={t.gate_zone} verifier={t.gate_verifier}")
    print(f"final path      : {reply.path}")
    print(f"latency         : {reply.latency_s:.2f}s")
    print(f"reply           : {reply.text}")
    print(f"stages          : " + ", ".join(f"{s.name}={s.duration_s:.3f}s"
                                              for s in t.stages))


def main() -> int:
    config_path = ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load data files.
    data_dir = ROOT / "data"
    faq_rows = _load_csv(data_dir / "bank-faq.csv")
    paraphrase_rows = _load_csv(data_dir / "paraphrase_index.csv")
    answer_bank_rows = _load_csv(data_dir / "answer_bank.csv")
    idk_rows = _load_csv(data_dir / "idk_templates.csv")

    faq_lookup = _load_faq_lookup(faq_rows)
    answer_bank = _load_answer_bank(answer_bank_rows)
    idk_bank = _load_idk(idk_rows)

    _print_separator("Setting up Ollama client and retriever")
    client = OllamaClient(config)

    # Warm up primary model so first user call is warm.
    print(f"warming up {config['models']['primary']}...")
    t0 = time.perf_counter()
    try:
        client.warmup(config["models"]["primary"])
        print(f"  warmup done in {time.perf_counter() - t0:.1f}s")
    except LLMUnavailable as e:
        print(f"  ERROR: cannot reach Ollama — {e}")
        print("  Make sure `ollama serve` is running and the model is pulled.")
        return 1

    # Build retriever index (real embeddings via bge-m3).
    print(f"building retriever index (bge-m3, {len(faq_rows)} FAQ + "
          f"{len(paraphrase_rows)} paraphrases)...")
    t0 = time.perf_counter()
    try:
        retriever = HybridRetriever(config, client).build_index(
            faq_rows, paraphrase_rows)
        print(f"  index built in {time.perf_counter() - t0:.1f}s")
    except LLMUnavailable as e:
        print(f"  ERROR: cannot embed — {e}")
        print("  Make sure bge-m3 is pulled: ollama pull bge-m3")
        return 1

    # Optional: write a trace file we can inspect later.
    trace_sink = TraceSink(str(ROOT / "results" / "smoke_trace.jsonl"))

    # Run each sample.
    for title, question in SAMPLES:
        _print_separator(title)
        print(f"question: {question}")
        try:
            reply = handle_query(question, retriever, client, config,
                                 faq_lookup, answer_bank, idk_bank,
                                 trace_sink=trace_sink)
            _print_trace(reply)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    _print_separator("Smoke test complete")
    print(f"trace written to {ROOT / 'results' / 'smoke_trace.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())