"""Run the pipeline on a CSV (dev_set or test_gold), compute metrics.

Dataset: dev_set (for tuning) or test_gold (final run).
Writes:
  results/<run_name>_<dataset>_outcomes.csv
  results/<run_name>_<dataset>_metrics.json

Usage:
  python eval/run_eval.py --dataset dev --run-name v1
  python eval/run_eval.py --dataset test --run-name final
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from nb_assist.llm import OllamaClient, LLMUnavailable
from nb_assist.retrieve import HybridRetriever
from nb_assist.pipeline import handle_query
from nb_assist.trace import TraceSink
from eval.metrics import compute_metrics


# A simple key-facts check: every number in the reply matches the source.
_NUMBER = re.compile(r"\d[\d,]*\.?\d*%?")


def _key_facts_check(reply: str, source_answer: str) -> bool:
    """True if every number in the reply appears in the source answer."""
    if not reply or not source_answer:
        return False
    src_digits = set(re.findall(r"\d+", source_answer))
    reply_digits = set(re.findall(r"\d+", reply))
    whitelist = {"1", "2", "3"}
    for d in reply_digits:
        if d in whitelist:
            continue
        if d not in src_digits:
            return False
    return True


def _language_match(reply: str, lang: str) -> bool:
    """Coarse script check: does the reply script match the detected language?"""
    if not reply:
        return False
    bn = bool(re.search(r"[\u0980-\u09ff]", reply))
    latin = bool(re.search(r"[A-Za-z]", reply))
    if lang == "banglish":
        return latin and not bn
    if lang == "en":
        return not bn
    if lang == "bn":
        return bn
    return True


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


def _resolve_dataset(name: str, config: dict) -> Path:
    paths = config["paths"]
    if name == "dev":
        return ROOT / paths["dev_set"]
    if name in ("test", "test_gold"):
        return ROOT / paths["test_gold"]
    raise ValueError(f"unknown dataset: {name}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["dev", "test"])
    p.add_argument("--run-name", default="run")
    p.add_argument("--no-verifier", action="store_true",
                   help="disable the LLM verifier (ablation A/B comparison)")
    p.add_argument("--no-slot-check", action="store_true",
                   help="disable slot check (ablation)")
    p.add_argument("--no-answer-bank", action="store_true",
                   help="ablation: use English answer only, runtime fallback")
    args = p.parse_args()

    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Configuration overrides for ablation runs.
    if args.no_verifier:
        config["gate"]["verifier_enabled"] = False
    if args.no_slot_check:
        config["gate"]["slot_check_enabled"] = False

    data_dir = ROOT / "data"
    faq_rows = _load_csv(data_dir / "bank-faq.csv")
    paraphrase_rows = _load_csv(data_dir / "paraphrase_index.csv")
    answer_bank_rows = _load_csv(data_dir / "answer_bank.csv")
    idk_rows = _load_csv(data_dir / "idk_templates.csv")

    faq_lookup = _load_faq_lookup(faq_rows)
    answer_bank = _load_answer_bank(answer_bank_rows)
    idk_bank = _load_idk(idk_rows)

    dataset_path = _resolve_dataset(args.dataset, config)
    dataset_rows = _load_csv(dataset_path)
    print(f"dataset: {dataset_path.name} ({len(dataset_rows)} rows)")
    print(f"run name: {args.run_name}")

    # Set up the Ollama client.
    client = OllamaClient(config)
    print(f"warming up {config['models']['primary']}...")
    t0 = time.perf_counter()
    try:
        client.warmup(config["models"]["primary"])
        print(f"  warmup done in {time.perf_counter() - t0:.1f}s")
    except LLMUnavailable as e:
        print(f"  ERROR: cannot reach ollama - {e}")
        return 1

    # Build the retriever index.
    print(f"building retriever index (bge-m3, {len(faq_rows)} FAQ + "
          f"{len(paraphrase_rows)} paraphrases)...")
    t0 = time.perf_counter()
    try:
        retriever = HybridRetriever(config, client).build_index(
            faq_rows, paraphrase_rows)
        print(f"  index built in {time.perf_counter() - t0:.1f}s")
    except LLMUnavailable as e:
        print(f"  ERROR: cannot embed - {e}")
        return 1

    # Output paths.
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / f"{args.run_name}_{args.dataset}_outcomes.csv"
    out_json = results_dir / f"{args.run_name}_{args.dataset}_metrics.json"
    trace_path = results_dir / f"{args.run_name}_{args.dataset}_trace.jsonl"
    sink = TraceSink(str(trace_path))

    outcomes = []
    for i, row in enumerate(dataset_rows, 1):
        qid = row.get("id", str(i))
        question = row["question"]
        lang_expected = row.get("language")
        expected_decision = row.get("expected_decision")
        gold_faq_id = row.get("gold_faq_id") or None

        try:
            reply = handle_query(question, retriever, client, config,
                                 faq_lookup, answer_bank, idk_bank,
                                 trace_sink=sink)
        except Exception as e:
            print(f"  [{i:>3}] {qid} ERROR {type(e).__name__}: {e}")
            outcomes.append({
                "id": qid, "language": lang_expected,
                "expected_decision": expected_decision,
                "gold_faq_id": gold_faq_id,
                "predicted_path": "error",
                "predicted_faq_id": None,
                "retrieval_top1_id": None,
                "answer_correctness": False,
                "language_match": False,
                "latency_s": 0.0,
                "gate_zone": "error",
                "verifier_said": None,
                "slot_flag": False,
                "reply_excerpt": "",
            })
            continue

        t = reply.trace
        top1 = t.retrieval_top[0]["faq_id"] if t.retrieval_top else None

        # Answer correctness: skip the handoff path (already screened
        # by the gate). For answered paths, compare numbers to source.
        if reply.path == "handoff":
            correct = None
        else:
            source = (answer_bank.get(top1, {}) if top1 else {}) \
                .get(reply.language) \
                or faq_lookup.get(top1, {}).get("answer", "")
            correct = _key_facts_check(reply.text, source)

        outcome = {
            "id": qid,
            "language": t.language,
            "expected_decision": expected_decision,
            "gold_faq_id": gold_faq_id,
            "predicted_path": reply.path,
            "predicted_faq_id": top1,
            "retrieval_top1_id": top1,
            "answer_correctness": correct,
            "language_match": _language_match(reply.text, reply.language),
            "latency_s": round(reply.latency_s, 3),
            "gate_zone": t.gate_zone,
            "verifier_said": t.gate_verifier,
            "slot_flag": False,
            "reply_excerpt": reply.text[:120],
        }
        outcomes.append(outcome)
        # Quick visual indicator of correctness per row.
        if expected_decision == "answer":
            ok = reply.path != "handoff"
        else:
            ok = reply.path == "handoff"
        marker = "ok" if ok else "MISS"
        print(f"  [{i:>3}/{len(dataset_rows)}] {marker} id={qid} "
              f"lang={t.language} path={reply.path} "
              f"top1={top1} zone={t.gate_zone} "
              f"lat={reply.latency_s:.2f}s")

    # Write outcomes CSV.
    fieldnames = list(outcomes[0].keys()) if outcomes else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(outcomes)
    print(f"\noutcomes -> {out_csv}")

    # Write metrics JSON.
    report = compute_metrics(outcomes)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"metrics  -> {out_json}")

    # Console summary.
    print("\n=== overall ===")
    for k, v in report.overall.items():
        if k == "path_mix":
            print(f"  path_mix: {v}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())