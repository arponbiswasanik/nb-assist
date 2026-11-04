"""Ablation study: run A, B, C, D configurations on the 40 test questions.

  A: gemma3:4b with full FAQ in context (no retrieval, no gate,
     no validators, no answer bank)
  B: + retrieval (no gate, no validators, no answer bank) -
     accept any retrieval result
  C: + gate (slot + thresholds + verifier, no validators, no answer bank)
  D: + validators + answer bank (full system, loaded from
     results/final_test_metrics.json)

Shares the retriever index across B and C. A doesn't use the retriever.

Writes:
  results/ablation_a_metrics.json
  results/ablation_b_metrics.json
  results/ablation_c_metrics.json
  results/ablation_summary.csv (4-level comparison table)

Usage:
  python eval/ablation.py
"""

from __future__ import annotations

import csv
import json
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
from eval.metrics import compute_metrics
from eval.online_compare import (
    _detect_idk, _language_match, _identify_faq_id, _key_facts_check,
    _build_faq_block, _build_prompt, _detect_language,
)


def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_faq_lookup(rows):
    return {int(r["id"]): {"question": r["question"], "answer": r["answer"]}
            for r in rows}


def _load_answer_bank(rows):
    out = {}
    for r in rows:
        out[int(r["faq_id"])] = {
            "en": r["answer_en"],
            "bn": r["answer_bn"],
            "banglish": r["answer_banglish"],
        }
    return out


def _no_gate_config(config):
    """Return a deep copy of config with gate effectively disabled."""
    cfg = json.loads(json.dumps(config))
    g = cfg["gate"]
    g["t_low"] = 0.0
    g["t_high"] = 0.0
    g["margin_min"] = 0.0
    g["slot_check_enabled"] = False
    g["verifier_enabled"] = False
    return cfg


def _no_validators_config(config):
    """Return a deep copy of config with all validators disabled."""
    cfg = json.loads(json.dumps(config))
    cfg["validators"] = {
        "script_check": False,
        "fact_check": False,
        "extra_content_check": False,
        "length_check": False,
        "loop_check": False,
        "retry_on_fail": 0,
        "fallback_to_verified": False,
    }
    return cfg


def _run_level_a(client, config, faq_rows, test_rows, gold_map):
    """Level A: gemma3:4b with full FAQ in context, no architecture."""
    print("\n=== level A: baseline (LLM with full FAQ, no architecture) ===")
    faq_block = _build_faq_block(faq_rows)

    outcomes = []
    for i, row in enumerate(test_rows, 1):
        qid = row.get("id", str(i))
        question = row["question"]
        messages = _build_prompt(faq_block, question)
        gold = gold_map.get(qid, {})

        t0 = time.perf_counter()
        try:
            res = client.chat(config["models"]["primary"], messages,
                              temperature=0.0, max_tokens=300)
            latency = time.perf_counter() - t0
            reply_text = res.text or ""
        except Exception as e:
            print(f"  [{i:>3}/{len(test_rows)}] id={qid} ERROR {e}")
            outcomes.append({
                "id": qid,
                "language": _detect_language(question),
                "expected_decision": gold.get("expected_decision"),
                "gold_faq_id": gold.get("gold_faq_id"),
                "predicted_path": "error",
                "predicted_faq_id": None,
                "retrieval_top1_id": None,
                "answer_correctness": False,
                "language_match": False,
                "latency_s": 0.0,
            })
            continue

        is_idk = _detect_idk(reply_text)
        path = "handoff" if is_idk else "generated"
        lang = _detect_language(question)
        lang_match = _language_match(reply_text, lang)
        if not is_idk:
            top1 = _identify_faq_id(reply_text, faq_rows)
            source = faq_rows[top1 - 1]["answer"] if top1 else ""
            correct = _key_facts_check(reply_text, source)
        else:
            top1 = None
            correct = None

        outcomes.append({
            "id": qid,
            "language": lang,
            "expected_decision": gold.get("expected_decision"),
            "gold_faq_id": gold.get("gold_faq_id"),
            "predicted_path": path,
            "predicted_faq_id": top1,
            "retrieval_top1_id": top1,
            "answer_correctness": correct,
            "language_match": lang_match,
            "latency_s": round(latency, 3),
        })
        print(f"  [{i:>3}/{len(test_rows)}] id={qid} path={path} "
              f"top1={top1} match={lang_match} lat={latency:.2f}s")

    return outcomes


def _run_pipeline_level(client, config, faq_lookup, answer_bank, idk_bank,
                        test_rows, gold_map, retriever, level_name):
    """Run the full pipeline for a given configuration."""
    print(f"\n=== {level_name} ===")
    outcomes = []
    for i, row in enumerate(test_rows, 1):
        qid = row.get("id", str(i))
        question = row["question"]
        gold = gold_map.get(qid, {})

        try:
            reply = handle_query(question, retriever, client, config,
                                 faq_lookup, answer_bank, idk_bank)
        except Exception as e:
            print(f"  [{i:>3}/{len(test_rows)}] id={qid} ERROR {e}")
            outcomes.append({
                "id": qid,
                "language": _detect_language(question),
                "expected_decision": gold.get("expected_decision"),
                "gold_faq_id": gold.get("gold_faq_id"),
                "predicted_path": "error",
                "predicted_faq_id": None,
                "retrieval_top1_id": None,
                "answer_correctness": False,
                "language_match": False,
                "latency_s": 0.0,
            })
            continue

        t = reply.trace
        top1 = t.retrieval_top[0]["faq_id"] if t.retrieval_top else None

        if reply.path == "handoff":
            correct = None
        else:
            source = (answer_bank.get(top1, {}) if top1 else {}) \
                .get(reply.language) \
                or faq_lookup.get(top1, {}).get("answer", "")
            correct = _key_facts_check(reply.text, source)

        outcomes.append({
            "id": qid,
            "language": t.language,
            "expected_decision": gold.get("expected_decision"),
            "gold_faq_id": gold.get("gold_faq_id"),
            "predicted_path": reply.path,
            "predicted_faq_id": top1,
            "retrieval_top1_id": top1,
            "answer_correctness": correct,
            "language_match": _language_match(reply.text, reply.language),
            "latency_s": round(reply.latency_s, 3),
        })
        exp = gold.get("expected_decision")
        if exp == "answer":
            ok = reply.path != "handoff"
        else:
            ok = reply.path == "handoff"
        marker = "ok" if ok else "MISS"
        print(f"  [{i:>3}/{len(test_rows)}] {marker} id={qid} "
              f"path={reply.path} top1={top1} lat={reply.latency_s:.2f}s")

    return outcomes


def main() -> int:
    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    faq_rows = _load_csv(ROOT / config["paths"]["faq"])
    paraphrase_rows = _load_csv(ROOT / config["paths"]["paraphrase_index"])
    idk_rows = _load_csv(ROOT / config["paths"]["idk_templates"])
    test_rows = _load_csv(ROOT / config["paths"]["test_questions"])
    gold_rows = _load_csv(ROOT / config["paths"]["test_gold"])
    answer_bank_rows = _load_csv(ROOT / config["paths"]["answer_bank"])

    faq_lookup = _load_faq_lookup(faq_rows)
    idk_bank = {r["language"]: r["text"] for r in idk_rows}
    answer_bank = _load_answer_bank(answer_bank_rows)
    gold_map = {r["id"]: r for r in gold_rows}

    print(f"FAQ: {len(faq_rows)} entries, paraphrases: {len(paraphrase_rows)}")
    print(f"test questions: {len(test_rows)}")

    client = OllamaClient(config)
    print(f"warming up {config['models']['primary']}...")
    t0 = time.perf_counter()
    try:
        client.warmup(config["models"]["primary"])
        print(f"  warmup done in {time.perf_counter() - t0:.1f}s")
    except LLMUnavailable as e:
        print(f"  ERROR: {e}")
        return 1

    print("building retriever index (shared across levels B and C)...")
    t0 = time.perf_counter()
    try:
        retriever = HybridRetriever(config, client).build_index(
            faq_rows, paraphrase_rows)
        print(f"  index built in {time.perf_counter() - t0:.1f}s")
    except LLMUnavailable as e:
        print(f"  ERROR: {e}")
        return 1

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Level A: baseline (no architecture)
    a_outcomes = _run_level_a(client, config, faq_rows, test_rows, gold_map)
    a_report = compute_metrics(a_outcomes)
    (results_dir / "ablation_a_metrics.json").write_text(
        json.dumps(a_report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")

    # Level B: + retrieval, no gate, no validators, no answer bank
    b_config = _no_validators_config(_no_gate_config(config))
    b_outcomes = _run_pipeline_level(
        client, b_config, faq_lookup, {}, idk_bank,
        test_rows, gold_map, retriever,
        "level B: + retrieval (no gate, no validators, no answer bank)")
    b_report = compute_metrics(b_outcomes)
    (results_dir / "ablation_b_metrics.json").write_text(
        json.dumps(b_report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")

    # Level C: + gate (full), no validators, no answer bank
    c_config = _no_validators_config(config)
    c_outcomes = _run_pipeline_level(
        client, c_config, faq_lookup, {}, idk_bank,
        test_rows, gold_map, retriever,
        "level C: + gate (slot + thresholds + verifier, no validators)")
    c_report = compute_metrics(c_outcomes)
    (results_dir / "ablation_c_metrics.json").write_text(
        json.dumps(c_report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")

    # Level D: full system - load from existing final_test_metrics.json
    d_path = results_dir / "final_test_metrics.json"
    if d_path.exists():
        d_report_dict = json.loads(d_path.read_text(encoding="utf-8"))
    else:
        print("\n=== level D (full system) - final_test_metrics.json missing, "
              "running fresh ===")
        d_outcomes = _run_pipeline_level(
            client, config, faq_lookup, answer_bank, idk_bank,
            test_rows, gold_map, retriever, "level D: full system")
        d_report = compute_metrics(d_outcomes)
        d_report_dict = d_report.to_dict()
        (results_dir / "ablation_d_metrics.json").write_text(
            json.dumps(d_report_dict, ensure_ascii=False, indent=2),
            encoding="utf-8")

    # Write ablation summary CSV.
    descriptions = {
        "A": "LLM with full FAQ in context, no architecture",
        "B": "+ retrieval (no gate, no validators, no answer bank)",
        "C": "+ gate (slot + thresholds + verifier, no validators)",
        "D": "+ validators + answer bank (full system)",
    }
    rows = []
    for level, report in [("A", a_report.to_dict()),
                         ("B", b_report.to_dict()),
                         ("C", c_report.to_dict()),
                         ("D", d_report_dict)]:
        o = report["overall"]
        rows.append({
            "level": level,
            "description": descriptions[level],
            "retrieval_top1_accuracy": o.get("retrieval_top1_accuracy"),
            "answer_correctness": o.get("answer_correctness"),
            "idk_recall": o.get("idk_recall"),
            "false_handoff_rate": o.get("false_handoff_rate"),
            "unsafe_answers_count": o.get("unsafe_answers_count"),
            "language_match": o.get("language_match"),
            "latency_p50_s": o.get("latency_p50_s"),
            "latency_p95_s": o.get("latency_p95_s"),
        })

    summary_path = results_dir / "ablation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nablation summary -> {summary_path}")

    # Print summary table.
    print("\n=== ablation study ===")
    print(f"  {'level':<5} {'description':<55} {'ret':>6} {'ans':>6} "
          f"{'idk':>6} {'fho':>6} {'unsf':>5} {'lang':>6} {'p50':>6}")
    for r in rows:
        print(f"  {r['level']:<5} {r['description'][:55]:<55} "
              f"{r['retrieval_top1_accuracy']!s:>6} "
              f"{r['answer_correctness']!s:>6} "
              f"{r['idk_recall']!s:>6} "
              f"{r['false_handoff_rate']!s:>6} "
              f"{r['unsafe_answers_count']!s:>5} "
              f"{r['language_match']!s:>6} "
              f"{r['latency_p50_s']!s:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())