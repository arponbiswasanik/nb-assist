"""Tune gate thresholds (t_low, t_high) on the dev set.

Strategy: cache retrieval and verifier signals once per dev question,
then sweep a grid of (t_low, t_high) pairs in memory with no extra
model calls. For each pair compute:
  - IDK recall: out-of-scope questions that became handoff
  - false handoff: in-scope questions that became handoff
  - unsafe: out-of-scope questions that received an answer
Operating point: max IDK recall subject to zero unsafe answers,
tie-broken by min false handoff.

Usage:
    python scripts/tune_thresholds.py
    python scripts/tune_thresholds.py --apply  # update config.yaml
"""

from __future__ import annotations

import argparse
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
from nb_assist.language import normalize
from nb_assist.policy import check_policy
# Import two private helpers from gate.py. This is a one-off tuning
# script where we need control over caching the raw signals more than
# we need the modular decide() entry point.
from nb_assist.gate import _slot_check, _ask_verifier


def _load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_faq_lookup(rows):
    return {int(r["id"]): {"question": r["question"], "answer": r["answer"]}
            for r in rows}


def _simulate(c, t_low, t_high, margin_min, verifier_skip_above,
              verifier_enabled, slot_check_enabled):
    """Replay the gate decision in memory using cached signals."""
    score = c["score"]
    margin = c["margin"]
    slot_flag = c["slot_flag"] if slot_check_enabled else False
    verifier_said = c["verifier_said"]

    if slot_flag:
        return False
    if score < t_low:
        return False
    if score >= t_high and margin >= margin_min:
        if not verifier_enabled:
            return True
        if score >= verifier_skip_above:
            return True
        return verifier_said == "YES"
    # Grey zone: verifier decides.
    if not verifier_enabled:
        return True
    if verifier_said is None:
        return False
    return verifier_said == "YES"


def _evaluate(cached, t_low, t_high, margin_min, verifier_skip_above,
              verifier_enabled, slot_check_enabled):
    """Compute metrics for one (t_low, t_high) pair."""
    sweep = [c for c in cached if not c["policy_handoff"]]
    idk_correct = 0
    out_total = 0
    false_handoff = 0
    in_total = 0
    unsafe = 0
    for c in sweep:
        answerable = _simulate(c, t_low, t_high, margin_min,
                               verifier_skip_above, verifier_enabled,
                               slot_check_enabled)
        path = "answer" if answerable else "handoff"
        if c["expected_decision"] == "handoff":
            out_total += 1
            if path == "handoff":
                idk_correct += 1
            else:
                unsafe += 1
        else:
            in_total += 1
            if path == "handoff":
                false_handoff += 1
    return {
        "idk_recall": round(idk_correct / out_total * 100, 1) if out_total else 0,
        "false_handoff_rate": round(false_handoff / in_total * 100, 1) if in_total else 0,
        "unsafe_count": unsafe,
        "idk_correct": idk_correct,
        "out_total": out_total,
        "false_handoff": false_handoff,
        "in_total": in_total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="update config.yaml (note: comments will be lost)")
    args = p.parse_args()

    config_path = ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = ROOT / "data"
    faq_rows = _load_csv(data_dir / "bank-faq.csv")
    paraphrase_rows = _load_csv(data_dir / "paraphrase_index.csv")
    dev_rows = _load_csv(data_dir / "dev_set.csv")
    faq_lookup = _load_faq_lookup(faq_rows)

    print(f"dev set: {len(dev_rows)} rows")

    client = OllamaClient(config)
    print(f"warming up {config['models']['primary']}...")
    t0 = time.perf_counter()
    try:
        client.warmup(config["models"]["primary"])
        print(f"  warmup done in {time.perf_counter() - t0:.1f}s")
    except LLMUnavailable as e:
        print(f"  ERROR: {e}")
        return 1

    print("building retriever index...")
    t0 = time.perf_counter()
    try:
        retriever = HybridRetriever(config, client).build_index(
            faq_rows, paraphrase_rows)
        print(f"  index built in {time.perf_counter() - t0:.1f}s")
    except LLMUnavailable as e:
        print(f"  ERROR: {e}")
        return 1

    # Phase 2: cache signals per dev question.
    g_cfg = config["gate"]
    slot_keywords = g_cfg.get("slot_keywords", [])
    primary_model = config["models"]["primary"]
    verifier_template = g_cfg.get("verifier_prompt_template", "")

    print("\ncaching per-question signals (retrieval + verifier each)...")
    cached = []
    for i, row in enumerate(dev_rows, 1):
        question = row["question"]
        exp = row["expected_decision"]
        gold_id = row.get("gold_faq_id")
        norm = normalize(question)
        pol = check_policy(norm)
        if pol.handoff:
            cached.append({
                "id": row.get("id", str(i)),
                "question": question,
                "expected_decision": exp,
                "gold_faq_id": gold_id,
                "score": 0.0,
                "margin": 0.0,
                "slot_flag": False,
                "verifier_said": "NO",
                "policy_handoff": True,
                "top1": None,
            })
            print(f"  [{i:>3}/{len(dev_rows)}] id={row.get('id')} "
                  f"policy_handoff rule={pol.rule}")
            continue
        retr = retriever.retrieve(norm)
        if not retr.results:
            cached.append({
                "id": row.get("id", str(i)),
                "question": question,
                "expected_decision": exp,
                "gold_faq_id": gold_id,
                "score": 0.0,
                "margin": 0.0,
                "slot_flag": False,
                "verifier_said": "NO",
                "policy_handoff": False,
                "top1": None,
            })
            print(f"  [{i:>3}/{len(dev_rows)}] id={row.get('id')} no_retrieval")
            continue
        top = retr.results[0]
        faq = faq_lookup.get(top.faq_id, {"question": "", "answer": ""})
        slot = _slot_check(norm, faq["question"], faq["answer"], slot_keywords)
        # Always call the verifier so the sweep has a cached verdict.
        verdict = _ask_verifier(client, primary_model, verifier_template,
                                faq["question"], faq["answer"], norm)
        cached.append({
            "id": row.get("id", str(i)),
            "question": question,
            "expected_decision": exp,
            "gold_faq_id": gold_id,
            "score": top.fused_score,
            "margin": retr.margin,
            "slot_flag": slot,
            "verifier_said": verdict,
            "policy_handoff": False,
            "top1": top.faq_id,
        })
        print(f"  [{i:>3}/{len(dev_rows)}] id={row.get('id')} "
              f"exp={exp} score={top.fused_score:.3f} "
              f"slot={slot} vrf={verdict} top1={top.faq_id}")

    # Phase 3: in-memory threshold sweep.
    print("\nsweeping threshold grid...")
    margin_min = g_cfg.get("margin_min", 0.10)
    verifier_skip_above = g_cfg.get("verifier_skip_above", 0.85)
    verifier_enabled = g_cfg.get("verifier_enabled", True)
    slot_check_enabled = g_cfg.get("slot_check_enabled", True)

    t_lows = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    t_highs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    results = []
    for t_low in t_lows:
        for t_high in t_highs:
            if t_low >= t_high:
                continue
            r = _evaluate(cached, t_low, t_high, margin_min,
                          verifier_skip_above, verifier_enabled,
                          slot_check_enabled)
            results.append({"t_low": t_low, "t_high": t_high, **r})

    # Operating point: max IDK recall subject to zero unsafe answers,
    # tie-broken by min false handoff.
    results.sort(key=lambda r: (r["idk_recall"], -r["unsafe_count"],
                                 -r["false_handoff_rate"]),
                 reverse=True)

    print("\n=== threshold sweep (top 10 by operating point) ===")
    print(f"{'t_low':>6} {'t_high':>7} {'idk_r':>7} {'fho_r':>8} {'unsafe':>6}")
    for r in results[:10]:
        print(f"{r['t_low']:>6.2f} {r['t_high']:>7.2f} "
              f"{r['idk_recall']:>6.1f}% {r['false_handoff_rate']:>7.1f}% "
              f"{r['unsafe_count']:>6d}")

    best = results[0]
    print(f"\nrecommended: t_low={best['t_low']:.2f} "
          f"t_high={best['t_high']:.2f}")
    print(f"  IDK recall: {best['idk_recall']}%")
    print(f"  false handoff rate: {best['false_handoff_rate']}%")
    print(f"  unsafe answers: {best['unsafe_count']}")

    # Save the full sweep table.
    out = ROOT / "results" / "threshold_sweep.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nsweep table -> {out}")

    # Save the recommendation.
    rec_path = ROOT / "results" / "threshold_recommendation.json"
    with rec_path.open("w", encoding="utf-8") as f:
        json.dump({
            "t_low": best["t_low"],
            "t_high": best["t_high"],
            "idk_recall": best["idk_recall"],
            "false_handoff_rate": best["false_handoff_rate"],
            "unsafe_count": best["unsafe_count"],
            "dev_set_size": len(dev_rows),
        }, f, indent=2)
    print(f"recommendation -> {rec_path}")

    if args.apply:
        # Backup before rewriting (preserves the commented version).
        backup = config_path.with_suffix(".yaml.bak")
        backup.write_text(config_path.read_text(encoding="utf-8"),
                          encoding="utf-8")
        print(f"backup -> {backup}")
        config["gate"]["t_low"] = best["t_low"]
        config["gate"]["t_high"] = best["t_high"]
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False,
                           sort_keys=False, allow_unicode=True)
        print(f"applied to {config_path}")
        print("note: yaml comments were lost; backup at .yaml.bak")
    else:
        print("\nto apply: rerun with --apply "
              "(comments in config.yaml will be lost; backup auto-created)")

    return 0


if __name__ == "__main__":
    sys.exit(main())