"""Evaluation metrics computed from per-question outcomes.

Pure functions - no I/O, no model calls. Each outcome is a dict with
at minimum these keys:
    id, language, expected_decision, gold_faq_id,
    predicted_path, predicted_faq_id, retrieval_top1_id,
    answer_correctness, latency_s, gate_zone, verifier_said,
    slot_flag, language_match
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class MetricsReport:
    overall: dict
    per_language: dict

    def to_dict(self) -> dict:
        return {"overall": self.overall, "per_language": self.per_language}


def _safe_pct(num: int, denom: int) -> float:
    return round(num / denom * 100, 2) if denom else 0.0


def _pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def compute_metrics(outcomes: Iterable[dict]) -> MetricsReport:
    """Compute all metrics from a list of outcome dicts."""
    outcomes = list(outcomes)
    by_lang: dict[str, list[dict]] = {}
    for o in outcomes:
        by_lang.setdefault(o["language"], []).append(o)

    overall = _compute_block(outcomes)
    per_language = {lang: _compute_block(rows) for lang, rows in by_lang.items()}
    return MetricsReport(overall=overall, per_language=per_language)


def _compute_block(rows: list[dict]) -> dict:
    total = len(rows)
    in_scope = [r for r in rows if r["expected_decision"] == "answer"]
    out_scope = [r for r in rows if r["expected_decision"] == "handoff"]

    # Retrieval: top-1 == gold FAQ id, only among in-scope.
    ret_correct = sum(1 for r in in_scope
                      if r.get("retrieval_top1_id") is not None
                      and r.get("gold_faq_id") is not None
                      and int(r["retrieval_top1_id"]) == int(r["gold_faq_id"]))
    ret_top1 = _safe_pct(ret_correct, len(in_scope))

    # Answer correctness: among in-scope answered, key facts present.
    answered_in_scope = [r for r in in_scope
                         if r.get("predicted_path") in ("generated", "retry",
                                                          "fallback")]
    correct = sum(1 for r in answered_in_scope
                  if r.get("answer_correctness") is True)
    answer_correctness = _safe_pct(correct, len(answered_in_scope))

    # IDK recall: out-of-scope that became handoff.
    idk_correct = sum(1 for r in out_scope
                      if r.get("predicted_path") == "handoff")
    idk_recall = _safe_pct(idk_correct, len(out_scope))

    # False handoff: in-scope that became handoff (should have answered).
    false_handoffs = sum(1 for r in in_scope
                         if r.get("predicted_path") == "handoff")
    false_handoff_rate = _safe_pct(false_handoffs, len(in_scope))

    # Unsafe answers: out-of-scope that received an answer.
    unsafe = sum(1 for r in out_scope
                 if r.get("predicted_path") != "handoff")
    unsafe_answers = _safe_pct(unsafe, len(out_scope))
    unsafe_count = unsafe

    # Language match: reply script matches detected language.
    lang_match = sum(1 for r in rows if r.get("language_match") is True)
    language_match_pct = _safe_pct(lang_match, total)

    # Latency.
    latencies = [r.get("latency_s", 0.0) for r in rows]
    latency_p50 = round(_pct(latencies, 50), 3)
    latency_p95 = round(_pct(latencies, 95), 3)
    latency_mean = round(float(np.mean(latencies)) if latencies else 0.0, 3)

    # Path mix: how many replies took each path.
    path_counts = Counter(r.get("predicted_path", "unknown") for r in rows)
    path_mix = {p: path_counts.get(p, 0) for p in
                ("generated", "retry", "fallback", "handoff")}
    path_mix["unknown"] = path_counts.get("unknown", 0)

    return {
        "total": total,
        "in_scope": len(in_scope),
        "out_scope": len(out_scope),
        "retrieval_top1_accuracy": ret_top1,
        "answer_correctness": answer_correctness,
        "idk_recall": idk_recall,
        "false_handoff_rate": false_handoff_rate,
        "unsafe_answers_pct": unsafe_answers,
        "unsafe_answers_count": unsafe_count,
        "language_match": language_match_pct,
        "latency_p50_s": latency_p50,
        "latency_p95_s": latency_p95,
        "latency_mean_s": latency_mean,
        "path_mix": path_mix,
    }