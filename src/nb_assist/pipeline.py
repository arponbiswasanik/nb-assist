"""Pipeline — wires S0 through S7 into one handle_query call.

Each stage is timed and recorded in the trace. The model never gets
the last word on a banking reply: failure at any stage fails closed
to an IDK template or to the verified answer verbatim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .language import normalize, detect_language, Lang
from .policy import check_policy
from .retrieve import HybridRetriever
from .gate import decide as gate_decide
from .generate import generate, _GenerationFailure
from .validate import run_all
from .trace import Trace, TraceSink


@dataclass
class Reply:
    text: str
    path: str            # 'generated' | 'retry' | 'fallback' | 'handoff'
    language: Lang
    latency_s: float
    trace: Trace


def _idk(lang: Lang, idk_bank: dict) -> str:
    return (idk_bank.get(lang)
            or idk_bank.get("en")
            or "I don't know, let me connect you to a person.")


def _finish_handoff(lang, idk_bank, trace, t_start, sink) -> Reply:
    text = _idk(lang, idk_bank)
    trace.final_path = "handoff"
    trace.final_text = text
    trace.total_duration_s = time.perf_counter() - t_start
    if sink:
        sink.write(trace)
    return Reply(text=text, path="handoff", language=lang,
                 latency_s=trace.total_duration_s, trace=trace)


def _finish(text, path, lang, trace, t_start, sink) -> Reply:
    trace.final_path = path
    trace.final_text = text
    trace.total_duration_s = time.perf_counter() - t_start
    if sink:
        sink.write(trace)
    return Reply(text=text, path=path, language=lang,
                 latency_s=trace.total_duration_s, trace=trace)


def _try_generate(client, config, norm, verified, lang, trace, attempt) -> tuple:
    """One generate+validate cycle. Returns (text, ok)."""
    t0 = time.perf_counter()
    try:
        result = generate(client, config, norm, verified, lang)
    except _GenerationFailure as e:
        trace.add_stage(f"generate_{attempt}", time.perf_counter() - t0,
                        {"path": "error", "error": str(e)})
        return None, False
    trace.add_stage(f"generate_{attempt}", time.perf_counter() - t0,
                    {"path": result.path})
    ok, reason = run_all(result.text, verified, lang, config)
    trace.add_stage(f"validate_{attempt}", 0.0,
                    {"passed": ok, "reason": reason})
    return result.text, ok


def handle_query(query: str,
                 retriever: HybridRetriever,
                 client,
                 config: dict,
                 faq_lookup: dict,
                 answer_bank: dict,
                 idk_bank: dict,
                 trace_sink: TraceSink | None = None) -> Reply:
    t_start = time.perf_counter()
    trace = Trace(query=query, normalized="", language="en")

    # S0 normalize
    t0 = time.perf_counter()
    norm = normalize(query)
    trace.normalized = norm
    trace.add_stage("normalize", time.perf_counter() - t0, {})

    # S1 detect language
    t0 = time.perf_counter()
    lang = detect_language(norm, config)
    trace.language = lang
    trace.add_stage("detect_language", time.perf_counter() - t0,
                    {"language": lang})

    # S2 policy pre-filter
    t0 = time.perf_counter()
    pol = check_policy(norm)
    trace.policy_handoff = pol.handoff
    trace.policy_rule = pol.rule
    trace.add_stage("policy", time.perf_counter() - t0,
                    {"handoff": pol.handoff, "rule": pol.rule})
    if pol.handoff:
        return _finish_handoff(lang, idk_bank, trace, t_start, trace_sink)

    # S3 retrieve
    t0 = time.perf_counter()
    retr = retriever.retrieve(norm)
    top_summary = [{"faq_id": r.faq_id, "fused": round(r.fused_score, 3)}
                   for r in retr.results[:3]]
    trace.retrieval_top = top_summary
    trace.add_stage("retrieve", time.perf_counter() - t0,
                    {"top3": top_summary, "margin": retr.margin})
    if not retr.results:
        return _finish_handoff(lang, idk_bank, trace, t_start, trace_sink)

    # S4 gate
    top = retr.results[0]
    faq_entry = faq_lookup.get(top.faq_id, {"question": "", "answer": ""})
    t0 = time.perf_counter()
    g = gate_decide(client, config, norm, faq_entry.get("question", ""),
                    faq_entry.get("answer", ""), retr)
    trace.gate_zone = g.zone
    trace.gate_verifier = g.verifier_said
    trace.add_stage("gate", time.perf_counter() - t0,
                    {"zone": g.zone, "verifier": g.verifier_said,
                     "answerable": g.answerable})
    if not g.answerable:
        return _finish_handoff(lang, idk_bank, trace, t_start, trace_sink)

    # S5 + S6 generate, validate, retry, fallback
    ab = answer_bank.get(top.faq_id, {})
    verified = ab.get(lang) or ab.get("en") or faq_entry.get("answer", "")

    text, ok = _try_generate(client, config, norm, verified, lang, trace, 1)
    if ok:
        return _finish(text, "generated", lang, trace, t_start, trace_sink)

    retries = config.get("validators", {}).get("retry_on_fail", 1)
    if retries >= 1:
        text, ok = _try_generate(client, config, norm, verified, lang, trace, 2)
        if ok:
            return _finish(text, "retry", lang, trace, t_start, trace_sink)

    if config.get("validators", {}).get("fallback_to_verified", True):
        return _finish(verified, "fallback", lang, trace, t_start, trace_sink)
    return _finish_handoff(lang, idk_bank, trace, t_start, trace_sink)