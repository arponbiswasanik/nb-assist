"""Answerability gate.

Three signals, three zones. The model never decides on its own
whether to answer; this gate does.

Signals:
  1. Slot check — extract salient tokens (places, product nouns) from
     the query and test whether the matched FAQ entry mentions them.
     Targets near-misses: high similarity, wrong intent.
  2. Score thresholds — below t_low => handoff; above t_high with
     no slot flag and a good margin => candidate; in between => grey.
  3. LLM verifier — for grey-zone candidates and below-t_high ones,
     ask the primary model a closed YES/NO question at temperature 0.
     A single token; about a second. NO => handoff.

Tunables (t_low, t_high, margin_min, slot_keywords, verifier_skip_above)
live in config.yaml and are calibrated on the dev set only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import OllamaClient, LLMError
from .retrieve import RetrievalOutput


@dataclass
class GateDecision:
    answerable: bool
    zone: str            # 'handoff_low' | 'candidate' | 'grey' | 'handoff_slot' | 'handoff_verifier' | 'handoff_error'
    slot_flag: bool
    verifier_said: str | None     # 'YES' | 'NO' | None (skipped or error)
    reason: str


_SLOT_RE = re.compile(r"\b([A-Za-z]+)\b")


def _slot_check(query: str, faq_question: str, faq_answer: str,
                slot_keywords: list[str]) -> bool:
    """Return True if a salient token in the query is absent from the FAQ entry.

    We only check keywords from config (Singapore, Sylhet, student,
    crypto, gold, dollar, visa, share). If any of them appears in the
    query but is absent from both the FAQ question and answer, we
    flag a near-miss.
    """
    q_lower = query.lower()
    faq_lower = (faq_question + " " + faq_answer).lower()
    for kw in slot_keywords:
        if kw.lower() in q_lower and kw.lower() not in faq_lower:
            return True
    return False


def _ask_verifier(client: OllamaClient, model: str, prompt_template: str,
                 faq_q: str, faq_a: str, query: str) -> str:
    """Ask the primary model a closed YES/NO question. Returns 'YES' or 'NO'."""
    faq_text = f"Q: {faq_q}\nA: {faq_a}"
    prompt = prompt_template.format(faq=faq_text, question=query)
    messages = [{"role": "user", "content": prompt}]
    res = client.chat(model, messages, temperature=0.0, max_tokens=3)
    text = (res.text or "").strip().upper()
    # Be lenient: pick the first YES/NO we see.
    if "YES" in text:
        return "YES"
    if "NO" in text:
        return "NO"
    # Ambiguous reply => treat as NO (fail closed).
    return "NO"


def decide(client: OllamaClient | None, config: dict,
           query: str, faq_question: str, faq_answer: str,
           retrieval: RetrievalOutput) -> GateDecision:
    """Decide whether the top-1 FAQ entry should be used to answer.

    client may be None when the verifier is disabled or skipped; we
    still return a decision in that case.
    """
    g = config["gate"]
    if not retrieval.results:
        return GateDecision(False, "handoff_low", False, None, "no retrieval results")

    top = retrieval.results[0]
    score = top.fused_score
    margin = retrieval.margin

    # Signal 1: slot check.
    slot_flag = False
    if g.get("slot_check_enabled", True):
        slot_flag = _slot_check(query, faq_question, faq_answer,
                                g.get("slot_keywords", []))
    if slot_flag:
        return GateDecision(False, "handoff_slot", True, None,
                            "slot keyword in query absent from FAQ")

    # Signal 2: score thresholds.
    t_low = g["t_low"]
    t_high = g["t_high"]
    margin_min = g.get("margin_min", 0.0)

    if score < t_low:
        return GateDecision(False, "handoff_low", False, None,
                            f"score {score:.3f} < t_low {t_low:.3f}")

    # Candidate zone: above t_high with a good margin.
    if score >= t_high and margin >= margin_min:
        # Signal 3: optional verifier skip on very high scores.
        if not g.get("verifier_enabled", True):
            return GateDecision(True, "candidate", False, None,
                                 "above t_high, verifier disabled")
        skip_above = g.get("verifier_skip_above", 1.01)
        if score >= skip_above:
            return GateDecision(True, "candidate", False, None,
                                f"score {score:.3f} >= skip_above, verifier skipped")
        # Otherwise still verify high-score candidates (cheap, safe).
        return _verify(client, config, query, faq_question, faq_answer)

    # Grey zone: t_low <= score < t_high, or margin too small.
    return _verify(client, config, query, faq_question, faq_answer)


def _verify(client: OllamaClient | None, config: dict,
            query: str, faq_question: str, faq_answer: str) -> GateDecision:
    """Run the LLM verifier. Fail closed on any client error."""
    g = config["gate"]
    if not g.get("verifier_enabled", True):
        return GateDecision(True, "candidate", False, None,
                            "verifier disabled, grey-zone accepted")
    if client is None:
        return GateDecision(False, "handoff_error", False, None,
                            "verifier enabled but no client provided")

    model = config["models"]["primary"]
    template = g.get("verifier_prompt_template", "")
    try:
        verdict = _ask_verifier(client, model, template,
                                faq_question, faq_answer, query)
    except LLMError as e:
        return GateDecision(False, "handoff_error", False, None,
                            f"verifier error: {type(e).__name__}")
    if verdict == "YES":
        return GateDecision(True, "candidate", False, "YES",
                            "verifier accepted")
    return GateDecision(False, "handoff_verifier", False, "NO",
                        "verifier rejected")