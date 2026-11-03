"""Unit tests for the gate. Mocked OllamaClient for the verifier."""

from unittest.mock import MagicMock

import pytest

from nb_assist.gate import decide, GateDecision
from nb_assist.retrieve import RetrievalResult, RetrievalOutput
from nb_assist.llm import LLMUnavailable


CFG = {
    "models": {"primary": "gemma3:4b", "embedding": "bge-m3"},
    "gate": {
        "t_low": 0.45,
        "t_high": 0.70,
        "margin_min": 0.10,
        "slot_check_enabled": True,
        "slot_keywords": ["Singapore", "Sylhet", "student",
                          "crypto", "gold", "dollar", "visa", "share"],
        "verifier_enabled": True,
        "verifier_skip_above": 0.85,
        "verifier_prompt_template":
            "Does this FAQ entry directly answer the question? "
            "Reply YES or NO only.\nFAQ: {faq}\nQuestion: {question}",
    },
}


def _retrieval(top_score, margin=0.20, faq_id=1):
    return RetrievalOutput(
        results=[
            RetrievalResult(faq_id=faq_id, fused_score=top_score,
                            dense_score=top_score, lexical_score=top_score,
                            rank=0),
            RetrievalResult(faq_id=faq_id + 1, fused_score=top_score - margin,
                            dense_score=0, lexical_score=0, rank=1),
        ],
        margin=margin,
    )


def _client_yes():
    c = MagicMock()
    c.chat.return_value = MagicMock(text="YES")
    return c


def _client_no():
    c = MagicMock()
    c.chat.return_value = MagicMock(text="NO")
    return c


def _client_error():
    c = MagicMock()
    c.chat.side_effect = LLMUnavailable("down")
    return c


def test_below_t_low_handoff():
    d = decide(None, CFG, "random question",
               "What is the loan rate?", "5 percent",
               _retrieval(0.30))
    assert not d.answerable
    assert d.zone == "handoff_low"


def test_slot_keyword_handoff():
    # Query mentions Singapore, FAQ does not.
    d = decide(_client_yes(), CFG, "do you have a Singapore branch",
               "What are your branch hours?",
               "Branches are open 10am to 4pm.",
               _retrieval(0.80, margin=0.20))
    assert not d.answerable
    assert d.zone == "handoff_slot"
    assert d.slot_flag is True


def test_no_slot_passes_when_keyword_present_in_faq():
    # Query mentions student; FAQ mentions student too => no slot flag.
    d = decide(_client_yes(), CFG, "am i eligible for a student loan",
               "What is personal loan eligibility?",
               "Personal loans require ages 25-60 and student status is not covered.",
               _retrieval(0.80, margin=0.20))
    assert d.slot_flag is False


def test_high_score_with_margin_skips_verifier_above_threshold():
    d = decide(_client_yes(), CFG, "what is the personal loan interest rate",
               "What is the personal loan interest rate?",
               "The rate is 12 percent.",
               _retrieval(0.90, margin=0.20))
    assert d.answerable
    assert d.zone == "candidate"
    assert d.verifier_said is None   # skipped


def test_high_score_below_skip_calls_verifier():
    c = _client_yes()
    d = decide(c, CFG, "personal loan",
               "What is the personal loan interest rate?",
               "The rate is 12 percent.",
               _retrieval(0.75, margin=0.20))
    assert d.answerable
    assert d.zone == "candidate"
    assert d.verifier_said == "YES"
    assert c.chat.called


def test_grey_zone_verifier_yes_accepts():
    d = decide(_client_yes(), CFG, "loan",
              "What is the personal loan interest rate?",
              "The rate is 12 percent.",
              _retrieval(0.55, margin=0.20))
    assert d.answerable
    assert d.zone == "candidate"
    assert d.verifier_said == "YES"


def test_grey_zone_verifier_no_handoff():
    d = decide(_client_no(), CFG, "loan",
              "What is the personal loan interest rate?",
              "The rate is 12 percent.",
              _retrieval(0.55, margin=0.20))
    assert not d.answerable
    assert d.zone == "handoff_verifier"
    assert d.verifier_said == "NO"


def test_grey_zone_no_margin_calls_verifier():
    d = decide(_client_yes(), CFG, "loan",
              "What is the personal loan interest rate?",
              "The rate is 12 percent.",
              _retrieval(0.75, margin=0.05))
    assert d.answerable
    assert d.zone == "candidate"
    assert d.verifier_said == "YES"


def test_verifier_error_fails_closed():
    d = decide(_client_error(), CFG, "loan",
              "What is the personal loan interest rate?",
              "The rate is 12 percent.",
              _retrieval(0.55, margin=0.20))
    assert not d.answerable
    assert d.zone == "handoff_error"


def test_no_client_when_verifier_needed_fails_closed():
    d = decide(None, CFG, "loan",
              "What is the personal loan interest rate?",
              "The rate is 12 percent.",
              _retrieval(0.55, margin=0.20))
    assert not d.answerable
    assert d.zone == "handoff_error"


def test_verifier_disabled_accepts_grey():
    cfg = dict(CFG)
    cfg["gate"] = dict(CFG["gate"])
    cfg["gate"]["verifier_enabled"] = False
    d = decide(None, cfg, "loan",
              "What is the personal loan interest rate?",
              "The rate is 12 percent.",
              _retrieval(0.55, margin=0.20))
    assert d.answerable
    assert d.zone == "candidate"


def test_ambiguous_verifier_reply_treated_as_no():
    c = MagicMock()
    c.chat.return_value = MagicMock(text="Maybe")
    d = decide(c, CFG, "loan",
              "What is the personal loan interest rate?",
              "The rate is 12 percent.",
              _retrieval(0.55, margin=0.20))
    assert not d.answerable
    assert d.zone == "handoff_verifier"


def test_no_retrieval_results_handoff():
    empty = RetrievalOutput(results=[], margin=0.0)
    d = decide(None, CFG, "x", "y", "z", empty)
    assert not d.answerable