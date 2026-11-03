"""Unit tests for the pipeline. Mocked retriever and client."""

from unittest.mock import MagicMock

from nb_assist.pipeline import handle_query
from nb_assist.retrieve import RetrievalResult, RetrievalOutput
from nb_assist.llm import LLMUnavailable


CFG = {
    "models": {"primary": "gemma3:4b", "embedding": "bge-m3"},
    "language": {"bn_char_threshold": 0.30, "ties_to": "en",
                 "banglish_lexicon": ["ami", "amar", "koto", "kivabe",
                                      "korbo", "ki"]},
    "gate": {
        "t_low": 0.45, "t_high": 0.70, "margin_min": 0.10,
        "slot_check_enabled": True,
        "slot_keywords": ["Singapore", "Sylhet", "student"],
        "verifier_enabled": True, "verifier_skip_above": 0.85,
        "verifier_prompt_template": "Q: {faq}\nQ: {question}\nYES or NO?",
    },
    "generation": {"temperature": 0.0, "max_tokens": 200, "num_fewshot": 2},
    "validators": {"script_check": True, "fact_check": True,
                   "extra_content_check": True, "length_check": True,
                   "loop_check": True, "retry_on_fail": 1,
                   "fallback_to_verified": True},
}


FAQ = {1: {"question": "How do I reset my card PIN?",
           "answer": "Open the NB Go app, go to Settings, then Card, then Reset PIN."}}
ANSWER_BANK = {1: {
    "en": "Open the NB Go app, go to Settings, then Card, then Reset PIN.",
    "banglish": "NB Go app khulin, settings e jan, tarpor card, tarpor reset PIN.",
    "bn": "NB Go অ্যাপ খুলুন, সেটিংসে যান, তারপর কার্ড, তারপর রিসেট পিন।",
}}
IDK = {"en": "I don't know, let me connect you to a person.",
       "banglish": "Eta ami jani na, apnake manusher sathe connect kore dichhi.",
       "bn": "আমি এই বিষয়ে নিশ্চিত নই।"}


def _retriever(score=0.80, margin=0.20, faq_id=1, empty=False):
    r = MagicMock()
    if empty:
        r.retrieve.return_value = RetrievalOutput(results=[], margin=0.0)
    else:
        r.retrieve.return_value = RetrievalOutput(
            results=[
                RetrievalResult(faq_id=faq_id, fused_score=score,
                                dense_score=score, lexical_score=score, rank=0),
                RetrievalResult(faq_id=faq_id + 1, fused_score=score - margin,
                                dense_score=0, lexical_score=0, rank=1),
            ],
            margin=margin,
        )
    return r


def _client_returns(text):
    c = MagicMock()
    c.chat.return_value = MagicMock(text=text)
    return c


def _client_sequence(*texts):
    c = MagicMock()
    c.chat.side_effect = [MagicMock(text=t) for t in texts]
    return c


# --- policy handoff path ------------------------------------------------

def test_policy_handoff_short_circuits_to_idk():
    r = _retriever()
    c = _client_returns("YES")
    reply = handle_query("what is my friend's account balance",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "handoff"
    assert "don't know" in reply.text.lower()
    assert c.chat.call_count == 0


# --- no retrieval results ----------------------------------------------

def test_no_retrieval_results_handoff():
    r = _retriever(empty=True)
    c = _client_returns("YES")
    reply = handle_query("random unrelated query",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "handoff"
    assert c.chat.call_count == 0


# --- gate rejects ------------------------------------------------------

def test_below_t_low_handoff():
    r = _retriever(score=0.30)
    c = _client_returns("YES")
    reply = handle_query("how do i reset my pin",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "handoff"
    assert c.chat.call_count == 0


def test_slot_keyword_handoff():
    r = _retriever(score=0.85)
    c = _client_returns("YES")
    reply = handle_query("do you have a singapore branch",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "handoff"
    assert c.chat.call_count == 0


def test_verifier_no_handoff():
    r = _retriever(score=0.55, margin=0.20)
    c = _client_returns("NO")
    reply = handle_query("how do i reset my pin",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "handoff"
    assert c.chat.call_count == 1


# --- happy path: generated --------------------------------------------

def test_happy_path_generated_banglish():
    r = _retriever(score=0.90, margin=0.20)  # above skip_above
    c = _client_returns("amar pin reset kore dilam NB Go app theke.")
    reply = handle_query("amar pin kivabe reset korbo?",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "generated"
    assert "amar pin" in reply.text.lower()
    assert reply.language == "banglish"


def test_high_score_skips_verifier():
    r = _retriever(score=0.90, margin=0.20)
    c = _client_returns("amar pin reset kore dilam.")
    handle_query("amar pin kivabe reset korbo?",
                 r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert c.chat.call_count == 1  # generate only, no verifier


# --- retry path --------------------------------------------------------

def test_retry_after_first_validation_failure():
    r = _retriever(score=0.55, margin=0.20)
    c = _client_sequence(
        "YES",
        "আমার পিন রিসেট করে দিলাম।",        # Bangla in banglish → script fail
        "amar pin reset kore dilam.",       # clean retry
    )
    reply = handle_query("amar pin kivabe reset korbo?",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "retry"
    assert "amar pin" in reply.text.lower()
    assert c.chat.call_count == 3


# --- fallback path -----------------------------------------------------

def test_fallback_after_both_validation_failures():
    r = _retriever(score=0.55, margin=0.20)
    c = _client_sequence(
        "YES",
        "আমার পিন রিসেট।",
        "আরো বাংলা টেক্সট।",
    )
    reply = handle_query("amar pin kivabe reset korbo?",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "fallback"
    assert reply.text == ANSWER_BANK[1]["banglish"]


def test_fallback_when_generation_raises():
    r = _retriever(score=0.55, margin=0.20)
    c = MagicMock()
    c.chat.side_effect = [
        MagicMock(text="YES"),
        LLMUnavailable("down"),
        LLMUnavailable("down again"),
    ]
    reply = handle_query("amar pin kivabe reset korbo?",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "fallback"
    assert reply.text == ANSWER_BANK[1]["banglish"]


def test_fallback_disabled_returns_handoff():
    cfg = dict(CFG)
    cfg["validators"] = dict(CFG["validators"])
    cfg["validators"]["fallback_to_verified"] = False
    r = _retriever(score=0.55, margin=0.20)
    c = _client_sequence(
        "YES",
        "আমার পিন রিসেট।",
        "আরো বাংলা টেক্সট।",
    )
    reply = handle_query("amar pin kivabe reset korbo?",
                         r, c, cfg, FAQ, ANSWER_BANK, IDK)
    assert reply.path == "handoff"


# --- trace populated ---------------------------------------------------

def test_trace_records_each_stage():
    r = _retriever(score=0.90, margin=0.20)
    c = _client_returns("amar pin reset kore dilam.")
    reply = handle_query("amar pin kivabe reset korbo?",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    names = [s.name for s in reply.trace.stages]
    assert "normalize" in names
    assert "detect_language" in names
    assert "policy" in names
    assert "retrieve" in names
    assert "gate" in names
    assert "generate_1" in names
    assert reply.trace.final_path == "generated"
    assert reply.trace.language == "banglish"


def test_trace_sink_called_when_provided(tmp_path):
    from nb_assist.trace import TraceSink
    import json
    p = tmp_path / "trace.jsonl"
    sink = TraceSink(str(p))
    r = _retriever(score=0.90, margin=0.20)
    c = _client_returns("amar pin reset kore dilam.")
    handle_query("amar pin kivabe reset korbo?",
                 r, c, CFG, FAQ, ANSWER_BANK, IDK, trace_sink=sink)
    assert p.exists()
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["final_path"] == "generated"


def test_latency_nonnegative():
    r = _retriever(score=0.90, margin=0.20)
    c = _client_returns("amar pin reset kore dilam.")
    reply = handle_query("amar pin",
                         r, c, CFG, FAQ, ANSWER_BANK, IDK)
    assert reply.latency_s >= 0.0