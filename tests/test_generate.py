"""Unit tests for the generator. Mocked OllamaClient."""

from unittest.mock import MagicMock

import pytest

from nb_assist.generate import generate, _GenerationFailure
from nb_assist.llm import LLMUnavailable


CFG = {
    "models": {"primary": "gemma3:4b", "embedding": "bge-m3"},
    "generation": {"temperature": 0.0, "max_tokens": 200, "num_fewshot": 2},
}


def _client_returns(text):
    c = MagicMock()
    c.chat.return_value = MagicMock(text=text)
    return c


def test_generate_returns_text_and_path():
    c = _client_returns("amar pin reset kore dilam")
    r = generate(c, CFG, "amar pin kivabe reset korbo?",
                 "amar pin reset kore dilam NB Go app theke.", "banglish")
    assert r.text == "amar pin reset kore dilam"
    assert r.path == "generated"
    assert r.attempt == 1
    assert r.latency_s >= 0.0


def test_generate_includes_script_rule_in_prompt():
    c = _client_returns("ok")
    generate(c, CFG, "amar card ki?", "answer here", "banglish")
    payload = c.chat.call_args.kwargs
    messages = payload["messages"]
    system = messages[0]["content"]
    assert "Latin letters" in system
    assert "Bangla script" in system


def test_generate_limits_fewshot():
    cfg = dict(CFG)
    cfg["generation"] = dict(CFG["generation"])
    cfg["generation"]["num_fewshot"] = 1
    c = _client_returns("ok")
    generate(c, cfg, "x", "y", "en")
    messages = c.chat.call_args.kwargs["messages"]
    # 1 system + 1 user/assistant pair (2) + 1 live user = 4
    assert len(messages) == 4


def test_generate_failure_raises_typed_error():
    c = MagicMock()
    c.chat.side_effect = LLMUnavailable("down")
    with pytest.raises(_GenerationFailure):
        generate(c, CFG, "q", "a", "en")


def test_generate_passes_temperature_and_max_tokens():
    c = _client_returns("ok")
    generate(c, CFG, "q", "a", "en")
    opts = c.chat.call_args.kwargs
    assert opts["temperature"] == 0.0
    assert opts["max_tokens"] == 200