"""Unit tests for the Ollama client, with mocked HTTP.

No real Ollama is contacted. Each test patches requests.post to
return a fixed response shape.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from nb_assist.llm import (
    OllamaClient, LLMUnavailable, LLMTimeout, LLMGenerationError,
)


CFG = {"ollama": {"host": "127.0.0.1", "port": 11434,
                  "timeout_s": 5, "keep_alive_s": 60}}


def _mock_response(status=200, json_body=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_body or {}
    r.text = ""
    return r


def test_chat_returns_text_and_timing():
    body = {
        "message": {"content": "hello"},
        "prompt_eval_duration": 1_000_000_000,
        "load_duration": 2_000_000_000,
        "total_duration": 3_500_000_000,
        "done": True,
    }
    with patch("nb_assist.llm.requests.post",
               return_value=_mock_response(200, body)):
        c = OllamaClient(CFG)
        r = c.chat("gemma3:4b", [{"role": "user", "content": "hi"}])
    assert r.text == "hello"
    assert r.prompt_eval_duration_s == 1.0
    assert r.load_duration_s == 2.0
    assert r.total_duration_s == 3.5
    assert r.done is True


def test_embed_returns_vectors_for_batch():
    body = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    with patch("nb_assist.llm.requests.post",
               return_value=_mock_response(200, body)):
        c = OllamaClient(CFG)
        v = c.embed("bge-m3", ["a", "b"])
    assert v == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_single_string_returns_one_vector():
    body = {"embeddings": [[0.5, 0.6]]}
    with patch("nb_assist.llm.requests.post",
               return_value=_mock_response(200, body)):
        c = OllamaClient(CFG)
        v = c.embed("bge-m3", "query")
    assert v == [[0.5, 0.6]]


def test_embed_empty_returns_empty():
    c = OllamaClient(CFG)
    assert c.embed("bge-m3", []) == []


def test_embed_missing_vectors_raises():
    body = {"embeddings": []}
    with patch("nb_assist.llm.requests.post",
               return_value=_mock_response(200, body)):
        c = OllamaClient(CFG)
        with pytest.raises(LLMGenerationError):
            c.embed("bge-m3", "x")


def test_connection_error_raises_unavailable():
    with patch("nb_assist.llm.requests.post",
               side_effect=requests.ConnectionError()):
        c = OllamaClient(CFG)
        with pytest.raises(LLMUnavailable):
            c.chat("gemma3:4b", [{"role": "user", "content": "x"}])


def test_timeout_raises_llm_timeout():
    with patch("nb_assist.llm.requests.post",
               side_effect=requests.Timeout()):
        c = OllamaClient(CFG)
        with pytest.raises(LLMTimeout):
            c.chat("gemma3:4b", [{"role": "user", "content": "x"}])


def test_5xx_raises_unavailable():
    with patch("nb_assist.llm.requests.post",
               return_value=_mock_response(503)):
        c = OllamaClient(CFG)
        with pytest.raises(LLMUnavailable):
            c.chat("gemma3:4b", [{"role": "user", "content": "x"}])


def test_4xx_raises_generation_error():
    with patch("nb_assist.llm.requests.post",
               return_value=_mock_response(404)):
        c = OllamaClient(CFG)
        with pytest.raises(LLMGenerationError):
            c.chat("gemma3:4b", [{"role": "user", "content": "x"}])


def test_warmup_uses_small_max_tokens():
    body = {"message": {"content": "ok"}, "done": True}
    with patch("nb_assist.llm.requests.post",
               return_value=_mock_response(200, body)) as mock_post:
        c = OllamaClient(CFG)
        r = c.warmup("gemma3:4b")
    assert r.text == "ok"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["options"]["num_predict"] == 1