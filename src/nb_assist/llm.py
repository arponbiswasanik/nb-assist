"""Thin Ollama HTTP client with timeouts and typed errors.

Two endpoints:
  - /api/chat       for the generator and the gate verifier
  - /api/embed     for retrieval embeddings (bge-m3)

Models are configurable; nothing is hard-coded. Network errors map
to a small set of typed exceptions so callers can fail closed without
catching broad Exception.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests


class LLMError(Exception):
    """Base for all LLM client errors."""


class LLMTimeout(LLMError):
    """Request exceeded the configured timeout."""


class LLMUnavailable(LLMError):
    """Ollama is not running, refused, or returned 5xx."""


class LLMGenerationError(LLMError):
    """Ollama returned a non-2xx (non-5xx) response or malformed body."""


@dataclass
class ChatResult:
    text: str
    prompt_eval_duration_s: float | None
    load_duration_s: float | None
    total_duration_s: float | None
    done: bool


def _ns_to_s(v):
    """Ollama returns durations in nanoseconds."""
    return v / 1e9 if v else None


class OllamaClient:
    """Minimal client over Ollama's local HTTP API."""

    def __init__(self, config: dict):
        o = config.get("ollama", {})
        self.host = o.get("host", "127.0.0.1")
        self.port = o.get("port", 11434)
        self.timeout_s = o.get("timeout_s", 60)
        self.keep_alive_s = o.get("keep_alive_s", 300)
        self.base_url = f"http://{self.host}:{self.port}"

    def chat(self, model: str, messages: list[dict], *,
             temperature: float = 0.0, max_tokens: int | None = None) -> ChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens if max_tokens is not None else -1,
            },
            "keep_alive": f"{self.keep_alive_s}s",
        }
        body = self._post("/api/chat", payload).json()
        msg = (body.get("message") or {}).get("content")
        if msg is None:
            raise LLMGenerationError(f"unexpected chat shape: {body}")
        return ChatResult(
            text=msg,
            prompt_eval_duration_s=_ns_to_s(body.get("prompt_eval_duration")),
            load_duration_s=_ns_to_s(body.get("load_duration")),
            total_duration_s=_ns_to_s(body.get("total_duration")),
            done=bool(body.get("done", False)),
        )

    def embed(self, model: str, text) -> list[list[float]]:
        """Return one vector per input. Single string -> list of one vector."""
        inputs = [text] if isinstance(text, str) else list(text)
        if not inputs:
            return []
        payload = {"model": model, "input": inputs}
        body = self._post("/api/embed", payload).json()
        vectors = body.get("embeddings") or []
        if not vectors:
            raise LLMGenerationError(f"no embeddings in response: {body}")
        return vectors

    def warmup(self, model: str) -> ChatResult:
        """Pre-load the model so the first real user call is warm.

        A tiny prompt; we don't care about the reply.
        """
        return self.chat(model, [{"role": "user", "content": "ping"}],
                         max_tokens=1)

    def _post(self, path: str, payload: dict) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
        except requests.ConnectionError as e:
            raise LLMUnavailable(
                f"cannot reach ollama at {self.base_url}: {e}") from e
        except requests.Timeout as e:
            raise LLMTimeout(
                f"ollama timeout after {self.timeout_s}s") from e
        if resp.status_code >= 500:
            raise LLMUnavailable(
                f"ollama 5xx: {resp.status_code} {resp.text[:200]}")
        if resp.status_code >= 400:
            raise LLMGenerationError(
                f"ollama {resp.status_code}: {resp.text[:200]}")
        return resp