"""Guarded generation.

The model only phrases the reply. It sees one verified answer in
the customer's language as the only source, plus the customer's
question and a per-language script instruction. Temperature 0,
bounded length. Two short few-shot examples per language nudge the
script in the right direction without giving the model room to
invent facts.

The validators in validate.py inspect every reply before it is sent.
On failure, the caller retries once, then falls back to the verified
answer verbatim. The model never gets a second chance to phrase the
same fact differently without a fresh prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from .language import Lang
from .llm import OllamaClient, LLMError


@dataclass
class GenerationResult:
    text: str
    path: str            # 'generated' | 'fallback' | 'retry'
    attempt: int         # 1 or 2
    latency_s: float


# Per-language script instruction. Banglish is Latin letters only.
_SCRIPT_RULE = {
    "en":        "Reply in English using Latin letters.",
    "bn":        "Reply in Bangla script only. Do not use Latin letters.",
    "banglish":  "Reply in Banglish only: Bangla written in Latin letters. "
                 "Do not use Bangla script.",
}

# Two few-shot examples per language. Each is (question, answer).
# Numbers and key facts are preserved verbatim; the model sees the
# style it should mimic.
_FEWSHOT = {
    "en": [
        ("How do I reset my card PIN?",
         "Open the NB Go app, go to Settings, then Card, then Reset PIN. "
         "You can also reset it at any Nirbhor Bank ATM using an OTP sent "
         "to your registered number."),
        ("What is the annual fee for the debit card?",
         "The first year is free. From the second year the annual fee is "
         "BDT 500."),
    ],
    "bn": [
        ("কার্ড পিন কীভাবে রিসেট করব?",
         "NB Go অ্যাপ খুলুন, সেটিংসে যান, তারপর কার্ড, তারপর রিসেট পিন। "
         "আপনি যেকোনো নির্ভর ব্যাংক এটিএম থেকে আপনার রেজিস্টার্ড নম্বরে পাঠানো "
         "OTP ব্যবহার করেও রিসেট করতে পারবেন।"),
        ("ডেবিট কার্ডের বার্ষিক ফি কত?",
         "প্রথম বছর ফ্রি। দ্বিতীয় বছর থেকে বার্ষিক ফি BDT 500।"),
    ],
    "banglish": [
        ("amar card PIN kivabe reset korbo?",
         "NB Go app khulin, settings e jan, tarpor card, tarpor reset PIN. "
         "Apni yekono Nirbhor Bank ATM theke apnar registered number e "
         "pathano OTP use koreo reset korte parben."),
        ("debit card er annual fee koto?",
         "Prothom bochor free. Ditiyo bochor theke bashorshik fee BDT 500."),
    ],
}


def _build_prompt(query: str, verified_answer: str, lang: Lang,
                  num_fewshot: int) -> list[dict]:
    """Assemble messages: system, few-shot pairs, then the live turn."""
    rule = _SCRIPT_RULE.get(lang, _SCRIPT_RULE["en"])
    system = (
        "You are a banking FAQ assistant. You answer only from the "
        "provided source text. Do not add facts that are not in the source. "
        "Keep every number, currency amount, phone number, date, and "
        "percentage exactly as written. Be concise and direct. "
        + rule
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    shots = _FEWSHOT.get(lang, [])[:num_fewshot]
    for q, a in shots:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({
        "role": "user",
        "content": (
            f"Source (the only allowed source of facts):\n"
            f"{verified_answer}\n\n"
            f"Customer question: {query}\n\n"
            f"Reply in {lang} using only the source above."
        ),
    })
    return messages


def generate(client: OllamaClient, config: dict, query: str,
             verified_answer: str, lang: Lang) -> GenerationResult:
    """Call the primary model once. Validation and retry live in the caller."""
    import time
    g_cfg = config.get("generation", {})
    model = config["models"]["primary"]
    temperature = g_cfg.get("temperature", 0.0)
    max_tokens = g_cfg.get("max_tokens", 200)
    num_fewshot = g_cfg.get("num_fewshot", 2)

    messages = _build_prompt(query, verified_answer, lang, num_fewshot)
    t0 = time.perf_counter()
    try:
        res = client.chat(model, messages=messages, temperature=temperature,
                          max_tokens=max_tokens)
        latency = time.perf_counter() - t0
        return GenerationResult(text=res.text or "", path="generated",
                                attempt=1, latency_s=latency)
    except LLMError as e:
        latency = time.perf_counter() - t0
        # Caller will fall back to the verified answer verbatim.
        raise _GenerationFailure(str(e), latency_s=latency) from e


class _GenerationFailure(Exception):
    def __init__(self, msg: str, latency_s: float = 0.0):
        super().__init__(msg)
        self.latency_s = latency_s