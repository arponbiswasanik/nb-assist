"""Deterministic validators for generated replies.

Each validator returns (passed: bool, reason: str | None). The caller
runs them in order; the first failure short-circuits and the caller
either retries once or falls back to the verified answer verbatim.

Checks:
  - script_check     reply script matches the requested language
  - fact_check       numbers/amounts/phones in reply appear in source
  - extra_check      no low-overlap sentence (catches invented lines)
  - length_check     reply not too short and not too long
  - loop_check       no repeated trigrams (catches generation loops)
"""

from __future__ import annotations

import re

from .language import Lang


_BN_CHAR = re.compile(r"[\u0980-\u09ff]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")

# Numbers, currency, phone, percent we expect to round-trip.
# Bangla digits are already normalized to ASCII in normalize(); the
# generator's few-shot examples use ASCII digits, so we only need to
# check ASCII digits here.
_NUMBER = re.compile(r"(?:BDT\s*)?\d[\d,]*%?")
_PHONE = re.compile(r"\b\d{3,5}\b")

# Sentence split: cheap but adequate for short banking replies.
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")
_TRIGRAM_WINDOW = 8


def script_check(reply: str, lang: Lang) -> tuple[bool, str | None]:
    """Reply script matches the requested language."""
    if not reply.strip():
        return False, "empty reply"
    has_bn = bool(_BN_CHARS.search(reply))
    has_latin = bool(_LATIN.search(reply))
    if lang == "banglish":
        if has_bn:
            return False, "banglish reply contains Bangla script"
        if not has_latin:
            return False, "banglish reply has no Latin letters"
    elif lang == "en":
        if has_bn:
            return False, "english reply contains Bangla script"
    elif lang == "bn":
        if not has_bn:
            return False, "bangla reply has no Bangla script"
        # Allow embedded Latin (NB Go, BDT, NID) but flag glued mixed
        # tokens like 'রিসেটreset' or 'PINভুলে'.
        # Crude check: a Bangla char immediately followed by a Latin
        # letter, or vice versa, inside the same word.
        if re.search(r"[\u0980-\u09ff][A-Za-z]|[A-Za-z][\u0980-\u09ff]", reply):
            return False, "mixed-script glued token in bangla reply"
    return True, None


def fact_check(reply: str, source: str) -> tuple[bool, str | None]:
    """Every number in the reply appears in the source (or a small whitelist)."""
    nums_reply = set(n.replace(" ", "").replace(",", "")
                     for n in _NUMBER.findall(reply))
    nums_source = set(n.replace(" ", "").replace(",", "")
                      for n in _NUMBER.findall(source))
    # Whitelist: trivial small numbers that often appear in templates
    # and few-shot ("1", "2") — keep narrow.
    whitelist = {"1", "2", "3"}
    for n in nums_reply:
        if n in whitelist:
            continue
        # Phone-like numbers: allow if all source digits contain the
        # same sequence, e.g. "16999" inside source "Call 16999".
        if n not in nums_source and n not in source:
            return False, f"number not in source: {n!r}"
    return True, None


def extra_check(reply: str, source: str) -> tuple[bool, str | None]:
    """No reply sentence with very low token overlap with the source."""
    src_tokens = set(re.findall(r"\w+", source.lower()))
    if not src_tokens:
        return True, None
    sentences = [s for s in _SENT_SPLIT.split(reply) if len(s) > 12]
    for s in sentences:
        toks = re.findall(r"\w+", s.lower())
        if not toks:
            continue
        overlap = sum(1 for t in toks if t in src_tokens) / len(toks)
        # A reply sentence that shares almost no tokens with the source
        # is likely an invented line.
        if overlap < 0.30:
            return False, f"low-overlap sentence: {s[:40]!r}"
    return True, None


def length_check(reply: str, lang: Lang) -> tuple[bool, str | None]:
    """Reply not too short and not too long."""
    n = len(reply.strip())
    if n < 8:
        return False, "reply too short"
    if n > 600:
        return False, "reply too long"
    return True, None


def loop_check(reply: str) -> tuple[bool, str | None]:
    """No repeated trigram window (catches degenerate generation loops)."""
    words = re.findall(r"\w+", reply)
    if len(words) < _TRIGRAM_WINDOW:
        return True, None
    seen = set()
    for i in range(len(words) - _TRIGRAM_WINDOW + 1):
        gram = tuple(words[i:i + _TRIGRAM_WINDOW])
        if gram in seen:
            return False, f"repeated window: {' '.join(gram[:3])}..."
        seen.add(gram)
    return True, None


def run_all(reply: str, source: str, lang: Lang,
            config: dict) -> tuple[bool, str | None]:
    """Run all enabled validators in order. First failure wins."""
    v = config.get("validators", {})
    checks: list[tuple[str, callable]] = []
    if v.get("script_check", True):
        checks.append(("script", lambda: script_check(reply, lang)))
    if v.get("fact_check", True):
        checks.append(("fact", lambda: fact_check(reply, source)))
    if v.get("extra_content_check", True):
        checks.append(("extra", lambda: extra_check(reply, source)))
    if v.get("length_check", True):
        checks.append(("length", lambda: length_check(reply, lang)))
    if v.get("loop_check", True):
        checks.append(("loop", lambda: loop_check(reply)))
    for name, fn in checks:
        passed, reason = fn()
        if not passed:
            return False, f"[{name}] {reason}"
    return True, None


# Re-export used by script_check; keep name singular for clarity.
_BN_CHARS = _BN_CHAR