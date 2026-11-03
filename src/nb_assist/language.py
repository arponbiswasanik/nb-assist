"""Language detection and text normalization.

Two responsibilities:
  - normalize(): Unicode NFC + lowercase Latin + Bangla digits to ASCII
  - detect_language(): deterministic bn / banglish / en by code

Code decides language, not the model. Reasons:
  - deterministic and testable
  - free (no model call)
  - the choice drives downstream retrieval and the script of the
    generated reply, so it must not depend on a small model that may
    drift.

The lexicon approach for Banglish is intentionally simple. Spelling
variants (kivabe / kibhabe / kemne) are handled by the normalizer and
retrieval layer; here we only need enough signal to route the query.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

Lang = Literal["en", "bn", "banglish"]

# Bangla Unicode block: U+0980 .. U+09FF (includes digits, vowels,
# consonants, signs). We count any codepoint in this range as Bangla.
_BN_RANGE = re.compile(r"[\u0980-\u09ff]")

# Bangla digits ০-৯ → ASCII 0-9.
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# English function words. Kept here, not in config, because they
# are stable and rarely need tuning. If they ever do, move to config.
_EN_FNWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "what", "how", "when", "where", "why", "who", "which", "my", "me",
    "i", "you", "we", "can", "will", "would", "should", "if", "and",
    "or", "to", "of", "for", "with", "in", "on", "at", "by",
}

_LATIN_WORD = re.compile(r"[A-Za-z]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Unicode NFC, lowercase Latin, Bangla digits to ASCII, trim, collapse spaces.

    Bangla characters are NOT lowercased (case is not a Bangla concept)
    and NOT transliterated (script must survive to the retrieval layer).
    """
    if not text:
        return ""
    # NFC first: composes combining chars, so া + ক becomes কা.
    text = unicodedata.normalize("NFC", text)
    # Lowercase only Latin letters. Bangla is unaffected by str.lower
    # in Python, so we can call it on the whole string safely.
    text = text.lower()
    text = text.translate(_BN_DIGITS)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def _bangla_share(text: str) -> float:
    if not text:
        return 0.0
    bn = sum(1 for _ in _BN_RANGE.finditer(text))
    total = len(text)
    return bn / total if total else 0.0


def detect_language(text: str, config: dict) -> Lang:
    """Return one of 'en', 'bn', 'banglish'.

    config keys used:
      language.bn_char_threshold  (float, 0..1)
      language.ties_to            ('en' | 'banglish' | 'bn')
      language.banglish_lexicon    (list[str])
    """
    norm = normalize(text)
    if not norm:
        return config.get("language", {}).get("ties_to", "en")

    lang_cfg = config.get("language", {})

    # Stage 1: dominant Bangla script?
    if _bangla_share(norm) >= lang_cfg.get("bn_char_threshold", 0.30):
        return "bn"

    # Stage 2: Latin only — Banglish vs English.
    words = [w.lower() for w in _LATIN_WORD.findall(norm)]
    if not words:
        return lang_cfg.get("ties_to", "en")

    bn_lex = set(w.lower() for w in lang_cfg.get("banglish_lexicon", []))
    bn_hits = sum(1 for w in words if w in bn_lex)
    en_hits = sum(1 for w in words if w in _EN_FNWORDS)

    if bn_hits > en_hits:
        return "banglish"
    if en_hits > bn_hits:
        return "en"
    return lang_cfg.get("ties_to", "en")