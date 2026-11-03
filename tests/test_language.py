"""Unit tests for language detection and normalization.

Inline fixtures only — no external test data files. Keeps the
language layer testable in isolation.
"""

import pytest

from nb_assist.language import normalize, detect_language


# Minimal config used by all tests. Mirrors config.yaml's language
# section. If config.yaml's lexicon grows, mirror it here too.
CFG = {
    "language": {
        "bn_char_threshold": 0.30,
        "ties_to": "en",
        "banglish_lexicon": [
            "ami", "amar", "koto", "kibhabe", "kivabe", "kemne",
            "korbo", "korchi", "ki", "jay", "hobe", "lage",
            "ache", "nai", "ta", "er",
        ],
    }
}


# --- normalize ----------------------------------------------------------

def test_normalize_empty():
    assert normalize("") == ""


def test_normalize_bangla_digits_to_ascii():
    assert normalize("আমার ব্যালেন্স ৫০০০ টাকা") == "আমার ব্যালেন্স 5000 টাকা"


def test_normalize_lowercase_latin():
    assert normalize("How Do I Reset My PIN?") == "how do i reset my pin?"


def test_normalize_collapse_whitespace():
    assert normalize("  amar   card   hariye   gese  ") == "amar card hariye gese"


# --- detect_language ----------------------------------------------------

def test_detect_bangla_script():
    assert detect_language("আমার কার্ড হারিয়ে গেছে, এখন কী করব?", CFG) == "bn"


def test_detect_english():
    assert detect_language("How do I reset my PIN?", CFG) == "en"


def test_detect_banglish():
    assert detect_language("amar PIN kivabe reset korbo?", CFG) == "banglish"


def test_detect_banglish_with_variant_spelling():
    # "kemne" is a known variant; should still register as Banglish.
    assert detect_language("card reset kemne korbo?", CFG) == "banglish"


def test_detect_mixed_script_dominant_bangla():
    # One short English word inside a Bangla sentence stays Bangla.
    assert detect_language("আমার PIN ভুলে গেছে, কীভাবে reset করব?", CFG) == "bn"


def test_detect_empty_string_default():
    assert detect_language("", CFG) == "en"


def test_detect_ties_go_to_en():
    # Pure noun sentence, no function words on either side.
    assert detect_language("credit card nirbhor bank", CFG) == "en"


def test_detect_mixed_latin_bangla_low_bangla_share():
    # Bangla share below threshold but present — still Latin routing.
    out = detect_language("amar card reset lagbe ASAP please", CFG)
    assert out in ("banglish", "en")