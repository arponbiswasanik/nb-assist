"""Unit tests for the deterministic validators."""

from nb_assist.validate import (
    script_check, fact_check, extra_check, length_check, loop_check,
    run_all,
)


CFG = {
    "validators": {
        "script_check": True,
        "fact_check": True,
        "extra_content_check": True,
        "length_check": True,
        "loop_check": True,
    }
}


# --- script_check -------------------------------------------------------

def test_script_banglish_clean_passes():
    ok, _ = script_check("amar card reset kore dilam", "banglish")
    assert ok


def test_script_banglish_with_bn_chars_fails():
    ok, reason = script_check("amar কার্ড reset kore dilam", "banglish")
    assert not ok
    assert "Bangla script" in reason


def test_script_english_clean_passes():
    ok, _ = script_check("Open the NB Go app and reset your PIN.", "en")
    assert ok


def test_script_english_with_bn_fails():
    ok, _ = script_check("Open the অ্যাপ and reset.", "en")
    assert not ok


def test_script_bangla_clean_passes():
    ok, _ = script_check("NB Go অ্যাপ খুলুন এবং পিন রিসেট করুন।", "bn")
    assert ok


def test_script_bangla_with_glued_token_fails():
    ok, _ = script_check("আমার PINভুলে গেছে।", "bn")
    assert not ok


def test_script_empty_fails():
    ok, _ = script_check("   ", "en")
    assert not ok


# --- fact_check ---------------------------------------------------------

def test_fact_check_numbers_match_passes():
    src = "The annual fee is BDT 500 from the second year."
    ok, _ = fact_check("The fee is BDT 500.", src)
    assert ok


def test_fact_check_invented_number_fails():
    src = "The annual fee is BDT 500."
    ok, reason = fact_check("The fee is BDT 999.", src)
    assert not ok
    assert "999" in reason


def test_fact_check_phone_from_source_passes():
    src = "Call 16999 to block the card."
    ok, _ = fact_check("Please call 16999.", src)
    assert ok


# --- extra_check --------------------------------------------------------

def test_extra_check_aligned_passes():
    src = "You need your NID or passport and two photos."
    ok, _ = extra_check("You need your NID or passport and two photos.", src)
    assert ok


def test_extra_check_invented_sentence_fails():
    src = "The minimum balance is BDT 1,000."
    ok, _ = extra_check(
        "The minimum balance is BDT 1,000. Gold is currently BDT 8,500 per vori.",
        src,
    )
    assert not ok


# --- length_check -------------------------------------------------------

def test_length_short_passes():
    ok, _ = length_check("Reset your PIN via the NB Go app settings menu.", "en")
    assert ok


def test_length_too_short_fails():
    ok, reason = length_check("ok", "en")
    assert not ok
    assert "short" in reason


def test_length_too_long_fails():
    ok, reason = length_check("x" * 700, "en")
    assert not ok
    assert "long" in reason


# --- loop_check ---------------------------------------------------------

def test_loop_check_clean_passes():
    text = "Open the app, go to settings, then card, then reset PIN."
    ok, _ = loop_check(text)
    assert ok


def test_loop_check_repeated_window_fails():
    text = "card card card card card card card card card card card card"
    ok, reason = loop_check(text)
    assert not ok
    assert "repeated" in reason


# --- run_all ------------------------------------------------------------

def test_run_all_clean_passes():
    src = "The annual fee is BDT 500 from the second year."
    reply = "The annual fee is BDT 500 from the second year."
    ok, _ = run_all(reply, src, "en", CFG)
    assert ok


def test_run_all_first_failure_wins():
    src = "The annual fee is BDT 500."
    # Script + fact both fail; script is checked first.
    reply = "কার্ডের ফি BDT 999।"
    ok, reason = run_all(reply, src, "en", CFG)
    assert not ok
    assert reason.startswith("[")