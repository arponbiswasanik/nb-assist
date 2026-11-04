"""Online model comparison: run the 40 test questions through a hosted
frontier model with the full FAQ in context, temperature 0.

This uploads the FAQ and the 40 test questions to the online API. The
40 questions are the bank-provided sample data; the brief explicitly
allows using them.

We use Groq's OpenAI-compatible API. The model gets the whole FAQ as
context, with instructions to answer only from it and to say "I don't
know" when the FAQ does not cover the question, replying in the
customer's language.

Free-tier rate limits (8000 TPM for gpt-oss-120b) require pacing and
retry-on-429. With 40 calls at ~2000 tokens each, the run takes about
10-15 minutes.

Writes:
  results/online_compare_outcomes.csv
  results/online_compare_metrics.json

Usage:
  set GROQ_API_KEY=gsk_xxx
  python eval/online_compare.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.metrics import compute_metrics


# Groq API endpoint (OpenAI-compatible chat completions).
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

# Pacing between successful calls (seconds). 2s = ~30 calls/min,
# well under the 8000 TPM limit for ~2000-token calls.
INTER_CALL_DELAY_S = 2.0


def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _build_faq_block(faq_rows: list[dict]) -> str:
    """Format the full FAQ as a numbered list the model can reference."""
    lines = []
    for i, r in enumerate(faq_rows, 1):
        lines.append(f"[{i}] Topic: {r['topic']}")
        lines.append(f"    Q: {r['question']}")
        lines.append(f"    A: {r['answer']}")
    return "\n".join(lines)


def _build_prompt(faq_block: str, question: str) -> list[dict]:
    """System + user message with the full FAQ and the live question."""
    system = (
        "You are a banking FAQ assistant for Nirbhor Bank. You answer "
        "customer questions strictly from the FAQ provided below. "
        "If the customer's question is not directly answered by any "
        "FAQ entry, reply exactly with: "
        "\"I don't know, let me connect you to a person.\" "
        "Do not invent facts or use outside knowledge.\n\n"
        "Reply in the same language and script the customer used:\n"
        "- English (Latin script) -> English (Latin script)\n"
        "- Bangla script -> Bangla script\n"
        "- Banglish (Bangla in Latin letters) -> Banglish (Latin letters)\n\n"
        "Keep every number, currency amount, phone number, date, and "
        "percentage exactly as it appears in the FAQ.\n\n"
        "FAQ:\n"
        f"{faq_block}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _call_groq(api_key: str, model: str, messages: list[dict],
               timeout_s: int = 60, max_retries: int = 5) -> tuple[str, float]:
    """Return (text, latency_s). Retries on 429 with backoff."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 300,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None
    total_latency = 0.0
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        resp = requests.post(GROQ_URL, json=payload, headers=headers,
                             timeout=timeout_s)
        total_latency += time.perf_counter() - t0
        if resp.status_code == 200:
            body = resp.json()
            text = body["choices"][0]["message"]["content"] or ""
            return text.strip(), total_latency
        if resp.status_code == 429:
            # Parse the suggested wait time from the error message.
            wait = 12.0 * (attempt + 1)  # 12s, 24s, 36s, 48s, 60s
            try:
                msg = resp.json().get("error", {}).get("message", "")
                m = re.search(r"try again in ([\d.]+)s", msg)
                if m:
                    wait = float(m.group(1)) + 1.0
            except Exception:
                pass
            print(f"      (429 rate-limited, waiting {wait:.1f}s, "
                  f"attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            last_error = f"429 after {max_retries} retries"
            continue
        # Non-429 error: don't retry.
        raise RuntimeError(
            f"Groq {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError(last_error or "Groq retries exhausted")


def _detect_idk(reply: str) -> bool:
    """Heuristic: did the model say it doesn't know?"""
    if not reply:
        return False
    r = reply.lower()
    return ("i don't know" in r
            or "let me connect you" in r
            or "আমি এই বিষয়ে নিশ্চিত নই" in r
            or "নিশ্চিত নই" in r
            or "eta ami jani na" in r
            or "jani na" in r)


def _language_match(reply: str, lang: str) -> bool:
    if not reply:
        return False
    bn = bool(re.search(r"[\u0980-\u09ff]", reply))
    latin = bool(re.search(r"[A-Za-z]", reply))
    if lang == "banglish":
        return latin and not bn
    if lang == "en":
        return not bn
    if lang == "bn":
        return bn
    return True


def _key_facts_check(reply: str, source: str) -> bool:
    """True if every number in the reply appears in the source."""
    if not reply or not source:
        return False
    src_digits = set(re.findall(r"\d+", source))
    reply_digits = set(re.findall(r"\d+", reply))
    whitelist = {"1", "2", "3"}
    for d in reply_digits:
        if d in whitelist:
            continue
        if d not in src_digits:
            return False
    return True


def _detect_language(text: str) -> str:
    """Tiny mirror of the local detector for parity in the report."""
    bn_count = sum(1 for c in text if "\u0980" <= c <= "\u09ff")
    if bn_count / max(len(text), 1) >= 0.30:
        return "bn"
    bn_lex = {"ami", "amar", "koto", "kivabe", "kibhabe", "kemne",
              "korbo", "korchi", "ki", "jay", "hobe", "lage", "ache",
              "nai", "ta", "er"}
    en_lex = {"the", "is", "what", "how", "my", "do", "can", "i"}
    words = [w.lower() for w in re.findall(r"[a-z]+", text.lower())]
    bn_hits = sum(1 for w in words if w in bn_lex)
    en_hits = sum(1 for w in words if w in en_lex)
    if bn_hits > en_hits:
        return "banglish"
    return "en"


def _identify_faq_id(reply: str, faq_rows: list[dict]) -> int | None:
    """Which FAQ entry does the reply most likely come from?

    For the online comparison we use a coarse heuristic: the FAQ
    entry whose answer shares the most digits and longest word n-grams
    with the reply. Good enough for the comparison table.
    """
    if not reply:
        return None
    reply_digits = set(re.findall(r"\d+", reply))
    reply_words = set(re.findall(r"\w+", reply.lower()))
    best_id = None
    best_score = -1
    for r in faq_rows:
        a = r["answer"]
        a_digits = set(re.findall(r"\d+", a))
        a_words = set(re.findall(r"\w+", a.lower()))
        digit_overlap = len(reply_digits & a_digits)
        word_overlap = len(reply_words & a_words)
        score = digit_overlap * 3 + word_overlap
        if score > best_score:
            best_score = score
            best_id = int(r["id"])
    return best_id if best_score > 0 else None


def main() -> int:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Run:")
        print("  set GROQ_API_KEY=gsk_xxx")
        return 1

    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Paths in config.yaml are relative to the repo root (already
    # include the data/ prefix), so resolve against ROOT, not data_dir.
    faq_rows = _load_csv(ROOT / config["paths"]["faq"])
    test_rows = _load_csv(ROOT / config["paths"]["test_questions"])
    print(f"FAQ: {len(faq_rows)} entries")
    print(f"test questions: {len(test_rows)}")

    faq_block = _build_faq_block(faq_rows)
    print(f"FAQ context size: {len(faq_block)} chars")

    # Try the default model; fall back if it errors out.
    model_to_use = DEFAULT_MODEL
    print(f"model: {model_to_use}")

    outcomes = []
    for i, row in enumerate(test_rows, 1):
        qid = row.get("id", str(i))
        question = row["question"]
        messages = _build_prompt(faq_block, question)
        try:
            reply_text, latency = _call_groq(api_key, model_to_use,
                                               messages)
        except RuntimeError as e:
            print(f"  [{i:>3}/{len(test_rows)}] id={qid} ERROR {e}")
            outcomes.append({
                "id": qid, "language": _detect_language(question),
                "predicted_path": "error",
                "predicted_faq_id": None,
                "answer_correctness": False,
                "language_match": False,
                "latency_s": 0.0,
                "reply_excerpt": "",
            })
            continue

        is_idk = _detect_idk(reply_text)
        path = "handoff" if is_idk else "generated"
        lang = _detect_language(question)
        lang_match = _language_match(reply_text, lang)
        if not is_idk:
            top1 = _identify_faq_id(reply_text, faq_rows)
            source = faq_rows[top1 - 1]["answer"] if top1 else ""
            correct = _key_facts_check(reply_text, source)
        else:
            top1 = None
            correct = None

        outcomes.append({
            "id": qid,
            "language": lang,
            "predicted_path": path,
            "predicted_faq_id": top1,
            "answer_correctness": correct,
            "language_match": lang_match,
            "latency_s": round(latency, 3),
            "reply_excerpt": reply_text[:120],
        })
        print(f"  [{i:>3}/{len(test_rows)}] id={qid} lang={lang} "
              f"path={path} top1={top1} match={lang_match} "
              f"lat={latency:.2f}s")
        # Pace requests to stay under the free-tier TPM limit.
        time.sleep(INTER_CALL_DELAY_S)

    # Write outcomes CSV.
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "online_compare_outcomes.csv"
    fieldnames = list(outcomes[0].keys()) if outcomes else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(outcomes)
    print(f"\noutcomes -> {out_csv}")

    # For metrics, we need expected_decision and gold_faq_id. These
    # come from test_gold.csv, which the model did not see. We attach
    # them here just for the metrics computation.
    gold_path = ROOT / config["paths"]["test_gold"]
    if gold_path.exists():
        gold_rows = _load_csv(gold_path)
        gold_map = {r["id"]: r for r in gold_rows}
        for o in outcomes:
            g = gold_map.get(o["id"], {})
            o["expected_decision"] = g.get("expected_decision")
            o["gold_faq_id"] = g.get("gold_faq_id") or None
            o["retrieval_top1_id"] = o["predicted_faq_id"]
        report = compute_metrics(outcomes)
        out_json = results_dir / "online_compare_metrics.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"metrics  -> {out_json}")
        print("\n=== online (Groq) overall ===")
        for k, v in report.overall.items():
            if k == "path_mix":
                print(f"  path_mix: {v}")
            else:
                print(f"  {k}: {v}")
    else:
        print("test_gold.csv not found; skipping metrics computation")
        report = None

    # Side-by-side with local results if available.
    local_json = results_dir / "final_test_metrics.json"
    if local_json.exists() and report is not None:
        local = json.loads(local_json.read_text(encoding="utf-8"))
        print("\n=== local (gemma3:4b) vs online (Groq gpt-oss-120b) ===")
        keys = ["retrieval_top1_accuracy", "answer_correctness",
                "idk_recall", "false_handoff_rate", "unsafe_answers_count",
                "language_match", "latency_p50_s", "latency_p95_s"]
        print(f"  {'metric':<25} {'local':>12} {'online':>12}")
        for k in keys:
            l = local["overall"].get(k, "n/a")
            o = report.overall.get(k, "n/a")
            print(f"  {k:<25} {l!s:>12} {o!s:>12}")

    return 0


if __name__ == "__main__":
    sys.exit(main())