"""Test Unit of eval/metrics.py"""

from eval.metrics import compute_metrics


def _o(id_, lang, exp, pred_path, gold_id=None, top1=None,
        correct=None, lang_match=True, lat=1.0):
    return {
        "id": id_, "language": lang, "expected_decision": exp,
        "gold_faq_id": gold_id, "predicted_path": pred_path,
        "retrieval_top1_id": top1, "answer_correctness": correct,
        "language_match": lang_match, "latency_s": lat,
    }


def test_empty_block_returns_zeros():
    r = compute_metrics([])
    assert r.overall["total"] == 0
    assert r.overall["retrieval_top1_accuracy"] == 0.0


def test_perfect_in_scope():
    rows = [
        _o("1", "en", "answer", "generated", gold_id=5, top1=5,
           correct=True, lat=2.0),
        _o("2", "en", "answer", "generated", gold_id=8, top1=8,
           correct=True, lat=3.0),
    ]
    r = compute_metrics(rows)
    assert r.overall["retrieval_top1_accuracy"] == 100.0
    assert r.overall["answer_correctness"] == 100.0
    assert r.overall["idk_recall"] == 0.0  # স্কোপের বাইরে নেই
    assert r.overall["unsafe_answers_count"] == 0


def test_perfect_handoff():
    rows = [
        _o("3", "bn", "handoff", "handoff", lat=0.5),
        _o("4", "banglish", "handoff", "handoff", lat=0.4),
    ]
    r = compute_metrics(rows)
    assert r.overall["idk_recall"] == 100.0
    assert r.overall["false_handoff_rate"] == 0.0


def test_unsafe_answer_counted():
    rows = [
        _o("3", "bn", "handoff", "generated", top1=18, correct=True,
           lat=2.0),
    ]
    r = compute_metrics(rows)
    assert r.overall["unsafe_answers_count"] == 1
    assert r.overall["unsafe_answers_pct"] == 100.0


def test_false_handoff():
    rows = [
        _o("1", "en", "answer", "handoff", gold_id=5, top1=5, lat=0.3),
        _o("2", "en", "answer", "generated", gold_id=8, top1=8,
           correct=True, lat=2.0),
    ]
    r = compute_metrics(rows)
    assert r.overall["false_handoff_rate"] == 50.0
    assert r.overall["answer_correctness"] == 100.0


def test_per_language_split():
    rows = [
        _o("1", "en", "answer", "generated", gold_id=5, top1=5,
           correct=True, lat=2.0),
        _o("2", "bn", "answer", "handoff", gold_id=8, top1=8, lat=0.4),
    ]
    r = compute_metrics(rows)
    assert set(r.per_language.keys()) == {"en", "bn"}
    assert r.per_language["en"]["false_handoff_rate"] == 0.0
    assert r.per_language["bn"]["false_handoff_rate"] == 100.0


def test_latency_percentiles():
    rows = [
        _o(str(i), "en", "answer", "generated", gold_id=1, top1=1,
           correct=True, lat=float(i)) for i in range(1, 21)
    ]
    r = compute_metrics(rows)
    assert r.overall["latency_p50_s"] <= r.overall["latency_p95_s"]
    assert r.overall["latency_mean_s"] > 0.0


def test_path_mix_counts_each_path():
    rows = [
        _o("1", "en", "answer", "generated", gold_id=1, top1=1, correct=True),
        _o("2", "en", "answer", "retry", gold_id=2, top1=2, correct=True),
        _o("3", "en", "answer", "fallback", gold_id=3, top1=3, correct=True),
        _o("4", "en", "handoff", "handoff"),
    ]
    r = compute_metrics(rows)
    assert r.overall["path_mix"]["generated"] == 1
    assert r.overall["path_mix"]["retry"] == 1
    assert r.overall["path_mix"]["fallback"] == 1
    assert r.overall["path_mix"]["handoff"] == 1


def test_language_match_overall_and_split():
    rows = [
        _o("1", "en", "answer", "generated", gold_id=1, top1=1,
           correct=True, lang_match=True),
        _o("2", "bn", "answer", "generated", gold_id=2, top1=2,
           correct=True, lang_match=False),
    ]
    r = compute_metrics(rows)
    assert r.overall["language_match"] == 50.0
    assert r.per_language["en"]["language_match"] == 100.0
    assert r.per_language["bn"]["language_match"] == 0.0