"""Unit tests for trace logging."""

import json

from nb_assist.trace import Trace, TraceSink, mask_digits


def test_mask_digits_replaces_all():
    assert mask_digits("call 16999") == "call XXXXX"
    assert mask_digits("amar pin") == "amar pin"
    assert mask_digits("") == ""


def test_trace_to_dict_masks_by_default():
    t = Trace(query="call 16999", normalized="call 16999", language="en")
    d = t.to_dict()
    assert "XXXXX" in d["query"]
    assert "XXXXX" in d["normalized"]


def test_trace_to_dict_unmasked_when_disabled():
    t = Trace(query="call 16999", normalized="call 16999", language="en")
    d = t.to_dict(mask_digits_flag=False)
    assert d["query"] == "call 16999"


def test_trace_to_dict_masks_final_text():
    t = Trace(query="q", normalized="q", language="en", final_text="BDT 500")
    d = t.to_dict()
    assert "XXX" in d["final_text"]


def test_trace_add_stage():
    t = Trace(query="q", normalized="q", language="en")
    t.add_stage("normalize", 0.001, {})
    assert len(t.stages) == 1
    assert t.stages[0].name == "normalize"


def test_trace_sink_writes_jsonl(tmp_path):
    p = tmp_path / "trace.jsonl"
    sink = TraceSink(str(p))
    t = Trace(query="call 16999", normalized="call 16999", language="en",
              final_path="generated", final_text="call 16999")
    sink.write(t)
    sink.write(t)
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert "XXXXX" in rec["query"]
    assert rec["final_path"] == "generated"


def test_trace_sink_no_path_is_no_op():
    sink = TraceSink(None)
    t = Trace(query="x", normalized="x", language="en")
    sink.write(t)  # must not raise


def test_trace_sink_store_raw_text_keeps_digits(tmp_path):
    p = tmp_path / "trace.jsonl"
    sink = TraceSink(str(p), store_raw_text=True)
    t = Trace(query="call 16999", normalized="call 16999", language="en")
    sink.write(t)
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["query"] == "call 16999"


def test_trace_sink_creates_parent_dir(tmp_path):
    p = tmp_path / "subdir" / "trace.jsonl"
    sink = TraceSink(str(p))
    t = Trace(query="x", normalized="x", language="en")
    sink.write(t)
    assert p.exists()