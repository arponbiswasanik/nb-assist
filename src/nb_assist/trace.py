"""Trace logging per query.

Per-stage scores, decision, path, latency. Privacy by default: digits
are masked and raw text is not stored unless explicitly enabled.
Output as JSONL when a path is configured; the Trace object is also
returned to the caller for UI display.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_DIGIT = re.compile(r"\d")


def mask_digits(text: str) -> str:
    return _DIGIT.sub("X", text) if text else text


@dataclass
class StageTrace:
    name: str
    duration_s: float
    result: dict[str, Any]


@dataclass
class Trace:
    query: str
    normalized: str
    language: str
    stages: list[StageTrace] = field(default_factory=list)
    policy_handoff: bool = False
    policy_rule: str | None = None
    retrieval_top: list | None = None
    gate_zone: str | None = None
    gate_verifier: str | None = None
    final_path: str | None = None
    final_text: str | None = None
    total_duration_s: float = 0.0

    def add_stage(self, name: str, duration_s: float, result: dict) -> None:
        self.stages.append(StageTrace(name, duration_s, result))

    def to_dict(self, mask_digits_flag: bool = True) -> dict:
        d = asdict(self)
        if mask_digits_flag:
            d["query"] = mask_digits(self.query)
            d["normalized"] = mask_digits(self.normalized)
            if d.get("final_text"):
                d["final_text"] = mask_digits(self.final_text)
        return d


class TraceSink:
    """Append traces to a JSONL file when configured."""

    def __init__(self, path: str | None = None,
                 mask_digits_flag: bool = True,
                 store_raw_text: bool = False):
        self.path = Path(path) if path else None
        self.mask_digits_flag = mask_digits_flag
        self.store_raw_text = store_raw_text
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: Trace) -> None:
        if not self.path:
            return
        mask = self.mask_digits_flag and not self.store_raw_text
        record = trace.to_dict(mask_digits_flag=mask)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")