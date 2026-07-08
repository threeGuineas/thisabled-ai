from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.build_safe_hardcase_dataset import build_rows
from src.data.dedup import find_duplicate_indices

ROOT = Path(__file__).resolve().parents[1]


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", text.lower())


def test_hardcases_are_balanced_and_unique():
    rows = build_rows()
    assert len(rows) == 960
    assert sum(row["label"] == 0 for row in rows) == 480
    assert sum(row["label"] == 1 for row in rows) == 480
    assert len({_norm(row["text"]) for row in rows}) == len(rows)

    v2_rows = build_rows(include_v2=True)
    assert len(v2_rows) == 1440
    assert sum(row["label"] == 0 for row in v2_rows) == 640
    assert sum(row["label"] == 1 for row in v2_rows) == 800


def test_hardcases_do_not_overlap_frozen_blind_sets():
    rows = build_rows()
    blind = []
    for name in ("safe_blind_v1.jsonl", "safe_blind_v2.jsonl", "safe_blind_v3.jsonl"):
        path = ROOT / "tests/fixtures" / name
        blind.extend(json.loads(line)["text"] for line in path.read_text().splitlines() if line)

    blind_norm = {_norm(text) for text in blind}
    assert not [row for row in rows if _norm(row["text"]) in blind_norm]
    assert not find_duplicate_indices(blind, [row["text"] for row in rows], threshold=0.8)

    v2_rows = build_rows(include_v2=True)
    assert not find_duplicate_indices(blind, [row["text"] for row in v2_rows], threshold=0.8)
