from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_safe_blind.py"
SPEC = importlib.util.spec_from_file_location("evaluate_safe_blind", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_blind_fixture_is_balanced_and_unique():
    cases = MODULE.load_cases(Path(__file__).parent / "fixtures" / "safe_blind_v1.jsonl")

    assert len(cases) == 60
    assert sum(case["label"] == 0 for case in cases) == 30
    assert sum(case["label"] == 1 for case in cases) == 30
    assert len({case["id"] for case in cases}) == 60


def test_metrics_uses_expected_confusion_matrix():
    rows = [
        {"label": 0, "prediction": 0},
        {"label": 0, "prediction": 1},
        {"label": 1, "prediction": 0},
        {"label": 1, "prediction": 1},
        {"label": 1, "prediction": 1},
    ]

    result = MODULE.metrics(rows)

    assert result["confusion_matrix"] == [[1, 1], [1, 2]]
    assert result["risk_recall"] == pytest.approx(2 / 3)
    assert result["specificity"] == pytest.approx(1 / 2)
    assert result["risk_precision"] == pytest.approx(2 / 3)


def test_load_cases_rejects_duplicate_id(tmp_path):
    path = tmp_path / "bad.jsonl"
    row = '{"id":"x","label":0,"slice":"s","receiver_is_minor":false,"text":"t"}'
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate case id"):
        MODULE.load_cases(path)
