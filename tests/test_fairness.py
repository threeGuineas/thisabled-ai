"""EXP-3 회귀 테스트 — fairness 모듈."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.fairness import (
    DISABILITY_KEYWORDS,
    MIN_SAMPLES,
    UNSMILE_GROUPS,
    evaluate_disability_domain,
    evaluate_kold_groups,
    evaluate_unsmile_groups,
)


def test_unsmile_groups_constant_has_7():
    assert len(UNSMILE_GROUPS) == 7
    assert "여성/가족" in UNSMILE_GROUPS
    assert "성소수자" in UNSMILE_GROUPS


def test_min_samples_threshold():
    assert MIN_SAMPLES == 30  # 통계 신뢰 임계치


def test_evaluate_disability_returns_two_groups():
    df = pd.DataFrame(
        {
            "text": ["장애인 친구다", "보통 문장", "휠체어 타고", "그냥 텍스트"] * 20,
            "label": [0, 1, 2, 0] * 20,
            "source": ["unsmile"] * 80,
        }
    )
    y_pred = np.array([0, 1, 2, 0] * 20)
    result = evaluate_disability_domain(df, y_pred)
    assert "disability_yes" in result["groups"]
    assert "disability_no" in result["groups"]
    assert result["groups"]["disability_yes"]["n"] > 0
    assert result["groups"]["disability_no"]["n"] > 0


def test_evaluate_unsmile_groups_below_min_skipped():
    test_df = pd.DataFrame(
        {
            "source": ["unsmile"] * 5,
            "source_id": ["0", "1", "2", "3", "4"],
            "label": [0, 1, 2, 0, 1],
            "text": ["a", "b", "c", "d", "e"],
        }
    )
    y_pred = np.array([0, 1, 2, 0, 1])
    unsmile_raw = pd.DataFrame({g: [1, 0, 0, 0, 0] for g in UNSMILE_GROUPS})
    result = evaluate_unsmile_groups(test_df, y_pred, unsmile_raw)
    # 모든 집단 < 30 → 모두 skipped
    for g in UNSMILE_GROUPS:
        assert result["groups"][g]["skipped"] == "n<30"


def test_evaluate_kold_groups_no_kold_in_test():
    test_df = pd.DataFrame({"source": ["unsmile"] * 10})
    y_pred = np.zeros(10)
    kold_raw = pd.DataFrame({"guid": ["a"], "GRP": ["g1"]})
    result = evaluate_kold_groups(test_df, y_pred, kold_raw)
    assert result["note"] == "no kold in test"


def test_disability_keywords_includes_main():
    for kw in ["장애", "휠체어", "발달장애", "활동지원"]:
        assert kw in DISABILITY_KEYWORDS
