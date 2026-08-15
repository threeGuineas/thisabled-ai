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


def test_single_class_slice_reports_recall_and_flags_unusable_f1():
    """정답이 위험(1) 하나뿐인 슬라이스에서 macro-F1 격차는 허수다.

    UnSmile·KOLD 보호집단은 혐오표현 대상 주석이라 이진 붕괴 후 전부 label=1 이다.
    전부 맞히면 f1=1.0, 몇 건만 놓치면 약 0.5로 떨어져 집단 간 비교가 불가능하므로
    recall 을 함께 내고 단일 클래스임을 표시해야 한다.
    """

    df = pd.DataFrame(
        {
            "text": ["장애인 비하 표현"] * 40 + ["다른 문장"] * 40,
            "label": [1] * 80,
            "source": ["unsmile"] * 80,
        }
    )
    # 장애 슬라이스는 전부 맞히고, 나머지는 절반만 맞힌다.
    y_pred = np.array([1] * 40 + [1] * 20 + [0] * 20)

    result = evaluate_disability_domain(df, y_pred)
    yes = result["groups"]["disability_yes"]
    no = result["groups"]["disability_no"]

    assert yes["single_class"] and no["single_class"]
    assert yes["n_neg"] == 0 and no["n_neg"] == 0
    assert yes["recall"] == 1.0
    assert no["recall"] == 0.5
    # f1 은 1.0 vs 약 0.33 으로 벌어지지만 이는 지표 오용이다.
    assert result["max_recall_gap"] == 0.5
    assert result["single_class_groups"] is True
    assert "max_recall_gap" in result["note"]


def test_both_class_slice_is_not_flagged_single_class():
    # 각 슬라이스 안에 위험(1)과 정상(0)이 모두 있어야 macro-F1 이 해석 가능하다.
    df = pd.DataFrame(
        {
            "text": ["장애인 비하", "장애인 소개", "욕설 문장", "보통 문장"] * 20,
            "label": [1, 0, 1, 0] * 20,
            "source": ["unsmile"] * 80,
        }
    )
    y_pred = np.array([1, 0, 1, 0] * 20)

    result = evaluate_disability_domain(df, y_pred)

    assert result["single_class_groups"] is False
    assert "note" not in result
    assert result["groups"]["disability_yes"]["recall"] == 1.0


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
