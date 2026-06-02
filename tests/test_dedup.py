"""Tests for src.data.dedup (MinHash 근사 중복 제거)."""

from __future__ import annotations

import pandas as pd
import pytest

datasketch = pytest.importorskip("datasketch")  # noqa: F841

from src.data.dedup import deduplicate_against, find_duplicate_indices  # noqa: E402


def test_exact_and_near_duplicates_detected():
    reference = ["너 진짜 죽고 싶냐? 오늘 학교 뒤로 와라.", "안녕하세요 반갑습니다"]
    candidates = [
        "너 진짜 죽고 싶냐? 오늘 학교 뒤로 와라.",  # 완전 동일 → 중복
        "너 진짜 죽고 싶냐? 오늘 학교 뒤로 와라!!",  # 거의 동일 → 중복
        "오늘 날씨가 참 맑고 좋네요 산책 가야지",  # 무관 → 비중복
    ]
    dup = find_duplicate_indices(reference, candidates, threshold=0.8)
    assert 0 in dup
    assert 1 in dup
    assert 2 not in dup


def test_deduplicate_against_drops_only_duplicates():
    reference = ["발달장애 청소년을 위한 쉼터입니다"]
    cand_df = pd.DataFrame(
        {
            "text": [
                "발달장애 청소년을 위한 쉼터입니다",  # 중복
                "전혀 다른 새로운 합성 문장입니다 완전히",  # 유지
            ],
            "label": [3, 3],
            "source": ["synthetic_emergency_v1", "synthetic_emergency_v1"],
        }
    )
    deduped, removed = deduplicate_against(reference, cand_df, threshold=0.8)
    assert removed == 1
    assert len(deduped) == 1
    assert deduped.iloc[0]["text"].startswith("전혀 다른")


def test_empty_candidate_is_noop():
    empty = pd.DataFrame(columns=["text", "label", "source"])
    deduped, removed = deduplicate_against(["ref"], empty)
    assert removed == 0
    assert deduped.empty


def test_empty_reference_removes_nothing():
    cand_df = pd.DataFrame({"text": ["a문장", "b문장"], "label": [0, 1]})
    deduped, removed = deduplicate_against([], cand_df, threshold=0.8)
    assert removed == 0
    assert len(deduped) == 2
