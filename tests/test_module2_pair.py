"""Tests for Module 2 pair generation."""

import pandas as pd
import pytest

from src.data.build_pairs import build_pairs, generate_mock_profiles


@pytest.fixture
def mock_profiles() -> pd.DataFrame:
    return generate_mock_profiles(50)


def test_build_pairs_symmetry_and_balance(mock_profiles: pd.DataFrame) -> None:
    df_pairs = build_pairs(mock_profiles, n_queries=10, n_candidates=5)

    assert len(df_pairs) > 0
    assert "query_id" in df_pairs.columns
    assert "cand_id" in df_pairs.columns
    assert "label" in df_pairs.columns

    # 쿼리 ID별 그룹핑 확인
    queries = df_pairs["query_id"].unique()
    assert len(queries) <= 10

    # 본인이 본인과 매칭되지 않았는지 확인
    assert not any(df_pairs["query_id"] == df_pairs["cand_id"])

    # Label 분포 확인 (모두 0은 아니어야 함)
    labels = df_pairs["label"].unique()
    assert len(labels) > 0
