"""Tests for Module 2 pair generation + EXP-2 leakage 차단 회귀 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.data.build_pairs import (
    FEATURE_MODES,
    build_pairs,
    generate_mock_profiles,
    split_users,
)


@pytest.fixture
def mock_profiles() -> pd.DataFrame:
    return generate_mock_profiles(50)


def test_build_pairs_symmetry_and_balance(mock_profiles: pd.DataFrame) -> None:
    df_pairs = build_pairs(mock_profiles, n_queries=10, n_candidates=5)
    assert len(df_pairs) > 0
    assert "query_id" in df_pairs.columns
    assert "cand_id" in df_pairs.columns
    assert "label" in df_pairs.columns
    queries = df_pairs["query_id"].unique()
    assert len(queries) <= 10
    assert not any(df_pairs["query_id"] == df_pairs["cand_id"])
    labels = df_pairs["label"].unique()
    assert len(labels) > 0


# === EXP-2: user-level split + leakage-free features ===


def test_split_users_disjoint(mock_profiles):
    train, test = split_users(mock_profiles, test_ratio=0.2, seed=42)
    train_ids = set(train["user_id"])
    test_ids = set(test["user_id"])
    assert train_ids.isdisjoint(test_ids), "EXP-2 핵심: train/test users disjoint"
    assert len(train) + len(test) == len(mock_profiles)


def test_split_users_deterministic(mock_profiles):
    a1, a2 = split_users(mock_profiles, test_ratio=0.2, seed=42)
    b1, b2 = split_users(mock_profiles, test_ratio=0.2, seed=42)
    pd.testing.assert_frame_equal(a1, b1)
    pd.testing.assert_frame_equal(a2, b2)


def test_build_pairs_only_uses_given_users(mock_profiles):
    """build_pairs는 넘긴 프로필 안에서만 페어를 만들어야 함 (split 무결성)."""
    train, _ = split_users(mock_profiles, test_ratio=0.2, seed=42)
    train_pairs = build_pairs(train, n_queries=5, n_candidates=3, seed=42)
    train_ids = set(train["user_id"])
    assert set(train_pairs["query_id"]).issubset(train_ids)
    assert set(train_pairs["cand_id"]).issubset(train_ids)


def test_feature_modes_constant():
    assert FEATURE_MODES == ("full", "embedding")


def test_engineer_features_embedding_excludes_rule_inputs(monkeypatch, mock_profiles):
    """embedding 모드는 label-결정 변수(region_match/age_diff/overlap)를 제외 → leakage 차단."""
    import src.data.build_pairs as bp

    monkeypatch.setattr(
        bp,
        "get_sbert_embeddings",
        lambda texts, model_name=None: np.zeros((len(texts), 8), dtype=np.float32),
    )

    pairs = build_pairs(mock_profiles, n_queries=5, n_candidates=3, seed=42)
    feats_embed, _ = bp.engineer_features(pairs, mode="embedding")
    feats_full, _ = bp.engineer_features(pairs, mode="full")

    leak_cols = {"f_region_match", "f_age_diff", "f_overlap"}
    embed_cols = set(feats_embed.columns)
    full_cols = set(feats_full.columns)

    assert embed_cols.isdisjoint(leak_cols), f"EXP-2 핵심: leakage 차단 — {embed_cols & leak_cols}"
    assert leak_cols.issubset(full_cols)


def test_engineer_features_invalid_mode_raises(mock_profiles):
    import src.data.build_pairs as bp

    pairs = build_pairs(mock_profiles, n_queries=3, n_candidates=2, seed=42)
    with pytest.raises(ValueError):
        bp.engineer_features(pairs, mode="invalid")
