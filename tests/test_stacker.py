"""Tests for LightGBMStacker."""

import numpy as np
import pandas as pd
import pytest

from src.models.stacker import LightGBMStacker


@pytest.fixture
def dummy_data() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.DataFrame(
        {
            "text": [
                "안녕하세요",
                "장애인 주차구역",
                "너무 화가 난다",
                "휠체어 접근성",
                "일반 텍스트",
            ],
            "source": ["unsmile", "kold", "synthetic_emergency_v1", "unknown", "unsmile"],
            "label": [0, 1, 2, 3, 0],
        }
    )
    logits = np.random.randn(5, 4)
    return df, logits


def test_stacker_build_meta_features(dummy_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    df, logits = dummy_data
    stacker = LightGBMStacker()

    meta_df, targets = stacker.build_meta_features(df, logits)

    assert meta_df.shape[0] == 5
    assert "logit_0" in meta_df.columns
    assert "logit_3" in meta_df.columns
    assert "source" in meta_df.columns
    assert "text_length_bucket" in meta_df.columns
    assert "has_disability" in meta_df.columns

    assert list(targets) == [0, 1, 2, 3, 0]

    # 장애 키워드 체크
    assert meta_df["has_disability"].tolist() == [0, 1, 0, 1, 0]


def test_stacker_fit_predict(dummy_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    df, logits = dummy_data
    stacker = LightGBMStacker({"objective": "multiclass", "num_class": 4, "verbose": -1})

    X, y = stacker.build_meta_features(df, logits)

    # 더미 훈련
    stacker.fit(X, y, X, y, num_boost_round=2)

    preds = stacker.predict(X)
    assert preds.shape == (5, 4)
