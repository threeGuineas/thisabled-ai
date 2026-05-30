"""Tests for src.data.build_dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.build_dataset import stratified_three_way_split


@pytest.fixture
def dummy_unified() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 10_000
    return pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(n)],
            "label": rng.choice([0, 1, 2], size=n, p=[0.5, 0.2, 0.3]).astype(np.int8),
            "source": rng.choice(["unsmile", "kold"], size=n),
            "source_id": [str(i) for i in range(n)],
        }
    )


def test_split_sizes_match_ratios(dummy_unified):
    tr, va, te = stratified_three_way_split(dummy_unified, ratios=(0.8, 0.1, 0.1), seed=42)
    total = len(dummy_unified)
    assert len(tr) + len(va) + len(te) == total
    assert abs(len(tr) / total - 0.8) < 0.01
    assert abs(len(va) / total - 0.1) < 0.01
    assert abs(len(te) / total - 0.1) < 0.01


def test_split_is_disjoint(dummy_unified):
    tr, va, te = stratified_three_way_split(dummy_unified, seed=42)
    tr_ids = set(tr["source_id"])
    va_ids = set(va["source_id"])
    te_ids = set(te["source_id"])
    assert tr_ids.isdisjoint(va_ids)
    assert tr_ids.isdisjoint(te_ids)
    assert va_ids.isdisjoint(te_ids)


def test_label_distribution_preserved(dummy_unified):
    """stratified split → 각 split의 라벨 비율이 원본과 거의 같아야 함."""
    tr, va, te = stratified_three_way_split(dummy_unified, seed=42)
    orig = dummy_unified["label"].value_counts(normalize=True).sort_index()
    for split in (tr, va, te):
        p = split["label"].value_counts(normalize=True).sort_index()
        for label in orig.index:
            assert abs(p[label] - orig[label]) < 0.01, f"label {label} 비율 차이 큼"


def test_split_is_deterministic(dummy_unified):
    a = stratified_three_way_split(dummy_unified, seed=42)
    b = stratified_three_way_split(dummy_unified, seed=42)
    for x, y in zip(a, b, strict=False):
        pd.testing.assert_frame_equal(x, y)


def test_invalid_ratios_raises(dummy_unified):
    with pytest.raises(ValueError):
        stratified_three_way_split(dummy_unified, ratios=(0.8, 0.1, 0.2))
