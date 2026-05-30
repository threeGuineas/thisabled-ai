"""Tests for src.data.label_mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.label_mapping import (
    LABEL_CAUTION,
    LABEL_EMERGENCY,
    LABEL_NORMAL,
    LABEL_WARNING,
    UNSMILE_HATE_COLS,
    build_unified_dataset,
    kold_dataframe_to_labels,
    kold_record_to_label,
    unsmile_dataframe_to_labels,
    unsmile_row_to_label,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


# === UnSmile per-row ===


def _us_row(clean=0, profanity=0, hate=None, person=0) -> dict:
    row = {"문장": "테스트", "clean": clean, "악플/욕설": profanity, "개인지칭": person}
    hate = hate or {}
    for col in UNSMILE_HATE_COLS:
        row[col] = hate.get(col, 0)
    return row


def test_unsmile_clean_is_normal():
    assert unsmile_row_to_label(_us_row(clean=1)) == LABEL_NORMAL


def test_unsmile_single_hate_is_warning():
    assert unsmile_row_to_label(_us_row(hate={"여성/가족": 1})) == LABEL_WARNING


def test_unsmile_multiple_hate_collapses_to_warning():
    assert unsmile_row_to_label(_us_row(hate={"여성/가족": 1, "성소수자": 1})) == LABEL_WARNING


def test_unsmile_profanity_only_is_caution():
    assert unsmile_row_to_label(_us_row(profanity=1)) == LABEL_CAUTION


def test_unsmile_profanity_with_hate_collapses_to_warning():
    """그룹 혐오 + 악플/욕설 동시: max severity collapse → 경고."""
    assert unsmile_row_to_label(_us_row(profanity=1, hate={"종교": 1})) == LABEL_WARNING


def test_unsmile_clean_overrides_other_signals():
    """clean=1이 다른 라벨과 동시 켜져 있어도 정상 우선 (정책 규칙 1)."""
    assert unsmile_row_to_label(_us_row(clean=1, profanity=1)) == LABEL_NORMAL


def test_unsmile_all_zero_is_normal_fallback():
    assert unsmile_row_to_label(_us_row()) == LABEL_NORMAL


# === KOLD per-record ===


def test_kold_off_false_is_normal():
    assert kold_record_to_label({"OFF": False, "TGT": None}) == LABEL_NORMAL


def test_kold_off_true_group_is_warning():
    assert kold_record_to_label({"OFF": True, "TGT": "group"}) == LABEL_WARNING


def test_kold_off_true_individual_is_caution():
    """학습 라벨은 주의(1) 유지 — 1:1 채팅 후처리 격상은 모델 밖."""
    assert kold_record_to_label({"OFF": True, "TGT": "individual"}) == LABEL_CAUTION


def test_kold_off_true_untargeted_is_caution():
    assert kold_record_to_label({"OFF": True, "TGT": "untargeted"}) == LABEL_CAUTION


def test_kold_off_true_other_is_caution():
    """TGT=other 1,402건 흡수 정책 (40건 샘플 검증 기반)."""
    assert kold_record_to_label({"OFF": True, "TGT": "other"}) == LABEL_CAUTION


# === Vectorized vs per-row 일치 ===


def test_unsmile_vectorized_matches_per_row():
    rows = [
        _us_row(clean=1),
        _us_row(hate={"남성": 1}),
        _us_row(profanity=1),
        _us_row(profanity=1, hate={"지역": 1, "종교": 1}),
        _us_row(),
    ]
    df = pd.DataFrame(rows)
    expected = [unsmile_row_to_label(r) for r in rows]
    assert list(unsmile_dataframe_to_labels(df)) == expected


def test_kold_vectorized_matches_per_record():
    records = [
        {"OFF": False, "TGT": None, "comment": "a", "guid": "0"},
        {"OFF": True, "TGT": "group", "comment": "b", "guid": "1"},
        {"OFF": True, "TGT": "individual", "comment": "c", "guid": "2"},
        {"OFF": True, "TGT": "untargeted", "comment": "d", "guid": "3"},
        {"OFF": True, "TGT": "other", "comment": "e", "guid": "4"},
    ]
    df = pd.DataFrame(records)
    expected = [kold_record_to_label(r) for r in records]
    assert list(kold_dataframe_to_labels(df)) == expected


# === Integration: 실제 시드 데이터 분포 EDA와 일치 ===


@pytest.fixture(scope="module")
def unsmile_full() -> pd.DataFrame:
    tr = RAW / "unsmile/unsmile_train_v1.0.tsv"
    va = RAW / "unsmile/unsmile_valid_v1.0.tsv"
    if not tr.exists() or not va.exists():
        pytest.skip("UnSmile 시드 데이터 없음 — scripts/download_seed_datasets.py 실행 필요")
    return pd.concat(
        [pd.read_csv(tr, sep="\t"), pd.read_csv(va, sep="\t")],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def kold_full() -> pd.DataFrame:
    p = RAW / "kold/kold_v1.json"
    if not p.exists():
        pytest.skip("KOLD 시드 데이터 없음 — scripts/download_seed_datasets.py 실행 필요")
    with p.open(encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def test_unsmile_distribution_matches_eda(unsmile_full):
    """EDA에서 측정한 분포와 정확히 일치해야 함 (구현 회귀 방지)."""
    labels = unsmile_dataframe_to_labels(unsmile_full)
    counts = labels.value_counts().to_dict()
    assert counts.get(LABEL_NORMAL, 0) == 4676
    assert counts.get(LABEL_CAUTION, 0) == 3929
    assert counts.get(LABEL_WARNING, 0) == 10137
    assert counts.get(LABEL_EMERGENCY, 0) == 0


def test_kold_distribution_matches_eda(kold_full):
    labels = kold_dataframe_to_labels(kold_full)
    counts = labels.value_counts().to_dict()
    assert counts.get(LABEL_NORMAL, 0) == 20119
    assert counts.get(LABEL_CAUTION, 0) == 7897
    assert counts.get(LABEL_WARNING, 0) == 12413
    assert counts.get(LABEL_EMERGENCY, 0) == 0


def test_build_unified_dataset_schema_and_size(unsmile_full, kold_full):
    df = build_unified_dataset(unsmile_full, kold_full)
    assert list(df.columns) == ["text", "label", "source", "source_id"]
    assert len(df) == len(unsmile_full) + len(kold_full)
    assert set(df["source"].unique()) == {"unsmile", "kold"}
    assert df["label"].isin([0, 1, 2, 3]).all()
    # 통합 분포도 검증
    counts = df["label"].value_counts().to_dict()
    assert counts[LABEL_NORMAL] == 24795
    assert counts[LABEL_CAUTION] == 11826
    assert counts[LABEL_WARNING] == 22550
