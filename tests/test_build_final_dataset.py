"""Tests for scripts/build_final_dataset.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_final_dataset import load_synthetic_splits


@pytest.fixture
def mock_synthesis_dir(tmp_path: Path) -> Path:
    """임시 합성 데이터 디렉토리와 더미 JSONL 생성."""
    synth_dir = tmp_path / "synthetic" / "emergency"

    categories = ["3a", "boundary"]
    splits = ["train", "val", "test"]

    for cat in categories:
        cat_dir = synth_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            jsonl_path = cat_dir / f"{split}.jsonl"
            items = []
            if cat == "3a":
                # 긴급 그루밍 데이터
                items = [
                    {
                        "text": f"grooming example {i} in {split}",
                        "label": 3,
                        "subcategory": "3a",
                        "split": split,
                        "source": "synthetic_emergency_v1",
                    }
                    for i in range(5)
                ]
            else:
                # boundary 데이터 (정상/주의/경고 반례)
                items = [
                    {
                        "text": f"boundary example {i} in {split}",
                        "label": i % 3,  # 0, 1, 2
                        "subcategory": "boundary",
                        "split": split,
                        "source": "synthetic_emergency_v1",
                        "reason": "dummy reason",
                    }
                    for i in range(3)
                ]

            with jsonl_path.open("w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return synth_dir


def test_load_synthetic_splits_schema_and_contents(mock_synthesis_dir: Path):
    # load_synthetic_splits 검증
    dfs = load_synthetic_splits(mock_synthesis_dir)

    assert "train" in dfs
    assert "val" in dfs
    assert "test" in dfs

    for split in ["train", "val", "test"]:
        df = dfs[split]
        # 3a (5건) + boundary (3건) = 총 8건
        assert len(df) == 8

        # 컬럼 명세 확인
        assert list(df.columns) == ["text", "label", "source", "source_id"]

        # source_id 고유성 검증
        assert df["source_id"].nunique() == 8
        assert df["source_id"].iloc[0].startswith("synth_")

        # 타입 검증
        assert df["text"].dtype == object
        assert pd.api.types.is_integer_dtype(df["label"])
        assert (df["label"] == 3).sum() == 5  # 3a
        assert (df["label"] < 3).sum() == 3  # boundary


def test_load_synthetic_splits_empty_dir(tmp_path: Path):
    empty_dir = tmp_path / "non_existent_synth_dir"
    dfs = load_synthetic_splits(empty_dir)

    for split in ["train", "val", "test"]:
        assert dfs[split].empty
        assert list(dfs[split].columns) == ["text", "label", "source", "source_id"]
