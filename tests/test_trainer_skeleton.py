"""Smoke tests for src.training — import·구조·config 로드만 검증 (실제 학습 X)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_imports_resolve():
    from src.training import dataset, trainer  # noqa: F401
    from src.training.trainer import FocalLossTrainer, load_config, train_module1  # noqa: F401


def test_config_loads_and_has_required_keys():
    from src.training.trainer import load_config

    cfg = load_config(ROOT / "configs" / "module1_kcelectra.yaml")
    for key in ("model", "training", "loss", "paths", "labels"):
        assert key in cfg, f"missing {key}"
    assert cfg["model"]["name"] == "beomi/KcELECTRA-base-v2022"
    assert cfg["model"]["num_labels"] == 4
    assert len(cfg["labels"]) == 4


class _FakeTokenizer:
    """transformers 의존 없이 Dataset 구조만 검증하기 위한 토크나이저 stub."""

    def __call__(self, text, truncation=True, max_length=128, padding=False, return_tensors=None):
        ids = [ord(c) % 1000 for c in text[:max_length]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_dataset_yields_expected_keys(tmp_path):
    from src.training.dataset import RiskTextDataset

    df = pd.DataFrame(
        {
            "text": ["테스트1", "안녕"],
            "label": [0, 2],
            "source": ["unsmile", "kold"],
            "source_id": ["a", "b"],
        }
    )
    p = tmp_path / "tiny.parquet"
    df.to_parquet(p, index=False)

    ds = RiskTextDataset(p, _FakeTokenizer(), max_length=16)
    assert len(ds) == 2
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].item() == 0


def test_processed_parquet_exists_and_has_schema():
    """build_processed_dataset.py 가 이미 실행된 상태여야 D2 학습이 바로 가능."""
    processed = ROOT / "data" / "processed"
    for split in ("train", "val", "test"):
        p = processed / f"{split}.parquet"
        if not p.exists():
            pytest.skip(f"{p} 없음 — scripts/build_processed_dataset.py 먼저 실행")
        df = pd.read_parquet(p)
        assert list(df.columns) == ["text", "label", "source", "source_id"]
        assert df["label"].isin([0, 1, 2, 3]).all()
        assert len(df) > 0
