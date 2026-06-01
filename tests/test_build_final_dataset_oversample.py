"""EXP-1 회귀 테스트 — build_final_dataset의 --synth-repeat + idempotent 검증."""

from __future__ import annotations

import pandas as pd

import scripts.build_final_dataset as bfd


def _make_seed_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"seed_{i}" for i in range(100)],
            "label": [i % 3 for i in range(100)],
            "source": ["unsmile"] * 50 + ["kold"] * 50,
            "source_id": [str(i) for i in range(100)],
        }
    )


def _make_synth_train(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"synth_{i}" for i in range(n)],
            "label": [3] * n,
            "source": ["synthetic_emergency_v1"] * n,
            "source_id": [f"synth_{i}" for i in range(n)],
        }
    )


def test_filter_seed_only_removes_synthetic():
    seed = _make_seed_train()
    synth = _make_synth_train(20)
    mixed = pd.concat([seed, synth], ignore_index=True)
    cleaned = bfd._filter_seed_only(mixed)
    assert len(cleaned) == 100
    assert not cleaned["source"].str.startswith("synthetic_").any()


def test_build_final_dataset_synth_repeat_oversamples(tmp_path, monkeypatch):
    """synth_repeat=N이면 합성을 N번 반복해서 train에 병합."""
    # PROCESSED_DIR을 임시 경로로 redirect
    monkeypatch.setattr(bfd, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(bfd, "SYNTH_DIR", tmp_path / "synth")

    seed = _make_seed_train()
    seed.to_parquet(tmp_path / "train.parquet", index=False)

    synth = _make_synth_train(10)

    monkeypatch.setattr(
        bfd,
        "load_synthetic_splits",
        lambda _d: {
            "train": synth,
            "val": pd.DataFrame(columns=synth.columns),
            "test": pd.DataFrame(columns=synth.columns),
        },
    )

    bfd.build_final_dataset(seed=42, synth_repeat=8)

    result = pd.read_parquet(tmp_path / "train.parquet")
    n_seed = (~result["source"].str.startswith("synthetic_")).sum()
    n_synth = result["source"].str.startswith("synthetic_").sum()
    assert n_seed == 100, "시드 보존"
    assert n_synth == 80, f"합성 10 × repeat 8 = 80, got {n_synth}"


def test_build_final_dataset_idempotent_no_accumulation(tmp_path, monkeypatch):
    """재실행해도 합성이 누적되지 않아야 함 (EXP-1 핵심 안전장치)."""
    monkeypatch.setattr(bfd, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(bfd, "SYNTH_DIR", tmp_path / "synth")

    seed = _make_seed_train()
    seed.to_parquet(tmp_path / "train.parquet", index=False)

    synth = _make_synth_train(10)
    monkeypatch.setattr(
        bfd,
        "load_synthetic_splits",
        lambda _d: {
            "train": synth,
            "val": pd.DataFrame(columns=synth.columns),
            "test": pd.DataFrame(columns=synth.columns),
        },
    )

    bfd.build_final_dataset(seed=42, synth_repeat=3)
    bfd.build_final_dataset(seed=42, synth_repeat=3)  # 두 번째 실행

    result = pd.read_parquet(tmp_path / "train.parquet")
    n_synth = result["source"].str.startswith("synthetic_").sum()
    assert n_synth == 30, f"두 번 실행해도 10×3 = 30 유지, got {n_synth}"


def test_build_final_dataset_repeat_zero_seed_only(tmp_path, monkeypatch):
    monkeypatch.setattr(bfd, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(bfd, "SYNTH_DIR", tmp_path / "synth")

    seed = _make_seed_train()
    seed.to_parquet(tmp_path / "train.parquet", index=False)

    monkeypatch.setattr(
        bfd,
        "load_synthetic_splits",
        lambda _d: {
            "train": _make_synth_train(10),
            "val": pd.DataFrame(),
            "test": pd.DataFrame(),
        },
    )

    bfd.build_final_dataset(seed=42, synth_repeat=0)

    result = pd.read_parquet(tmp_path / "train.parquet")
    assert len(result) == 100
    assert not result["source"].str.startswith("synthetic_").any()
