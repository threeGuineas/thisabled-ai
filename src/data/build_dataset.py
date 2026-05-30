"""시드 데이터 → train/val/test split → ``data/processed/*.parquet``.

매핑 정책: [docs/label_mapping.md](../../docs/label_mapping.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.label_mapping import build_unified_dataset
from src.data.loaders import save_dataset

DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test


def load_seed_dataframes(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw/ 에서 UnSmile (train+valid concat) 과 KOLD 를 DataFrame으로 로드."""
    us_tr = pd.read_csv(raw_dir / "unsmile/unsmile_train_v1.0.tsv", sep="\t")
    us_va = pd.read_csv(raw_dir / "unsmile/unsmile_valid_v1.0.tsv", sep="\t")
    unsmile = pd.concat([us_tr, us_va], ignore_index=True)

    with (raw_dir / "kold/kold_v1.json").open(encoding="utf-8") as f:
        kold = pd.DataFrame(json.load(f))

    return unsmile, kold


def stratified_three_way_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """label 기준 stratified 80/10/10 split.

    2-step:
        1) train(=ratios[0]) vs temp(=1-ratios[0])
        2) temp → val(=ratios[1]/(1-ratios[0])) vs test
    """
    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_r + test_r),
        stratify=df["label"],
        random_state=seed,
    )
    val_share = val_r / (val_r + test_r)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share),
        stratify=temp_df["label"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_and_save(
    raw_dir: Path,
    out_dir: Path,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """End-to-end: load seeds → unify → stratified split → save parquet."""
    unsmile, kold = load_seed_dataframes(raw_dir)
    unified = build_unified_dataset(unsmile, kold)
    train_df, val_df, test_df = stratified_three_way_split(unified, ratios, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": save_dataset(train_df, out_dir / "train.parquet"),
        "val": save_dataset(val_df, out_dir / "val.parquet"),
        "test": save_dataset(test_df, out_dir / "test.parquet"),
    }
    return paths
