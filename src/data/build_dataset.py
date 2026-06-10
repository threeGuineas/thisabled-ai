"""시드 데이터 → train/val/test split → ``data/processed/*.parquet``.

매핑 정책: [docs/label_mapping.md](../../docs/label_mapping.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.label_mapping import (
    LABEL_EMERGENCY,
    LABEL_WARNING,
    aihub_record_to_label,
    kold_dataframe_to_labels,
    unsmile_dataframe_to_labels,
)
from src.data.loaders import save_dataset

DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test
AIHUB_TARGET_SIZE = 45000


def load_holdout_conv_ids(eval_dir: Path) -> set[str]:
    """실데이터 hold-out에 쓰인 AI-Hub conv_id 집합 로드.

    ``scripts/build_real_holdout.py``가 ``data/eval/aihub_holdout_conv_ids.txt``로
    남긴 conv_id를 train/val/test 빌드에서 배제하기 위한 누수 가드. 파일이 없으면
    빈 집합(가드 미적용).
    """
    p = eval_dir / "aihub_holdout_conv_ids.txt"
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_seed_dataframes(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw/ 에서 UnSmile 과 KOLD 를 DataFrame으로 로드."""
    us_tr = pd.read_csv(raw_dir / "unsmile/unsmile_train_v1.0.tsv", sep="\t")
    us_va = pd.read_csv(raw_dir / "unsmile/unsmile_valid_v1.0.tsv", sep="\t")
    unsmile = pd.concat([us_tr, us_va], ignore_index=True)

    with (raw_dir / "kold/kold_v1.json").open(encoding="utf-8") as f:
        kold = pd.DataFrame(json.load(f))

    return unsmile, kold


def load_aihub_dataframe(raw_dir: Path) -> pd.DataFrame:
    """raw/ 에서 AI-Hub 558 데이터를 로드하고 파싱."""
    aihub_dir = raw_dir / "aihub_558/147.텍스트 윤리검증 데이터/01.데이터"
    json_paths = list(aihub_dir.glob("*/라벨링데이터/aihub/extracted/**/*.json"))

    records = []
    for p in json_paths:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
            for convo in data:
                conv_id = convo["id"]
                for sent in convo["sentences"]:
                    records.append(
                        {
                            "text": sent["text"],
                            "label": aihub_record_to_label(sent),
                            "source": "aihub",
                            "source_id": sent["id"],
                            "conv_id": conv_id,
                        }
                    )

    return pd.DataFrame(records)


def downsample_aihub(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    """AI-Hub 데이터를 다운샘플링하되, 긴급(3)/경고(2)는 전량 보존하고 정상(0)/주의(1)만 줄임."""
    urgent_mask = df["label"].isin([LABEL_EMERGENCY, LABEL_WARNING])
    urgent_conv_ids = df[urgent_mask]["conv_id"].unique()

    urgent_df = df[df["conv_id"].isin(urgent_conv_ids)]
    remaining_df = df[~df["conv_id"].isin(urgent_conv_ids)]

    current_size = len(urgent_df)
    if current_size >= target_size:
        return urgent_df

    needed_size = target_size - current_size
    avg_conv_len = (
        len(remaining_df) / len(remaining_df["conv_id"].unique()) if len(remaining_df) > 0 else 1
    )
    needed_convs = int(needed_size / avg_conv_len)

    remaining_conv_ids = remaining_df["conv_id"].unique()
    np.random.seed(seed)
    if needed_convs < len(remaining_conv_ids):
        selected_conv_ids = np.random.choice(remaining_conv_ids, needed_convs, replace=False)
    else:
        selected_conv_ids = remaining_conv_ids

    selected_remaining_df = remaining_df[remaining_df["conv_id"].isin(selected_conv_ids)]

    final_df = pd.concat([urgent_df, selected_remaining_df], ignore_index=True)
    return final_df


def build_unified_dataset(
    unsmile_df: pd.DataFrame,
    kold_df: pd.DataFrame,
    aihub_df: pd.DataFrame,
) -> pd.DataFrame:
    """세 데이터셋을 통합 스키마로 변환."""
    us = pd.DataFrame(
        {
            "text": unsmile_df["문장"].astype(str),
            "label": unsmile_dataframe_to_labels(unsmile_df).astype(np.int8),
            "source": "unsmile",
            "source_id": unsmile_df.index.astype(str),
            "conv_id": "us_" + unsmile_df.index.astype(str),
        }
    )
    ko = pd.DataFrame(
        {
            "text": kold_df["comment"].astype(str),
            "label": kold_dataframe_to_labels(kold_df).astype(np.int8),
            "source": "kold",
            "source_id": kold_df["guid"].astype(str)
            if "guid" in kold_df
            else kold_df.index.astype(str),
            "conv_id": "ko_"
            + (kold_df["guid"].astype(str) if "guid" in kold_df else kold_df.index.astype(str)),
        }
    )
    return pd.concat([us, ko, aihub_df], ignore_index=True)


def stratified_group_split(
    df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """conv_id 기준 StratifiedGroupKFold 80/10/10 split."""
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = list(sgkf.split(df, df["label"], df["conv_id"]))
    test_idx = folds[0][1]
    train_val_idx = folds[0][0]

    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)

    sgkf_val = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=seed)
    folds_val = list(sgkf_val.split(train_val_df, train_val_df["label"], train_val_df["conv_id"]))
    val_idx = folds_val[0][1]
    train_idx = folds_val[0][0]

    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def build_and_save(
    raw_dir: Path,
    out_dir: Path,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    eval_dir: Path | None = None,
) -> dict[str, Path]:
    """End-to-end: load seeds → unify → stratified group split → save parquet.

    누수 가드: ``eval_dir/aihub_holdout_conv_ids.txt``(기본 ``data/eval``)에 있는 conv_id는
    train/val/test 어디에도 들어가지 않도록 AI-Hub 풀에서 먼저 제외한다. 이 가드가 동작하려면
    ``scripts/build_real_holdout.py``를 **먼저** 실행해 conv_id 파일이 존재해야 한다.
    """
    if eval_dir is None:
        eval_dir = raw_dir.parent / "eval"
    unsmile, kold = load_seed_dataframes(raw_dir)
    aihub_raw = load_aihub_dataframe(raw_dir)

    holdout_ids = load_holdout_conv_ids(eval_dir)
    if holdout_ids:
        before = len(aihub_raw)
        aihub_raw = aihub_raw[~aihub_raw["conv_id"].astype(str).isin(holdout_ids)].reset_index(
            drop=True
        )
        print(
            f"[누수 가드] hold-out conv_id {len(holdout_ids):,}개 → "
            f"AI-Hub {before - len(aihub_raw):,}행 제외 (잔존 {len(aihub_raw):,}행)"
        )
    else:
        print(
            f"⚠ [누수 가드] {eval_dir / 'aihub_holdout_conv_ids.txt'} 없음 — "
            "hold-out 배제 미적용. scripts/build_real_holdout.py 를 먼저 실행하세요."
        )

    aihub_downsampled = downsample_aihub(aihub_raw, AIHUB_TARGET_SIZE, seed)
    unified = build_unified_dataset(unsmile, kold, aihub_downsampled)

    train_df, val_df, test_df = stratified_group_split(unified, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": save_dataset(train_df, out_dir / "train.parquet"),
        "val": save_dataset(val_df, out_dir / "val.parquet"),
        "test": save_dataset(test_df, out_dir / "test.parquet"),
    }
    return paths
