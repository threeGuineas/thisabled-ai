"""EXP-3 단계 2: F1 격차 큰 보호집단의 train 샘플 oversample.

fairness.json을 읽어 F1 < threshold인 집단 식별 → 해당 집단의 train.parquet 샘플
N회 반복 → train.parquet 재저장.

[정책]
- val/test는 건드리지 않음 (시드 일관성)
- 합성 데이터는 건드리지 않음
- idempotent: 매번 source != synthetic_* AND source != fairness_oversample_*
  필터 후 재구축

Usage:
    python scripts/oversample_fairness.py --threshold 0.65 --multiplier 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import save_dataset  # noqa: E402
from src.evaluation.fairness import (  # noqa: E402
    DISABILITY_KEYWORDS,
)


def _filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    """oversample row를 제거 (idempotent)."""
    return df[~df["source"].str.startswith("fairness_oversample_", na=False)].reset_index(drop=True)


def _find_unsmile_rows_in_group(
    train_df: pd.DataFrame, unsmile_raw: pd.DataFrame, group: str
) -> pd.DataFrame:
    """train_df 중 source=unsmile + raw에서 해당 group 라벨이 1인 row만 반환."""
    us_mask = train_df["source"] == "unsmile"
    us = train_df[us_mask].copy()
    us["raw_idx"] = us["source_id"].astype(int)
    raw_subset = unsmile_raw[[group]].reset_index(drop=True)
    us = us.join(raw_subset, on="raw_idx", how="left")
    return us[us[group] == 1].drop(columns=["raw_idx", group])


def _find_disability_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    mask = train_df["text"].astype(str).apply(lambda t: any(kw in t for kw in DISABILITY_KEYWORDS))
    return train_df[mask].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fairness-json",
        type=str,
        default=str(ROOT / "reports/validation_reports/module1/fairness.json"),
    )
    parser.add_argument(
        "--train-parquet", type=str, default=str(ROOT / "data/processed/train.parquet")
    )
    parser.add_argument(
        "--unsmile-train", type=str, default=str(ROOT / "data/raw/unsmile/unsmile_train_v1.0.tsv")
    )
    parser.add_argument(
        "--unsmile-valid", type=str, default=str(ROOT / "data/raw/unsmile/unsmile_valid_v1.0.tsv")
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="F1이 이 값 미만인 집단을 oversample 대상으로 선정",
    )
    parser.add_argument("--multiplier", type=int, default=4, help="대상 집단을 N회 oversample")
    args = parser.parse_args()

    fairness = json.loads(Path(args.fairness_json).read_text())

    train_df = _filter_clean(pd.read_parquet(args.train_parquet))
    print(f"clean train 베이스: {len(train_df):,}")

    unsmile_raw = pd.concat(
        [pd.read_csv(args.unsmile_train, sep="\t"), pd.read_csv(args.unsmile_valid, sep="\t")],
        ignore_index=True,
    )

    extras: list[pd.DataFrame] = []
    targeted: list[str] = []

    # 1) UnSmile 7집단 중 F1 < threshold
    for group_name, info in fairness.get("unsmile_7_groups", {}).get("groups", {}).items():
        if info.get("f1") is None:
            continue
        if info["f1"] < args.threshold:
            rows = _find_unsmile_rows_in_group(train_df, unsmile_raw, group_name)
            if rows.empty:
                continue
            print(f"  UnSmile/{group_name} F1={info['f1']:.3f} → {len(rows)}건 × {args.multiplier}")
            for i in range(args.multiplier - 1):
                copy = rows.copy()
                copy["source"] = f"fairness_oversample_unsmile_{group_name}"
                copy["source_id"] = copy["source_id"].astype(str) + f"_fo{i}"
                extras.append(copy)
            targeted.append(f"unsmile:{group_name}")

    # 2) 장애 도메인 격차
    dis_info = fairness.get("disability_domain", {}).get("groups", {})
    if dis_info.get("disability_yes", {}).get("f1") is not None:
        f1_yes = dis_info["disability_yes"]["f1"]
        f1_no = dis_info.get("disability_no", {}).get("f1", 1.0)
        if f1_yes + 0.02 < f1_no:  # 장애 그룹이 0.02 이상 낮으면 oversample
            rows = _find_disability_rows(train_df)
            if not rows.empty:
                print(
                    f"  Disability F1={f1_yes:.3f} vs no={f1_no:.3f} → {len(rows)}건 × {args.multiplier}"
                )
                for i in range(args.multiplier - 1):
                    copy = rows.copy()
                    copy["source"] = "fairness_oversample_disability"
                    copy["source_id"] = copy["source_id"].astype(str) + f"_fo{i}"
                    extras.append(copy)
                targeted.append("disability")

    if not extras:
        print("⚠ 임계치 미달 집단 없음 — oversample 안 함")
        return 0

    merged = pd.concat([train_df] + extras, ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
    save_dataset(merged, Path(args.train_parquet))

    n_added = sum(len(e) for e in extras)
    print("\n=== oversample 완료 ===")
    print(f"clean({len(train_df):,}) + extras({n_added:,}) = {len(merged):,}")
    print(f"대상 집단: {targeted}")
    print(f"라벨 분포: {dict(merged['label'].value_counts().sort_index())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
