"""통합 최종 데이터셋 빌드 (시드 데이터 + 제미나이 합성 데이터).

[정책 - 2026-06-01]
- Train: 시드 Train + 합성 Train × N (oversample) 병합
- Val / Test: 시드 데이터만 유지 (오염 방지)
- 합성 Hold-out: 합성 Val + Test를 묶어서 별도 parquet로 분리

[EXP-1, 2026-06-02]
- --synth-repeat N: 합성 train을 N회 반복 oversample (긴급(3) 학습 강화).
- idempotent: 매 실행마다 기존 train.parquet에서 합성(source='synthetic_*')을
  제거한 뒤 새로 병합 → 재실행해도 합성 누적 안 됨.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dedup import deduplicate_against  # noqa: E402
from src.data.loaders import save_dataset  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
SYNTH_DIR = ROOT / "data" / "synthetic" / "emergency"
DEDUP_THRESHOLD = 0.8  # MinHash Jaccard 임계값: 이 이상 유사한 합성 행은 시드와 중복 처리


def load_synthetic_splits(synth_dir: Path) -> dict[str, pd.DataFrame]:
    """synthetic_dir 에서 모든 카테고리의 JSONL 데이터를 읽어 split별 DataFrame으로 병합."""
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    if not synth_dir.exists():
        print(f"⚠ 합성 데이터 디렉토리가 존재하지 않습니다: {synth_dir}")
        return {k: pd.DataFrame(columns=["text", "label", "source", "source_id"]) for k in splits}

    for jsonl_path in synth_dir.rglob("*.jsonl"):
        split = jsonl_path.stem  # "train", "val", "test"
        if split not in splits:
            continue
        try:
            with jsonl_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    splits[split].append(item)
        except Exception as e:
            print(f"⚠ {jsonl_path.name} 로드 실패: {e}")

    dfs = {}
    for split, items in splits.items():
        if not items:
            dfs[split] = pd.DataFrame(columns=["text", "label", "source", "source_id"])
            continue

        df = pd.DataFrame(items)
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)
        df["source"] = df.get("source", "synthetic_emergency_v1")
        df["source_id"] = df.apply(
            lambda r, _s=split: f"synth_{r.get('subcategory', 'unknown')}_{_s}_{r.name}",
            axis=1,
        )
        dfs[split] = df[["text", "label", "source", "source_id"]]
        print(f"  → 합성 [{split}]: {len(dfs[split])}건 로드 완료")

    return dfs


def _filter_seed_only(df: pd.DataFrame) -> pd.DataFrame:
    """idempotent를 위해 기존 합성 row를 제거하고 시드만 반환."""
    return df[~df["source"].str.startswith("synthetic_", na=False)].reset_index(drop=True)


def build_final_dataset(seed: int = 42, synth_repeat: int = 1) -> None:
    print(f"=== 통합 최종 데이터셋 빌드 (synth_repeat={synth_repeat}) ===")

    # 1. 합성 데이터 로드
    print(f"1. 합성 데이터 로딩 중... ({SYNTH_DIR})")
    synth_dfs = load_synthetic_splits(SYNTH_DIR)

    # 2. Train 병합 (시드 + 합성 × N)
    train_path = PROCESSED_DIR / "train.parquet"
    if train_path.exists():
        seed_train = _filter_seed_only(pd.read_parquet(train_path))
        synth_train = synth_dfs["train"]

        # 누수 차단: 시드 train과 근사 중복인 합성 행을 oversample 이전에 제거.
        if not synth_train.empty and synth_repeat > 0:
            synth_train, n_removed = deduplicate_against(
                seed_train["text"], synth_train, threshold=DEDUP_THRESHOLD
            )
            print(
                f"  → MinHash 중복 제거(threshold={DEDUP_THRESHOLD}): "
                f"시드 train과 근사 중복 합성 {n_removed}건 제거 → 합성 {len(synth_train)}건 잔존"
            )

        if not synth_train.empty and synth_repeat > 0:
            synth_train_repeated = pd.concat([synth_train] * synth_repeat, ignore_index=True)
            merged_train = pd.concat([seed_train, synth_train_repeated], ignore_index=True)
            merged_train = merged_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            save_dataset(merged_train, train_path)
            print(
                f"\n[train] 시드({len(seed_train)}) + 합성({len(synth_train)}×{synth_repeat}={len(synth_train_repeated)}) "
                f"병합 -> {len(merged_train)}건"
            )
            # 라벨 분포 출력 (긴급 비율 확인용)
            dist = merged_train["label"].value_counts().sort_index()
            total = len(merged_train)
            print(f"  라벨 분포: {dict(dist)} (긴급 비율 {dist.get(3, 0)/total*100:.1f}%)")
        else:
            save_dataset(seed_train, train_path)  # 시드 only 저장 (idempotent)
            print("\n[train] 합성 없음/중복제거 후 0건 또는 repeat=0 → 시드 데이터만 유지")
    else:
        print("❌ 시드 train.parquet 파일이 없습니다.")

    # 3. Val / Test 유지 (합성 데이터 섞지 않음)
    # 이미 build_processed_dataset.py 에서 생성된 순수 시드 파일을 그대로 둠
    print("\n[val / test] 오염 방지를 위해 시드 데이터만 유지합니다.")

    # 4. 합성 Hold-out 평가셋 별도 저장
    synth_val = synth_dfs["val"]
    synth_test = synth_dfs["test"]
    synth_holdout = pd.concat([synth_val, synth_test], ignore_index=True)
    if not synth_holdout.empty:
        holdout_path = PROCESSED_DIR / "synthetic_holdout.parquet"
        save_dataset(synth_holdout, holdout_path)
        print(
            f"\n[synthetic_holdout] 합성 Val+Test {len(synth_holdout)}건 분리 저장 완료 -> {holdout_path.name}"
        )

    print("\n=== 최종 데이터셋 통합 빌드 완료! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="셔플 RNG 시드 고정")
    parser.add_argument(
        "--synth-repeat",
        type=int,
        default=1,
        help="합성 train을 N회 반복 oversample (EXP-1: 8 권장). 0이면 합성 미포함",
    )
    args = parser.parse_args()

    build_final_dataset(seed=args.seed, synth_repeat=args.synth_repeat)
