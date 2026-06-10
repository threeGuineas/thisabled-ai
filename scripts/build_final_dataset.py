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
EVAL_DIR = ROOT / "data" / "eval"
AIHUB_TRAIN_JSONL = EVAL_DIR / "aihub_train.jsonl"  # D-2 보완(B): 실데이터 train 투입분
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
    """idempotent를 위해 기존 주입(합성·AI-Hub실데이터) row를 제거하고 시드만 반환."""
    src = df["source"].astype(str)
    injected = src.str.startswith("synthetic_") | src.str.startswith("aihub_real")
    return df[~injected].reset_index(drop=True)


def _load_aihub_train() -> pd.DataFrame:
    cols = ["text", "label", "source", "source_id"]
    if not AIHUB_TRAIN_JSONL.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_json(AIHUB_TRAIN_JSONL, lines=True)
    return df[cols]


def build_final_dataset(seed: int = 42, synth_repeat: int = 1, include_aihub: bool = False) -> None:
    print(
        f"=== 통합 최종 데이터셋 빌드 (synth_repeat={synth_repeat}, include_aihub={include_aihub}) ==="
    )

    # 1. 합성 데이터 로드
    print(f"1. 합성 데이터 로딩 중... ({SYNTH_DIR})")
    synth_dfs = load_synthetic_splits(SYNTH_DIR)

    # 2. Train 병합 (시드 + 합성×N + (선택)AI-Hub 실데이터)
    train_path = PROCESSED_DIR / "train.parquet"
    if train_path.exists():
        seed_train = _filter_seed_only(pd.read_parquet(train_path))
        parts = [seed_train]
        desc = [f"시드({len(seed_train)})"]

        synth_train = synth_dfs["train"]
        if not synth_train.empty and synth_repeat > 0:
            # 누수 차단: 시드 train과 근사 중복인 합성 행을 oversample 이전에 제거.
            synth_train, n_removed = deduplicate_against(
                seed_train["text"], synth_train, threshold=DEDUP_THRESHOLD
            )
            print(
                f"  → MinHash 중복 제거(threshold={DEDUP_THRESHOLD}): "
                f"시드 train과 근사 중복 합성 {n_removed}건 제거 → 합성 {len(synth_train)}건 잔존"
            )
            if not synth_train.empty:
                synth_rep = pd.concat([synth_train] * synth_repeat, ignore_index=True)
                parts.append(synth_rep)
                desc.append(f"합성({len(synth_train)}×{synth_repeat}={len(synth_rep)})")

        if include_aihub:
            aihub_train = _load_aihub_train()
            if aihub_train.empty:
                print(f"  ⚠ {AIHUB_TRAIN_JSONL.name} 없음 → AI-Hub 투입 생략")
            else:
                aihub_train, n_dup = deduplicate_against(
                    seed_train["text"], aihub_train, threshold=DEDUP_THRESHOLD
                )
                parts.append(aihub_train)
                desc.append(f"AI-Hub실({len(aihub_train)}, 시드중복 {n_dup}제거)")

        merged_train = (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        save_dataset(merged_train, train_path)
        dist = merged_train["label"].value_counts().sort_index()
        total = len(merged_train)
        print(f"\n[train] {' + '.join(desc)} 병합 -> {total}건")
        print(f"  라벨 분포: {dict(dist)} (긴급 비율 {dist.get(3, 0)/total*100:.1f}%)")
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
    parser.add_argument(
        "--include-aihub-train",
        action="store_true",
        help="D-2 보완(B): data/eval/aihub_train.jsonl(실데이터, hold-out 배제분)을 train에 투입",
    )
    args = parser.parse_args()

    build_final_dataset(
        seed=args.seed, synth_repeat=args.synth_repeat, include_aihub=args.include_aihub_train
    )
