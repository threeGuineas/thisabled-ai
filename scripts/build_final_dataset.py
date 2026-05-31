"""통합 최종 데이터셋 빌드 (시드 데이터 + 제미나이 합성 데이터).

data/processed/{train,val,test}.parquet 에 존재하는 시드 데이터와
data/synthetic/emergency/ 하위의 합성 데이터(*.jsonl)를 split 별로 결합하여
동일한 data/processed/ 하위에 덮어씁니다.
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

PROCESSED_DIR = ROOT / "data" / "processed"
SYNTH_DIR = ROOT / "data" / "synthetic" / "emergency"


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
        # 스키마 통일화 및 가공
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)

        # source_id 고유 식별자 생성
        # synth_<category>_<split>_<index>
        df["source"] = df.get("source", "synthetic_emergency_v1")
        df["source_id"] = df.apply(
            lambda r, _s=split: f"synth_{r.get('subcategory', 'unknown')}_{_s}_{r.name}",
            axis=1,
        )

        # 최종 스키마만 유지
        dfs[split] = df[["text", "label", "source", "source_id"]]
        print(f"  → 합성 [{split}]: {len(dfs[split])}건 로드 완료")

    return dfs


def build_final_dataset(seed: int = 42) -> None:
    print("=== 통합 최종 데이터셋 빌드 시작 ===")

    # 1. 합성 데이터 로드
    print(f"1. 합성 데이터 로딩 중... ({SYNTH_DIR})")
    synth_dfs = load_synthetic_splits(SYNTH_DIR)

    # 2. 각 split별 병합 및 덮어쓰기
    for split in ["train", "val", "test"]:
        seed_path = PROCESSED_DIR / f"{split}.parquet"
        if not seed_path.exists():
            print(f"❌ 시드 Parquet 파일이 없습니다: {seed_path}")
            sys.exit(1)

        # 시드 데이터 로드
        seed_df = pd.read_parquet(seed_path)
        synth_df = synth_dfs[split]

        print(f"\n[{split}] 병합 정보:")
        print(f"  - 시드 데이터: {len(seed_df)}건")
        print(f"  - 합성 데이터: {len(synth_df)}건")

        # 3. 결합
        if not synth_df.empty:
            merged_df = pd.concat([seed_df, synth_df], ignore_index=True)
        else:
            merged_df = seed_df

        # 4. Shuffle (RNG 고정)
        merged_df = merged_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        # 5. 저장 (덮어쓰기)
        save_dataset(merged_df, seed_path)
        print(f"  → 최종 병합 성공! 총 {len(merged_df)}건 저장 완료: {seed_path.relative_to(ROOT)}")

        # 6. 간단 분포 출력
        dist = merged_df["label"].value_counts().sort_index()
        dist_str = ", ".join(f"라벨 {lbl}: {cnt}건" for lbl, cnt in dist.items())
        print(f"  → 분포: [{dist_str}]")

    print("\n=== 최종 데이터셋 통합 빌드 완료! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="셔플 RNG 시드 고정")
    args = parser.parse_args()

    build_final_dataset(seed=args.seed)
