"""D-2 보완(B): AI-Hub 실데이터 학습셋 추출 (hold-out conv 완전 배제).

배경: 합성-only 긴급 학습이 실긴급에 전이되지 않음(real-holdout 긴급 Recall 0.005,
합성↔실 갭 0.995). 보완책으로 hold-out에 쓰이지 않은 AI-Hub conv에서 4-class 학습셋을
만들어 train에 섞는다 → "실데이터를 train에 넣으면 실 Recall이 오르는가" 검증(B).

누수 가드(이중):
1. `data/eval/aihub_holdout_conv_ids.txt`의 conv는 전량 제외 → hold-out과 conv 분리.
2. hold-out 텍스트와 MinHash 근사 중복 제거.

재현성: seed=42. 작은 jsonl로 커밋 → Colab clone 재현(raw는 git 밖).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_dataset import load_aihub_dataframe  # noqa: E402
from src.data.dedup import find_duplicate_indices  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
HOLDOUT_CONV = EVAL_DIR / "aihub_holdout_conv_ids.txt"
OUT_PATH = EVAL_DIR / "aihub_train.jsonl"

DEFAULT_SEED = 42
MAX_EMERGENCY = 8000  # 투입 긴급(3) 상한
MAX_OTHER = 12000  # 투입 0/1/2 합계 상한 (도메인 적응용, 라벨 비례 샘플)


def build_aihub_train(
    raw_dir: Path,
    holdout_conv_path: Path,
    seed: int = DEFAULT_SEED,
    max_emergency: int = MAX_EMERGENCY,
    max_other: int = MAX_OTHER,
) -> pd.DataFrame:
    df = load_aihub_dataframe(raw_dir)

    holdout_convs = set(holdout_conv_path.read_text(encoding="utf-8").split())
    before = len(df)
    df = df[~df["conv_id"].isin(holdout_convs)].reset_index(drop=True)
    print(f"  hold-out conv {len(holdout_convs)}개 제외: {before} → {len(df)}행")

    emg = df[df["label"] == 3]
    if len(emg) > max_emergency:
        emg = emg.sample(n=max_emergency, random_state=seed)
    others = df[df["label"] != 3]
    if len(others) > max_other:
        others = others.groupby("label", group_keys=False).apply(
            lambda g: g.sample(n=int(round(max_other * len(g) / len(others))), random_state=seed)
        )
    out = pd.concat([emg, others], ignore_index=True).sample(frac=1.0, random_state=seed)
    out = out[["text", "label"]].reset_index(drop=True)
    out["source"] = "aihub_real"
    out["source_id"] = ["aihub_real_" + str(i) for i in range(len(out))]

    # hold-out 텍스트와 근사 중복 제거(이중 가드)
    holdout_jsonl = EVAL_DIR / "aihub_real_holdout.jsonl"
    if holdout_jsonl.exists():
        ho_texts = pd.read_json(holdout_jsonl, lines=True)["text"].tolist()
        dup = find_duplicate_indices(ho_texts, out["text"].tolist(), threshold=0.9)
        if dup:
            out = out.iloc[[i for i in range(len(out)) if i not in dup]].reset_index(drop=True)
            print(f"  hold-out과 근사 중복 {len(dup)}건 제거")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-emergency", type=int, default=MAX_EMERGENCY)
    parser.add_argument("--max-other", type=int, default=MAX_OTHER)
    args = parser.parse_args()

    print("=== AI-Hub 실데이터 train 추출 (hold-out 배제) ===")
    if not HOLDOUT_CONV.exists():
        print(f"❌ {HOLDOUT_CONV} 없음 — 먼저 scripts/build_real_holdout.py 실행")
        return 1

    df = build_aihub_train(RAW_DIR, HOLDOUT_CONV, args.seed, args.max_emergency, args.max_other)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(
                json.dumps(
                    {
                        "text": r["text"],
                        "label": int(r["label"]),
                        "source": r["source"],
                        "source_id": r["source_id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    dist = dict(df["label"].value_counts().sort_index())
    print(f"  ✔ {OUT_PATH.relative_to(ROOT)} — {len(df)}행 {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
