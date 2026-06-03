"""D-2: 비순환 실데이터 hold-out 평가셋 빌드.

문제: 지금까지 긴급(3) 평가가 '합성으로 학습 → 같은 분포 합성으로 평가'하는 순환 구조라
일반화를 보장하지 못했다. 또 시드 val/test(UnSmile+KOLD)에는 긴급이 0건이라 긴급 Recall이
'측정 불가'였다.

해결:
- **AI-Hub 558 실긴급(label=3)**을 conv 단위로 train과 분리해 4-class 실데이터 hold-out 구성
  → 학습(합성 긴급)과 평가(실데이터 긴급)를 분리해 합성→실데이터 일반화를 측정.
- **BEEP(Korean HateSpeech)**을 0/1/2 실데이터 hold-out으로 추가(긴급 없음) → 비긴급 클래스의
  도메인 일반화·합성 과적합 진단.

둘 다 작은 jsonl로 export해 gitignore 예외로 커밋 → Colab clone만으로 재현.

누수 가드:
- AI-Hub hold-out에 쓰인 conv_id를 별도 파일로 남겨 train 빌드에서 배제(같은 대화 누수 방지).
- 합성 긴급과 MinHash 근사 중복 제거(합성 train과 실데이터 eval이 겹치지 않도록).

재현성: seed=42 고정.

실행:
    python scripts/build_real_holdout.py            # AI-Hub + BEEP 모두
    python scripts/build_real_holdout.py --skip-beep  # AI-Hub만(오프라인)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_dataset import load_aihub_dataframe  # noqa: E402
from src.data.dedup import find_duplicate_indices  # noqa: E402
from src.data.label_mapping import (  # noqa: E402
    LABEL_CAUTION,
    LABEL_NORMAL,
    LABEL_WARNING,
)

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
SYNTH_DIR = ROOT / "data" / "synthetic" / "emergency"

DEFAULT_SEED = 42
DEFAULT_AIHUB_EMERGENCY = 1500  # hold-out에 담을 긴급(3) 문장 목표 수
DEFAULT_BEEP_SIZE = 1500

# BEEP(Korean HateSpeech, kocohub) → 4단계 위험도 매핑 (긴급 없음)
BEEP_BASE = "https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled"
BEEP_HATE_MAP = {"none": LABEL_NORMAL, "offensive": LABEL_CAUTION, "hate": LABEL_WARNING}


def load_synthetic_emergency_texts(synth_dir: Path) -> list[str]:
    """합성 emergency jsonl의 텍스트 전체(dedup 기준)."""
    texts: list[str] = []
    if not synth_dir.exists():
        return texts
    for p in synth_dir.rglob("*.jsonl"):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    texts.append(str(json.loads(line).get("text", "")))
                except json.JSONDecodeError:
                    continue
    return texts


def build_aihub_holdout(
    raw_dir: Path,
    target_emergency: int = DEFAULT_AIHUB_EMERGENCY,
    seed: int = DEFAULT_SEED,
    synth_texts: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """AI-Hub 실긴급을 conv 단위로 분리한 4-class hold-out.

    긴급(3)을 포함한 대화를 무작위(seed) 선택해 긴급 문장이 target에 도달할 때까지 모으고,
    선택된 대화의 **모든 문장**(0/1/2/3)을 hold-out으로 삼는다(현실적 평가 분포).

    Returns:
        ``(holdout_df, holdout_conv_ids)``.
    """
    df = load_aihub_dataframe(raw_dir)
    emergency_convs = df[df["label"] == 3]["conv_id"].unique()

    rng = np.random.default_rng(seed)
    shuffled = emergency_convs.copy()
    rng.shuffle(shuffled)

    selected: list[str] = []
    n_emg = 0
    emg_per_conv = df[df["label"] == 3]["conv_id"].value_counts()
    for cid in shuffled:
        selected.append(cid)
        n_emg += int(emg_per_conv.get(cid, 0))
        if n_emg >= target_emergency:
            break

    holdout = df[df["conv_id"].isin(selected)].reset_index(drop=True)

    # 합성 긴급과 근사 중복 제거 (합성 train ↔ 실데이터 eval 분리)
    if synth_texts:
        dup_idx = find_duplicate_indices(synth_texts, holdout["text"].tolist(), threshold=0.8)
        if dup_idx:
            keep = [i for i in range(len(holdout)) if i not in dup_idx]
            print(f"  AI-Hub hold-out: 합성과 근사 중복 {len(dup_idx)}건 제거")
            holdout = holdout.iloc[keep].reset_index(drop=True)

    return holdout, sorted(selected)


def fetch_beep(raw_dir: Path) -> dict[str, Path]:
    """BEEP labeled train/dev tsv 다운로드(없을 때만)."""
    beep_dir = raw_dir / "korean_hatespeech"
    beep_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "dev"):
        dest = beep_dir / f"{split}.tsv"
        if not dest.exists():
            url = f"{BEEP_BASE}/{split}.tsv"
            print(f"  [BEEP/{split}] 다운로드: {url}")
            urllib.request.urlretrieve(url, dest)  # noqa: S310
        paths[split] = dest
    return paths


def build_beep_holdout(
    raw_dir: Path, target: int = DEFAULT_BEEP_SIZE, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """BEEP을 0/1/2 실데이터 hold-out으로(긴급 없음)."""
    paths = fetch_beep(raw_dir)
    rows: list[dict] = []
    for split, p in paths.items():
        with p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, rec in enumerate(reader):
                text = (rec.get("comments") or "").strip()
                hate = (rec.get("hate") or "").strip().lower()
                if not text or hate not in BEEP_HATE_MAP:
                    continue
                rows.append(
                    {
                        "text": text,
                        "label": BEEP_HATE_MAP[hate],
                        "source": "beep",
                        "source_id": f"beep_{split}_{i}",
                    }
                )
    df = pd.DataFrame(rows)
    if len(df) > target:
        df = df.sample(n=target, random_state=seed).reset_index(drop=True)
    return df


def export_jsonl(df: pd.DataFrame, path: Path, cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({c: r[c] for c in cols}, ensure_ascii=False) + "\n")
    print(
        f"  ✔ {path.relative_to(ROOT)} — {len(df)}행 {dict(df['label'].value_counts().sort_index())}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--aihub-emergency", type=int, default=DEFAULT_AIHUB_EMERGENCY)
    parser.add_argument("--beep-size", type=int, default=DEFAULT_BEEP_SIZE)
    parser.add_argument("--skip-beep", action="store_true", help="오프라인이면 BEEP 다운로드 생략")
    args = parser.parse_args()

    print("=== D-2 실데이터 hold-out 빌드 ===")
    synth_texts = load_synthetic_emergency_texts(SYNTH_DIR)
    print(f"합성 dedup 기준 텍스트 {len(synth_texts)}건 로드")

    print("\n[1] AI-Hub 실긴급 hold-out")
    aihub, conv_ids = build_aihub_holdout(RAW_DIR, args.aihub_emergency, args.seed, synth_texts)
    export_jsonl(
        aihub, EVAL_DIR / "aihub_real_holdout.jsonl", ["text", "label", "source", "source_id"]
    )
    conv_path = EVAL_DIR / "aihub_holdout_conv_ids.txt"
    conv_path.write_text("\n".join(conv_ids) + "\n", encoding="utf-8")
    print(f"  ✔ {conv_path.relative_to(ROOT)} — {len(conv_ids)}개 conv_id (train 배제용)")

    if not args.skip_beep:
        print("\n[2] BEEP 0/1/2 hold-out")
        try:
            beep = build_beep_holdout(RAW_DIR, args.beep_size, args.seed)
            export_jsonl(
                beep, EVAL_DIR / "beep_real_holdout.jsonl", ["text", "label", "source", "source_id"]
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ BEEP 빌드 실패(네트워크?): {type(e).__name__}: {e}")
            print("    → 네트워크 가능한 환경에서 `python scripts/build_real_holdout.py` 재실행")
    else:
        print("\n[2] BEEP 생략(--skip-beep)")

    print("\n=== 완료 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
