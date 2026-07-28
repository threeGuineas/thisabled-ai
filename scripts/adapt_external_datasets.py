"""외부 데이터셋 → SAFE 이진 학습용 jsonl 어댑터 (R1: DKTC + K-MHaS + APEACH).

산출:
  data/synthetic/dktc.jsonl        {text, source:"dktc", split_role:"threat", cls, conv_idx, win_idx}
                                   → configs extra_caution_jsonl 경로 (전량 label=1 주의)
  data/synthetic/kmhas.jsonl       {text, label(0/1), source:"kmhas"}
  data/synthetic/apeach.jsonl      {text, label(0/1), source:"apeach"}
                                   → configs extra_train_jsonl 경로 (라벨 보존)

누수 차단: 세 소스 모두 실데이터 홀드아웃(aihub/beep) + 소비 blind(v1..v9) +
grooming 개발셋과
MinHash(0.8) 및 정규화 exact 로 교차 dedup 후 저장. (trainer의 extra_* 경로는
build_final_dataset 의 홀드아웃 dedup을 거치지 않으므로 여기서 선제 차단한다.)

실행: python scripts/adapt_external_datasets.py
근거 문서: docs/safe_데이터셋_학습적용_플랜.md
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dedup import deduplicate_against  # noqa: E402

SYNTH_DIR = ROOT / "data" / "synthetic"
EVAL_DIR = ROOT / "data" / "eval"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
DKTC_CSV = ROOT / "data" / "raw" / "dktc" / "train.csv"
DKTC_URL = "https://raw.githubusercontent.com/tunib-ai/DKTC/main/data/train.csv"

DEDUP_THRESHOLD = 0.8
KMHAS_CAP = 15000  # 79k 전량은 base(~48k)를 압도 → 균형 위해 이진 층화 샘플
SEED = 42

_EXACT_NOISE = re.compile(r"[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ]+", re.IGNORECASE)
CONSUMED_BLIND_VERSIONS = tuple(range(1, 10))
GROOMING_DEV_FILES = (
    SYNTH_DIR / "emergency" / "3a" / "val.jsonl",
    SYNTH_DIR / "emergency" / "3a" / "test.jsonl",
)


def _norm(t: object) -> str:
    """공백·구두점·대소문자를 무시하되 한글 자모는 보존하는 exact key."""

    return _EXACT_NOISE.sub("", str(t)).lower()


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def load_forbidden_texts() -> pd.Series:
    """홀드아웃 + 소비 blind 원문 — 신규 소스에서 이와 겹치면 제거(누수 차단)."""
    texts: list[str] = []
    src_files = [
        EVAL_DIR / "aihub_real_holdout.jsonl",
        EVAL_DIR / "beep_real_holdout.jsonl",
        *(FIXTURE_DIR / f"safe_blind_v{version}.jsonl" for version in CONSUMED_BLIND_VERSIONS),
        *GROOMING_DEV_FILES,
    ]
    for p in src_files:
        if not p.exists():
            print(f"  · 참조 없음(건너뜀): {_display_path(p)}")
            continue
        n = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = row.get("text")
                if t:
                    texts.append(str(t))
                    n += 1
        print(f"  · forbidden 참조 {n:5d}건 ← {_display_path(p)}")
    print(f"  → forbidden 총 {len(texts)}건")
    return pd.Series(texts, dtype="object")


def dedup_report(df: pd.DataFrame, forbidden: pd.Series, name: str) -> pd.DataFrame:
    """정규화 exact + MinHash near-dup 두 단계로 forbidden 제거."""
    before = len(df)
    forb_norm = {key for text in forbidden if (key := _norm(text))}
    candidate_keys = df["text"].map(_norm)
    df = df[~candidate_keys.isin(forb_norm) & candidate_keys.ne("")].reset_index(drop=True)
    exact_removed = before - len(df)
    if len(df):
        df, near_removed = deduplicate_against(forbidden, df, threshold=DEDUP_THRESHOLD)
    else:
        near_removed = 0
    # 자기 자신 내부 중복(정규화 exact)도 제거
    df = df.loc[~df["text"].map(_norm).duplicated()].reset_index(drop=True)
    print(
        f"[{name}] {before} → {len(df)}  (홀드아웃/blind exact -{exact_removed}, near -{near_removed})"
    )
    return df


# ---------- DKTC ----------
def ensure_dktc_csv() -> Path:
    """공개 DKTC train CSV를 필요할 때만 내려받고 필수 스키마를 검증한다."""
    if DKTC_CSV.exists() and DKTC_CSV.stat().st_size > 0:
        print(f"[dktc] 기존 원본 사용: {DKTC_CSV.relative_to(ROOT)}")
        return DKTC_CSV

    DKTC_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = DKTC_CSV.with_suffix(".csv.part")
    print(f"[dktc] 다운로드: {DKTC_URL}")
    try:
        request = urllib.request.Request(DKTC_URL, headers={"User-Agent": "thisabled-ai/SAFE-v6"})
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1 << 20):
                output.write(chunk)
        columns = set(pd.read_csv(temporary, nrows=1).columns)
        required = {"idx", "class", "conversation"}
        if missing := required - columns:
            raise ValueError(f"DKTC CSV 필수 열 누락: {sorted(missing)}")
        temporary.replace(DKTC_CSV)
    except (OSError, urllib.error.URLError, ValueError, pd.errors.ParserError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"DKTC 다운로드/검증 실패 ({DKTC_URL}): {exc}") from exc

    print(f"[dktc] 다운로드 완료: {DKTC_CSV.stat().st_size:,} bytes")
    return DKTC_CSV


def build_dktc(
    window: int = 3, stride: int = 2, per_conv: int = 4, min_words: int = 6
) -> pd.DataFrame:
    df = pd.read_csv(ensure_dktc_csv())
    rows: list[dict] = []
    for _, r in df.iterrows():
        conv_idx = int(r["idx"])
        cls = str(r["class"])
        turns = [t.strip() for t in str(r["conversation"]).split("\n") if t.strip()]
        wins = []
        for i in range(0, max(1, len(turns) - window + 1), stride):
            chunk = " ".join(turns[i : i + window])
            if len(chunk.split()) >= min_words:
                wins.append(chunk)
            if len(wins) >= per_conv:
                break
        for j, w in enumerate(wins):
            rows.append(
                {
                    "text": w,
                    "label": 1,  # DKTC 4클래스 전량 위협 → 주의. extra_train 경로(라벨 보존)로 주입.
                    "source": "dktc",
                    "split_role": "threat",
                    "cls": cls,
                    "conv_idx": conv_idx,
                    "win_idx": j,
                }
            )
    out = pd.DataFrame(rows)
    print(f"[dktc] 대화 {len(df)} → window {len(out)} (클래스별 {dict(out['cls'].value_counts())})")
    return out


# ---------- K-MHaS / APEACH ----------
def build_kmhas() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("jeanlee/kmhas_korean_hate_speech")["train"]
    df = pd.DataFrame({"text": ds["text"], "labels": ds["label"]})
    # label 8 = not-hate(정상). 그 외 포함 시 주의.
    df["label"] = df["labels"].map(lambda ls: 0 if list(ls) == [8] else 1)
    df["text"] = df["text"].map(lambda t: str(t).strip())
    df = df[df["text"].str.len() > 0].drop_duplicates("text")
    # 이진 층화 샘플로 base 압도 방지
    if len(df) > KMHAS_CAP:
        frac = KMHAS_CAP / len(df)
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=SEED))
            .reset_index(drop=True)
        )
    df["source"] = "kmhas"
    print(
        f"[kmhas] 표본 {len(df)} (정상 {int((df.label==0).sum())} / 주의 {int((df.label==1).sum())})"
    )
    return df[["text", "label", "source"]]


def build_apeach() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("jason9693/APEACH")
    frames = []
    for split in ds:
        frames.append(pd.DataFrame({"text": ds[split]["text"], "label": ds[split]["class"]}))
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].map(lambda t: str(t).strip())
    df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["source"] = "apeach"
    print(f"[apeach] {len(df)} (정상 {int((df.label==0).sum())} / 주의 {int((df.label==1).sum())})")
    return df[["text", "label", "source"]]


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"  ✔ 저장 {len(df):5d}행 → {path.relative_to(ROOT)}")


def main() -> None:
    print("=== forbidden(홀드아웃+blind) 로드 ===")
    forbidden = load_forbidden_texts()

    print("\n=== DKTC ===")
    dktc = build_dktc()
    dktc = dedup_report(dktc, forbidden, "dktc")
    write_jsonl(dktc, SYNTH_DIR / "dktc.jsonl")

    print("\n=== K-MHaS ===")
    kmhas = build_kmhas()
    kmhas = dedup_report(kmhas, forbidden, "kmhas")
    write_jsonl(kmhas, SYNTH_DIR / "kmhas.jsonl")

    print("\n=== APEACH ===")
    apeach = build_apeach()
    apeach = dedup_report(apeach, forbidden, "apeach")
    write_jsonl(apeach, SYNTH_DIR / "apeach.jsonl")

    print("\n=== 요약 ===")
    print(f"dktc(주의 전량) {len(dktc)} · kmhas {len(kmhas)} · apeach {len(apeach)}")


if __name__ == "__main__":
    csv.field_size_limit(10**7)
    main()
