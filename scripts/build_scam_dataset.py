"""SAFE v7 사기 학습 데이터 생성 파이프라인 — 합성(A3) → LLM 검수(B2) → 저장.

실행: GEMINI_API_KEY가 .env 또는 환경변수에 있어야 한다.
    python scripts/build_scam_dataset.py --per-subtype 40
결과: data/synthetic/scam/train.jsonl  (LLM 검수 통과분만, 스키마 {text,label,slice,...}).

합성/검수 로직은 src/data/scam_synthesis.py에 있고 여기서는 실제 Gemini를 주입해 실행만 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.llm_client import GeminiClient  # noqa: E402
from src.data.scam_synthesis import synthesize_scam, verify_labels  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "synthetic" / "scam" / "train.jsonl"


def _load_api_key() -> str:
    if not os.getenv("GEMINI_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY 없음 — .env 또는 환경변수 설정 필요")
    return key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-subtype", type=int, default=40, help="유형별 합성 개수")
    parser.add_argument("--no-benign", action="store_true", help="경계 반례(정상) 미생성")
    parser.add_argument("--no-verify", action="store_true", help="LLM 검수 생략(디버그)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synth-model", default="gemini-flash-latest")
    args = parser.parse_args()

    api_key = _load_api_key()
    synth_llm = GeminiClient(api_key, model=args.synth_model, temperature=0.95)

    print(f"1. 합성 (유형별 {args.per_subtype}개)...")
    examples = synthesize_scam(
        synth_llm, per_subtype=args.per_subtype, include_benign=not args.no_benign
    )
    print(
        f"   합성 {len(examples)}건 (사기 {sum(e['label'] == 1 for e in examples)} / "
        f"정상 {sum(e['label'] == 0 for e in examples)})"
    )

    if args.no_verify:
        kept, stats = examples, {"kept": len(examples), "dropped": 0, "unparsed": 0}
    else:
        print("2. LLM 라벨 검수 (B2)...")
        verify_llm = GeminiClient(api_key, model=args.synth_model, temperature=0.0)
        kept, stats = verify_labels(verify_llm, examples)
        print(f"   통과 {stats['kept']} / 탈락 {stats['dropped']} / 미파싱 {stats['unparsed']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for example in kept:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"저장: {args.out} ({len(kept)}건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
