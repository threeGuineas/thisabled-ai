"""PAN12 그루밍 코퍼스 → 번역 증강용 턴 추출 (계획서 ①단계).

입력: data/raw/pan12/ 아래에 압축 해제된 PAN12 파일들
  - 학습 코퍼스 XML (conversations/messages)
  - predator ID 목록 txt (한 줄에 한 ID)
출력: data/processed/pan12_extracted.jsonl
  {"text_en", "split_role": "predator"|"normal", "conv_id", "turn_idx"}

규칙 (docs/grooming_번역_증강_계획.md):
- predator 발화 턴만 '주의' 후보로. 3단어 미만·URL 포함·기호뿐인 턴 제거.
- 대화당 최대 --per-conv 턴 (편중 방지).
- 정상 풀: predator가 등장하지 않는 대화의 턴, 동일 필터, predator 분량과 1:1.
- seed 42 고정.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "pan12"
OUT_PATH = ROOT / "data" / "processed" / "pan12_extracted.jsonl"

URL_RE = re.compile(r"https?://|www\.|\.com|\.net", re.I)
ALNUM_RE = re.compile(r"[a-zA-Z]")


def find_inputs() -> tuple[Path, Path]:
    """training 코퍼스 XML과 predator ID txt를 자동 탐색."""
    xmls = sorted(RAW_DIR.rglob("*training*corpus*.xml")) or sorted(RAW_DIR.rglob("*.xml"))
    preds = sorted(RAW_DIR.rglob("*predator*id*.txt")) or sorted(RAW_DIR.rglob("*predator*.txt"))
    if not xmls or not preds:
        sys.exit(
            f"입력 파일을 못 찾음 — {RAW_DIR} 아래에 zip을 풀었는지 확인.\n"
            f"xml 후보: {[str(p) for p in xmls]}\npredator txt 후보: {[str(p) for p in preds]}"
        )
    # 가장 큰 xml = 본 코퍼스
    xml = max(xmls, key=lambda p: p.stat().st_size)
    return xml, preds[0]


def usable(text: str) -> bool:
    words = text.split()
    if len(words) < 3:
        return False
    if URL_RE.search(text):
        return False
    if not ALNUM_RE.search(text):  # 기호·숫자뿐
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=4000, help="predator 턴 목표 수")
    ap.add_argument("--per-conv", type=int, default=10, help="대화당 최대 턴")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    xml_path, pred_path = find_inputs()
    predator_ids = {line.strip() for line in pred_path.read_text().splitlines() if line.strip()}
    print(f"corpus: {xml_path.name} / predators: {len(predator_ids)}명")

    predator_turns: list[dict] = []
    normal_turns: list[dict] = []

    # 대용량 XML — iterparse로 스트리밍
    for _, conv in ET.iterparse(str(xml_path), events=("end",)):
        if conv.tag != "conversation":
            continue
        conv_id = conv.get("id", "")
        msgs = []
        has_pred = False
        for m in conv.findall("message"):
            author = (m.findtext("author") or "").strip()
            text = (m.findtext("text") or "").strip()
            is_pred = author in predator_ids
            has_pred = has_pred or is_pred
            if text and usable(text):
                msgs.append({"author_is_pred": is_pred, "text": text})
        if msgs:
            pool = [m for m in msgs if m["author_is_pred"]] if has_pred else msgs
            role = "predator" if has_pred else "normal"
            if len(pool) > args.per_conv:
                pool = rng.sample(pool, args.per_conv)
            bucket = predator_turns if has_pred else normal_turns
            for i, m in enumerate(pool):
                bucket.append(
                    {"text_en": m["text"], "split_role": role, "conv_id": conv_id, "turn_idx": i}
                )
        conv.clear()  # 메모리 해제

    rng.shuffle(predator_turns)
    rng.shuffle(normal_turns)
    predator_turns = predator_turns[: args.target]
    normal_turns = normal_turns[: len(predator_turns)]  # 1:1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in predator_turns + normal_turns:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"predator {len(predator_turns)} / normal {len(normal_turns)} → {OUT_PATH}")
    print("다음 단계: python scripts/translate_pan12.py")


if __name__ == "__main__":
    main()
