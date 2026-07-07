"""PAN12 그루밍 코퍼스 → 번역 증강용 대화 window 추출 (계획서 ①단계).

입력: data/raw/pan12/ 아래에 압축 해제된 PAN12 파일들
  - 학습 코퍼스 XML (conversations/messages)
  - predator ID 목록 txt (한 줄에 한 ID)
출력: data/processed/pan12_extracted.jsonl
  {"text_en", "split_role": "predator"|"normal", "conv_id", "win_idx", "n_turns"}

설계 (docs/grooming_번역_증강_계획.md — 대화 window 방식):
- 개별 턴이 아니라 **같은 화자의 연속 발화 W턴을 한 덩어리(window)**로 묶는다.
  그루밍은 신뢰형성→고립→요구가 여러 턴에 걸쳐 나타나므로 'hey!!' 같은 짧은 턴도
  맥락 안에서 신호가 살아난다. 실제 채팅 메시지 흐름과도 유사.
- predator window = predator 발화 연속분 → '주의' 후보.
- 정상 window = predator 미등장 대화에서 한 화자의 연속 발화 → '정상'.
- window 단위 필터: 합쳐진 텍스트가 최소 단어 수 이상, URL 없음. (개별 턴 필터는 완화)
- 대화당 최대 --per-conv window (편중 방지), 1:1, seed 42.
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
    """**training 코퍼스**의 XML과 predator ID txt를 짝 맞춰 탐색.

    주의: raw/pan12 아래에 training(164M)·test(377M) 두 코퍼스가 함께 있다. XML과 predator
    파일은 반드시 같은 코퍼스에서 골라야 한다(교차 선택 시 predator 매칭 실패 → 전량 normal).
    학습 데이터 증강이므로 training 코퍼스를 사용한다.
    """
    xmls = sorted(RAW_DIR.rglob("*training*corpus*.xml"))
    preds = sorted(
        p for p in RAW_DIR.rglob("*predators*.txt") if "training" in p.name and "diff" not in p.name
    )
    if not xmls or not preds:
        sys.exit(
            f"training 코퍼스 입력을 못 찾음 — {RAW_DIR} 아래 압축 해제 확인.\n"
            f"xml 후보: {[str(p) for p in xmls]}\npredator txt 후보: {[str(p) for p in preds]}"
        )
    xml = max(xmls, key=lambda p: p.stat().st_size)  # training corpus xml
    return xml, preds[0]


def usable_turn(text: str) -> bool:
    """개별 턴 — window 구성 재료. 시스템 알림·빈 줄·URL만 제거 (짧은 턴은 유지)."""
    if not text or URL_RE.search(text):
        return False
    return bool(ALNUM_RE.search(text))


def usable_window(text: str, min_words: int) -> bool:
    """묶인 window — 최소 단어 수 이상이어야 학습 신호가 됨."""
    return len(text.split()) >= min_words


def make_windows(msgs: list[dict], author_filter, size: int) -> list[str]:
    """대상 화자의 발화를 대화 순서대로 모아 size턴씩 묶는다.

    실제 채팅은 화자가 번갈아 말해 '연속 같은 화자' 구간이 대부분 1턴이므로, 연속성에
    의존하지 않고 대상 화자(predator) 발화를 **순서대로 이어붙여** 청킹한다. 상대 발화가
    사이에 껴도 predator의 말 흐름(신뢰형성→고립→요구)은 순서대로 보존된다.
    author_filter(is_pred)==True인 턴만 대상.
    """
    picked = [m["text"] for m in msgs if author_filter(m["is_pred"])]
    return [" ".join(picked[i : i + size]) for i in range(0, len(picked), size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=250, help="predator window 목표 수 (파일럿 250)")
    ap.add_argument("--per-conv", type=int, default=5, help="대화당 최대 window")
    ap.add_argument("--window", type=int, default=4, help="window 당 연속 턴 수")
    ap.add_argument("--min-words", type=int, default=6, help="window 최소 단어 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    xml_path, pred_path = find_inputs()
    predator_ids = {line.strip() for line in pred_path.read_text().splitlines() if line.strip()}
    print(f"corpus: {xml_path.name} / predators: {len(predator_ids)}명")

    predator_wins: list[dict] = []
    normal_wins: list[dict] = []

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
            if usable_turn(text):
                msgs.append({"is_pred": is_pred, "text": text})
        conv.clear()  # 메모리 해제 먼저
        if not msgs:
            continue

        if has_pred:
            wins = make_windows(msgs, lambda p: p, args.window)
            role, bucket = "predator", predator_wins
        else:
            # 정상 대화: 첫 화자 기준 연속 발화 (author 정보는 window엔 불필요)
            wins = make_windows(msgs, lambda p: True, args.window)
            role, bucket = "normal", normal_wins

        wins = [w for w in wins if usable_window(w, args.min_words)]
        if len(wins) > args.per_conv:
            wins = rng.sample(wins, args.per_conv)
        for i, w in enumerate(wins):
            bucket.append(
                {
                    "text_en": w,
                    "split_role": role,
                    "conv_id": conv_id,
                    "win_idx": i,
                    "n_turns": min(args.window, w.count(" ") + 1),
                }
            )

    rng.shuffle(predator_wins)
    rng.shuffle(normal_wins)
    predator_wins = predator_wins[: args.target]
    normal_wins = normal_wins[: len(predator_wins)]  # 1:1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in predator_wins + normal_wins:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"predator {len(predator_wins)} / normal {len(normal_wins)} window → {OUT_PATH}")
    print("다음 단계: python scripts/translate_pan12.py")


if __name__ == "__main__":
    main()
