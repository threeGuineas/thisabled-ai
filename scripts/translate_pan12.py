"""PAN12 추출 턴 → 한국어 현지화 번역 (계획서 ②단계).

입력: data/processed/pan12_extracted.jsonl (scripts/extract_pan12.py 산출)
출력: data/synthetic/pan12_translated.jsonl
  {"text": <한국어>, "source": "pan12", "split_role", "conv_id", "turn_idx"}

- GPT-4o-mini 배치 번역 (환경변수 OPENAI_API_KEY, .env 자동 로드)
- 현지화: 한국 메신저 대화체(반말·축약), 이름/지역 익명 치환, 문화 맥락 변환
- 재개 가능: 출력 파일에 이미 있는 (conv_id, turn_idx)는 건너뜀 → 중단돼도 재실행하면 이어감
- 안전장치: 거부 응답([REFUSED]) 감지 시 1회 재시도 후 스킵, 비용 추정 출력
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "pan12_extracted.jsonl"
OUT_PATH = ROOT / "data" / "synthetic" / "pan12_translated.jsonl"
BATCH = 20  # 한 요청에 묶을 턴 수
MODEL = os.getenv("PAN12_TRANSLATE_MODEL", "gpt-4o-mini")

SYSTEM = """당신은 아동·청소년 온라인 보호 연구를 위한 데이터 구축 번역가입니다.
온라인 그루밍 탐지 모델 학습용으로, 영어 채팅 발화를 한국어로 번역합니다.
유해한 내용이 포함되지만 이는 탐지 모델이 배워야 할 신호이므로 완곡하게 바꾸지 말고
의미와 위험 신호를 보존하세요. 규칙:
1. 직역 금지 — 한국 메신저 대화체로 현지화 (반말, 축약, 자연스러운 구어)
2. 사람 이름→한국식 가명, 지명·학교→한국식, 달러→만원 등 문화 맥락 변환
3. 위험 신호(비밀 유도, 만남 제안, 사진 요구, 칭찬을 통한 신뢰 형성 등)는 뉘앙스 그대로 보존
4. 각 문장을 JSON 배열로만 응답: ["번역1", "번역2", ...] (입력 순서 유지, 개수 동일)
5. 번역이 불가능한 항목만 "[SKIP]"으로 표기"""


def load_done() -> set[tuple[str, int]]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            done.add((r["conv_id"], r["turn_idx"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def translate_batch(client, rows: list[dict]) -> list[str | None]:
    numbered = json.dumps([r["text_en"] for r in rows], ensure_ascii=False)
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": numbered},
            ],
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        # 코드펜스 제거
        if content.startswith("```"):
            content = content.strip("`").lstrip("json").strip()
        try:
            out = json.loads(content)
            if isinstance(out, list) and len(out) == len(rows):
                return [None if t == "[SKIP]" else t for t in out]
        except json.JSONDecodeError:
            pass
        time.sleep(2 * (attempt + 1))
    return [None] * len(rows)


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY 없음 — .env 확인")
    from openai import OpenAI

    client = OpenAI()

    rows = [json.loads(line) for line in IN_PATH.read_text(encoding="utf-8").splitlines()]
    done = load_done()
    todo = [r for r in rows if (r["conv_id"], r["turn_idx"]) not in done]
    est_tokens = sum(len(r["text_en"].split()) for r in todo) * 2.5  # 대략치(입출력)
    print(f"전체 {len(rows)} / 완료 {len(done)} / 남음 {len(todo)}")
    print(f"예상 토큰 ~{est_tokens / 1e6:.2f}M → gpt-4o-mini 기준 ~${est_tokens / 1e6 * 0.4:.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            results = translate_batch(client, batch)
            for r, ko in zip(batch, results, strict=True):
                if ko is None:
                    skipped += 1
                    continue
                f.write(
                    json.dumps(
                        {
                            "text": ko,
                            "source": "pan12",
                            "split_role": r["split_role"],
                            "conv_id": r["conv_id"],
                            "turn_idx": r["turn_idx"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)} (스킵 {skipped})")

    print(f"완료 → {OUT_PATH} (스킵 {skipped}건)")
    print("다음 단계: 무작위 200건 수동 검수 (계획서 ③) 후 병합")


if __name__ == "__main__":
    main()
