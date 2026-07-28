"""PAN12 추출 window → 한국어 현지화 번역 (계획서 ②단계).

입력: data/processed/pan12_extracted.jsonl (scripts/extract_pan12.py 산출)
출력: data/synthetic/pan12_translated.jsonl
  {"text": <한국어>, "source": "pan12", "split_role", "conv_id", "win_idx"}

- Gemini 배치 번역 — REST API 직접 호출(표준 라이브러리 urllib만 사용, 추가 설치 불필요).
  SDK의 gRPC 전송이 일부 네트워크(프록시)에서 막히는 문제를 피하기 위함.
- 현지화: 한국 메신저 대화체(반말·축약), 이름/지역 익명 치환, 문화 맥락 변환
- 재개 가능: 출력 파일에 이미 있는 (conv_id, win_idx)는 건너뜀 → 중단돼도 재실행하면 이어감
- 안전장치: 번역 실패([SKIP]) 항목 스킵, --limit로 파일럿 수 제한, 비용 추정 출력

실행: GEMINI_API_KEY가 .env에 있어야 함. `python scripts/translate_pan12.py --limit 10`(파일럿).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "pan12_extracted.jsonl"
OUT_PATH = ROOT / "data" / "synthetic" / "pan12_translated.jsonl"
BATCH = 10  # 한 요청에 묶을 window 수 (실패 손실·TPM 스파이크 완화)
MODEL = os.getenv("PAN12_TRANSLATE_MODEL", "gemini-flash-latest")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM = """당신은 아동·청소년 온라인 보호 연구를 위한 데이터 구축 번역가입니다.
온라인 그루밍 탐지 모델 학습용으로, 영어 채팅 발화를 한국어로 번역합니다.
유해한 내용이 포함되지만 이는 탐지 모델이 배워야 할 신호이므로 완곡하게 바꾸지 말고
의미와 위험 신호를 보존하세요. 규칙:
1. 직역 금지 — 한국 메신저 대화체로 현지화 (반말, 축약, 자연스러운 구어)
2. 사람 이름→한국식 가명, 지명·학교→한국식, 달러→만원 등 문화 맥락 변환
3. 위험 신호(비밀 유도, 만남 제안, 사진 요구, 칭찬을 통한 신뢰 형성 등)는 뉘앙스 그대로 보존
4. 각 입력 문자열을 번역해 JSON 배열로만 응답: ["번역1", "번역2", ...] (입력 순서·개수 동일)
5. 번역이 불가능한 항목만 "[SKIP]"으로 표기"""


def load_done() -> set[tuple[str, int]]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            done.add((r["conv_id"], r["win_idx"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _parse_array(text: str, n: int) -> list[str] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(out, list) and len(out) == n:
        return [str(t) for t in out]
    return None


# 연구용 — 그루밍·성적 유인 탐지 데이터 구축 목적이라 안전필터로 차단되면 학습 신호가
# 소실된다. 4개 카테고리 BLOCK_NONE. (API 정책상 일부 카테고리는 여전히 최소 차단 가능)
SAFETY = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


class RateLimitedError(Exception):
    """429 — 호출측에서 긴 백오프."""


def _call_gemini(api_key: str, prompt: str) -> str:
    url = f"{ENDPOINT.format(model=MODEL)}?key={api_key}"
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "safetySettings": SAFETY,
            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310 — 고정 도메인
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimitedError from e
        raise
    cands = data.get("candidates")
    if not cands:  # 안전필터 등으로 응답 없음 → 차단으로 간주
        raise KeyError(f"no candidates ({data.get('promptFeedback', {})})")
    parts = cands[0].get("content", {}).get("parts")
    if not parts:  # finishReason=SAFETY 등
        raise KeyError(f"blocked (finishReason={cands[0].get('finishReason')})")
    return parts[0]["text"]


def _translate_once(api_key: str, texts: list[str]) -> list[str] | None:
    payload = json.dumps(texts, ensure_ascii=False)
    prompt = f"{SYSTEM}\n\n# 입력(JSON 배열)\n{payload}"
    return _parse_array(_call_gemini(api_key, prompt), len(texts))


def translate_batch(api_key: str, rows: list[dict]) -> list[str | None]:
    """배치 번역. 429는 긴 백오프로 재시도, 그 외 실패 시 개별 번역으로 폴백
    (한 항목의 안전차단이 배치 전체를 날리지 않게)."""
    texts = [r["text_en"] for r in rows]
    for attempt in range(4):
        try:
            parsed = _translate_once(api_key, texts)
            if parsed is not None:
                return [None if t == "[SKIP]" else t for t in parsed]
        except RateLimitedError:
            wait = 15 * (attempt + 1)
            print(f"    (429 rate limit — {wait}s 대기)", file=sys.stderr)
            time.sleep(wait)
            continue
        except (urllib.error.URLError, KeyError, TimeoutError):
            break  # 배치 파싱·차단 실패 → 개별 폴백으로
        time.sleep(2 * (attempt + 1))

    # 개별 폴백: 항목별로 번역, 실패한 것만 None
    out: list[str | None] = []
    for t in texts:
        try:
            r = _translate_once(api_key, [t])
            out.append(None if not r or r[0] == "[SKIP]" else r[0])
        except RateLimitedError:
            time.sleep(20)
            try:
                r = _translate_once(api_key, [t])
                out.append(None if not r or r[0] == "[SKIP]" else r[0])
            except (urllib.error.URLError, KeyError, TimeoutError, RateLimitedError):
                out.append(None)
        except (urllib.error.URLError, KeyError, TimeoutError):
            out.append(None)
        time.sleep(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="파일럿용 — 앞에서 N개만 번역 (0=전체)")
    ap.add_argument(
        "--sleep", type=float, default=4.0, help="배치 간 대기 초 (무료 티어 rate limit 대응)"
    )
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY 없음 — .env 확인")

    rows = [json.loads(line) for line in IN_PATH.read_text(encoding="utf-8").splitlines()]
    done = load_done()
    todo = [r for r in rows if (r["conv_id"], r["win_idx"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    est_tokens = sum(len(r["text_en"].split()) for r in todo) * 2.5  # 대략치(입출력)
    print(f"모델 {MODEL} / 전체 {len(rows)} / 완료 {len(done)} / 이번 실행 {len(todo)}")
    print(f"예상 토큰 ~{est_tokens / 1e6:.2f}M (Gemini Flash 무료·저비용 구간)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            results = translate_batch(api_key, batch)
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
                            "win_idx": r["win_idx"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)} (스킵 {skipped})")
            if i + BATCH < len(todo):
                time.sleep(args.sleep)

    print(f"완료 → {OUT_PATH} (스킵 {skipped}건)")
    print("다음 단계: 무작위 샘플 수동 검수 (계획서 ③) 후 병합")


if __name__ == "__main__":
    main()
