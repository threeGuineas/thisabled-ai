"""SAFE·MATCH 서빙 서버 계약 스모크 테스트.

사용: 두 서버 기동 후 `python serving/smoke_test.py`
      (백엔드 mock에 돌려도 통과해야 함 — 같은 계약이므로)
"""

from __future__ import annotations

import statistics
import sys
import time

import httpx

SAFE_URL = "http://localhost:9001"
MATCH_URL = "http://localhost:9002"
WAIT_READY_SECONDS = 180  # 모델 로딩(수십 초) 동안 포트가 닫혀 있으므로 /health 폴링


def wait_ready(name: str, url: str) -> None:
    deadline = time.time() + WAIT_READY_SECONDS
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=3).status_code == 200:
                print(f"[{name}] ready: {url}")
                return
        except httpx.HTTPError:
            pass
        print(f"[{name}] 기동 대기 중... ({url})")
        time.sleep(5)
    sys.exit(f"❌ [{name}] {WAIT_READY_SECONDS}초 내에 준비되지 않음 — 서버 로그 확인")


SAFE_CASES = [
    # (text, receiver_is_minor, 기대 verdict — 규칙 mock·실모델 공통 상식선)
    ("오늘 날씨 진짜 좋다! 산책 갈래?", False, "safe"),
    ("계좌번호 알려주면 돈 보내줄게. 급한 거니까 빨리", False, "flagged"),
    ("우리 단둘이 몰래 만나자. 아무한테도 말하지 마", True, "flagged"),
    ("너 같은 건 살 가치도 없어. 죽여버린다", False, "flagged"),
]

MATCH_BODY = {
    "me": {
        "user_id": "u-me",
        "bio": "영화와 야구를 좋아해요. 일상 이야기 나눌 친구 찾아요.",
        "tags": ["movie", "walking"],
        "age_band": "25~34세",
        "ui_mode": "visual",
    },
    "candidates": [
        {
            "user_id": "u-sim",
            "bio": "영화 보는 게 취미예요. 편하게 수다 떨 친구 구해요.",
            "tags": ["movie", "drama"],
            "age_band": "25~34세",
            "ui_mode": "hearing",
        },
        {
            "user_id": "u-diff",
            "bio": "주식 정보 공유방 운영합니다. 수익 인증 가능.",
            "tags": ["gym"],
            "age_band": "55세 이상",
            "ui_mode": "visual",
        },
        {
            "user_id": "u-empty",
            "bio": "",
            "tags": [],
            "age_band": "35-44",
            "ui_mode": "",
        },
    ],
}

FORBIDDEN_REASON_WORDS = ["장애", "시각", "청각", "발달", "모드"]  # MATCH-04


def test_safe() -> list[str]:
    errs = []
    lat = []
    for text, minor, expected in SAFE_CASES:
        t0 = time.perf_counter()
        r = httpx.post(
            f"{SAFE_URL}/analyze", json={"text": text, "receiver_is_minor": minor}, timeout=10
        )
        lat.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        body = r.json()
        if body.get("verdict") not in ("safe", "flagged"):
            errs.append(f"[SAFE] verdict 형식 위반: {body}")
        elif body["verdict"] != expected:
            errs.append(
                f"[SAFE] 판정 불일치 (참고): {text[:20]!r} → {body['verdict']} (기대 {expected})"
            )
    print(
        f"[SAFE] latency ms: median={statistics.median(lat):.0f} max={max(lat):.0f} "
        f"(백엔드 타임아웃 예산 2000ms)"
    )
    return errs


def test_match() -> list[str]:
    errs = []
    t0 = time.perf_counter()
    r = httpx.post(f"{MATCH_URL}/score", json=MATCH_BODY, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    results = r.json().get("results")
    print(f"[MATCH] latency ms: {ms:.0f} / results: {results}")
    if not isinstance(results, list) or len(results) != 3:
        return [f"[MATCH] results 형식 위반: {results}"]
    for item in results:
        if not {"user_id", "score", "reasons"} <= item.keys():
            errs.append(f"[MATCH] 키 누락: {item}")
        for reason in item.get("reasons", []):
            if any(w in reason for w in FORBIDDEN_REASON_WORDS):
                errs.append(f"[MATCH] MATCH-04 위반 — 사유에 금지어: {reason!r}")
    scores = {i["user_id"]: i["score"] for i in results}
    if scores.get("u-sim", 0) <= scores.get("u-diff", 1):
        errs.append(
            f"[MATCH] 랭킹 상식 위반 (참고): 유사 후보 u-sim({scores.get('u-sim')}) "
            f"<= 상이 후보 u-diff({scores.get('u-diff')})"
        )
    return errs


if __name__ == "__main__":
    wait_ready("SAFE", SAFE_URL)
    wait_ready("MATCH", MATCH_URL)
    problems = test_safe() + test_match()
    hard = [p for p in problems if "(참고)" not in p]
    for p in problems:
        print(("⚠️  " if "(참고)" in p else "❌ ") + p)
    if hard:
        sys.exit(1)
    print("✅ 계약 스모크 테스트 통과")
