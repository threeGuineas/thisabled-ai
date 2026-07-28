"""MATCH v2 서빙 스모크 (in-process, 포트 불필요).

app을 TestClient 컨텍스트로 띄우면 lifespan이 모델(HF 또는 로컬)과 SBERT를 로드한다.
그 뒤 /health·/score 계약과 MATCH-04 금지어 비노출을 검증한다.

HF에서 실제 v2 모델을 pull해 검증(배포와 동일 경로):
    HF_TOKEN=<read> MATCH_FEATURE_SCHEMA=match-input-v2 \
    MATCH_HF_REPO=soyuncj/module2 MATCH_HF_FILE=module2_lambdamart_v2.pkl \
    MATCH_HF_REVISION=ecb31a428e74dfc393617a6a4a95ecc4cb7e6d67 \
    .venv/bin/python scripts/smoke_match_v2.py

로컬 번들로 스크립트 자체만 검증(HF 없이):
    MATCH_FEATURE_SCHEMA=match-input-v2 MATCH_MODEL_PATH=<bundle.pkl> \
    .venv/bin/python scripts/smoke_match_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from serving.match_server.app import app  # noqa: E402

FORBIDDEN_REASON_WORDS = ["장애", "시각", "청각", "발달", "모드"]  # MATCH-04

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


def main() -> int:
    errs: list[str] = []
    with TestClient(app) as client:  # lifespan → 모델(HF/로컬)·SBERT 로드
        health = client.get("/health").json()
        print("health:", health)
        if health.get("feature_schema") != "match-input-v2":
            errs.append(f"feature_schema != match-input-v2 (got {health.get('feature_schema')})")
        if not health.get("loaded"):
            errs.append("모델이 로드되지 않음")

        response = client.post("/score", json=MATCH_BODY)
        print("score status:", response.status_code)
        if response.status_code != 200:
            errs.append(f"/score {response.status_code}: {response.text[:200]}")
        else:
            results = response.json().get("results", [])
            print("results:", results)
            if len(results) != 3:
                errs.append(f"results 개수 {len(results)} != 3")
            for item in results:
                if not {"user_id", "score", "reasons"} <= item.keys():
                    errs.append(f"키 누락: {item}")
                for reason in item.get("reasons", []):
                    if any(word in reason for word in FORBIDDEN_REASON_WORDS):
                        errs.append(f"MATCH-04 위반 — 사유 금지어: {reason!r}")
            scores = {item["user_id"]: item["score"] for item in results}
            # 랭킹 상식(유사>상이)은 실 HF 모델일 때만 유의미 — 참고 출력.
            if scores.get("u-sim", 0.0) <= scores.get("u-diff", 1.0):
                print(
                    f"[참고] 랭킹: u-sim({scores.get('u-sim')}) <= u-diff({scores.get('u-diff')}) "
                    "— fake-trained 번들/불일치 조합이면 무시"
                )

    if errs:
        print("\n".join("❌ " + e for e in errs))
        return 1
    print("✅ MATCH v2 스모크 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
