"""MATCH 모델 서빙 서버 — 백엔드 mock(match_model) 대체 실서버.

계약 (백엔드 mocks/match_model.py 와 동일, 변경 금지):
    POST /score {me:{user_id,bio,tags,age_band,ui_mode}, candidates:[...]}
        → {"results": [{user_id, score, reasons}]}
    GET  /health

모델: models/checkpoints/module2_lambdamart_embedding.pkl
      (jhgan/ko-sroberta-multitask 임베딩 → f_cosine, f_l2, f_dis_match → LambdaMART)

서빙 시 특성 구성 (학습 embedding 모드와 동일 순서):
    f_cosine, f_l2  : me.bio vs cand.bio SBERT 임베딩 (bio 없으면 태그 문자열로 폴백)
    f_dis_match     : 학습의 disability_type 일치를 ui_mode 일치로 대응
                      (서버 내부 특성 전용, 사유로 절대 노출하지 않음 — MATCH-04)
score: LambdaMART raw score → sigmoid로 [0,1] 정규화.
reasons: 일반화 문장만 (관심사/연령대/소개 유사 — 명세 MATCH-03).
"""

from __future__ import annotations

import math
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_PATH = Path(
    os.getenv("MATCH_MODEL_PATH", "models/checkpoints/module2_lambdamart_embedding.pkl")
)
SBERT_NAME = os.getenv("MATCH_SBERT_NAME", "jhgan/ko-sroberta-multitask")
COSINE_REASON_MIN = float(os.getenv("MATCH_COSINE_REASON_MIN", "0.5"))

FEATURE_COLS = ["f_cosine", "f_l2", "f_dis_match"]  # 학습과 동일 순서 (build_pairs.py)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_PATH, "rb") as f:
        _state["ranker"] = pickle.load(f)
    _state["sbert"] = SentenceTransformer(SBERT_NAME)
    _state["sbert"].encode(["워밍업"])  # 첫 요청 지연 방지
    yield
    _state.clear()


app = FastAPI(title="match-model (SBERT+LambdaMART)", lifespan=lifespan)


class Features(BaseModel):
    user_id: str
    bio: str = ""
    tags: list[str] = []
    age_band: str = ""
    ui_mode: str = ""


class ScoreIn(BaseModel):
    me: Features
    candidates: list[Features]


def _profile_text(p: Features) -> str:
    """임베딩 입력 텍스트 — bio 우선, 비어 있으면 관심사 태그로 폴백 (MATCH-02-8)."""
    text = p.bio.strip()
    if p.tags:
        text = f"{text} 관심사: {', '.join(p.tags)}" if text else f"관심사: {', '.join(p.tags)}"
    return text or "정보 없음"


def _reasons(me: Features, cand: Features, cosine: float) -> list[str]:
    reasons = []
    if set(me.tags) & set(cand.tags):
        reasons.append("관심사가 비슷해요")
    if me.age_band and me.age_band == cand.age_band:
        reasons.append("비슷한 연령대예요")
    if not reasons and cosine >= COSINE_REASON_MIN:
        reasons.append("소개 내용이 비슷해요")
    return reasons  # ui_mode·장애 관련 문구 금지 (MATCH-04)


@app.post("/score")
async def score(body: ScoreIn):
    if not body.candidates:
        return {"results": []}

    texts = [_profile_text(body.me)] + [_profile_text(c) for c in body.candidates]
    embs = _state["sbert"].encode(texts, batch_size=32)
    me_emb, cand_embs = embs[0], embs[1:]

    denom = np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(me_emb) + 1e-8
    cosine = (cand_embs @ me_emb) / denom
    l2 = np.linalg.norm(cand_embs - me_emb, axis=1)
    dis_match = np.array(
        [1.0 if c.ui_mode and c.ui_mode == body.me.ui_mode else 0.0 for c in body.candidates]
    )

    features = pd.DataFrame(
        {
            "f_cosine": cosine.astype(np.float32),
            "f_l2": l2.astype(np.float32),
            "f_dis_match": dis_match.astype(np.float32),
        },
        columns=FEATURE_COLS,
    )
    raw = _state["ranker"].predict(features)

    results = [
        {
            "user_id": c.user_id,
            "score": round(1.0 / (1.0 + math.exp(-float(r))), 4),
            "reasons": _reasons(body.me, c, float(cos)),
        }
        for c, r, cos in zip(body.candidates, raw, cosine, strict=False)
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results}


@app.get("/health")
async def health():
    return {"status": "ok", "model": str(MODEL_PATH), "loaded": "ranker" in _state}
