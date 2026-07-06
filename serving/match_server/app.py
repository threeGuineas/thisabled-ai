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
score: 블렌드 = W_MODEL·sigmoid(LambdaMART) + W_TAG·태그교집합 + W_AGE·연령대일치.
    학습 입력(지역·나이·장애유형이 명시된 템플릿 자기소개)과 서빙 입력(bio+tags)의
    스키마 드리프트를 보완 — 태그 교집합·연령대 일치는 학습 라벨 규칙(overlap,
    age_diff)과 동일 계열 신호라 정합적. 근본 해결(계약 스키마 재학습)은 후속.
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
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_PATH = Path(
    os.getenv("MATCH_MODEL_PATH", "models/checkpoints/module2_lambdamart_embedding.pkl")
)
# 로컬에 pkl이 없으면 HF private repo에서 다운로드 (HF_TOKEN env 필요)
MATCH_HF_REPO = os.getenv("MATCH_HF_REPO", "")
MATCH_HF_FILE = os.getenv("MATCH_HF_FILE", "module2_lambdamart_embedding.pkl")
SBERT_NAME = os.getenv("MATCH_SBERT_NAME", "jhgan/ko-sroberta-multitask")
COSINE_REASON_MIN = float(os.getenv("MATCH_COSINE_REASON_MIN", "0.5"))

# 점수 블렌드 가중치 (운영 설정값)
W_MODEL = float(os.getenv("MATCH_W_MODEL", "0.5"))
W_TAG = float(os.getenv("MATCH_W_TAG", "0.3"))
W_AGE = float(os.getenv("MATCH_W_AGE", "0.2"))

FEATURE_COLS = ["f_cosine", "f_l2", "f_dis_match"]  # 학습과 동일 순서 (build_pairs.py)

_state: dict = {}


def _resolve_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    if MATCH_HF_REPO:
        return Path(hf_hub_download(repo_id=MATCH_HF_REPO, filename=MATCH_HF_FILE))
    raise FileNotFoundError(
        f"모델 없음: {MODEL_PATH} — 로컬 배치 또는 MATCH_HF_REPO(+HF_TOKEN) 설정 필요"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(_resolve_model_path(), "rb") as f:
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

    results = []
    for c, r, cos in zip(body.candidates, raw, cosine, strict=False):
        model_score = 1.0 / (1.0 + math.exp(-float(r)))
        tag_score = min(len(set(body.me.tags) & set(c.tags)), 3) / 3
        age_score = 1.0 if body.me.age_band and body.me.age_band == c.age_band else 0.0
        blended = W_MODEL * model_score + W_TAG * tag_score + W_AGE * age_score
        results.append(
            {
                "user_id": c.user_id,
                "score": round(blended, 4),
                "model_score": round(model_score, 4),  # 계약 외 부가 필드 (데모·리포트용)
                "reasons": _reasons(body.me, c, float(cos)),
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results}


@app.get("/health")
async def health():
    return {"status": "ok", "model": str(MODEL_PATH), "loaded": "ranker" in _state}
