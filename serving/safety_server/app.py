"""SAFE 모델 서빙 서버 — 백엔드 mock(safety_model) 대체 실서버.

계약 (백엔드 CLAUDE.md / mocks/safety_model.py 와 동일, 변경 금지):
    POST /analyze {text, receiver_is_minor} → {"verdict": "safe"|"flagged"}
    GET  /health

모델: models/checkpoints/module1_ce (KcELECTRA-base-v2022 fine-tune, 4-class CE).
비고: LightGBM 스태커는 meta feature에 학습 데이터 전용 `source` 컬럼이 필요해
서빙에서 제외하고 PLM softmax를 직접 사용한다 (보고서에 명시).

4-class → binary 매핑:
    P(주의) + P(경고) + P(긴급) >= threshold → flagged
    threshold: 성인 SAFE_FLAG_THRESHOLD (기본 0.5)
               미성년 수신자 SAFE_FLAG_THRESHOLD_MINOR (기본 0.35, §4.5 민감 판정)
    운영 설정값 — env로 조정 (명세 4.5 "임계값은 운영 설정값으로 관리").
"""

from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.getenv("SAFE_MODEL_DIR", "models/checkpoints/module1_ce"))
THRESHOLD = float(os.getenv("SAFE_FLAG_THRESHOLD", "0.5"))
THRESHOLD_MINOR = float(os.getenv("SAFE_FLAG_THRESHOLD_MINOR", "0.35"))
MAX_LENGTH = int(os.getenv("SAFE_MAX_LENGTH", "128"))

# 모델 헤드 크기에 맞춰 기동 시 결정 (이진 2 = 정상/주의, 4-class = 정상/주의/경고/긴급).
# 서빙 verdict는 어느 쪽이든 P(주의 이상 위험) = sum(probs[1:])로 동일하게 산출되므로
# 한 서버가 두 모델을 모두 지원한다 (재학습 전환기 안전).
LABELS_BY_N = {2: ["정상", "주의"], 4: ["정상", "주의", "경고", "긴급"]}

# ── 규칙 보조 레이어 (SAFE-02 유형 1: 금전 요구·사기) ──────────────────────
# 학습 시드(Unsmile/KOLD)가 혐오표현 중심이라 금전 사기 커버리지가 약함(스모크에서
# risk_prob 0.07 확인). 모델 판정에 OR로만 결합 — 플래그를 추가할 뿐 해제하지 않으므로
# 재현율만 올라감. 오탐 트레이드오프와 함께 보고서에 하이브리드 구성으로 명시.
RULE_ASSIST = os.getenv("SAFE_RULE_ASSIST", "0") == "1"

_MONEY = r"(돈|현금|계좌\s*번호|계좌|송금|입금|이체|상품권|기프트\s*카드|문상|코인|비트코인|수익금)"
_ACTION = r"(알려|보내|빌려|부쳐|넣어|이체|입금|찍어|줘|주면|필요|급하)"
SCAM_PATTERNS = [
    re.compile(_MONEY + r".{0,15}" + _ACTION),
    re.compile(_ACTION + r".{0,15}" + _MONEY),
    re.compile(r"(인증\s*번호|비밀\s*번호|OTP)\s*.{0,10}(알려|보내|불러)"),
    re.compile(r"(투자|리딩방|수익)\s*.{0,12}(보장|확정|고수익|배로)"),
]


def _rule_hit(text: str) -> bool:
    return RULE_ASSIST and any(p.search(text) for p in SCAM_PATTERNS)


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    n = int(model.config.num_labels)
    _state["labels"] = LABELS_BY_N.get(n, [str(i) for i in range(n)])
    _state["num_labels"] = n
    _infer("워밍업 문장입니다.")  # 첫 요청 지연 방지
    yield
    _state.clear()


app = FastAPI(title="safety-model (KcELECTRA)", lifespan=lifespan)


class AnalyzeIn(BaseModel):
    text: str
    receiver_is_minor: bool = False


def _infer(text: str) -> list[float]:
    enc = _state["tokenizer"](text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.inference_mode():
        logits = _state["model"](**enc).logits[0]
    return F.softmax(logits, dim=-1).tolist()


@app.post("/analyze")
async def analyze(body: AnalyzeIn):
    t0 = time.perf_counter()
    probs = _infer(body.text)
    p_risk = sum(probs[1:])  # 이진=P(주의), 4-class=P(주의+경고+긴급)
    threshold = THRESHOLD_MINOR if body.receiver_is_minor else THRESHOLD
    rule_hit = _rule_hit(body.text)
    verdict = "flagged" if (p_risk >= threshold or rule_hit) else "safe"
    labels = _state["labels"]
    # 계약 필수 키는 verdict 하나 — 나머지는 데모·리포트용 부가 정보 (백엔드는 무시)
    return {
        "verdict": verdict,
        "rule_assist": rule_hit,
        "risk_prob": round(p_risk, 4),
        "level": labels[int(max(range(len(probs)), key=lambda i: probs[i]))],
        "probs": {labels[i]: round(p, 4) for i, p in enumerate(probs)},
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": str(MODEL_DIR),
        "loaded": "model" in _state,
        "num_labels": _state.get("num_labels"),
        "labels": _state.get("labels"),
    }
