"""MATCH 모델 서빙 서버 — 기존 5필드 계약과 MATCH-02 확장 입력을 함께 지원한다.

기존 백엔드 계약:
    POST /score {me:{user_id,bio,tags,age_band,ui_mode}, candidates:[...]}

선택 확장 필드:
    age_years, authored_items, liked_items
    candidates[].relationship

구형 LambdaMART에는 학습 당시와 같은 3개 열만 투영한다. 확장 특성은 입력 파이프라인에서
생성하지만, 해당 열로 재학습된 모델이 배포되기 전까지 구형 모델에 섞지 않는다.
"""

from __future__ import annotations

import math
import os
import pickle
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from src.data.matching_input import (
    CandidateInput,
    CandidateRelationship,
    ContentSignal,
    InputValidationError,
    MatchingInputPolicy,
    UserSnapshot,
    build_recommendation_reasons,
    prepare_match_inputs,
    validate_user_snapshot,
)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _find_default_config_path(*, cwd: Path | None = None, module_file: Path | None = None) -> Path:
    module_dir = (module_file or Path(__file__)).resolve().parent
    candidates = (
        (cwd or Path.cwd()) / "configs/module2_matching.yaml",
        module_dir / "configs/module2_matching.yaml",
        module_dir.parent / "configs/module2_matching.yaml",
        module_dir.parent.parent / "configs/module2_matching.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("configs/module2_matching.yaml not found")


CONFIG_PATH = Path(os.getenv("MATCH_CONFIG_PATH") or _find_default_config_path())
with open(CONFIG_PATH, encoding="utf-8") as config_file:
    _CONFIG = yaml.safe_load(config_file)
if not isinstance(_CONFIG, dict) or not isinstance(_CONFIG.get("input"), dict):
    raise ValueError("MATCH config must contain an input mapping")

_INPUT_CONFIG = _CONFIG["input"]
_EMBEDDING_CONFIG = _CONFIG.get("embedding", {})
_SERVING_CONFIG = _CONFIG.get("serving", {})
_FEATURE_CONFIG = _CONFIG.get("features", {})
_PATH_CONFIG = _CONFIG.get("paths", {})

_configured_tag_ids = frozenset(str(tag) for tag in _INPUT_CONFIG.get("allowed_tag_ids", []))
if raw_tag_ids := os.getenv("MATCH_ALLOWED_TAG_IDS"):
    _configured_tag_ids = frozenset(tag.strip() for tag in raw_tag_ids.split(",") if tag.strip())
if not _configured_tag_ids:
    raise ValueError("MATCH allowed_tag_ids must not be empty")

# 명시적 MATCH_MODEL_PATH가 있으면 로컬 파일을 강제한다. 없으면 config 기본 경로를 쓰되,
# MATCH_HF_REPO가 지정되면 HF를 우선한다(스키마 전환 시 로컬 구형 파일이 가리지 않도록).
_EXPLICIT_MODEL_PATH = os.getenv("MATCH_MODEL_PATH")
MODEL_PATH = Path(
    _EXPLICIT_MODEL_PATH
    or str(_PATH_CONFIG.get("model_path", "models/checkpoints/module2_lambdamart_embedding.pkl"))
)
MATCH_HF_REPO = os.getenv("MATCH_HF_REPO", "")
MATCH_HF_FILE = os.getenv("MATCH_HF_FILE", "module2_lambdamart_embedding.pkl")
# 서빙은 학습 산출물의 특정 커밋을 고정한다(train/serve 일관성). 빈 값이면 최신.
MATCH_HF_REVISION = os.getenv("MATCH_HF_REVISION", "") or None
SBERT_NAME = os.getenv(
    "MATCH_SBERT_NAME",
    str(_EMBEDDING_CONFIG.get("sbert_model_name", "jhgan/ko-sroberta-multitask")),
)

W_MODEL = _env_float("MATCH_W_MODEL", float(_SERVING_CONFIG.get("model_weight", 0.5)))
W_TAG = _env_float("MATCH_W_TAG", float(_SERVING_CONFIG.get("tag_weight", 0.3)))
W_AGE = _env_float("MATCH_W_AGE", float(_SERVING_CONFIG.get("age_weight", 0.2)))
if any(weight < 0.0 for weight in (W_MODEL, W_TAG, W_AGE)) or not math.isclose(
    W_MODEL + W_TAG + W_AGE, 1.0, abs_tol=1e-9
):
    raise ValueError("MATCH score weights must be non-negative and sum to 1")

LEGACY_FEATURE_COLS = list(
    _FEATURE_CONFIG.get("legacy_model_features", ["f_cosine", "f_l2", "f_dis_match"])
)
if LEGACY_FEATURE_COLS != ["f_cosine", "f_l2", "f_dis_match"]:
    raise ValueError("legacy_model_features must match the deployed model schema")

V2_FEATURE_COLS = list(_FEATURE_CONFIG.get("v2_pair_features", []))
FEATURE_SCHEMA = os.getenv(
    "MATCH_FEATURE_SCHEMA", str(_FEATURE_CONFIG.get("feature_schema", "legacy-v1"))
)
if FEATURE_SCHEMA not in {"legacy-v1", "match-input-v2"}:
    raise ValueError("MATCH_FEATURE_SCHEMA must be 'legacy-v1' or 'match-input-v2'")
if FEATURE_SCHEMA == "match-input-v2" and not V2_FEATURE_COLS:
    raise ValueError("v2_pair_features must be set for match-input-v2 schema")
NOT_ENOUGH = "추천 정보가 부족합니다"


INPUT_POLICY = MatchingInputPolicy(
    bio_max_chars=int(_INPUT_CONFIG["bio_max_chars"]),
    max_tags=int(_INPUT_CONFIG["max_tags"]),
    max_tag_chars=int(_INPUT_CONFIG["max_tag_chars"]),
    authored_lookback_days=_env_int(
        "MATCH_AUTHORED_LOOKBACK_DAYS", int(_INPUT_CONFIG["authored_lookback_days"])
    ),
    liked_lookback_days=_env_int(
        "MATCH_LIKED_LOOKBACK_DAYS", int(_INPUT_CONFIG["liked_lookback_days"])
    ),
    max_authored_items=_env_int(
        "MATCH_MAX_AUTHORED_ITEMS", int(_INPUT_CONFIG["max_authored_items"])
    ),
    max_liked_items=_env_int("MATCH_MAX_LIKED_ITEMS", int(_INPUT_CONFIG["max_liked_items"])),
    max_content_chars=_env_int("MATCH_MAX_CONTENT_CHARS", int(_INPUT_CONFIG["max_content_chars"])),
    max_candidates=_env_int("MATCH_MAX_CANDIDATES", int(_INPUT_CONFIG["max_candidates"])),
    embedding_batch_size=_env_int(
        "MATCH_EMBEDDING_BATCH_SIZE", int(_EMBEDDING_CONFIG["batch_size"])
    ),
    rejection_cooldown_days=int(_INPUT_CONFIG["rejection_cooldown_days"]),
    allowed_tag_ids=_configured_tag_ids,
    allowed_ui_modes=frozenset(_INPUT_CONFIG["allowed_ui_modes"]),
    allowed_content_sources=frozenset(_INPUT_CONFIG["allowed_content_sources"]),
    content_reason_min=_env_float(
        "MATCH_CONTENT_REASON_MIN", float(_SERVING_CONFIG["content_reason_min"])
    ),
    max_reasons=int(_SERVING_CONFIG["max_reasons"]),
    tag_reason_min_overlap=_env_int(
        "MATCH_TAG_REASON_MIN_OVERLAP", int(_SERVING_CONFIG["tag_reason_min_overlap"])
    ),
    age_reason_max_diff=_env_int(
        "MATCH_AGE_REASON_MAX_DIFF", int(_SERVING_CONFIG["age_reason_max_diff"])
    ),
)

_state: dict = {}


def _resolve_model_path() -> Path:
    # 1) 명시적 로컬 경로가 있으면 우선.
    if _EXPLICIT_MODEL_PATH and MODEL_PATH.exists():
        return MODEL_PATH
    # 2) HF repo가 지정되면 HF에서 받는다(기본 로컬 구형 파일이 v2 전환을 가리지 않도록).
    if MATCH_HF_REPO:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=MATCH_HF_REPO, filename=MATCH_HF_FILE, revision=MATCH_HF_REVISION
            )
        )
    # 3) 그 외 config 기본 로컬 경로.
    if MODEL_PATH.exists():
        return MODEL_PATH
    raise FileNotFoundError(
        f"모델 없음: {MODEL_PATH} — 로컬 배치 또는 MATCH_HF_REPO(+HF_TOKEN) 설정 필요"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 무거운 NLP 의존성은 서버 기동 시 로드하여 계약 단위 테스트와 분리한다.
    from sentence_transformers import SentenceTransformer

    resolved_path = _resolve_model_path()
    _state["model_source"] = str(resolved_path)
    with open(resolved_path, "rb") as file:
        loaded = pickle.load(file)
    # v2 pickle은 {model, columns, ...} 번들, 구형은 bare 모델이다.
    if isinstance(loaded, dict) and "model" in loaded:
        _state["ranker"] = loaded["model"]
        _state["ranker_columns"] = list(loaded.get("columns", V2_FEATURE_COLS))
    else:
        _state["ranker"] = loaded
    # v2 스키마인데 v2 번들(열 목록 포함)이 아니면, 특성 수 불일치로 예측이 깨지기 전에 막는다.
    if FEATURE_SCHEMA == "match-input-v2" and "ranker_columns" not in _state:
        raise RuntimeError(
            "MATCH_FEATURE_SCHEMA=match-input-v2 인데 로드된 모델이 v2 번들이 아닙니다. "
            "MATCH_HF_REPO/FILE/REVISION로 v2 pickle을 가리키거나 MATCH_MODEL_PATH를 확인하세요."
        )
    _state["sbert"] = SentenceTransformer(SBERT_NAME)
    _state["sbert"].encode(["워밍업"])
    yield
    _state.clear()


app = FastAPI(title="match-model (SBERT+LambdaMART)", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def sanitized_request_validation_error(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Pydantic 기본 422가 입력값을 반사하지 않도록 필드 위치만 반환한다."""

    fields = sorted(
        {
            ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            for error in exc.errors()
        }
        - {""}
    )
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "VALIDATION_ERROR", "fields": fields}},
    )


@app.exception_handler(InputValidationError)
async def sanitized_domain_validation_error(
    _request: Request, exc: InputValidationError
) -> JSONResponse:
    error: dict[str, str] = {"code": exc.code}
    if exc.field_name is not None:
        error["field"] = exc.field_name
    return JSONResponse(status_code=422, content={"error": error})


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContentItemIn(StrictRequestModel):
    content_id: str = Field(min_length=1, max_length=128)
    source_type: str = Field(min_length=1, max_length=16)
    text: str = Field(max_length=10_000)
    created_at: datetime
    is_deleted: StrictBool = False
    is_accessible: StrictBool = True
    is_blocked_author: StrictBool = False
    is_like_active: StrictBool = True


class RelationshipIn(StrictRequestModel):
    blocked_either_direction: StrictBool = False
    already_friends: StrictBool = False
    last_rejected_at: datetime | None = None
    common_friend_count: int = Field(default=0, strict=True, ge=0, le=1_000_000)


class Features(StrictRequestModel):
    # Wire 이름 `tags`는 현재 백엔드와의 호환을 위해 유지한다.
    user_id: str = Field(min_length=1, max_length=128)
    bio: str = Field(default="", max_length=INPUT_POLICY.bio_max_chars)
    tags: list[str] = Field(default_factory=list, max_length=INPUT_POLICY.max_tags)
    age_years: int | None = Field(default=None, strict=True, ge=14, le=120)
    age_band: str = Field(default="", max_length=16)
    ui_mode: str = Field(default="", max_length=64)
    authored_items: list[ContentItemIn] = Field(
        default_factory=list, max_length=INPUT_POLICY.max_authored_items
    )
    liked_items: list[ContentItemIn] = Field(
        default_factory=list, max_length=INPUT_POLICY.max_liked_items
    )
    # 누락은 현재 백엔드가 이미 후보를 선필터링한 legacy 입력을 뜻한다.
    relationship: RelationshipIn | None = None


class ScoreIn(StrictRequestModel):
    me: Features
    candidates: list[Features] = Field(max_length=INPUT_POLICY.max_candidates)


def _content_signal(item: ContentItemIn) -> ContentSignal:
    return ContentSignal(
        content_id=item.content_id,
        source_type=item.source_type,
        text=item.text,
        created_at=item.created_at,
        is_deleted=item.is_deleted,
        is_accessible=item.is_accessible,
        is_blocked_author=item.is_blocked_author,
        is_like_active=item.is_like_active,
    )


def _user_snapshot(features: Features) -> UserSnapshot:
    return UserSnapshot(
        user_id=features.user_id,
        bio=features.bio,
        tag_ids=tuple(features.tags),
        age_years=features.age_years,
        age_band=features.age_band,
        ui_mode=features.ui_mode,
        authored_items=tuple(_content_signal(item) for item in features.authored_items),
        liked_items=tuple(_content_signal(item) for item in features.liked_items),
    )


def _candidate_input(features: Features) -> CandidateInput:
    relation = features.relationship or RelationshipIn()
    return CandidateInput(
        profile=_user_snapshot(features),
        relationship=CandidateRelationship(
            candidate_id=features.user_id,
            blocked_either_direction=relation.blocked_either_direction,
            already_friends=relation.already_friends,
            last_rejected_at=relation.last_rejected_at,
            common_friend_count=relation.common_friend_count,
        ),
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@app.post("/score")
async def score(body: ScoreIn):
    me_snapshot = _user_snapshot(body.me)
    if not body.candidates:
        validate_user_snapshot(me_snapshot, INPUT_POLICY)
        return {"results": []}

    batch = prepare_match_inputs(
        me_snapshot,
        tuple(_candidate_input(candidate) for candidate in body.candidates),
        encoder=_state["sbert"],
        policy=INPUT_POLICY,
        as_of=datetime.now(UTC),
    )
    if batch.status == "insufficient_signal":
        return {"results": [], "message": NOT_ENOUGH}
    if not batch.candidates:
        return {"results": []}

    if FEATURE_SCHEMA == "match-input-v2":
        feature_cols = _state.get("ranker_columns") or V2_FEATURE_COLS
        model_input = pd.DataFrame(
            [
                {col: candidate.features[col] for col in feature_cols}
                for candidate in batch.candidates
            ],
            columns=feature_cols,
        )
    else:
        model_input = pd.DataFrame(
            [
                {
                    "f_cosine": candidate.features["f_cosine"],
                    "f_l2": candidate.features["f_l2"],
                    # 모델 파일의 구형 열 이름만 유지한다. 실제 입력 의미는 UI 모드 일치다.
                    "f_dis_match": candidate.features["f_ui_mode_match"],
                }
                for candidate in batch.candidates
            ],
            columns=LEGACY_FEATURE_COLS,
        )
    raw_scores = np.asarray(_state["ranker"].predict(model_input), dtype=float).reshape(-1)
    if len(raw_scores) != len(batch.candidates) or not np.isfinite(raw_scores).all():
        raise RuntimeError("invalid ranker output")

    results = []
    for candidate, raw_score in zip(batch.candidates, raw_scores, strict=True):
        features = candidate.features
        model_score = _sigmoid(float(raw_score))
        if FEATURE_SCHEMA == "match-input-v2":
            # v2 모델은 tag·age·콘텐츠·친구 특성을 이미 학습했으므로 서빙에서 재가산하지 않는다.
            blended = model_score
        else:
            tag_score = min(features["f_tag_overlap"], 3.0) / 3.0
            age_score = features["f_age_band_match"]
            blended = W_MODEL * model_score + W_TAG * tag_score + W_AGE * age_score
        results.append(
            {
                "user_id": candidate.user_id,
                "score": round(blended, 4),
                "reasons": build_recommendation_reasons(features, policy=INPUT_POLICY),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return {"results": results}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _state.get("model_source", str(MODEL_PATH)),
        "loaded": "ranker" in _state,
        "feature_schema": FEATURE_SCHEMA,
    }
