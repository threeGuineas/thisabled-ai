"""MATCH-02 허용 신호를 검증·선별·집계하는 입력 파이프라인.

이 모듈은 DB나 FastAPI에 의존하지 않는다. 인증된 백엔드가 만든 스냅샷을 받아
금지 데이터와 비활성 콘텐츠를 Sentence-BERT 경계 전에 제거하고, 사용자 쌍의
LightGBM 입력 특성을 생성한다.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Literal, Protocol

import numpy as np

AGE_BAND_14_18 = "14~18세"
AGE_BAND_19_24 = "19~24세"
AGE_BAND_25_34 = "25~34세"
AGE_BAND_35_44 = "35~44세"
AGE_BAND_45_54 = "45~54세"
AGE_BAND_55_PLUS = "55세 이상"

AGE_BANDS = (
    AGE_BAND_14_18,
    AGE_BAND_19_24,
    AGE_BAND_25_34,
    AGE_BAND_35_44,
    AGE_BAND_45_54,
    AGE_BAND_55_PLUS,
)

# 기존 백엔드/스모크 테스트 표기를 MATCH 명세의 공개 표기로 정규화한다.
AGE_BAND_ALIASES = {
    "14-18": AGE_BAND_14_18,
    "14~18": AGE_BAND_14_18,
    "14-18세": AGE_BAND_14_18,
    AGE_BAND_14_18: AGE_BAND_14_18,
    "19-24": AGE_BAND_19_24,
    "19~24": AGE_BAND_19_24,
    "19-24세": AGE_BAND_19_24,
    AGE_BAND_19_24: AGE_BAND_19_24,
    "25-34": AGE_BAND_25_34,
    "25~34": AGE_BAND_25_34,
    "25-34세": AGE_BAND_25_34,
    AGE_BAND_25_34: AGE_BAND_25_34,
    "35-44": AGE_BAND_35_44,
    "35~44": AGE_BAND_35_44,
    "35-44세": AGE_BAND_35_44,
    AGE_BAND_35_44: AGE_BAND_35_44,
    "45-54": AGE_BAND_45_54,
    "45~54": AGE_BAND_45_54,
    "45-54세": AGE_BAND_45_54,
    AGE_BAND_45_54: AGE_BAND_45_54,
    "55+": AGE_BAND_55_PLUS,
    "55세+": AGE_BAND_55_PLUS,
    AGE_BAND_55_PLUS: AGE_BAND_55_PLUS,
}

_EMAIL_PATTERN = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:(?:\+?82[-.\s]?)?0?1[016789]|0\d{1,2})[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"
)
_INTERNATIONAL_PHONE_PATTERN = re.compile(
    r"(?<!\d)\+\d{1,3}[-.\s]?(?:\(?\d{1,4}\)?[-.\s]?){2,5}\d{2,4}(?!\d)"
)
_MESSENGER_ID_PATTERN = re.compile(
    r"(?i)(?:카카오톡|카톡|텔레그램|인스타그램|인스타|라인|디스코드|"
    r"(?<![A-Za-z0-9_])(?:kakao|telegram|instagram|insta|line|discord)(?![A-Za-z0-9_]))"
    r"\s*(?:(?:(?:아이디|id|계정)(?:으로|로|는|은|이|가)?|"
    r"(?:으로|로|는|은|이|가))\s*[:：]?\s*)?"
    r"[@A-Za-z0-9_.-]{3,}"
)
_MESSENGER_URL_PATTERN = re.compile(
    r"(?i)(?:https?://)?(?:open\.kakao\.com|t\.me|line\.me|discord\.gg)/\S+"
)
_CONTACT_PATTERNS = (
    _EMAIL_PATTERN,
    _PHONE_PATTERN,
    _INTERNATIONAL_PHONE_PATTERN,
    _MESSENGER_ID_PATTERN,
    _MESSENGER_URL_PATTERN,
)

ALLOWED_RECOMMENDATION_REASONS = frozenset(
    {
        "관심사가 비슷해요",
        "관심 있는 콘텐츠가 비슷해요",
        "공통 친구가 있어요",
        "비슷한 연령대예요",
    }
)


class TextEncoder(Protocol):
    """SentenceTransformer와 테스트 fake가 공유하는 최소 인터페이스."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray: ...


class InputValidationError(ValueError):
    """민감한 입력값을 포함하지 않는 경계 검증 오류."""

    def __init__(self, code: str, *, field_name: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.field_name = field_name


@dataclass(frozen=True, slots=True)
class MatchingInputPolicy:
    """운영에서 조정 가능한 MATCH 입력 상한과 기간."""

    bio_max_chars: int = 300
    max_tags: int = 10
    max_tag_chars: int = 64
    authored_lookback_days: int = 90
    liked_lookback_days: int = 90
    max_authored_items: int = 100
    max_liked_items: int = 100
    max_content_chars: int = 2_000
    max_candidates: int = 200
    embedding_batch_size: int = 64
    rejection_cooldown_days: int = 30
    allowed_tag_ids: frozenset[str] | None = None
    allowed_ui_modes: frozenset[str] = frozenset({"visual", "hearing", "developmental"})
    allowed_content_sources: frozenset[str] = frozenset({"post", "comment"})
    content_reason_min: float = 0.65
    max_reasons: int = 4
    # 화면 문구가 모델 기여와 어긋나지 않도록 잡은 하한.
    # artifacts/match_v2_shap.json 기준으로 겹침 1개는 추천의 94.7%에 문구가 붙으면서
    # SHAP 뒷받침률이 57.3%였고, 3개로 올리면 노출 42.3%에 뒷받침률 97.7%가 된다.
    tag_reason_min_overlap: int = 3
    # 연령 문구는 f_age_band_match(기여 4.2%)가 아니라 실제 최대 기여 특성인
    # f_age_diff(31.0%)에 건다. 5세 이내에서 SHAP 뒷받침률 100%.
    age_reason_max_diff: int = 5

    def __post_init__(self) -> None:
        positive_fields = (
            "bio_max_chars",
            "max_tags",
            "max_tag_chars",
            "authored_lookback_days",
            "liked_lookback_days",
            "max_authored_items",
            "max_liked_items",
            "max_content_chars",
            "max_candidates",
            "embedding_batch_size",
            "rejection_cooldown_days",
            "max_reasons",
            "tag_reason_min_overlap",
            "age_reason_max_diff",
        )
        if any(getattr(self, name) <= 0 for name in positive_fields):
            raise ValueError("MATCH policy limits must be positive")
        if not 0.0 <= self.content_reason_min <= 1.0:
            raise ValueError("content_reason_min must be between 0 and 1")
        if self.tag_reason_min_overlap > self.max_tags:
            raise ValueError("tag_reason_min_overlap cannot exceed max_tags")


@dataclass(frozen=True, slots=True)
class ContentSignal:
    content_id: str
    source_type: str
    text: str
    created_at: datetime
    is_deleted: bool = False
    is_accessible: bool = True
    is_blocked_author: bool = False
    is_like_active: bool = True


@dataclass(frozen=True, slots=True)
class UserSnapshot:
    user_id: str
    bio: str = ""
    tag_ids: tuple[str, ...] = ()
    age_years: int | None = None
    age_band: str | None = None
    ui_mode: str = ""
    authored_items: tuple[ContentSignal, ...] = ()
    liked_items: tuple[ContentSignal, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateRelationship:
    candidate_id: str
    blocked_either_direction: bool = False
    already_friends: bool = False
    last_rejected_at: datetime | None = None
    common_friend_count: int = 0


@dataclass(frozen=True, slots=True)
class CandidateInput:
    profile: UserSnapshot
    relationship: CandidateRelationship


@dataclass(frozen=True, slots=True)
class PreparedTextSignals:
    user_id: str
    tag_ids: tuple[str, ...]
    age_years: int | None
    age_band: str | None
    ui_mode: str
    profile_text: str | None
    authored_texts: tuple[str, ...]
    liked_texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedUserFeatures:
    user_id: str
    tag_ids: tuple[str, ...]
    age_years: int | None
    age_band: str | None
    ui_mode: str
    profile_vector: np.ndarray | None
    authored_vector: np.ndarray | None
    liked_vector: np.ndarray | None
    effective_vector: np.ndarray | None
    authored_count: int
    liked_count: int


@dataclass(frozen=True, slots=True)
class PreparedCandidate:
    user_id: str
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class ExcludedCandidate:
    candidate_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class PreparedMatchBatch:
    query: PreparedUserFeatures
    candidates: tuple[PreparedCandidate, ...]
    excluded: tuple[ExcludedCandidate, ...]
    status: Literal["ok", "insufficient_signal", "no_eligible_candidates"]


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    return re.sub(r"\s+", " ", normalized)


def contains_contact_info(text: str) -> bool:
    """전화번호·이메일·외부 메신저 주소/ID 형태가 있는지만 반환한다."""

    normalized = unicodedata.normalize("NFKC", text)
    return any(pattern.search(normalized) is not None for pattern in _CONTACT_PATTERNS)


def normalize_age_band(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value)
    if not normalized:
        return None
    try:
        return AGE_BAND_ALIASES[normalized]
    except KeyError as exc:
        raise InputValidationError("INVALID_AGE_BAND", field_name="age_band") from exc


def age_band_for_age(age_years: int) -> str:
    if 14 <= age_years <= 18:
        return AGE_BAND_14_18
    if 19 <= age_years <= 24:
        return AGE_BAND_19_24
    if 25 <= age_years <= 34:
        return AGE_BAND_25_34
    if 35 <= age_years <= 44:
        return AGE_BAND_35_44
    if 45 <= age_years <= 54:
        return AGE_BAND_45_54
    if 55 <= age_years <= 120:
        return AGE_BAND_55_PLUS
    raise InputValidationError("INVALID_AGE", field_name="age_years")


def validate_user_snapshot(snapshot: UserSnapshot, policy: MatchingInputPolicy) -> UserSnapshot:
    """사용자 스냅샷을 정규화하고 모델 경계 규칙을 검증한다."""

    if not isinstance(snapshot.user_id, str):
        raise InputValidationError("INVALID_USER_ID", field_name="user_id")
    user_id = _normalize_text(snapshot.user_id)
    if not user_id or len(user_id) > 128:
        raise InputValidationError("INVALID_USER_ID", field_name="user_id")

    if not isinstance(snapshot.bio, str):
        raise InputValidationError("INVALID_BIO", field_name="bio")
    bio = _normalize_text(snapshot.bio)
    if len(bio) > policy.bio_max_chars:
        raise InputValidationError("BIO_TOO_LONG", field_name="bio")
    if bio and contains_contact_info(bio):
        raise InputValidationError("CONTACT_INFO_DETECTED", field_name="bio")

    if len(snapshot.tag_ids) > policy.max_tags:
        raise InputValidationError("TOO_MANY_TAGS", field_name="tag_ids")
    tags: list[str] = []
    seen_tags: set[str] = set()
    for raw_tag in snapshot.tag_ids:
        if not isinstance(raw_tag, str):
            raise InputValidationError("INVALID_TAG", field_name="tag_ids")
        tag = _normalize_text(raw_tag)
        if not tag or len(tag) > policy.max_tag_chars:
            raise InputValidationError("INVALID_TAG", field_name="tag_ids")
        if contains_contact_info(tag):
            raise InputValidationError("CONTACT_INFO_DETECTED", field_name="tag_ids")
        if tag not in seen_tags:
            tags.append(tag)
            seen_tags.add(tag)
    if policy.allowed_tag_ids is not None and any(
        tag not in policy.allowed_tag_ids for tag in tags
    ):
        raise InputValidationError("UNKNOWN_TAG", field_name="tag_ids")

    age_years = snapshot.age_years
    if isinstance(age_years, bool) or (age_years is not None and not isinstance(age_years, int)):
        raise InputValidationError("INVALID_AGE", field_name="age_years")
    age_band = normalize_age_band(snapshot.age_band)
    if age_years is None and age_band is None:
        raise InputValidationError("MISSING_AGE", field_name="age_band")
    if age_years is not None:
        derived_band = age_band_for_age(age_years)
        if age_band is not None and age_band != derived_band:
            raise InputValidationError("AGE_BAND_MISMATCH", field_name="age_band")
        age_band = derived_band

    if not isinstance(snapshot.ui_mode, str):
        raise InputValidationError("INVALID_UI_MODE", field_name="ui_mode")
    ui_mode = _normalize_text(snapshot.ui_mode)
    if len(ui_mode) > 64 or (ui_mode and ui_mode not in policy.allowed_ui_modes):
        raise InputValidationError("INVALID_UI_MODE", field_name="ui_mode")

    return replace(
        snapshot,
        user_id=user_id,
        bio=bio,
        tag_ids=tuple(tags),
        age_years=age_years,
        age_band=age_band,
        ui_mode=ui_mode,
    )


def _validate_timestamp(value: datetime, *, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise InputValidationError("INVALID_TIMESTAMP", field_name=field_name)


def select_content_signals(
    items: Sequence[ContentSignal],
    *,
    kind: Literal["authored", "liked"],
    policy: MatchingInputPolicy,
    as_of: datetime,
) -> tuple[ContentSignal, ...]:
    """활성·접근 가능·기간 내 게시물/댓글만 최신순으로 반환한다."""

    _validate_timestamp(as_of, field_name="as_of")
    if kind == "authored":
        lookback_days = policy.authored_lookback_days
        max_items = policy.max_authored_items
    elif kind == "liked":
        lookback_days = policy.liked_lookback_days
        max_items = policy.max_liked_items
    else:
        raise ValueError("kind must be 'authored' or 'liked'")

    cutoff = as_of - timedelta(days=lookback_days)
    # content_id별 최신 상태를 먼저 확정해야 최신 삭제/취소 tombstone 뒤의 과거 활성
    # 레코드가 되살아나지 않는다.
    versioned: list[ContentSignal] = []
    for item in items:
        if not isinstance(item, ContentSignal):
            raise InputValidationError("INVALID_CONTENT", field_name=f"{kind}_items")
        _validate_timestamp(item.created_at, field_name="created_at")
        if item.created_at < cutoff or item.created_at > as_of:
            continue
        if not isinstance(item.content_id, str):
            raise InputValidationError("INVALID_CONTENT_ID", field_name="content_id")
        content_id = _normalize_text(item.content_id)
        if not content_id or len(content_id) > 128:
            raise InputValidationError("INVALID_CONTENT_ID", field_name="content_id")
        versioned.append(replace(item, content_id=content_id))

    versioned.sort(
        key=lambda item: (
            item.created_at,
            item.is_deleted
            or not item.is_accessible
            or item.is_blocked_author
            or (kind == "liked" and not item.is_like_active),
            item.content_id,
        ),
        reverse=True,
    )
    current_items: list[ContentSignal] = []
    seen_ids: set[str] = set()
    for item in versioned:
        if item.content_id in seen_ids:
            continue
        current_items.append(item)
        seen_ids.add(item.content_id)

    eligible: list[ContentSignal] = []
    for item in current_items:
        if item.source_type not in policy.allowed_content_sources:
            raise InputValidationError("INVALID_CONTENT_SOURCE", field_name="source_type")
        if item.is_deleted or not item.is_accessible or item.is_blocked_author:
            continue
        if kind == "liked" and not item.is_like_active:
            continue
        if not isinstance(item.text, str):
            raise InputValidationError("INVALID_CONTENT_TEXT", field_name="text")
        text = _normalize_text(item.text)
        if not text or contains_contact_info(text):
            continue
        eligible.append(
            replace(
                item,
                content_id=item.content_id,
                text=text[: policy.max_content_chars],
            )
        )

    return tuple(eligible[:max_items])


def _profile_text(snapshot: UserSnapshot) -> str | None:
    tag_text = f"관심사: {', '.join(snapshot.tag_ids)}" if snapshot.tag_ids else ""
    if snapshot.bio and tag_text:
        return f"{snapshot.bio} {tag_text}"
    return snapshot.bio or tag_text or None


def prepare_text_signals(
    snapshot: UserSnapshot,
    *,
    policy: MatchingInputPolicy,
    as_of: datetime,
) -> PreparedTextSignals:
    validated = validate_user_snapshot(snapshot, policy)
    authored = select_content_signals(
        validated.authored_items,
        kind="authored",
        policy=policy,
        as_of=as_of,
    )
    liked = select_content_signals(
        validated.liked_items,
        kind="liked",
        policy=policy,
        as_of=as_of,
    )
    return PreparedTextSignals(
        user_id=validated.user_id,
        tag_ids=validated.tag_ids,
        age_years=validated.age_years,
        age_band=validated.age_band,
        ui_mode=validated.ui_mode,
        profile_text=_profile_text(validated),
        authored_texts=tuple(item.text for item in authored),
        liked_texts=tuple(item.text for item in liked),
    )


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _aggregate_vectors(vectors: Sequence[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    normalized = np.stack([_unit_vector(vector) for vector in vectors])
    return _unit_vector(normalized.mean(axis=0))


def encode_prepared_users(
    users: Sequence[PreparedTextSignals],
    *,
    encoder: TextEncoder,
    policy: MatchingInputPolicy,
) -> tuple[PreparedUserFeatures, ...]:
    """여러 사용자의 모든 허용 텍스트를 한 번의 encoder 배치로 집계한다."""

    texts: list[str] = []
    references: list[tuple[int, Literal["profile", "authored", "liked"]]] = []
    for user_index, prepared in enumerate(users):
        if prepared.profile_text is not None:
            texts.append(prepared.profile_text)
            references.append((user_index, "profile"))
        for text in prepared.authored_texts:
            texts.append(text)
            references.append((user_index, "authored"))
        for text in prepared.liked_texts:
            texts.append(text)
            references.append((user_index, "liked"))

    if texts:
        encoded = np.asarray(
            encoder.encode(
                texts,
                batch_size=policy.embedding_batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if (
            encoded.ndim != 2
            or encoded.shape[0] != len(texts)
            or encoded.shape[1] == 0
            or not np.isfinite(encoded).all()
        ):
            raise InputValidationError("ENCODER_OUTPUT_INVALID")
    else:
        encoded = np.empty((0, 0), dtype=np.float32)

    buckets: list[dict[str, list[np.ndarray]]] = [
        {"profile": [], "authored": [], "liked": []} for _ in users
    ]
    for vector, (user_index, component) in zip(encoded, references, strict=True):
        buckets[user_index][component].append(vector)

    results: list[PreparedUserFeatures] = []
    for prepared, bucket in zip(users, buckets, strict=True):
        # 레거시 LambdaMART의 L2 분포를 유지하기 위해 profile은 raw SBERT 벡터로 보존한다.
        profile_vector = (
            np.asarray(bucket["profile"][0], dtype=np.float32) if bucket["profile"] else None
        )
        authored_vector = _aggregate_vectors(bucket["authored"])
        liked_vector = _aggregate_vectors(bucket["liked"])
        effective_vector = _aggregate_vectors(
            [
                vector
                for vector in (profile_vector, authored_vector, liked_vector)
                if vector is not None
            ]
        )
        results.append(
            PreparedUserFeatures(
                user_id=prepared.user_id,
                tag_ids=prepared.tag_ids,
                age_years=prepared.age_years,
                age_band=prepared.age_band,
                ui_mode=prepared.ui_mode,
                profile_vector=profile_vector,
                authored_vector=authored_vector,
                liked_vector=liked_vector,
                effective_vector=effective_vector,
                authored_count=len(prepared.authored_texts),
                liked_count=len(prepared.liked_texts),
            )
        )
    return tuple(results)


def _minor_status(age_band: str | None, age_years: int | None) -> bool | None:
    normalized = normalize_age_band(age_band)
    if normalized is None and age_years is not None:
        if isinstance(age_years, bool) or not isinstance(age_years, int):
            raise InputValidationError("INVALID_AGE", field_name="age_years")
        normalized = age_band_for_age(age_years)
    if normalized is None:
        return None
    return normalized == AGE_BAND_14_18


def candidate_exclusion_reason(
    me: UserSnapshot,
    candidate: UserSnapshot,
    relationship: CandidateRelationship,
    *,
    as_of: datetime,
    rejection_cooldown_days: int = 30,
) -> str | None:
    """점수·임베딩 계산 전에 적용할 후보 제외 사유를 반환한다."""

    _validate_timestamp(as_of, field_name="as_of")
    me_id = _normalize_text(me.user_id)
    candidate_id = _normalize_text(candidate.user_id)
    relation_id = _normalize_text(relationship.candidate_id)
    if candidate_id != relation_id:
        raise InputValidationError("CANDIDATE_RELATION_MISMATCH", field_name="candidate_id")
    if me_id == candidate_id:
        return "self"
    if relationship.blocked_either_direction:
        return "blocked"
    if relationship.already_friends:
        return "already_friends"
    if relationship.last_rejected_at is not None:
        _validate_timestamp(relationship.last_rejected_at, field_name="last_rejected_at")
        if as_of - relationship.last_rejected_at < timedelta(days=rejection_cooldown_days):
            return "recent_rejection"
    me_minor = _minor_status(me.age_band, me.age_years)
    candidate_minor = _minor_status(candidate.age_band, candidate.age_years)
    if me_minor is not None and candidate_minor is not None and me_minor != candidate_minor:
        return "minor_adult_separation"
    return None


def _vector_pair(left: np.ndarray | None, right: np.ndarray | None) -> tuple[float, float, float]:
    if left is None or right is None:
        return 0.0, 0.0, 0.0
    cosine = float(np.clip(np.dot(_unit_vector(left), _unit_vector(right)), -1.0, 1.0))
    l2 = float(np.linalg.norm(left - right))
    return cosine, l2, 1.0


def build_pair_features(
    query: PreparedUserFeatures,
    candidate: PreparedUserFeatures,
    relationship: CandidateRelationship,
) -> dict[str, float]:
    """신규 스키마 특성과 구형 모델 호환용 cosine/l2를 함께 만든다."""

    effective_cosine, effective_l2, effective_available = _vector_pair(
        query.effective_vector, candidate.effective_vector
    )
    profile_cosine, profile_l2, profile_available = _vector_pair(
        query.profile_vector, candidate.profile_vector
    )
    authored_cosine, _authored_l2, authored_available = _vector_pair(
        query.authored_vector, candidate.authored_vector
    )
    liked_cosine, _liked_l2, liked_available = _vector_pair(
        query.liked_vector, candidate.liked_vector
    )
    liked_authored_cosine, _cross_l2, liked_authored_available = _vector_pair(
        query.liked_vector, candidate.authored_vector
    )

    query_tags = set(query.tag_ids)
    candidate_tags = set(candidate.tag_ids)
    overlap = len(query_tags & candidate_tags)
    union = len(query_tags | candidate_tags)
    age_available = query.age_years is not None and candidate.age_years is not None
    band_available = query.age_band is not None and candidate.age_band is not None

    return {
        # 기존 LambdaMART는 profile 텍스트의 raw SBERT cosine/L2로 학습되었다.
        "f_cosine": profile_cosine,
        "f_l2": profile_l2,
        "f_effective_cosine": effective_cosine,
        "f_effective_l2": effective_l2,
        "f_effective_available": effective_available,
        "f_profile_cosine": profile_cosine,
        "f_profile_l2": profile_l2,
        "f_profile_available": profile_available,
        "f_authored_cosine": authored_cosine,
        "f_authored_available": authored_available,
        "f_liked_cosine": liked_cosine,
        "f_liked_available": liked_available,
        "f_liked_authored_cosine": liked_authored_cosine,
        "f_liked_authored_available": liked_authored_available,
        "f_tag_overlap": float(overlap),
        "f_tag_jaccard": float(overlap / union) if union else 0.0,
        "f_common_friend_count": float(relationship.common_friend_count),
        "f_age_diff": (float(abs(query.age_years - candidate.age_years)) if age_available else 0.0),
        "f_age_available": float(age_available),
        "f_age_band_match": float(band_available and query.age_band == candidate.age_band),
        "f_age_band_available": float(band_available),
        "f_ui_mode_match": float(bool(query.ui_mode) and query.ui_mode == candidate.ui_mode),
        "f_authored_count": float(candidate.authored_count),
        "f_liked_count": float(candidate.liked_count),
    }


def build_recommendation_reasons(
    features: dict[str, float],
    *,
    policy: MatchingInputPolicy | None = None,
) -> list[str]:
    """원문·UI 모드·장애 정보를 쓰지 않는 일반화 사유만 반환한다."""

    active_policy = policy or MatchingInputPolicy()
    reasons: list[str] = []
    if features.get("f_tag_overlap", 0.0) >= active_policy.tag_reason_min_overlap:
        reasons.append("관심사가 비슷해요")

    content_similar = any(
        features.get(available_key, 0.0) > 0
        and features.get(similarity_key, 0.0) >= active_policy.content_reason_min
        for similarity_key, available_key in (
            ("f_liked_cosine", "f_liked_available"),
            ("f_authored_cosine", "f_authored_available"),
            ("f_liked_authored_cosine", "f_liked_authored_available"),
        )
    )
    if content_similar:
        reasons.append("관심 있는 콘텐츠가 비슷해요")
    if features.get("f_common_friend_count", 0.0) > 0:
        reasons.append("공통 친구가 있어요")
    if (
        features.get("f_age_available", 0.0) > 0
        and features.get("f_age_diff", float("inf")) <= active_policy.age_reason_max_diff
    ):
        reasons.append("비슷한 연령대예요")
    # "소개 내용이 비슷해요"는 폐기했다. 임계값 0.5에서 추천의 92.5%에 붙으면서 SHAP
    # 뒷받침률이 36.1%였고, 뒷받침률을 올리려 0.75로 높이면 노출이 6.5%로 떨어져
    # 어느 값에서도 정보 가치가 없었다(artifacts/match_v2_shap.json).

    safe_reasons = [reason for reason in reasons if reason in ALLOWED_RECOMMENDATION_REASONS]
    return safe_reasons[: active_policy.max_reasons]


def _validate_relationship(relationship: CandidateRelationship) -> CandidateRelationship:
    if type(relationship.blocked_either_direction) is not bool:
        raise InputValidationError("INVALID_BLOCK_STATUS", field_name="blocked_either_direction")
    if type(relationship.already_friends) is not bool:
        raise InputValidationError("INVALID_FRIEND_STATUS", field_name="already_friends")
    if isinstance(relationship.common_friend_count, bool) or not isinstance(
        relationship.common_friend_count, int
    ):
        raise InputValidationError("INVALID_COMMON_FRIEND_COUNT", field_name="common_friend_count")
    if relationship.common_friend_count < 0:
        raise InputValidationError("INVALID_COMMON_FRIEND_COUNT", field_name="common_friend_count")
    if relationship.last_rejected_at is not None:
        _validate_timestamp(relationship.last_rejected_at, field_name="last_rejected_at")
    return relationship


def prepare_match_inputs(
    me: UserSnapshot,
    candidates: Sequence[CandidateInput],
    *,
    encoder: TextEncoder,
    policy: MatchingInputPolicy,
    as_of: datetime,
) -> PreparedMatchBatch:
    """후보 제외 후 허용 텍스트만 일괄 임베딩하고 페어 특성을 만든다."""

    _validate_timestamp(as_of, field_name="as_of")
    if len(candidates) > policy.max_candidates:
        raise InputValidationError("TOO_MANY_CANDIDATES", field_name="candidates")

    validated_me = validate_user_snapshot(me, policy)
    eligible: list[tuple[UserSnapshot, CandidateRelationship]] = []
    excluded: list[ExcludedCandidate] = []
    seen_candidate_ids: set[str] = set()
    for item in candidates:
        if not isinstance(item, CandidateInput):
            raise InputValidationError("INVALID_CANDIDATE", field_name="candidates")
        candidate_id = _normalize_text(item.profile.user_id)
        if candidate_id in seen_candidate_ids:
            raise InputValidationError("DUPLICATE_CANDIDATE", field_name="candidate_id")
        seen_candidate_ids.add(candidate_id)

        relationship = _validate_relationship(item.relationship)
        reason = candidate_exclusion_reason(
            validated_me,
            item.profile,
            relationship,
            as_of=as_of,
            rejection_cooldown_days=policy.rejection_cooldown_days,
        )
        if reason is not None:
            excluded.append(ExcludedCandidate(candidate_id=candidate_id, reason=reason))
            continue
        eligible.append((validate_user_snapshot(item.profile, policy), relationship))

    prepared_texts = [prepare_text_signals(validated_me, policy=policy, as_of=as_of)]
    prepared_texts.extend(
        prepare_text_signals(profile, policy=policy, as_of=as_of) for profile, _ in eligible
    )
    encoded_users = encode_prepared_users(prepared_texts, encoder=encoder, policy=policy)
    query_features = encoded_users[0]

    prepared_candidates = tuple(
        PreparedCandidate(
            user_id=candidate_features.user_id,
            features=build_pair_features(query_features, candidate_features, relationship),
        )
        for candidate_features, (_profile, relationship) in zip(
            encoded_users[1:], eligible, strict=True
        )
    )

    has_query_signal = query_features.effective_vector is not None or bool(query_features.ui_mode)
    has_pair_signal = any(
        candidate.features["f_common_friend_count"] > 0 for candidate in prepared_candidates
    )
    if not has_query_signal and not has_pair_signal:
        status: Literal["ok", "insufficient_signal", "no_eligible_candidates"] = (
            "insufficient_signal"
        )
    elif not prepared_candidates:
        status = "no_eligible_candidates"
    else:
        status = "ok"

    return PreparedMatchBatch(
        query=query_features,
        candidates=prepared_candidates,
        excluded=tuple(excluded),
        status=status,
    )
