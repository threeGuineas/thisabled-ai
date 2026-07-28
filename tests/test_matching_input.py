"""MATCH-02 입력 파이프라인의 경계·개인정보·후보 제외 회귀 테스트."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.data.matching_input import (
    AGE_BAND_14_18,
    AGE_BAND_19_24,
    AGE_BAND_25_34,
    CandidateInput,
    CandidateRelationship,
    ContentSignal,
    InputValidationError,
    MatchingInputPolicy,
    UserSnapshot,
    build_recommendation_reasons,
    candidate_exclusion_reason,
    prepare_match_inputs,
    select_content_signals,
    validate_user_snapshot,
)

NOW = datetime(2026, 7, 20, 12, tzinfo=UTC)


class RecordingEncoder:
    """문장별 고정 벡터를 반환하고 실제 전달 문자열을 기록한다."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        self.calls.append(list(sentences))
        return np.asarray(
            [self.vectors.get(text, [1.0, 0.0, 0.0]) for text in sentences],
            dtype=np.float32,
        )


def user(
    user_id: str,
    *,
    bio: str = "영화 이야기를 좋아해요.",
    tags: tuple[str, ...] = ("영화",),
    age: int | None = 27,
    age_band: str | None = AGE_BAND_25_34,
    ui_mode: str = "visual",
    authored: tuple[ContentSignal, ...] = (),
    liked: tuple[ContentSignal, ...] = (),
) -> UserSnapshot:
    return UserSnapshot(
        user_id=user_id,
        bio=bio,
        tag_ids=tags,
        age_years=age,
        age_band=age_band,
        ui_mode=ui_mode,
        authored_items=authored,
        liked_items=liked,
    )


def content(
    content_id: str,
    text: str,
    *,
    days_ago: int = 1,
    source_type: str = "post",
    deleted: bool = False,
    accessible: bool = True,
    blocked_author: bool = False,
    like_active: bool = True,
) -> ContentSignal:
    return ContentSignal(
        content_id=content_id,
        source_type=source_type,
        text=text,
        created_at=NOW - timedelta(days=days_ago),
        is_deleted=deleted,
        is_accessible=accessible,
        is_blocked_author=blocked_author,
        is_like_active=like_active,
    )


@pytest.fixture
def policy() -> MatchingInputPolicy:
    return MatchingInputPolicy(
        authored_lookback_days=90,
        liked_lookback_days=90,
        max_authored_items=2,
        max_liked_items=2,
        max_content_chars=2_000,
        max_candidates=20,
        allowed_tag_ids=frozenset({"영화", "야구", "걷기", "게임"}),
    )


@pytest.mark.parametrize(
    "bio",
    [
        "연락은 test@example.com 으로 주세요",
        "제 번호는 010-1234-5678이에요",
        "카톡 아이디: hello_friend",
        "카톡 아이디는 hello_friend",
        "카톡 abc123으로 연락해요",
        "인스타 insta_friend로 연락해요",
        "인스타그램 계정은 cool_friend",
        "카톡으로 hello123",
        "텔레그램은 secret123",
        "인스타는 cool_friend",
        "라인으로 friend123",
        "해외 번호는 +1 415-555-2671입니다",
        "오픈채팅 https://open.kakao.com/o/abc123",
    ],
)
def test_bio_contact_info_is_rejected_without_echoing_value(
    bio: str, policy: MatchingInputPolicy
) -> None:
    with pytest.raises(InputValidationError) as exc_info:
        validate_user_snapshot(user("me", bio=bio), policy)

    assert exc_info.value.code == "CONTACT_INFO_DETECTED"
    assert bio not in str(exc_info.value)
    assert "010-1234-5678" not in str(exc_info.value)
    assert "test@example.com" not in str(exc_info.value)


@pytest.mark.parametrize(
    "bio",
    [
        "lineage 게임을 좋아해요",
        "온라인으로 친구와 게임해요",
        "인스타그램 사진을 구경해요",
    ],
)
def test_non_contact_messenger_words_are_not_rejected(
    bio: str, policy: MatchingInputPolicy
) -> None:
    assert validate_user_snapshot(user("me", bio=bio), policy).bio == bio


def test_bio_and_tags_enforce_boundary_and_tag_registry(policy: MatchingInputPolicy) -> None:
    validated = validate_user_snapshot(user("me", bio="가" * 300, tags=("영화", "영화")), policy)
    assert len(validated.bio) == 300
    assert validated.tag_ids == ("영화",)

    with pytest.raises(InputValidationError, match="BIO_TOO_LONG"):
        validate_user_snapshot(user("me", bio="가" * 301), policy)

    with pytest.raises(InputValidationError, match="TOO_MANY_TAGS"):
        validate_user_snapshot(user("me", tags=tuple(f"태그{i}" for i in range(11))), policy)

    with pytest.raises(InputValidationError, match="UNKNOWN_TAG"):
        validate_user_snapshot(user("me", tags=("없는태그",)), policy)

    with pytest.raises(InputValidationError, match="CONTACT_INFO_DETECTED"):
        validate_user_snapshot(user("me", tags=("tag-owner@example.com",)), policy)


def test_age_band_must_match_age_and_raw_birth_date_is_not_a_field(
    policy: MatchingInputPolicy,
) -> None:
    with pytest.raises(InputValidationError, match="AGE_BAND_MISMATCH"):
        validate_user_snapshot(user("me", age=18, age_band=AGE_BAND_19_24), policy)

    assert "birth" not in UserSnapshot.__dataclass_fields__
    assert "date_of_birth" not in UserSnapshot.__dataclass_fields__


def test_age_information_is_required(policy: MatchingInputPolicy) -> None:
    with pytest.raises(InputValidationError, match="MISSING_AGE"):
        validate_user_snapshot(user("me", age=None, age_band=None), policy)


def test_backend_age_band_alias_is_normalized(policy: MatchingInputPolicy) -> None:
    validated = validate_user_snapshot(user("me", age=27, age_band="25-34"), policy)
    assert validated.age_band == AGE_BAND_25_34


def test_unknown_ui_mode_is_rejected(policy: MatchingInputPolicy) -> None:
    with pytest.raises(InputValidationError, match="INVALID_UI_MODE"):
        validate_user_snapshot(user("me", ui_mode="arbitrary-mode"), policy)


def test_content_selection_removes_ineligible_and_contact_items(
    policy: MatchingInputPolicy,
) -> None:
    items = (
        content("keep-new", "영화 후기", days_ago=1),
        content("deleted", "삭제 콘텐츠", deleted=True),
        content("private", "비공개 콘텐츠", accessible=False),
        content("blocked", "차단 사용자 글", blocked_author=True),
        content("contact", "메일 foo@example.com", days_ago=2),
        content("old", "오래된 글", days_ago=91),
        content("keep-second", "야구 후기", days_ago=3, source_type="comment"),
        content("over-limit", "걷기 후기", days_ago=4),
    )

    selected = select_content_signals(items, kind="authored", policy=policy, as_of=NOW)
    assert [item.content_id for item in selected] == ["keep-new", "keep-second"]


def test_liked_content_requires_active_like(policy: MatchingInputPolicy) -> None:
    items = (
        content("cancelled", "영화", like_active=False),
        content("active", "야구", like_active=True),
    )
    selected = select_content_signals(items, kind="liked", policy=policy, as_of=NOW)
    assert [item.content_id for item in selected] == ["active"]


def test_newer_tombstone_prevents_older_content_from_reappearing(
    policy: MatchingInputPolicy,
) -> None:
    authored = (
        content("same", "이전 활성 글", days_ago=2),
        content("same", "삭제됨", days_ago=1, deleted=True),
    )
    liked = (
        content("same-like", "이전 좋아요 글", days_ago=2),
        content("same-like", "취소됨", days_ago=1, like_active=False),
    )

    assert select_content_signals(authored, kind="authored", policy=policy, as_of=NOW) == ()
    assert select_content_signals(liked, kind="liked", policy=policy, as_of=NOW) == ()


def test_non_post_or_comment_source_is_rejected(policy: MatchingInputPolicy) -> None:
    with pytest.raises(InputValidationError, match="INVALID_CONTENT_SOURCE"):
        select_content_signals(
            (content("chat", "1:1 채팅", source_type="chat"),),
            kind="authored",
            policy=policy,
            as_of=NOW,
        )


def test_naive_content_timestamp_is_rejected(policy: MatchingInputPolicy) -> None:
    item = ContentSignal(
        content_id="naive",
        source_type="post",
        text="시간대 없는 글",
        created_at=datetime(2026, 7, 20, 12),
    )
    with pytest.raises(InputValidationError, match="INVALID_TIMESTAMP"):
        select_content_signals((item,), kind="authored", policy=policy, as_of=NOW)


@pytest.mark.parametrize(
    ("me_profile", "candidate", "relation", "expected"),
    [
        (user("same"), user("same"), CandidateRelationship("same"), "self"),
        (
            user("me"),
            user("cand"),
            CandidateRelationship("cand", blocked_either_direction=True),
            "blocked",
        ),
        (
            user("me"),
            user("cand"),
            CandidateRelationship("cand", already_friends=True),
            "already_friends",
        ),
        (
            user("me"),
            user("cand"),
            CandidateRelationship("cand", last_rejected_at=NOW - timedelta(days=29)),
            "recent_rejection",
        ),
        (
            user("minor", age=18, age_band=AGE_BAND_14_18),
            user("adult", age=19, age_band=AGE_BAND_19_24),
            CandidateRelationship("adult"),
            "minor_adult_separation",
        ),
    ],
)
def test_candidate_exclusion_rules(
    me_profile: UserSnapshot,
    candidate: UserSnapshot,
    relation: CandidateRelationship,
    expected: str,
) -> None:
    assert candidate_exclusion_reason(me_profile, candidate, relation, as_of=NOW) == expected


def test_rejection_at_exactly_30_days_is_eligible() -> None:
    relation = CandidateRelationship("cand", last_rejected_at=NOW - timedelta(days=30))
    assert candidate_exclusion_reason(user("me"), user("cand"), relation, as_of=NOW) is None


def test_rejection_just_inside_30_days_is_excluded() -> None:
    relation = CandidateRelationship(
        "cand", last_rejected_at=NOW - timedelta(days=30) + timedelta(microseconds=1)
    )
    assert (
        candidate_exclusion_reason(user("me"), user("cand"), relation, as_of=NOW)
        == "recent_rejection"
    )


def test_minor_adult_separation_is_symmetric() -> None:
    adult = user("adult", age=19, age_band=AGE_BAND_19_24)
    minor = user("minor", age=18, age_band=AGE_BAND_14_18)
    assert (
        candidate_exclusion_reason(adult, minor, CandidateRelationship("minor"), as_of=NOW)
        == "minor_adult_separation"
    )


def test_minor_adult_separation_uses_age_when_band_is_missing() -> None:
    adult = user("adult", age=19, age_band=None)
    minor = user("minor", age=18, age_band=None)
    assert (
        candidate_exclusion_reason(adult, minor, CandidateRelationship("minor"), as_of=NOW)
        == "minor_adult_separation"
    )
    assert (
        candidate_exclusion_reason(minor, adult, CandidateRelationship("adult"), as_of=NOW)
        == "minor_adult_separation"
    )


def test_relationship_flags_must_be_real_booleans(policy: MatchingInputPolicy) -> None:
    encoder = RecordingEncoder()
    candidate = CandidateInput(
        profile=user("cand"),
        relationship=CandidateRelationship("cand", blocked_either_direction="false"),  # type: ignore[arg-type]
    )
    with pytest.raises(InputValidationError, match="INVALID_BLOCK_STATUS"):
        prepare_match_inputs(user("me"), (candidate,), encoder=encoder, policy=policy, as_of=NOW)


def test_excluded_candidate_never_reaches_encoder(policy: MatchingInputPolicy) -> None:
    encoder = RecordingEncoder()
    excluded_text = "연락은 leaked@example.com 으로 주세요"
    candidates = (
        CandidateInput(
            profile=user("blocked", bio=excluded_text),
            relationship=CandidateRelationship("blocked", blocked_either_direction=True),
        ),
        CandidateInput(
            profile=user("ok", bio="영화와 야구를 좋아해요", tags=("영화", "야구")),
            relationship=CandidateRelationship("ok", common_friend_count=1),
        ),
    )

    batch = prepare_match_inputs(user("me"), candidates, encoder=encoder, policy=policy, as_of=NOW)

    assert [item.candidate_id for item in batch.excluded] == ["blocked"]
    assert [item.user_id for item in batch.candidates] == ["ok"]
    assert excluded_text not in [text for call in encoder.calls for text in call]


def test_pipeline_batches_components_and_builds_pair_features(policy: MatchingInputPolicy) -> None:
    me_authored = content("me-post", "내가 쓴 영화 후기")
    me_liked = content("me-like", "좋아한 야구 글")
    cand_authored = content("cand-post", "후보의 영화 후기")
    cand_liked = content("cand-like", "후보가 좋아한 야구 글")

    vectors = {
        "영화 이야기를 좋아해요. 관심사: 영화, 야구": [1.0, 0.0, 0.0],
        "영화 이야기를 좋아해요. 관심사: 영화, 걷기": [1.0, 0.0, 0.0],
        "내가 쓴 영화 후기": [0.0, 1.0, 0.0],
        "후보의 영화 후기": [0.0, 1.0, 0.0],
        "좋아한 야구 글": [0.0, 0.0, 1.0],
        "후보가 좋아한 야구 글": [0.0, 0.0, 1.0],
    }
    encoder = RecordingEncoder(vectors)
    me = user("me", tags=("영화", "야구"), authored=(me_authored,), liked=(me_liked,))
    candidate = CandidateInput(
        profile=user(
            "cand",
            tags=("영화", "걷기"),
            authored=(cand_authored,),
            liked=(cand_liked,),
        ),
        relationship=CandidateRelationship("cand", common_friend_count=2),
    )

    batch = prepare_match_inputs(me, (candidate,), encoder=encoder, policy=policy, as_of=NOW)

    assert len(encoder.calls) == 1
    assert batch.status == "ok"
    assert len(batch.candidates) == 1
    pair = batch.candidates[0]
    assert pair.features["f_cosine"] == pytest.approx(1.0)
    assert pair.features["f_profile_cosine"] == pytest.approx(1.0)
    assert pair.features["f_authored_cosine"] == pytest.approx(1.0)
    assert pair.features["f_liked_cosine"] == pytest.approx(1.0)
    assert pair.features["f_tag_overlap"] == 1.0
    assert pair.features["f_common_friend_count"] == 2.0
    assert pair.features["f_age_band_match"] == 1.0
    assert pair.features["f_ui_mode_match"] == 1.0


def test_legacy_l2_uses_raw_profile_embedding_distance(policy: MatchingInputPolicy) -> None:
    vectors = {
        "기준 소개 관심사: 영화": [2.0, 0.0, 0.0],
        "후보 소개 관심사: 영화": [1.0, 0.0, 0.0],
    }
    encoder = RecordingEncoder(vectors)
    me = user("me", bio="기준 소개")
    candidate = CandidateInput(
        profile=user("cand", bio="후보 소개"),
        relationship=CandidateRelationship("cand"),
    )

    batch = prepare_match_inputs(me, (candidate,), encoder=encoder, policy=policy, as_of=NOW)

    assert batch.candidates[0].features["f_cosine"] == pytest.approx(1.0)
    assert batch.candidates[0].features["f_l2"] == pytest.approx(1.0)


def test_duplicate_candidate_id_is_rejected(policy: MatchingInputPolicy) -> None:
    encoder = RecordingEncoder()
    candidates = (
        CandidateInput(user("same"), CandidateRelationship("same")),
        CandidateInput(user("same"), CandidateRelationship("same", already_friends=True)),
    )
    with pytest.raises(InputValidationError, match="DUPLICATE_CANDIDATE"):
        prepare_match_inputs(user("me"), candidates, encoder=encoder, policy=policy, as_of=NOW)


def test_missing_bio_uses_tags_without_placeholder_text(policy: MatchingInputPolicy) -> None:
    encoder = RecordingEncoder({"관심사: 영화": [1.0, 0.0, 0.0]})
    me = user("me", bio="", tags=("영화",))
    candidate = CandidateInput(
        profile=user("cand", bio="", tags=("영화",)),
        relationship=CandidateRelationship("cand"),
    )

    batch = prepare_match_inputs(me, (candidate,), encoder=encoder, policy=policy, as_of=NOW)

    sent = [text for call in encoder.calls for text in call]
    assert sent == ["관심사: 영화", "관심사: 영화"]
    assert "정보 없음" not in sent
    assert batch.status == "ok"


def test_no_signal_returns_insufficient_signal(policy: MatchingInputPolicy) -> None:
    encoder = RecordingEncoder()
    me = user("me", bio="", tags=(), age=27, age_band=AGE_BAND_25_34, ui_mode="")
    candidate = CandidateInput(
        profile=user("cand", bio="", tags=(), age=28, age_band=AGE_BAND_25_34, ui_mode=""),
        relationship=CandidateRelationship("cand"),
    )

    batch = prepare_match_inputs(me, (candidate,), encoder=encoder, policy=policy, as_of=NOW)

    assert batch.status == "insufficient_signal"
    assert encoder.calls == []


def test_recommendation_reasons_are_generalized_and_ui_mode_is_never_exposed() -> None:
    reasons = build_recommendation_reasons(
        {
            "f_tag_overlap": 2.0,
            "f_common_friend_count": 1.0,
            "f_age_band_match": 1.0,
            "f_liked_cosine": 0.9,
            "f_liked_available": 1.0,
            "f_profile_cosine": 0.8,
            "f_profile_available": 1.0,
            "f_ui_mode_match": 1.0,
        }
    )

    assert reasons == [
        "관심사가 비슷해요",
        "관심 있는 콘텐츠가 비슷해요",
        "공통 친구가 있어요",
        "비슷한 연령대예요",
    ]
    assert all("모드" not in reason and "장애" not in reason for reason in reasons)
