"""P1/P2 합성 생성기·라벨 회귀 테스트.

핵심 검증:
- 생성 스냅샷 전수가 validate_user_snapshot을 통과하고 금지 필드가 없다.
- seed 고정 시 결정성.
- 라벨이 잠재 변수 단조성을 따른다(sanity check).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.build_pairs_v2 import (
    LabelWeights,
    build_pairs,
    interest_similarity,
    latent_score,
    score_to_label,
)
from src.data.build_profiles_v2 import (
    GenerationConfig,
    LatentUser,
    generate_population,
    load_allowed_tags,
)
from src.data.matching_input import (
    MatchingInputPolicy,
    UserSnapshot,
    prepare_match_inputs,
    validate_user_snapshot,
)

ALLOWED_TAGS = load_allowed_tags()
POLICY = MatchingInputPolicy(allowed_tag_ids=frozenset(ALLOWED_TAGS))
SMALL = GenerationConfig(n_users=200, seed=7)


def test_snapshots_pass_validation_and_have_no_forbidden_fields():
    users = generate_population(SMALL, policy=POLICY)
    assert len(users) == 200
    forbidden = {"region", "disability_type", "birthdate", "nickname", "profile_image"}
    for u in users:
        # 검증기를 다시 통과해야 하며(멱등) 예외가 없어야 한다.
        validate_user_snapshot(u.snapshot, POLICY)
        assert isinstance(u.snapshot, UserSnapshot)
        assert set(vars(type(u.snapshot))).isdisjoint(forbidden)
        assert all(tag in ALLOWED_TAGS for tag in u.snapshot.tag_ids)
        assert len(u.snapshot.tag_ids) <= POLICY.max_tags
        assert len(u.snapshot.bio) <= POLICY.bio_max_chars


def test_generation_is_deterministic():
    a = generate_population(SMALL, policy=POLICY)
    b = generate_population(SMALL, policy=POLICY)
    assert [u.snapshot for u in a] == [u.snapshot for u in b]
    assert all(
        np.array_equal(x.latent.interest_weights, y.latent.interest_weights)
        for x, y in zip(a, b, strict=True)
    )


def test_missing_signal_profiles_exist():
    users = generate_population(GenerationConfig(n_users=500, seed=1), policy=POLICY)
    assert any(u.snapshot.bio == "" for u in users)
    assert any(len(u.snapshot.authored_items) == 0 for u in users)
    assert any(len(u.snapshot.liked_items) == 0 for u in users)
    # 최소 신호(태그)는 항상 존재
    assert all(len(u.snapshot.tag_ids) >= 1 for u in users)


def test_latent_fields_not_leaked_into_snapshot():
    users = generate_population(SMALL, policy=POLICY)
    # slotted dataclass의 필드는 __slots__로 확인한다.
    slots = set(type(users[0].snapshot).__slots__)
    assert "interest_weights" not in slots
    assert "social_cluster" not in slots


def _latent(weights, age, cluster):
    return LatentUser("x", np.asarray(weights, dtype=np.float64), age, cluster)


def test_label_monotonic_in_interest_similarity():
    w = LabelWeights()
    base = _latent([1.0, 0.0, 0.0], 30, 0)
    same = _latent([1.0, 0.0, 0.0], 30, 0)
    orthogonal = _latent([0.0, 0.0, 1.0], 30, 0)
    s_same = latent_score(base, same, 0, w)
    s_diff = latent_score(base, orthogonal, 0, w)
    assert s_same > s_diff
    assert interest_similarity(base, same) == pytest.approx(1.0)
    assert interest_similarity(base, orthogonal) == pytest.approx(0.0)


def test_score_to_label_boundaries():
    w = LabelWeights(thresholds=(0.30, 0.50, 0.70))
    assert score_to_label(0.71, w) == 3
    assert score_to_label(0.70, w) == 3
    assert score_to_label(0.50, w) == 2
    assert score_to_label(0.30, w) == 1
    assert score_to_label(0.29, w) == 0


def test_build_pairs_produces_grouped_labeled_records():
    users = generate_population(GenerationConfig(n_users=120, seed=3), policy=POLICY)
    pairs = build_pairs(users, n_queries=20, n_candidates=10, seed=5)
    assert pairs
    # 자기 자신은 후보로 들어오지 않는다.
    assert all(p.query_id != p.cand_id for p in pairs)
    # 라벨은 0~3.
    assert all(0 <= p.label <= 3 for p in pairs)
    # 관계 candidate_id가 후보 스냅샷과 일치(검증기 계약).
    for p in pairs:
        assert p.candidate_input.relationship.candidate_id == p.candidate_input.profile.user_id
    # 라벨 분포가 한 값에 몰리지 않는다(생성기 신호가 살아있음).
    labels = {p.label for p in pairs}
    assert len(labels) >= 2


def test_build_pairs_deterministic():
    users = generate_population(GenerationConfig(n_users=120, seed=3), policy=POLICY)
    a = build_pairs(users, n_queries=20, n_candidates=10, seed=5)
    b = build_pairs(users, n_queries=20, n_candidates=10, seed=5)
    assert [(p.query_id, p.cand_id, p.label) for p in a] == [
        (p.query_id, p.cand_id, p.label) for p in b
    ]


class _HashEncoder:
    """텍스트 해시 기반 결정적 벡터. SBERT 대역(P3 노트북 로직 스모크용)."""

    def encode(self, sentences, *, batch_size, show_progress_bar):
        rows = []
        for text in sentences:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            rows.append(rng.normal(size=16))
        return np.asarray(rows, dtype=np.float32)


def test_generated_users_flow_through_matching_pipeline():
    """생성 스냅샷(실코퍼스 콘텐츠 포함)이 prepare_match_inputs 전 경로를 통과한다."""

    users = generate_population(GenerationConfig(n_users=120, seed=9), policy=POLICY)
    pairs = build_pairs(users, n_queries=10, n_candidates=8, seed=9)
    by_query: dict[str, list] = {}
    for p in pairs:
        by_query.setdefault(p.query_id, []).append(p)
    # 후보가 가장 많은 쿼리 하나로 파이프라인을 돌린다.
    query_id = max(by_query, key=lambda q: len(by_query[q]))
    group = by_query[query_id]
    me = group[0].query_snapshot
    candidates = [p.candidate_input for p in group]

    batch = prepare_match_inputs(
        me,
        candidates,
        encoder=_HashEncoder(),
        policy=POLICY,
        as_of=GenerationConfig().as_of,
    )
    assert batch.status in {"ok", "no_eligible_candidates", "insufficient_signal"}
    for cand in batch.candidates:
        # v2 스키마 특성이 실제로 생성된다.
        for key in ("f_effective_cosine", "f_tag_jaccard", "f_common_friend_count"):
            assert key in cand.features
        assert -1.0 <= cand.features["f_effective_cosine"] <= 1.0
