"""MATCH 모듈② v2 재학습용 라벨·페어 합성 (P2).

라벨(0~3)은 **잠재 변수에서만** 계산한다(build_profiles_v2.LatentUser):
- 잠재 관심 유사도 (interest_weights 코사인)
- 나이 근접
- 잠재 사회 그래프 (social_cluster 공유 → 공통 친구)

관측 특성(태그 교집합, cosine, bio 텍스트 등)은 라벨 식에 넣지 않는다. 관측은 잠재의
노이즈 낀 실현이므로, 모델은 관측→잠재를 추정하는 실제 과제를 학습한다(EXP-2 순환성 방지).
ui_mode·장애 관련 변수는 라벨에 넣지 않는다(보호 속성 비의존).

라벨 계수(LabelWeights)와 임계값은 P1.5 외부 데이터(SNAP Pokec)로 보정 가능한 지점이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.build_profiles_v2 import LatentUser, SyntheticUser
from src.data.matching_input import CandidateInput, CandidateRelationship, UserSnapshot

AGE_PROXIMITY_SCALE = 15.0  # 이 나이차에서 근접도가 0에 수렴


@dataclass(frozen=True, slots=True)
class LabelWeights:
    """잠재 호환성 → 라벨 점수 가중치. Pokec 보정 대상(P1.5)."""

    interest: float = 0.55
    age: float = 0.20
    social: float = 0.25
    # score → 라벨(0~3) 경계. 기본 생성 분포(n=2000, seed=42)의 score 분위수
    # p65/p88/p96 ≈ 0.17/0.27/0.40에 맞춰 등급 라벨이 고르게 나오도록 보정했다.
    # P1.5(Pokec)에서 실측 friendship 상관으로 재보정 가능한 지점이다.
    thresholds: tuple[float, float, float] = (0.17, 0.27, 0.40)


def interest_similarity(latent_q: LatentUser, latent_c: LatentUser) -> float:
    a = latent_q.interest_weights
    b = latent_c.interest_weights
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def age_proximity(latent_q: LatentUser, latent_c: LatentUser) -> float:
    diff = abs(latent_q.age - latent_c.age)
    return float(max(0.0, 1.0 - diff / AGE_PROXIMITY_SCALE))


def social_affinity(latent_q: LatentUser, latent_c: LatentUser, common_friends: int) -> float:
    """같은 잠재 클러스터 + 공통 친구 수에서 사회적 근접도를 만든다."""

    cluster_bonus = 0.6 if latent_q.social_cluster == latent_c.social_cluster else 0.0
    friend_term = 0.4 * (1.0 - np.exp(-common_friends / 3.0))
    return float(min(1.0, cluster_bonus + friend_term))


def sample_common_friends(
    rng: np.random.Generator, latent_q: LatentUser, latent_c: LatentUser
) -> int:
    """잠재 클러스터 공유 여부에 따라 공통 친구 수를 뽑는다."""

    if latent_q.social_cluster == latent_c.social_cluster:
        return int(rng.poisson(4.0))
    return int(rng.poisson(0.4))


def latent_score(
    latent_q: LatentUser,
    latent_c: LatentUser,
    common_friends: int,
    weights: LabelWeights,
) -> float:
    return (
        weights.interest * interest_similarity(latent_q, latent_c)
        + weights.age * age_proximity(latent_q, latent_c)
        + weights.social * social_affinity(latent_q, latent_c, common_friends)
    )


def score_to_label(score: float, weights: LabelWeights) -> int:
    low, mid, high = weights.thresholds
    if score >= high:
        return 3
    if score >= mid:
        return 2
    if score >= low:
        return 1
    return 0


@dataclass(frozen=True, slots=True)
class PairRecord:
    query_id: str
    cand_id: str
    query_snapshot: UserSnapshot
    candidate_input: CandidateInput
    label: int
    latent_score: float  # sanity check(oracle 특성)용


def build_pairs(
    users: list[SyntheticUser],
    *,
    n_queries: int,
    n_candidates: int,
    weights: LabelWeights | None = None,
    seed: int = 42,
) -> list[PairRecord]:
    """쿼리별 후보 슬레이트와 잠재 라벨을 만든다.

    users는 이미 disjoint split된 풀이어야 한다(호출자가 train/test 분리).
    """

    weights = weights or LabelWeights()
    rng = np.random.default_rng(seed)
    n_users = len(users)
    n_queries_eff = min(n_queries, n_users)
    n_candidates_eff = min(n_candidates, n_users - 1)

    query_idx = rng.choice(n_users, size=n_queries_eff, replace=False)
    records: list[PairRecord] = []
    for qi in query_idx:
        q = users[int(qi)]
        cand_idx = rng.choice(n_users, size=min(n_candidates_eff + 1, n_users), replace=False)
        added = 0
        for ci in cand_idx:
            if int(ci) == int(qi):
                continue
            if added >= n_candidates_eff:
                break
            c = users[int(ci)]
            common_friends = sample_common_friends(rng, q.latent, c.latent)
            score = latent_score(q.latent, c.latent, common_friends, weights)
            label = score_to_label(score, weights)
            relationship = CandidateRelationship(
                candidate_id=c.snapshot.user_id,
                blocked_either_direction=False,
                already_friends=False,
                last_rejected_at=None,
                common_friend_count=common_friends,
            )
            records.append(
                PairRecord(
                    query_id=q.snapshot.user_id,
                    cand_id=c.snapshot.user_id,
                    query_snapshot=q.snapshot,
                    candidate_input=CandidateInput(profile=c.snapshot, relationship=relationship),
                    label=label,
                    latent_score=score,
                )
            )
            added += 1
    return records
