"""MATCH 모듈② v2 재학습셋 구성 (P3 지원).

노트북(notebooks/module2_retrain_v2_colab.ipynb)은 이 모듈을 import해 오케스트레이션만
한다. 특성 생성은 반드시 matching_input.build_pair_features를 그대로 호출해 train/serve
일관성을 지킨다(별도 재구현 금지).

핵심 흐름:
1. split_users: user-level disjoint split(cold-start).
2. embed_users: 전체 사용자를 encoder로 1회 배치 임베딩 → user_id별 PreparedUserFeatures.
3. build_feature_frame: 페어별 build_pair_features → (X, y, groups). 서빙과 동일한
   candidate_exclusion_reason으로 제외 대상 페어를 학습에서도 뺀다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.data.build_pairs_v2 import PairRecord
from src.data.build_profiles_v2 import DEFAULT_CONFIG_PATH, SyntheticUser
from src.data.matching_input import (
    MatchingInputPolicy,
    PreparedUserFeatures,
    TextEncoder,
    build_pair_features,
    candidate_exclusion_reason,
    encode_prepared_users,
    prepare_text_signals,
)


@dataclass(frozen=True, slots=True)
class PairFrameTrace:
    """생성된 한 페어가 평가 행으로 이어지는 과정을 보존한다."""

    query_position: int
    candidate_position: int
    query_id: str
    candidate_id: str
    label: int
    latent_score: float
    common_friend_count: int
    blocked_either_direction: bool
    already_friends: bool
    last_rejected_at: datetime | None
    status: str
    exclusion_reason: str | None
    feature_row_index: int | None


@dataclass(frozen=True, slots=True)
class QueryFrameTrace:
    """쿼리별 필터 전후 개수와 NDCG 포함 여부."""

    query_position: int
    query_id: str
    pre_filter_count: int
    kept_count: int
    metric_included: bool


@dataclass(frozen=True, slots=True)
class DetailedFeatureFrame:
    """기존 학습 배열과 재현성 평가용 정렬 정보를 함께 담는다."""

    x: np.ndarray
    y: np.ndarray
    group_sizes: list[int]
    query_ids: list[str]
    row_query_ids: list[str]
    candidate_ids: list[str]
    latent_scores: np.ndarray
    pair_traces: tuple[PairFrameTrace, ...]
    query_traces: tuple[QueryFrameTrace, ...]


def load_v2_feature_columns(config_path: Path | None = None) -> list[str]:
    """config features.v2_pair_features를 학습 열의 단일 소스로 로드한다."""

    path = config_path or DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    columns = list(config["features"]["v2_pair_features"])
    if not columns:
        raise ValueError("v2_pair_features must not be empty")
    return columns


def split_users(
    users: list[SyntheticUser], *, test_ratio: float = 0.2, seed: int = 42
) -> tuple[list[SyntheticUser], list[SyntheticUser]]:
    """train/test 사용자를 disjoint하게 나눈다(cold-start 평가)."""

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(users))
    n_test = int(len(users) * test_ratio)
    test_idx = set(int(i) for i in idx[:n_test])
    train = [u for i, u in enumerate(users) if i not in test_idx]
    test = [u for i, u in enumerate(users) if i in test_idx]
    return train, test


def embed_users(
    users: list[SyntheticUser],
    *,
    encoder: TextEncoder,
    policy: MatchingInputPolicy,
    as_of: datetime,
) -> dict[str, PreparedUserFeatures]:
    """전체 사용자를 1회 배치 임베딩해 user_id별 특성으로 반환한다."""

    prepared = [prepare_text_signals(u.snapshot, policy=policy, as_of=as_of) for u in users]
    features = encode_prepared_users(prepared, encoder=encoder, policy=policy)
    return {f.user_id: f for f in features}


def build_feature_frame_detailed(
    pairs: list[PairRecord],
    feat_by_id: dict[str, PreparedUserFeatures],
    *,
    columns: list[str],
    snapshot_by_id: dict[str, SyntheticUser],
    as_of: datetime,
    rejection_cooldown_days: int = 30,
) -> DetailedFeatureFrame:
    """페어를 쿼리별로 묶고 평가팩에 필요한 행 정렬·필터 추적을 보존한다.

    서빙과 동일한 candidate_exclusion_reason으로 제외 대상(미성년-성인 등) 페어는 뺀다.
    """

    grouped: dict[str, list[PairRecord]] = {}
    for pair in pairs:
        grouped.setdefault(pair.query_id, []).append(pair)

    rows: list[list[float]] = []
    labels: list[int] = []
    group_sizes: list[int] = []
    query_ids: list[str] = []
    row_query_ids: list[str] = []
    candidate_ids: list[str] = []
    latent_scores: list[float] = []
    pair_traces: list[PairFrameTrace] = []
    query_traces: list[QueryFrameTrace] = []
    seen_pairs: set[tuple[str, str]] = set()

    for query_position, (query_id, recs) in enumerate(grouped.items()):
        query_feat = feat_by_id.get(query_id)
        if query_feat is None:
            for candidate_position, pair in enumerate(recs):
                pair_key = (query_id, pair.cand_id)
                if pair_key in seen_pairs:
                    raise ValueError(f"duplicate pair: {query_id}/{pair.cand_id}")
                seen_pairs.add(pair_key)
                pair_traces.append(
                    PairFrameTrace(
                        query_position=query_position,
                        candidate_position=candidate_position,
                        query_id=query_id,
                        candidate_id=pair.cand_id,
                        label=int(pair.label),
                        latent_score=float(pair.latent_score),
                        common_friend_count=int(
                            pair.candidate_input.relationship.common_friend_count
                        ),
                        blocked_either_direction=bool(
                            pair.candidate_input.relationship.blocked_either_direction
                        ),
                        already_friends=bool(pair.candidate_input.relationship.already_friends),
                        last_rejected_at=pair.candidate_input.relationship.last_rejected_at,
                        status="excluded",
                        exclusion_reason="query_features_missing",
                        feature_row_index=None,
                    )
                )
            query_traces.append(
                QueryFrameTrace(
                    query_position=query_position,
                    query_id=query_id,
                    pre_filter_count=len(recs),
                    kept_count=0,
                    metric_included=False,
                )
            )
            continue
        me = snapshot_by_id[query_id].snapshot
        kept = 0
        kept_labels: list[int] = []
        for candidate_position, pair in enumerate(recs):
            cand_id = pair.cand_id
            pair_key = (query_id, cand_id)
            if pair_key in seen_pairs:
                raise ValueError(f"duplicate pair: {query_id}/{cand_id}")
            seen_pairs.add(pair_key)
            cand_feat = feat_by_id.get(cand_id)
            if cand_feat is None:
                pair_traces.append(
                    PairFrameTrace(
                        query_position=query_position,
                        candidate_position=candidate_position,
                        query_id=query_id,
                        candidate_id=cand_id,
                        label=int(pair.label),
                        latent_score=float(pair.latent_score),
                        common_friend_count=int(
                            pair.candidate_input.relationship.common_friend_count
                        ),
                        blocked_either_direction=bool(
                            pair.candidate_input.relationship.blocked_either_direction
                        ),
                        already_friends=bool(pair.candidate_input.relationship.already_friends),
                        last_rejected_at=pair.candidate_input.relationship.last_rejected_at,
                        status="excluded",
                        exclusion_reason="candidate_features_missing",
                        feature_row_index=None,
                    )
                )
                continue
            relationship = pair.candidate_input.relationship
            reason = candidate_exclusion_reason(
                me,
                pair.candidate_input.profile,
                relationship,
                as_of=as_of,
                rejection_cooldown_days=rejection_cooldown_days,
            )
            if reason is not None:
                pair_traces.append(
                    PairFrameTrace(
                        query_position=query_position,
                        candidate_position=candidate_position,
                        query_id=query_id,
                        candidate_id=cand_id,
                        label=int(pair.label),
                        latent_score=float(pair.latent_score),
                        common_friend_count=int(relationship.common_friend_count),
                        blocked_either_direction=bool(relationship.blocked_either_direction),
                        already_friends=bool(relationship.already_friends),
                        last_rejected_at=relationship.last_rejected_at,
                        status="excluded",
                        exclusion_reason=reason,
                        feature_row_index=None,
                    )
                )
                continue
            feats = build_pair_features(query_feat, cand_feat, relationship)
            feature_row_index = len(rows)
            rows.append([float(feats[col]) for col in columns])
            labels.append(pair.label)
            row_query_ids.append(query_id)
            candidate_ids.append(cand_id)
            latent_scores.append(float(pair.latent_score))
            kept_labels.append(int(pair.label))
            pair_traces.append(
                PairFrameTrace(
                    query_position=query_position,
                    candidate_position=candidate_position,
                    query_id=query_id,
                    candidate_id=cand_id,
                    label=int(pair.label),
                    latent_score=float(pair.latent_score),
                    common_friend_count=int(relationship.common_friend_count),
                    blocked_either_direction=bool(relationship.blocked_either_direction),
                    already_friends=bool(relationship.already_friends),
                    last_rejected_at=relationship.last_rejected_at,
                    status="kept",
                    exclusion_reason=None,
                    feature_row_index=feature_row_index,
                )
            )
            kept += 1
        if kept > 0:
            group_sizes.append(kept)
            query_ids.append(query_id)
        query_traces.append(
            QueryFrameTrace(
                query_position=query_position,
                query_id=query_id,
                pre_filter_count=len(recs),
                kept_count=kept,
                metric_included=kept >= 2 and max(kept_labels, default=0) > 0,
            )
        )

    x = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, len(columns)), np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return DetailedFeatureFrame(
        x=x,
        y=y,
        group_sizes=group_sizes,
        query_ids=query_ids,
        row_query_ids=row_query_ids,
        candidate_ids=candidate_ids,
        latent_scores=np.asarray(latent_scores, dtype=np.float64),
        pair_traces=tuple(pair_traces),
        query_traces=tuple(query_traces),
    )


def build_feature_frame(
    pairs: list[PairRecord],
    feat_by_id: dict[str, PreparedUserFeatures],
    *,
    columns: list[str],
    snapshot_by_id: dict[str, SyntheticUser],
    as_of: datetime,
    rejection_cooldown_days: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """기존 호출자를 위한 (X, y, group_sizes, query_ids) 호환 래퍼."""

    detailed = build_feature_frame_detailed(
        pairs,
        feat_by_id,
        columns=columns,
        snapshot_by_id=snapshot_by_id,
        as_of=as_of,
        rejection_cooldown_days=rejection_cooldown_days,
    )
    return detailed.x, detailed.y, detailed.group_sizes, detailed.query_ids


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, group_sizes: list[int], k: int) -> float:
    """그룹(쿼리)별 NDCG@k 평균. sklearn을 query 단위로 호출한다."""

    from sklearn.metrics import ndcg_score

    if isinstance(k, bool) or not isinstance(k, int | np.integer) or int(k) <= 0:
        raise ValueError("k must be a positive integer")
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred length mismatch")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("y_true and y_pred must be finite")
    groups: list[int] = []
    for size in group_sizes:
        if isinstance(size, bool) or not isinstance(size, int | np.integer) or int(size) <= 0:
            raise ValueError("group sizes must be positive integers")
        groups.append(int(size))
    if sum(groups) != len(true):
        raise ValueError("group sizes must cover every prediction row")

    scores: list[float] = []
    offset = 0
    for g in groups:
        if g >= 2:
            yt = true[offset : offset + g].reshape(1, -1)
            yp = pred[offset : offset + g].reshape(1, -1)
            if yt.max() > 0:
                scores.append(ndcg_score(yt, yp, k=k))
        offset += g
    return float(np.mean(scores)) if scores else 0.0
