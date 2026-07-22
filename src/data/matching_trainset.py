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


def build_feature_frame(
    pairs: list[PairRecord],
    feat_by_id: dict[str, PreparedUserFeatures],
    *,
    columns: list[str],
    snapshot_by_id: dict[str, SyntheticUser],
    as_of: datetime,
    rejection_cooldown_days: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """페어를 쿼리별로 묶어 (X, y, group_sizes, query_ids)를 만든다.

    서빙과 동일한 candidate_exclusion_reason으로 제외 대상(미성년-성인 등) 페어는 뺀다.
    """

    grouped: dict[str, list[PairRecord]] = {}
    for pair in pairs:
        grouped.setdefault(pair.query_id, []).append(pair)

    rows: list[list[float]] = []
    labels: list[int] = []
    group_sizes: list[int] = []
    query_ids: list[str] = []

    for query_id, recs in grouped.items():
        query_feat = feat_by_id.get(query_id)
        if query_feat is None:
            continue
        me = snapshot_by_id[query_id].snapshot
        kept = 0
        for pair in recs:
            cand_id = pair.cand_id
            cand_feat = feat_by_id.get(cand_id)
            if cand_feat is None:
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
                continue
            feats = build_pair_features(query_feat, cand_feat, relationship)
            rows.append([float(feats[col]) for col in columns])
            labels.append(pair.label)
            kept += 1
        if kept > 0:
            group_sizes.append(kept)
            query_ids.append(query_id)

    x = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, len(columns)), np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return x, y, group_sizes, query_ids


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, group_sizes: list[int], k: int) -> float:
    """그룹(쿼리)별 NDCG@k 평균. sklearn을 query 단위로 호출한다."""

    from sklearn.metrics import ndcg_score

    scores: list[float] = []
    offset = 0
    for g in group_sizes:
        if g >= 2:
            yt = y_true[offset : offset + g].reshape(1, -1)
            yp = y_pred[offset : offset + g].reshape(1, -1)
            if yt.max() > 0:
                scores.append(ndcg_score(yt, yp, k=k))
        offset += g
    return float(np.mean(scores)) if scores else 0.0
