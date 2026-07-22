"""P2 라벨 sanity check (SBERT 불필요, 로컬 실행용).

목적: 합성 라벨이 (a) 잠재 오라클로는 완벽 정렬되고(NDCG≈1),
(b) 관측 프록시로는 랜덤보다 낫지만 완벽하지 않은(<1) 구간에 있음을 실측한다.
이는 EXP-2 라벨 순환성이 재발하지 않았음을 임베딩 학습 전에 확인하는 값싼 게이트다.

임베딩(cosine) 특성을 포함한 본 학습·평가는 P3 Colab 노트북에서 수행한다.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_pairs_v2 import PairRecord, build_pairs  # noqa: E402
from src.data.build_profiles_v2 import (  # noqa: E402
    GenerationConfig,
    generate_population,
    load_allowed_tags,
)
from src.data.matching_input import MatchingInputPolicy  # noqa: E402


def _group(records: list[PairRecord]) -> dict[str, list[PairRecord]]:
    groups: dict[str, list[PairRecord]] = defaultdict(list)
    for r in records:
        groups[r.query_id].append(r)
    return groups


def _tag_overlap(r: PairRecord) -> float:
    q = set(r.query_snapshot.tag_ids)
    c = set(r.candidate_input.profile.tag_ids)
    return float(len(q & c))


def _common_friends(r: PairRecord) -> float:
    return float(r.candidate_input.relationship.common_friend_count)


def _mean_ndcg(groups: dict[str, list[PairRecord]], scorer, k: int) -> float:
    scores = []
    rng = np.random.default_rng(0)
    for recs in groups.values():
        if len(recs) < 2:
            continue
        y_true = np.asarray([r.label for r in recs], dtype=float).reshape(1, -1)
        if y_true.max() == 0:
            continue
        y_pred = np.asarray([scorer(r, rng) for r in recs], dtype=float).reshape(1, -1)
        scores.append(ndcg_score(y_true, y_pred, k=k))
    return float(np.mean(scores)) if scores else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--n-queries", type=int, default=300)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    tags = load_allowed_tags()
    policy = MatchingInputPolicy(allowed_tag_ids=frozenset(tags))
    users = generate_population(GenerationConfig(n_users=args.n_users, seed=42), policy=policy)
    pairs = build_pairs(users, n_queries=args.n_queries, n_candidates=args.candidates, seed=42)
    groups = _group(pairs)

    labels = np.asarray([p.label for p in pairs])
    dist = {int(v): int((labels == v).sum()) for v in range(4)}

    oracle = _mean_ndcg(groups, lambda r, _rng: r.latent_score, args.k)
    tag_only = _mean_ndcg(groups, lambda r, _rng: _tag_overlap(r), args.k)
    friends_only = _mean_ndcg(groups, lambda r, _rng: _common_friends(r), args.k)
    combo = _mean_ndcg(groups, lambda r, _rng: _tag_overlap(r) + 0.5 * _common_friends(r), args.k)
    random_ndcg = _mean_ndcg(groups, lambda _r, rng: rng.random(), args.k)

    print(f"pairs={len(pairs)} queries={len(groups)} label_dist={dist}")
    print(f"NDCG@{args.k}:")
    print(f"  oracle (latent_score)      = {oracle:.4f}")
    print(f"  observed tag_overlap only  = {tag_only:.4f}")
    print(f"  observed common_friends    = {friends_only:.4f}")
    print(f"  observed tag+friends combo = {combo:.4f}")
    print(f"  random baseline            = {random_ndcg:.4f}")

    ok = oracle > 0.98 and random_ndcg < combo < 0.98 and combo > random_ndcg
    print(f"SANITY: {'PASS' if ok else 'REVIEW'} " f"(oracle≈1, random < observed < 1 기대)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
