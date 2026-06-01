"""모듈 ② LambdaMART 기반 사용자 호환성 랭킹.

[EXP-2, 2026-06-02] Data leakage 차단:
- user-level split (train_users 80% / test_users 20%, disjoint)
- feature mode "embedding" — label 결정 변수(region_match/age_diff/overlap) 제외
- train pairs로 학습 → test pairs(cold-start)로 NDCG@5, NDCG@10 측정

비교를 위해 mode="full"도 측정해 sanity check (NDCG ≈ 1.0이어야 정상).

Usage:
    python scripts/train_lambdarank.py
    python scripts/train_lambdarank.py --mode full
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_pairs import (  # noqa: E402
    build_pairs,
    engineer_features,
    generate_mock_profiles,
    split_users,
)


def _ndcg_per_query(y_true, y_pred, groups, k):
    """그룹별 NDCG@k 계산 후 평균. sklearn ndcg_score를 query 단위로 호출."""
    scores = []
    offset = 0
    for g in groups:
        if g < 2:
            offset += g
            continue
        yt = np.asarray(y_true[offset : offset + g]).reshape(1, -1)
        yp = np.asarray(y_pred[offset : offset + g]).reshape(1, -1)
        try:
            scores.append(ndcg_score(yt, yp, k=k))
        except ValueError:
            pass  # 모두 0 라벨이면 skip
        offset += g
    return float(np.mean(scores)) if scores else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["embedding", "full"],
        default="embedding",
        help="embedding (EXP-2 권장, leakage 차단) / full (sanity check)",
    )
    parser.add_argument("--n-profiles", type=int, default=1000)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--train-queries", type=int, default=160)
    parser.add_argument("--test-queries", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=15)
    args = parser.parse_args()

    print(f"=== [모듈 ②] LambdaMART (mode={args.mode}) ===")

    print(f"1. 프로필 {args.n_profiles}건 생성 + user-level split")
    df_profiles = generate_mock_profiles(args.n_profiles)
    train_profiles, test_profiles = split_users(df_profiles, test_ratio=args.test_ratio)
    print(f"   train_users={len(train_profiles)}, test_users={len(test_profiles)} (disjoint)")

    print("2. train/test 페어 별도 구성 (cold-start)")
    train_pairs = build_pairs(
        train_profiles, n_queries=args.train_queries, n_candidates=args.candidates, seed=42
    )
    test_pairs = build_pairs(
        test_profiles, n_queries=args.test_queries, n_candidates=args.candidates, seed=43
    )
    print(f"   train pairs={len(train_pairs)}, test pairs={len(test_pairs)}")

    print(f"3. 피처 엔지니어링 (mode={args.mode})")
    X_train, g_train = engineer_features(train_pairs, mode=args.mode)  # noqa: N806
    X_test, g_test = engineer_features(test_pairs, mode=args.mode)  # noqa: N806
    y_train = train_pairs["label"].values
    y_test = test_pairs["label"].values
    print(f"   features: {list(X_train.columns)} ({X_train.shape[1]}-dim)")

    print("4. LightGBM LambdaMART 훈련")
    train_data = lgb.Dataset(X_train, label=y_train, group=g_train)
    test_data = lgb.Dataset(X_test, label=y_test, group=g_test, reference=train_data)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 10,
        "seed": 42,
        "verbosity": -1,
    }
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        valid_names=["test"],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(10)],
    )

    print("5. Cold-start test 평가")
    y_pred = model.predict(X_test)
    metrics = {
        "mode": args.mode,
        "ndcg@5": _ndcg_per_query(y_test, y_pred, g_test, 5),
        "ndcg@10": _ndcg_per_query(y_test, y_pred, g_test, 10),
        "n_train_users": len(train_profiles),
        "n_test_users": len(test_profiles),
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "n_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
    }
    print(f"\n=== Cold-start Test NDCG (mode={args.mode}) ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out_dir = ROOT / "models" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"module2_lambdamart_{args.mode}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    metrics_dir = ROOT / "reports" / "validation_reports" / "module2"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"ranker_{args.mode}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\n✓ 모델: {out_path.relative_to(ROOT)}")
    print(f"✓ 메트릭: {metrics_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
