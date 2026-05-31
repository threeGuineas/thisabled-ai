"""D4: LambdaMART 기반 사용자 호환성 랭킹 예측.

주어진 모듈 ② 매칭 스키마 명세서(docs/matching_schema.md)를 바탕으로
사용자 프로필 더미 데이터를 생성, 쿼리-후보 페어를 구성하고 SBERT 임베딩 및
메타데이터 피처 엔지니어링 후 LightGBM Ranker를 훈련합니다.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.data.build_pairs import build_pairs, engineer_features, generate_mock_profiles


def main() -> int:
    print("=== [모듈 ②] 사용자 호환성 랭킹 LambdaMART 훈련 시작 ===")

    print("1. 사용자 프로필(Dummy) 1,000건 생성")
    df_profiles = generate_mock_profiles(1000)

    print("2. Query-Candidate 랭킹 페어 구성 및 Relevance 합성")
    df_pairs = build_pairs(df_profiles, n_queries=200, n_candidates=15)

    print(f"3. 피처 엔지니어링 시작 (총 {len(df_pairs)} 페어)")
    features, groups = engineer_features(df_pairs)
    labels = df_pairs["label"].values

    print("4. LightGBM LambdaMART 모델 훈련")
    # 예제 단순화를 위해 Train 전체 사용 (실제론 분리 필요)
    train_data = lgb.Dataset(features, label=labels, group=groups)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 10,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
    )

    out_dir = ROOT / "models" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "module2_lambdamart.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✔ LambdaMART 랭커 저장 완료: {out_path.relative_to(ROOT)}")
    print("=== D4 파이프라인 완성! ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
