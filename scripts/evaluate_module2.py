"""D4: 모듈 ② LambdaMART 최종 성능(NDCG@K) 평가 스크립트."""

import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_pairs import build_pairs, engineer_features, generate_mock_profiles


def evaluate_module2(model_path: Path) -> None:
    print("=== [모듈 ②] LambdaMART 평가 시작 ===")

    if not model_path.exists():
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return

    print("1. 모델 로드...")
    with open(model_path, "rb") as f:
        model: lgb.Booster = pickle.load(f)

    # 평가용 새로운 Mock 프로필 및 페어 생성 (훈련과 다른 시드 적용)
    np.random.seed(999)
    print("2. 평가용 Test 셋 생성...")
    df_profiles = generate_mock_profiles(500)
    df_pairs = build_pairs(df_profiles, n_queries=100, n_candidates=20)

    print(f"3. 피처 엔지니어링 (총 {len(df_pairs)} 페어)...")
    X_test, groups_test = engineer_features(df_pairs)
    y_test = df_pairs["label"].values

    print("4. 예측 진행...")
    y_pred = model.predict(X_test)
    df_pairs["pred"] = y_pred

    print("5. NDCG@K 메트릭 계산...")
    # 쿼리별 NDCG 계산
    ndcg_1, ndcg_5, ndcg_10 = [], [], []

    for q_id, group in df_pairs.groupby("query_id"):
        true_rel = np.asarray([group["label"].values])
        pred_rel = np.asarray([group["pred"].values])

        # relevance 합이 0이면 (모든 후보가 관련 없음) 계산 불가 -> 무시
        if np.sum(true_rel) == 0:
            continue

        ndcg_1.append(ndcg_score(true_rel, pred_rel, k=1))
        ndcg_5.append(ndcg_score(true_rel, pred_rel, k=5))
        ndcg_10.append(ndcg_score(true_rel, pred_rel, k=10))

    final_ndcg_1 = np.mean(ndcg_1)
    final_ndcg_5 = np.mean(ndcg_5)
    final_ndcg_10 = np.mean(ndcg_10)

    print("\n[평가 결과]")
    print(f"  - NDCG@1  : {final_ndcg_1:.4f}")
    print(f"  - NDCG@5  : {final_ndcg_5:.4f}")
    print(f"  - NDCG@10 : {final_ndcg_10:.4f}")

    print("\n✔ 평가 완료.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(ROOT / "models" / "checkpoints" / "module2_lambdamart.pkl"),
    )
    args = parser.parse_args()

    evaluate_module2(Path(args.model_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
