"""모듈② 매칭의 Demographic Parity(장애유형별) 측정.

project_facts 게이트: 모듈② Demographic Parity diff ≤ 0.10.

배경: 호환성 ground-truth 라벨은 region/age/interest 규칙으로만 정해지고 **장애유형에는
의존하지 않는다**(build_pairs). 그러나 임베딩 모델은 `f_dis_match`(query·cand 장애 일치)를
피처로 쓰므로, 모델이 '같은 장애끼리'를 더/덜 추천하는 집단 편향이 생길 수 있다.

측정: 각 query의 후보를 모델 점수로 정렬해 **top-K 추천**을 만들고, 후보의 장애유형 g별로
  selection_rate[g] = (top-K에 든 g 후보 수) / (후보 슬레이트의 g 후보 총수)
를 구한다. **DP diff = max_g − min_g**. (선택률 격차가 작을수록 공정.)

mock 프로필 기반(실 사용자 데이터 없음)이라 결과는 '합성 타당성' 수준으로 해석한다.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_pairs import (  # noqa: E402
    build_pairs,
    engineer_features,
    generate_mock_profiles,
)
from src.utils.tracking import log_metrics, mlflow_run  # noqa: E402

DEFAULT_K = 10
DEFAULT_SEED = 42


def evaluate_dp(
    model_path: Path, out_path: Path, k: int = DEFAULT_K, seed: int = DEFAULT_SEED
) -> dict:
    print(f"=== [모듈②] Demographic Parity(장애유형) 측정 (top-K={k}) ===")
    with model_path.open("rb") as f:
        model = pickle.load(f)

    profiles = generate_mock_profiles(500)
    pairs = build_pairs(profiles, n_queries=100, n_candidates=20, seed=seed)
    X, _ = engineer_features(pairs, mode="embedding")  # noqa: N806
    pairs = pairs.copy()
    pairs["score"] = model.predict(X)

    # 후보 장애유형 결합
    dis_by_user = profiles.set_index("user_id")["disability_type"].to_dict()
    pairs["cand_disability"] = pairs["cand_id"].map(dis_by_user)

    # query별 top-K 추천 마킹
    pairs["rank"] = pairs.groupby("query_id")["score"].rank(ascending=False, method="first")
    pairs["in_topk"] = (pairs["rank"] <= k).astype(int)

    # 장애유형별 선택률 = top-K 진입 / 후보 등장
    grp = pairs.groupby("cand_disability")["in_topk"].agg(["sum", "count"])
    grp["selection_rate"] = grp["sum"] / grp["count"]
    rates = grp["selection_rate"].to_dict()
    dp_diff = float(grp["selection_rate"].max() - grp["selection_rate"].min())

    # 같은-장애 favoritism 보조 지표
    same = pairs[pairs["dis_match"] == 1]["in_topk"].mean()
    diff = pairs[pairs["dis_match"] == 0]["in_topk"].mean()

    result = {
        "k": k,
        "n_queries": int(pairs["query_id"].nunique()),
        "n_pairs": int(len(pairs)),
        "selection_rate_by_disability": {g: round(float(r), 4) for g, r in rates.items()},
        "demographic_parity_difference": round(dp_diff, 4),
        "same_disability_topk_rate": round(float(same), 4),
        "diff_disability_topk_rate": round(float(diff), 4),
        "gate_dp_le_0.10": bool(dp_diff <= 0.10),
        "note": "mock 프로필 기반 — 실 사용자 데이터 없음(합성 타당성).",
    }

    print(f"  장애유형별 선택률: {result['selection_rate_by_disability']}")
    print(f"  DP diff = {dp_diff:.4f}  (게이트 ≤0.10: {'통과' if dp_diff <= 0.10 else '미달'})")
    print(f"  같은장애 top-K율 {same:.4f} vs 다른장애 {diff:.4f}")

    with mlflow_run("thisabled-module2", run_name=f"dp-fairness/k{k}"):
        log_metrics(
            {
                "module2_dp_diff": dp_diff,
                "module2_same_dis_topk": float(same),
                "module2_diff_dis_topk": float(diff),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✔ 저장: {out_path.relative_to(ROOT)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(ROOT / "models" / "checkpoints" / "module2_lambdamart_embedding.pkl"),
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(ROOT / "reports" / "validation_reports" / "module2" / "fairness_dp.json"),
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()
    evaluate_dp(Path(args.model_path), Path(args.out_path), k=args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
