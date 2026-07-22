"""MATCH 모듈② v2 LambdaMART 재학습 오케스트레이터 (P3).

노트북(notebooks/module2_retrain_v2_colab.ipynb)은 `run(encoder=SentenceTransformer(...))`로
이 함수를 호출한다. 로컬 스모크는 fake encoder로 같은 함수를 호출해 학습·평가 경로를 검증한다.

산출물(out_dir): module2_lambdamart_v2.pkl {model, columns, params, metrics},
metrics_v2.json. Colab에서 zip으로 묶어 내려받아 로컬 P4/P5에서 검증한다.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_pairs_v2 import LabelWeights, build_pairs  # noqa: E402
from src.data.build_profiles_v2 import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    GenerationConfig,
    generate_population,
    load_allowed_tags,
)
from src.data.matching_input import MatchingInputPolicy, TextEncoder  # noqa: E402
from src.data.matching_trainset import (  # noqa: E402
    build_feature_frame,
    embed_users,
    load_v2_feature_columns,
    ndcg_at_k,
    split_users,
)

# 콘텐츠 신호 특성(ablation ③에서 제외 대상)
CONTENT_COLUMNS = frozenset(
    {
        "f_authored_cosine",
        "f_authored_available",
        "f_liked_cosine",
        "f_liked_available",
        "f_liked_authored_cosine",
    }
)
LEGACY_COLUMNS = ["f_cosine", "f_l2"]  # 구형 3열 중 임베딩 기반 2열(f_dis_match는 v2에 없음)


def _load_lgbm_params(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return dict(config["ranker"]["lgbm_params"])


def _train_and_eval(
    frames: dict[str, Any], columns: list[str], params: dict[str, Any]
) -> tuple[lgb.Booster, dict[str, float]]:
    col_idx = [frames["columns"].index(c) for c in columns]
    x_tr = frames["x_train"][:, col_idx]
    x_te = frames["x_test"][:, col_idx]
    train_ds = lgb.Dataset(x_tr, label=frames["y_train"], group=frames["g_train"])
    n_estimators = int(params.get("n_estimators", 300))
    train_params = {k: v for k, v in params.items() if k != "n_estimators"}
    train_params.setdefault("verbose", -1)
    booster = lgb.train(train_params, train_ds, num_boost_round=n_estimators)
    pred_te = booster.predict(x_te)
    metrics = {
        "ndcg@5": ndcg_at_k(frames["y_test"], pred_te, frames["g_test"], 5),
        "ndcg@10": ndcg_at_k(frames["y_test"], pred_te, frames["g_test"], 10),
        "n_features": len(columns),
    }
    return booster, metrics


def run(
    *,
    encoder: TextEncoder,
    n_users: int = 10_000,
    n_train_queries: int = 4_000,
    n_test_queries: int = 1_000,
    n_candidates: int = 20,
    seed: int = 42,
    out_dir: Path | None = None,
    config_path: Path | None = None,
    ablations: bool = True,
    n_estimators: int | None = None,
) -> dict[str, Any]:
    config_path = config_path or DEFAULT_CONFIG_PATH
    out_dir = out_dir or (ROOT / "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = load_allowed_tags(config_path)
    policy = MatchingInputPolicy(allowed_tag_ids=frozenset(tags))
    columns = load_v2_feature_columns(config_path)
    params = _load_lgbm_params(config_path)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    as_of = GenerationConfig().as_of

    # 프레임은 v2 열 + legacy ablation 열(f_cosine/f_l2)의 상위집합으로 만든다.
    # 각 모델은 자기 부분집합만 선택한다(저장 모델은 v2 columns만 사용).
    frame_columns = list(dict.fromkeys([*columns, *LEGACY_COLUMNS]))

    users = generate_population(GenerationConfig(n_users=n_users, seed=seed), policy=policy)
    train_users, test_users = split_users(users, test_ratio=0.2, seed=seed)

    weights = LabelWeights()
    train_pairs = build_pairs(
        train_users,
        n_queries=n_train_queries,
        n_candidates=n_candidates,
        weights=weights,
        seed=seed,
    )
    test_pairs = build_pairs(
        test_users,
        n_queries=n_test_queries,
        n_candidates=n_candidates,
        weights=weights,
        seed=seed + 1,
    )

    train_feat = embed_users(train_users, encoder=encoder, policy=policy, as_of=as_of)
    test_feat = embed_users(test_users, encoder=encoder, policy=policy, as_of=as_of)
    train_snap = {u.snapshot.user_id: u for u in train_users}
    test_snap = {u.snapshot.user_id: u for u in test_users}

    x_tr, y_tr, g_tr, _ = build_feature_frame(
        train_pairs, train_feat, columns=frame_columns, snapshot_by_id=train_snap, as_of=as_of
    )
    x_te, y_te, g_te, _ = build_feature_frame(
        test_pairs, test_feat, columns=frame_columns, snapshot_by_id=test_snap, as_of=as_of
    )
    frames = {
        "columns": frame_columns,
        "x_train": x_tr,
        "y_train": y_tr,
        "g_train": g_tr,
        "x_test": x_te,
        "y_test": y_te,
        "g_test": g_te,
    }

    booster, metrics = _train_and_eval(frames, columns, params)
    result: dict[str, Any] = {
        "columns": columns,
        "params": params,
        "seed": seed,
        "n_users": n_users,
        "train_pairs": int(len(y_tr)),
        "test_pairs": int(len(y_te)),
        "metrics": {"v2_full": metrics},
        "gain_importance": dict(
            sorted(
                zip(
                    columns,
                    booster.feature_importance(importance_type="gain").tolist(),
                    strict=True,
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }

    if ablations:
        v2_no_content = [c for c in columns if c not in CONTENT_COLUMNS]
        legacy = [c for c in LEGACY_COLUMNS if c in frames["columns"]]
        _, m_no_content = _train_and_eval(frames, v2_no_content, params)
        _, m_legacy = _train_and_eval(frames, legacy, params)
        result["metrics"]["v2_no_content"] = m_no_content
        result["metrics"]["legacy_embedding"] = m_legacy

    model_path = out_dir / "module2_lambdamart_v2.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {"model": booster, "columns": columns, "params": params, "metrics": result["metrics"]},
            handle,
        )
    (out_dir / "metrics_v2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["model_path"] = str(model_path)
    return result


def _build_sbert(config_path: Path):
    from sentence_transformers import SentenceTransformer

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    name = config["embedding"]["sbert_model_name"]
    model = SentenceTransformer(name)

    class _Adapter:
        def encode(self, sentences, *, batch_size, show_progress_bar):
            return np.asarray(
                model.encode(sentences, batch_size=batch_size, show_progress_bar=show_progress_bar),
                dtype=np.float32,
            )

    return _Adapter()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=10_000)
    parser.add_argument("--train-queries", type=int, default=4_000)
    parser.add_argument("--test-queries", type=int, default=1_000)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--no-ablations", action="store_true")
    args = parser.parse_args()

    encoder = _build_sbert(DEFAULT_CONFIG_PATH)
    result = run(
        encoder=encoder,
        n_users=args.n_users,
        n_train_queries=args.train_queries,
        n_test_queries=args.test_queries,
        n_candidates=args.candidates,
        seed=args.seed,
        out_dir=args.out_dir,
        ablations=not args.no_ablations,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"saved: {result['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
