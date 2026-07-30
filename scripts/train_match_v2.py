"""MATCH 모듈② v2 LambdaMART 재학습 오케스트레이터 (P3).

노트북(notebooks/module2_retrain_v2_colab.ipynb)은 `run(encoder=SentenceTransformer(...))`로
이 함수를 호출한다. 로컬 스모크는 fake encoder로 같은 함수를 호출해 학습·평가 경로를 검증한다.

산출물(out_dir/match_v2_run_*): module2_lambdamart_v2.pkl, metrics_v2.json,
match_v2_eval_pack/. 완성된 버전만 out_dir/match_v2_current.json이 가리킨다.
Colab에서 내려받아 CPU 환경의 단계별 재현성을 검증한다.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import tempfile
from dataclasses import asdict
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
from src.data.match_eval_pack import (  # noqa: E402
    RUN_MANIFEST_SCHEMA,
    RUN_POINTER_SCHEMA,
    CapturingEncoder,
    sha256_file,
    write_match_eval_pack,
)
from src.data.matching_input import (  # noqa: E402
    MatchingInputPolicy,
    TextEncoder,
    prepare_text_signals,
)
from src.data.matching_trainset import (  # noqa: E402
    build_feature_frame,
    build_feature_frame_detailed,
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


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    data = _canonical_json_bytes(payload)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _publish_current_pointer(path: Path, payload: dict[str, Any]) -> None:
    existing = path.exists() or path.is_symlink()
    if existing:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("MATCH current pointer exists but is not a regular file")
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("MATCH current pointer is not valid JSON") from exc
        if not isinstance(previous, dict) or previous.get("schema_version") != RUN_POINTER_SCHEMA:
            raise RuntimeError("refusing to replace an unrecognized MATCH current pointer")

    data = _canonical_json_bytes(payload)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if existing:
            os.replace(temp_path, path)
            temp_path = None
        else:
            os.link(temp_path, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


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
    write_eval_pack: bool = True,
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
    generation_config = GenerationConfig(n_users=n_users, seed=seed)
    as_of = generation_config.as_of

    # 프레임은 v2 열 + legacy ablation 열(f_cosine/f_l2)의 상위집합으로 만든다.
    # 각 모델은 자기 부분집합만 선택한다(저장 모델은 v2 columns만 사용).
    frame_columns = list(dict.fromkeys([*columns, *LEGACY_COLUMNS]))

    users = generate_population(generation_config, policy=policy)
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
    test_capture = CapturingEncoder(encoder) if write_eval_pack else None
    test_feat = embed_users(
        test_users,
        encoder=test_capture or encoder,
        policy=policy,
        as_of=as_of,
    )
    train_snap = {u.snapshot.user_id: u for u in train_users}
    test_snap = {u.snapshot.user_id: u for u in test_users}

    x_tr, y_tr, g_tr, _ = build_feature_frame(
        train_pairs, train_feat, columns=frame_columns, snapshot_by_id=train_snap, as_of=as_of
    )
    test_frame = build_feature_frame_detailed(
        test_pairs, test_feat, columns=frame_columns, snapshot_by_id=test_snap, as_of=as_of
    )
    x_te, y_te, g_te = test_frame.x, test_frame.y, test_frame.group_sizes
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

    run_dir = Path(tempfile.mkdtemp(dir=out_dir, prefix="match_v2_run_"))
    model_path = run_dir / "module2_lambdamart_v2.pkl"
    metrics_path = run_dir / "metrics_v2.json"
    eval_pack_path = run_dir / "match_v2_eval_pack"
    run_manifest_path = run_dir / "run_manifest.json"
    current_pointer_path = out_dir / "match_v2_current.json"

    with model_path.open("wb") as handle:
        pickle.dump(
            {"model": booster, "columns": columns, "params": params, "metrics": result["metrics"]},
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    if write_eval_pack:
        if (
            test_capture is None
            or test_capture.sentences is None
            or test_capture.embeddings is None
        ):
            raise RuntimeError("test encoder inputs were not captured")
        prepared_test_signals = tuple(
            prepare_text_signals(user.snapshot, policy=policy, as_of=as_of) for user in test_users
        )
        model_col_idx = [frame_columns.index(column) for column in columns]
        model_x_test = x_te[:, model_col_idx]
        raw_scores = np.asarray(booster.predict(model_x_test), dtype=np.float64)
        eval_pack_manifest = write_match_eval_pack(
            eval_pack_path,
            model_path=model_path,
            columns=columns,
            detailed_frame=test_frame,
            x_test=model_x_test,
            raw_scores=raw_scores,
            test_users=test_users,
            prepared_signals=prepared_test_signals,
            captured_sentences=test_capture.sentences,
            text_embeddings=test_capture.embeddings,
            user_features=test_feat,
            metrics=metrics,
            policy=policy,
            encoder=encoder,
            run_metadata={
                "seed": seed,
                "n_users": n_users,
                "n_train_queries": n_train_queries,
                "n_test_queries": n_test_queries,
                "n_candidates": n_candidates,
                "test_ratio": 0.2,
                "as_of": as_of,
                "generation_config": asdict(generation_config),
                "label_weights": asdict(weights),
                "lgbm_params": params,
            },
            config_path=config_path,
        )
        result["eval_pack_path"] = str(eval_pack_path)
        result["eval_pack_sha256"] = eval_pack_manifest["pack_content_sha256"]

    model_sha256 = sha256_file(model_path)
    result["run_dir"] = str(run_dir)
    result["model_path"] = str(model_path)
    result["model_sha256"] = model_sha256
    result["metrics_path"] = str(metrics_path)
    result["run_manifest_path"] = str(run_manifest_path)
    result["current_pointer_path"] = str(current_pointer_path)
    _write_new_json(metrics_path, result)
    metrics_sha256 = sha256_file(metrics_path)
    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": "complete",
        "model": {
            "path": model_path.name,
            "sha256": model_sha256,
        },
        "metrics": {
            "path": metrics_path.name,
            "sha256": metrics_sha256,
        },
        "evaluation_pack": (
            {
                "path": eval_pack_path.name,
                "sha256": result["eval_pack_sha256"],
            }
            if write_eval_pack
            else None
        ),
    }
    _write_new_json(run_manifest_path, run_manifest)
    run_manifest_sha256 = sha256_file(run_manifest_path)
    relative_run = Path(run_dir.name)
    _publish_current_pointer(
        current_pointer_path,
        {
            "schema_version": RUN_POINTER_SCHEMA,
            "run_dir": relative_run.as_posix(),
            "run_manifest": {
                "path": (relative_run / run_manifest_path.name).as_posix(),
                "sha256": run_manifest_sha256,
            },
            "model": {
                "path": (relative_run / model_path.name).as_posix(),
                "sha256": model_sha256,
            },
            "metrics": {
                "path": (relative_run / metrics_path.name).as_posix(),
                "sha256": metrics_sha256,
            },
            "evaluation_pack": (
                {
                    "path": (relative_run / eval_pack_path.name).as_posix(),
                    "sha256": result["eval_pack_sha256"],
                }
                if write_eval_pack
                else None
            ),
        },
    )
    return result


def _build_sbert(config_path: Path):
    from sentence_transformers import SentenceTransformer

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    name = config["embedding"]["sbert_model_name"]
    model = SentenceTransformer(name)
    first_module = model._first_module()
    resolved_revision = getattr(getattr(first_module, "auto_model", None), "config", None)
    resolved_revision = getattr(resolved_revision, "_commit_hash", None)
    if (
        not isinstance(resolved_revision, str)
        or re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", resolved_revision) is None
    ):
        raise RuntimeError("SBERT immutable resolved revision could not be determined")
    resolved_revision = resolved_revision.lower()
    try:
        parameter_dtype = str(next(model.parameters()).dtype)
    except StopIteration:
        parameter_dtype = None

    class _Adapter:
        def encode(self, sentences, *, batch_size, show_progress_bar):
            return np.asarray(
                model.encode(sentences, batch_size=batch_size, show_progress_bar=show_progress_bar),
                dtype=np.float32,
            )

        def reproducibility_metadata(self):
            return {
                "model_name": name,
                "resolved_revision": resolved_revision,
                "device": str(model.device),
                "parameter_dtype": parameter_dtype,
                "configured_max_seq_length": config["embedding"].get("max_seq_length"),
                "actual_max_seq_length": int(model.max_seq_length),
                "pooling": config["embedding"].get("pooling"),
                "normalize_embeddings": False,
            }

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
    parser.add_argument("--no-eval-pack", action="store_true")
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
        write_eval_pack=not args.no_eval_pack,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"saved: {result['model_path']}")
    print(f"metrics: {result['metrics_path']}")
    print(f"current: {result['current_pointer_path']}")
    if "eval_pack_path" in result:
        print(f"eval pack: {result['eval_pack_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
