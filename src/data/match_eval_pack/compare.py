"""MATCH v2 평가팩 단계 비교, 고정 모델 재평가, target encoder 재생성."""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.data.build_pairs_v2 import PairRecord
from src.data.build_profiles_v2 import DEFAULT_CORPUS_PATH, LatentUser, SyntheticUser
from src.data.match_eval_pack.format import (
    EXACT_STAGE_ORDER,
    MAX_MODEL_BYTES,
    CapturingEncoder,
    EvalPackError,
    EvalPackIntegrityError,
    canonical_float_array,
    write_match_eval_pack,
)
from src.data.match_eval_pack.validation import (
    LoadedMatchEvalPack,
    _read_bounded_regular_file,
    load_match_eval_pack,
    validate_sha256,
)
from src.data.matching_input import (
    CandidateInput,
    CandidateRelationship,
    ContentSignal,
    MatchingInputPolicy,
    TextEncoder,
    UserSnapshot,
    prepare_text_signals,
)
from src.data.matching_trainset import build_feature_frame_detailed, embed_users, ndcg_at_k


def _validate_tolerances(*values: float) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("comparison tolerances must be finite and non-negative")


def _read_trusted_model_payload(
    model_path: Path,
    *,
    expected_model_sha256: str,
) -> tuple[bytes, str]:
    expected = validate_sha256(
        expected_model_sha256,
        field_name="expected_model_sha256",
    )
    model_path = Path(model_path)
    payload = _read_bounded_regular_file(
        model_path,
        max_bytes=MAX_MODEL_BYTES,
    )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise EvalPackIntegrityError("model SHA-256 differs from external trust anchor")
    return payload, actual


def _deserialize_model_bundle(payload: bytes) -> dict[str, Any]:
    try:
        bundle = pickle.loads(payload)
    except Exception as exc:
        raise EvalPackIntegrityError("trusted model bundle cannot be deserialized") from exc
    if not isinstance(bundle, dict) or "model" not in bundle or "columns" not in bundle:
        raise EvalPackIntegrityError("model bundle must contain model and columns")
    columns = bundle["columns"]
    if (
        not isinstance(columns, list)
        or not columns
        or len(columns) != len(set(columns))
        or any(not isinstance(column, str) or not column for column in columns)
    ):
        raise EvalPackIntegrityError("model bundle columns are invalid")
    return bundle


def load_trusted_model_bundle(
    model_path: Path,
    *,
    expected_model_sha256: str,
) -> dict[str, Any]:
    """별도로 고정한 SHA와 일치하는 bounded bytes만 모델 bundle로 역직렬화한다."""

    payload, _actual = _read_trusted_model_payload(
        model_path,
        expected_model_sha256=expected_model_sha256,
    )
    return _deserialize_model_bundle(payload)


def _load_trusted_bundle(
    model_path: Path,
    *,
    expected_model_sha256: str,
    pack_model_sha256: str,
) -> dict[str, Any]:
    """외부 trust anchor와 pack 좌표가 모두 일치한 bounded bytes만 역직렬화한다."""

    payload, actual = _read_trusted_model_payload(
        model_path,
        expected_model_sha256=expected_model_sha256,
    )
    if actual != pack_model_sha256:
        raise EvalPackIntegrityError("model SHA-256 differs from evaluation pack")
    return _deserialize_model_bundle(payload)


def _metrics_for(pack: LoadedMatchEvalPack, scores: np.ndarray) -> dict[str, float]:
    return {
        "ndcg@5": ndcg_at_k(
            pack.frame["y_test"],
            scores,
            pack.frame["group_sizes"].tolist(),
            5,
        ),
        "ndcg@10": ndcg_at_k(
            pack.frame["y_test"],
            scores,
            pack.frame["group_sizes"].tolist(),
            10,
        ),
    }


def _runtime_metric_deltas(pack: LoadedMatchEvalPack) -> dict[str, float]:
    current = _metrics_for(pack, pack.frame["raw_scores"])
    stored = pack.manifest["evaluation"]["metrics"]
    return {key: float(current[key] - stored[key]) for key in current}


def _embedding_drift_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    cosine_chunks: list[np.ndarray] = []
    max_abs = 0.0
    for start in range(0, len(reference), 4096):
        stop = min(start + 4096, len(reference))
        left = np.asarray(reference[start:stop], dtype=np.float64)
        right = np.asarray(candidate[start:stop], dtype=np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(left - right), initial=0.0)))
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        valid = (left_norm > 0) & (right_norm > 0)
        cosine = np.ones(len(left), dtype=np.float64)
        cosine[valid] = np.sum(left[valid] * right[valid], axis=1) / (
            left_norm[valid] * right_norm[valid]
        )
        cosine_chunks.append(cosine)
    values = np.concatenate(cosine_chunks) if cosine_chunks else np.ones(0, dtype=np.float64)
    return {
        "max_abs": max_abs,
        "cosine_min": float(np.min(values, initial=1.0)),
        "cosine_p001": float(np.quantile(values, 0.001)) if len(values) else 1.0,
    }


def _provenance_difference_keys(
    reference: LoadedMatchEvalPack,
    candidate: LoadedMatchEvalPack,
) -> list[str]:
    ref_provenance = reference.manifest["provenance"]
    cand_provenance = candidate.manifest["provenance"]
    return sorted(
        key
        for key in set(ref_provenance) | set(cand_provenance)
        if ref_provenance.get(key) != cand_provenance.get(key)
    )


def compare_match_eval_packs(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    expected_reference_sha256: str,
    expected_candidate_sha256: str,
    score_atol: float = 1e-10,
    metric_atol: float = 1e-6,
) -> dict[str, Any]:
    """두 pack의 최초 불일치와 현재 runtime metric 재계산 차이를 보고한다."""

    _validate_tolerances(score_atol, metric_atol)
    reference = load_match_eval_pack(
        reference_dir,
        expected_pack_sha256=expected_reference_sha256,
    )
    candidate = load_match_eval_pack(
        candidate_dir,
        expected_pack_sha256=expected_candidate_sha256,
    )
    ref_stages = reference.manifest["stages"]
    cand_stages = candidate.manifest["stages"]
    provenance_differences = _provenance_difference_keys(reference, candidate)
    for stage in EXACT_STAGE_ORDER:
        if ref_stages[stage] != cand_stages[stage]:
            details: dict[str, Any] = {
                "reference_sha256": ref_stages[stage],
                "candidate_sha256": cand_stages[stage],
                "provenance_differences": provenance_differences,
            }
            if stage == "text_embeddings":
                details.update(
                    _embedding_drift_summary(
                        reference.text_embeddings,
                        candidate.text_embeddings,
                    )
                )
                details["reference_encoder"] = reference.manifest["provenance"]["encoder"]
                details["candidate_encoder"] = candidate.manifest["provenance"]["encoder"]
                details["reference_runtime"] = reference.manifest["provenance"]["runtime"]
                details["candidate_runtime"] = candidate.manifest["provenance"]["runtime"]
            return {"status": "drift", "first_mismatch": stage, "details": details}

    ref_scores = reference.frame["raw_scores"]
    cand_scores = candidate.frame["raw_scores"]
    if ref_scores.shape != cand_scores.shape or not np.allclose(
        ref_scores,
        cand_scores,
        rtol=0.0,
        atol=score_atol,
    ):
        max_abs = (
            float(np.max(np.abs(ref_scores - cand_scores), initial=0.0))
            if ref_scores.shape == cand_scores.shape
            else None
        )
        return {
            "status": "drift",
            "first_mismatch": "model_scores",
            "details": {
                "reference_shape": list(ref_scores.shape),
                "candidate_shape": list(cand_scores.shape),
                "max_abs": max_abs,
                "atol": score_atol,
                "provenance_differences": provenance_differences,
            },
        }

    ref_runtime_deltas = _runtime_metric_deltas(reference)
    cand_runtime_deltas = _runtime_metric_deltas(candidate)
    stored_deltas = {
        key: float(
            candidate.manifest["evaluation"]["metrics"][key]
            - reference.manifest["evaluation"]["metrics"][key]
        )
        for key in ("ndcg@5", "ndcg@10")
    }
    metric_drift = (
        ref_stages["metrics"] != cand_stages["metrics"]
        or any(abs(delta) > metric_atol for delta in stored_deltas.values())
        or any(abs(delta) > metric_atol for delta in ref_runtime_deltas.values())
        or any(abs(delta) > metric_atol for delta in cand_runtime_deltas.values())
    )
    if metric_drift:
        return {
            "status": "drift",
            "first_mismatch": "metrics",
            "details": {
                "stored_metric_deltas": stored_deltas,
                "reference_runtime_deltas": ref_runtime_deltas,
                "candidate_runtime_deltas": cand_runtime_deltas,
                "atol": metric_atol,
                "reference_sha256": ref_stages["metrics"],
                "candidate_sha256": cand_stages["metrics"],
                "provenance_differences": provenance_differences,
            },
        }

    return {
        "status": "match",
        "first_mismatch": None,
        "details": {
            "score_max_abs": float(np.max(np.abs(ref_scores - cand_scores), initial=0.0)),
            "stored_metric_deltas": stored_deltas,
            "reference_runtime_deltas": ref_runtime_deltas,
            "candidate_runtime_deltas": cand_runtime_deltas,
            "reference_pack_content_sha256": reference.manifest["pack_content_sha256"],
            "candidate_pack_content_sha256": candidate.manifest["pack_content_sha256"],
            "provenance_differences": provenance_differences,
        },
    }


def evaluate_model_against_pack(
    model_path: Path,
    pack_dir: Path,
    *,
    expected_model_sha256: str,
    expected_pack_sha256: str,
    score_atol: float = 1e-10,
    metric_atol: float = 1e-6,
) -> dict[str, Any]:
    """외부에서 pin한 동일 모델을 frozen X에 재적용한다."""

    _validate_tolerances(score_atol, metric_atol)
    pack = load_match_eval_pack(
        pack_dir,
        expected_pack_sha256=expected_pack_sha256,
    )
    bundle = _load_trusted_bundle(
        model_path,
        expected_model_sha256=expected_model_sha256,
        pack_model_sha256=pack.manifest["model"]["sha256"],
    )
    columns = list(bundle["columns"])
    if columns != pack.manifest["evaluation"]["columns"]:
        raise EvalPackIntegrityError("model bundle column order differs from evaluation pack")
    model = bundle["model"]
    if hasattr(model, "num_feature") and int(model.num_feature()) != len(columns):
        raise EvalPackIntegrityError("model feature count differs from evaluation pack")
    predictions = canonical_float_array(model.predict(pack.frame["x_test"]), np.dtype("<f8"))
    stored_scores = pack.frame["raw_scores"]
    if predictions.shape != stored_scores.shape:
        raise EvalPackIntegrityError("model prediction length mismatch")
    score_max_abs = float(np.max(np.abs(predictions - stored_scores), initial=0.0))
    metrics = _metrics_for(pack, predictions)
    stored_metrics = pack.manifest["evaluation"]["metrics"]
    metric_deltas = {key: float(metrics[key] - stored_metrics[key]) for key in metrics}
    passed = score_max_abs <= score_atol and all(
        abs(delta) <= metric_atol for delta in metric_deltas.values()
    )
    return {
        "status": "match" if passed else "drift",
        "model_sha256": expected_model_sha256,
        "pack_content_sha256": expected_pack_sha256,
        "score_max_abs": score_max_abs,
        "score_atol": score_atol,
        "metrics": metrics,
        "stored_metrics": stored_metrics,
        "metric_deltas": metric_deltas,
        "metric_atol": metric_atol,
    }


def _parse_datetime(value: Any, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise EvalPackIntegrityError("timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise EvalPackIntegrityError("invalid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvalPackIntegrityError("timestamps must be timezone-aware")
    return parsed


def _content_signal(record: dict[str, Any]) -> ContentSignal:
    created_at = _parse_datetime(record.get("created_at"))
    if created_at is None:
        raise EvalPackIntegrityError("content timestamp is required")
    return ContentSignal(
        content_id=record["content_id"],
        source_type=record["source_type"],
        text=record["text"],
        created_at=created_at,
        is_deleted=record["is_deleted"],
        is_accessible=record["is_accessible"],
        is_blocked_author=record["is_blocked_author"],
        is_like_active=record["is_like_active"],
    )


def _restore_users(pack: LoadedMatchEvalPack) -> list[SyntheticUser]:
    users: list[SyntheticUser] = []
    for record in pack.user_records:
        snapshot_record = record["snapshot"]
        latent_record = record["latent"]
        snapshot = UserSnapshot(
            user_id=snapshot_record["user_id"],
            bio=snapshot_record["bio"],
            tag_ids=tuple(snapshot_record["tag_ids"]),
            age_years=snapshot_record["age_years"],
            age_band=snapshot_record["age_band"],
            ui_mode=snapshot_record["ui_mode"],
            authored_items=tuple(
                _content_signal(item) for item in snapshot_record["authored_items"]
            ),
            liked_items=tuple(_content_signal(item) for item in snapshot_record["liked_items"]),
        )
        weights = canonical_float_array(
            latent_record["interest_weights"],
            np.dtype("<f8"),
        )
        users.append(
            SyntheticUser(
                snapshot=snapshot,
                latent=LatentUser(
                    user_id=latent_record["user_id"],
                    interest_weights=weights,
                    age=int(latent_record["age"]),
                    social_cluster=int(latent_record["social_cluster"]),
                ),
            )
        )
    return users


def _restore_pairs(
    pack: LoadedMatchEvalPack,
    users_by_id: dict[str, SyntheticUser],
) -> list[PairRecord]:
    pairs: list[PairRecord] = []
    for record in pack.pair_records:
        relationship_record = record["relationship"]
        rejected_at = _parse_datetime(
            relationship_record.get("last_rejected_at"),
            optional=True,
        )
        query = users_by_id[record["query_id"]]
        candidate = users_by_id[record["candidate_id"]]
        relationship = CandidateRelationship(
            candidate_id=relationship_record["candidate_id"],
            blocked_either_direction=relationship_record["blocked_either_direction"],
            already_friends=relationship_record["already_friends"],
            last_rejected_at=rejected_at,
            common_friend_count=int(relationship_record["common_friend_count"]),
        )
        pairs.append(
            PairRecord(
                query_id=query.snapshot.user_id,
                cand_id=candidate.snapshot.user_id,
                query_snapshot=query.snapshot,
                candidate_input=CandidateInput(
                    profile=candidate.snapshot,
                    relationship=relationship,
                ),
                label=int(record["label"]),
                latent_score=float(record["latent_score"]),
            )
        )
    return pairs


def _restore_policy(pack: LoadedMatchEvalPack) -> MatchingInputPolicy:
    values = dict(pack.manifest["provenance"]["policy"])
    for name in ("allowed_tag_ids", "allowed_ui_modes", "allowed_content_sources"):
        if values.get(name) is not None:
            values[name] = frozenset(values[name])
    return MatchingInputPolicy(**values)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = Path(left).expanduser().resolve(strict=False)
    right_resolved = Path(right).expanduser().resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def replay_match_eval_pack(
    reference_dir: Path,
    *,
    model_path: Path,
    expected_model_sha256: str,
    expected_reference_sha256: str,
    encoder: TextEncoder,
    output_dir: Path,
    config_path: Path,
    corpus_path: Path | None = None,
    score_atol: float = 1e-10,
    metric_atol: float = 1e-6,
) -> dict[str, Any]:
    """reference raw snapshot과 고정 모델을 target encoder 환경에서 끝까지 재생성한다."""

    _validate_tolerances(score_atol, metric_atol)
    reference_dir = Path(reference_dir)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    config_path = Path(config_path)
    resolved_corpus_path = Path(corpus_path or DEFAULT_CORPUS_PATH)
    for protected in (reference_dir, model_path, config_path, resolved_corpus_path):
        if _paths_overlap(output_dir, protected):
            raise EvalPackError(
                "replay output must not overlap reference, model, config, or corpus paths"
            )
    reference = load_match_eval_pack(
        reference_dir,
        expected_pack_sha256=expected_reference_sha256,
    )
    bundle = _load_trusted_bundle(
        model_path,
        expected_model_sha256=expected_model_sha256,
        pack_model_sha256=reference.manifest["model"]["sha256"],
    )
    columns = list(bundle["columns"])
    if columns != reference.manifest["evaluation"]["columns"]:
        raise EvalPackIntegrityError("model columns differ from reference pack")
    model = bundle["model"]
    if hasattr(model, "num_feature") and int(model.num_feature()) != len(columns):
        raise EvalPackIntegrityError("model feature count differs from reference pack")

    users = _restore_users(reference)
    users_by_id = {user.snapshot.user_id: user for user in users}
    pairs = _restore_pairs(reference, users_by_id)
    policy = _restore_policy(reference)
    as_of = _parse_datetime(reference.manifest["provenance"]["run"]["as_of"])
    if as_of is None:
        raise EvalPackIntegrityError("reference as_of timestamp is required")
    capture = CapturingEncoder(encoder)
    features = embed_users(users, encoder=capture, policy=policy, as_of=as_of)
    if capture.sentences is None or capture.embeddings is None:
        raise EvalPackError("target encoder inputs were not captured")
    prepared = tuple(
        prepare_text_signals(user.snapshot, policy=policy, as_of=as_of) for user in users
    )
    frame = build_feature_frame_detailed(
        pairs,
        features,
        columns=columns,
        snapshot_by_id=users_by_id,
        as_of=as_of,
        rejection_cooldown_days=policy.rejection_cooldown_days,
    )
    scores = canonical_float_array(model.predict(frame.x), np.dtype("<f8"))
    if scores.shape != frame.y.shape:
        raise EvalPackIntegrityError("replayed model prediction length mismatch")
    metrics = {
        "ndcg@5": ndcg_at_k(frame.y, scores, frame.group_sizes, 5),
        "ndcg@10": ndcg_at_k(frame.y, scores, frame.group_sizes, 10),
    }
    candidate_manifest = write_match_eval_pack(
        output_dir,
        model_path=model_path,
        columns=columns,
        detailed_frame=frame,
        x_test=frame.x,
        raw_scores=scores,
        test_users=users,
        prepared_signals=prepared,
        captured_sentences=capture.sentences,
        text_embeddings=capture.embeddings,
        user_features=features,
        metrics=metrics,
        policy=policy,
        encoder=encoder,
        run_metadata=reference.manifest["provenance"]["run"],
        config_path=config_path,
        corpus_path=resolved_corpus_path,
    )
    report = compare_match_eval_packs(
        reference_dir,
        output_dir,
        expected_reference_sha256=expected_reference_sha256,
        expected_candidate_sha256=candidate_manifest["pack_content_sha256"],
        score_atol=score_atol,
        metric_atol=metric_atol,
    )
    report["candidate_pack"] = str(Path(output_dir))
    report["reference_pack_sha256"] = expected_reference_sha256
    report["candidate_pack_sha256"] = candidate_manifest["pack_content_sha256"]
    report["replayed_metrics"] = metrics
    return report
