"""MATCH v2 평가팩의 고정 포맷과 원자적 writer."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.data.build_profiles_v2 import DEFAULT_CORPUS_PATH, SyntheticUser
from src.data.matching_input import (
    MatchingInputPolicy,
    PreparedTextSignals,
    PreparedUserFeatures,
    TextEncoder,
)
from src.data.matching_trainset import DetailedFeatureFrame, ndcg_at_k

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "match-v2-eval-pack/v2"
MANIFEST_FILE = "manifest.json"
FRAME_FILE = "frame.npz"
TEXT_EMBEDDINGS_FILE = "text_embeddings.npy"
USER_EMBEDDINGS_FILE = "user_embeddings.npz"
TEST_USERS_FILE = "test_users.jsonl"
TEST_PAIRS_FILE = "test_pairs.jsonl"
PAIR_TRACE_FILE = "pair_trace.jsonl"
PREPARED_TEXTS_FILE = "prepared_texts.jsonl"
QUERY_METRICS_FILE = "query_metrics.jsonl"
PACK_FILES = (
    FRAME_FILE,
    TEXT_EMBEDDINGS_FILE,
    USER_EMBEDDINGS_FILE,
    TEST_USERS_FILE,
    TEST_PAIRS_FILE,
    PAIR_TRACE_FILE,
    PREPARED_TEXTS_FILE,
    QUERY_METRICS_FILE,
)
FRAME_KEYS = (
    "x_test",
    "y_test",
    "group_sizes",
    "group_offsets",
    "raw_scores",
    "group_query_ids",
    "row_query_ids",
    "candidate_ids",
    "latent_scores",
)
USER_EMBEDDING_KEYS = (
    "user_ids",
    "profile_embeddings",
    "profile_available",
    "authored_embeddings",
    "authored_available",
    "liked_embeddings",
    "liked_available",
    "effective_embeddings",
    "effective_available",
    "authored_count",
    "liked_count",
)
EXACT_STAGE_ORDER = (
    "data",
    "prepared_texts",
    "text_embeddings",
    "user_embeddings",
    "features",
    "model",
)
MAX_ROWS = 100_000
MAX_FEATURES = 256
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_USER_EMBEDDINGS_BYTES = 96 * 1024 * 1024
MAX_TEXT_EMBEDDINGS_BYTES = 96 * 1024 * 1024
MAX_JSONL_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_PACK_BYTES = 256 * 1024 * 1024
FILE_SIZE_LIMITS = {
    FRAME_FILE: MAX_FRAME_BYTES,
    TEXT_EMBEDDINGS_FILE: MAX_TEXT_EMBEDDINGS_BYTES,
    USER_EMBEDDINGS_FILE: MAX_USER_EMBEDDINGS_BYTES,
    TEST_USERS_FILE: MAX_JSONL_BYTES,
    TEST_PAIRS_FILE: 32 * 1024 * 1024,
    PAIR_TRACE_FILE: 16 * 1024 * 1024,
    PREPARED_TEXTS_FILE: MAX_JSONL_BYTES,
    QUERY_METRICS_FILE: 8 * 1024 * 1024,
}


class EvalPackError(ValueError):
    """평가팩 생성·구조 검증 오류."""


class EvalPackIntegrityError(EvalPackError):
    """평가팩 파일 또는 semantic hash 불일치."""


class CapturingEncoder:
    """실제 encoder 입력과 출력을 한 번만 포착하는 래퍼."""

    def __init__(self, delegate: TextEncoder):
        self.delegate = delegate
        self.sentences: tuple[str, ...] | None = None
        self.embeddings: np.ndarray | None = None

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        if self.sentences is not None:
            raise EvalPackError("capturing encoder expected exactly one encode call")
        encoded = np.asarray(
            self.delegate.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            ),
            dtype=np.float32,
        )
        self.sentences = tuple(sentences)
        self.embeddings = canonical_float_array(encoded, np.dtype("<f4"))
        return encoded


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """파일을 스트리밍해 lowercase SHA-256을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, set | frozenset):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return (text + "\n").encode("utf-8")


def strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise EvalPackIntegrityError(f"non-finite JSON constant: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise EvalPackIntegrityError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def canonical_float_array(value: Any, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype, order="C").copy(order="C")
    if not np.isfinite(array).all():
        raise EvalPackError("floating arrays must be finite")
    array[array == 0] = 0.0
    return array


def _canonical_int_array(value: Any, dtype: np.dtype) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in {"i", "u"}:
        raise EvalPackError("integer arrays must not be silently cast from another dtype")
    return np.asarray(source, dtype=dtype, order="C").copy(order="C")


def _canonical_text_array(values: Sequence[str]) -> np.ndarray:
    if any(not isinstance(value, str) or not value for value in values):
        raise EvalPackError("identifier arrays must contain non-empty strings")
    return np.asarray(list(values), dtype=str)


def _hash_parts(parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def semantic_array_sha256(name: str, array: np.ndarray) -> str:
    """dtype·shape·순서를 포함한 배열 semantic SHA-256."""

    contiguous = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"name": name, "dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    if contiguous.dtype.kind in {"U", "S"}:
        return _hash_parts((header, canonical_json_bytes(contiguous.tolist())))
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    view = memoryview(contiguous).cast("B")
    digest.update(len(view).to_bytes(8, "big"))
    for start in range(0, len(view), 1024 * 1024):
        digest.update(view[start : start + 1024 * 1024])
    return digest.hexdigest()


def semantic_arrays_sha256(
    arrays: Mapping[str, np.ndarray],
    keys: Sequence[str],
) -> str:
    return _hash_parts(
        tuple(
            canonical_json_bytes({"key": key, "sha256": semantic_array_sha256(key, arrays[key])})
            for key in keys
        )
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for record in records:
                handle.write(canonical_json_bytes(dict(record)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_save_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    keys: Sequence[str],
) -> None:
    if tuple(arrays) != tuple(keys):
        raise EvalPackError("NPZ arrays must use the declared deterministic key order")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _runtime_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "torch",
        "sentence-transformers",
        "transformers",
        "tokenizers",
        "safetensors",
        "huggingface-hub",
        "pyyaml",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_provenance() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run_git("status", "--porcelain")
    return {
        "revision": run_git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _source_hashes() -> dict[str, str]:
    relative_paths = (
        "configs/module2_matching.yaml",
        "notebooks/module2_retrain_v2_colab.ipynb",
        "requirements-colab.txt",
        "scripts/evaluate_match_v2_pack.py",
        "scripts/replay_match_v2_pack.py",
        "scripts/train_match_v2.py",
        "scripts/verify_match_v2_parity.py",
        "src/data/build_pairs_v2.py",
        "src/data/build_profiles_v2.py",
        "src/data/match_eval_pack/__init__.py",
        "src/data/match_eval_pack/compare.py",
        "src/data/match_eval_pack/format.py",
        "src/data/match_eval_pack/run_pointer.py",
        "src/data/match_eval_pack/validation.py",
        "src/data/matching_input.py",
        "src/data/matching_trainset.py",
    )
    return {
        relative: sha256_file(ROOT / relative)
        for relative in relative_paths
        if (ROOT / relative).is_file()
    }


def describe_encoder(
    encoder: TextEncoder,
    *,
    output_dimension: int,
    batch_size: int,
) -> dict[str, Any]:
    delegate = getattr(encoder, "delegate", encoder)
    metadata: dict[str, Any] = {
        "adapter_type": f"{type(delegate).__module__}.{type(delegate).__qualname__}",
        "output_dimension": int(output_dimension),
        "batch_size": int(batch_size),
    }
    supplied = getattr(delegate, "reproducibility_metadata", None)
    if callable(supplied):
        supplied = supplied()
    if isinstance(supplied, Mapping):
        metadata.update(dict(supplied))
    return metadata


def flatten_prepared_texts(
    prepared_signals: Sequence[PreparedTextSignals],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """encode_prepared_users와 동일한 순서로 text row를 만든다."""

    records: list[dict[str, Any]] = []
    sentences: list[str] = []
    for prepared in prepared_signals:
        components: tuple[tuple[str, Sequence[str]], ...] = (
            ("profile", (prepared.profile_text,) if prepared.profile_text is not None else ()),
            ("authored", prepared.authored_texts),
            ("liked", prepared.liked_texts),
        )
        for component, texts in components:
            for component_index, text in enumerate(texts):
                row = len(sentences)
                sentences.append(text)
                records.append(
                    {
                        "text_row": row,
                        "user_id": prepared.user_id,
                        "component": component,
                        "component_index": component_index,
                        "text": text,
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                )
    return records, tuple(sentences)


def _content_record(item: Any) -> dict[str, Any]:
    return {
        "content_id": item.content_id,
        "source_type": item.source_type,
        "text": item.text,
        "created_at": item.created_at.isoformat(),
        "is_deleted": item.is_deleted,
        "is_accessible": item.is_accessible,
        "is_blocked_author": item.is_blocked_author,
        "is_like_active": item.is_like_active,
    }


def _test_user_records(
    test_users: Sequence[SyntheticUser],
    prepared_signals: Sequence[PreparedTextSignals],
) -> list[dict[str, Any]]:
    if len(test_users) != len(prepared_signals):
        raise EvalPackError("test user and prepared signal count mismatch")
    records: list[dict[str, Any]] = []
    for position, (user, prepared) in enumerate(zip(test_users, prepared_signals, strict=True)):
        snapshot = user.snapshot
        if snapshot.user_id != prepared.user_id:
            raise EvalPackError("prepared signal user order mismatch")
        records.append(
            {
                "user_position": position,
                "user_id": snapshot.user_id,
                "snapshot": {
                    "user_id": snapshot.user_id,
                    "bio": snapshot.bio,
                    "tag_ids": list(snapshot.tag_ids),
                    "age_years": snapshot.age_years,
                    "age_band": snapshot.age_band,
                    "ui_mode": snapshot.ui_mode,
                    "authored_items": [_content_record(item) for item in snapshot.authored_items],
                    "liked_items": [_content_record(item) for item in snapshot.liked_items],
                },
                "latent": {
                    "user_id": user.latent.user_id,
                    "interest_weights": user.latent.interest_weights.tolist(),
                    "age": user.latent.age,
                    "social_cluster": user.latent.social_cluster,
                },
            }
        )
    return records


def _pair_records(
    frame: DetailedFeatureFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    for trace in frame.pair_traces:
        identity = {"query_id": trace.query_id, "candidate_id": trace.candidate_id}
        pair_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        input_records.append(
            {
                "pair_id": pair_id,
                "query_position": trace.query_position,
                "candidate_position": trace.candidate_position,
                "query_id": trace.query_id,
                "candidate_id": trace.candidate_id,
                "label": trace.label,
                "latent_score": trace.latent_score,
                "relationship": {
                    "candidate_id": trace.candidate_id,
                    "common_friend_count": trace.common_friend_count,
                    "blocked_either_direction": trace.blocked_either_direction,
                    "already_friends": trace.already_friends,
                    "last_rejected_at": (
                        trace.last_rejected_at.isoformat()
                        if trace.last_rejected_at is not None
                        else None
                    ),
                },
            }
        )
        trace_records.append(
            {
                "pair_id": pair_id,
                "filter_status": trace.status,
                "exclusion_reason": trace.exclusion_reason,
                "feature_row_index": trace.feature_row_index,
            }
        )
    return input_records, trace_records


def _query_metric_records(
    frame: DetailedFeatureFrame,
    raw_scores: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    kept_trace = {trace.query_id: trace for trace in frame.query_traces if trace.kept_count > 0}
    for query_id, size in zip(frame.query_ids, frame.group_sizes, strict=True):
        labels = frame.y[offset : offset + size]
        scores = raw_scores[offset : offset + size]
        included = size >= 2 and int(labels.max(initial=0)) > 0
        records.append(
            {
                "query_id": query_id,
                "group_size": size,
                "metric_included": included,
                "ndcg@5": ndcg_at_k(labels, scores, [size], 5) if included else None,
                "ndcg@10": ndcg_at_k(labels, scores, [size], 10) if included else None,
                "pre_filter_count": kept_trace[query_id].pre_filter_count,
            }
        )
        offset += size
    return records


def _stack_optional_vectors(
    user_ids: Sequence[str],
    features: Mapping[str, PreparedUserFeatures],
    *,
    attribute: str,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(user_ids), dimension), dtype="<f4")
    available = np.zeros(len(user_ids), dtype=np.bool_)
    for row, user_id in enumerate(user_ids):
        value = getattr(features[user_id], attribute)
        if value is None:
            continue
        vector = canonical_float_array(value, np.dtype("<f4"))
        if vector.shape != (dimension,):
            raise EvalPackError(f"{attribute} dimension mismatch for {user_id}")
        matrix[row] = vector
        available[row] = True
    return matrix, available


def _user_embedding_arrays(
    test_users: Sequence[SyntheticUser],
    features: Mapping[str, PreparedUserFeatures],
    *,
    dimension: int,
) -> dict[str, np.ndarray]:
    user_ids = [user.snapshot.user_id for user in test_users]
    if len(user_ids) != len(set(user_ids)):
        raise EvalPackError("duplicate test user IDs")
    if any(user_id not in features for user_id in user_ids):
        raise EvalPackError("missing prepared user features")
    arrays: dict[str, np.ndarray] = {"user_ids": _canonical_text_array(user_ids)}
    for prefix in ("profile", "authored", "liked", "effective"):
        matrix, available = _stack_optional_vectors(
            user_ids,
            features,
            attribute=f"{prefix}_vector",
            dimension=dimension,
        )
        arrays[f"{prefix}_embeddings"] = matrix
        arrays[f"{prefix}_available"] = available
    arrays["authored_count"] = np.asarray(
        [features[user_id].authored_count for user_id in user_ids],
        dtype="<i4",
    )
    arrays["liked_count"] = np.asarray(
        [features[user_id].liked_count for user_id in user_ids],
        dtype="<i4",
    )
    return {key: arrays[key] for key in USER_EMBEDDING_KEYS}


def _frame_arrays(
    frame: DetailedFeatureFrame,
    *,
    x_test: np.ndarray,
    raw_scores: np.ndarray,
    columns: Sequence[str],
) -> dict[str, np.ndarray]:
    if not columns or len(columns) != len(set(columns)):
        raise EvalPackError("model columns must be non-empty and unique")
    x = canonical_float_array(x_test, np.dtype("<f4"))
    y = _canonical_int_array(frame.y, np.dtype("<i4"))
    groups = _canonical_int_array(frame.group_sizes, np.dtype("<i4"))
    scores = canonical_float_array(raw_scores, np.dtype("<f8"))
    latent_scores = canonical_float_array(frame.latent_scores, np.dtype("<f8"))
    n_rows = x.shape[0] if x.ndim == 2 else -1
    if n_rows <= 0 or n_rows > MAX_ROWS or x.shape[1] != len(columns):
        raise EvalPackError("invalid x_test shape")
    if len(columns) > MAX_FEATURES:
        raise EvalPackError("feature count outside supported range")
    if y.shape != (n_rows,) or scores.shape != (n_rows,) or latent_scores.shape != (n_rows,):
        raise EvalPackError("row-aligned array length mismatch")
    if groups.ndim != 1 or len(groups) != len(frame.query_ids) or np.any(groups <= 0):
        raise EvalPackError("invalid group sizes")
    if int(groups.astype(np.int64).sum()) != n_rows:
        raise EvalPackError("group sizes do not cover x_test")
    if np.any((y < 0) | (y > 3)):
        raise EvalPackError("labels must be integers in [0, 3]")
    expected_queries = [
        query_id
        for query_id, size in zip(frame.query_ids, frame.group_sizes, strict=True)
        for _ in range(size)
    ]
    if expected_queries != frame.row_query_ids or len(frame.candidate_ids) != n_rows:
        raise EvalPackError("row identifiers do not match group boundaries")
    offsets = np.concatenate(
        (np.asarray([0], dtype="<i8"), np.cumsum(groups, dtype=np.int64).astype("<i8"))
    )
    arrays = {
        "x_test": x,
        "y_test": y,
        "group_sizes": groups,
        "group_offsets": offsets,
        "raw_scores": scores,
        "group_query_ids": _canonical_text_array(frame.query_ids),
        "row_query_ids": _canonical_text_array(frame.row_query_ids),
        "candidate_ids": _canonical_text_array(frame.candidate_ids),
        "latent_scores": latent_scores,
    }
    return {key: arrays[key] for key in FRAME_KEYS}


def _file_metadata(pack_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": sha256_file(pack_dir / name), "bytes": (pack_dir / name).stat().st_size}
        for name in PACK_FILES
    }


def stage_hashes(
    *,
    files: Mapping[str, Mapping[str, Any]],
    frame: Mapping[str, np.ndarray],
    user_embeddings: Mapping[str, np.ndarray],
    text_embeddings: np.ndarray,
    columns: Sequence[str],
    model_sha256: str,
    metrics: Mapping[str, float],
) -> dict[str, str]:
    data_hash = _hash_parts(
        (
            str(files[TEST_USERS_FILE]["sha256"]).encode(),
            str(files[TEST_PAIRS_FILE]["sha256"]).encode(),
            semantic_arrays_sha256(
                frame,
                (
                    "y_test",
                    "group_sizes",
                    "group_offsets",
                    "group_query_ids",
                    "row_query_ids",
                    "candidate_ids",
                    "latent_scores",
                ),
            ).encode(),
        )
    )
    return {
        "data": data_hash,
        "prepared_texts": str(files[PREPARED_TEXTS_FILE]["sha256"]),
        "text_embeddings": semantic_array_sha256("text_embeddings", text_embeddings),
        "user_embeddings": semantic_arrays_sha256(user_embeddings, USER_EMBEDDING_KEYS),
        "features": _hash_parts(
            (
                canonical_json_bytes(list(columns)),
                semantic_array_sha256("x_test", frame["x_test"]).encode(),
                str(files[PAIR_TRACE_FILE]["sha256"]).encode(),
            )
        ),
        "model": model_sha256,
        "model_scores": semantic_array_sha256("raw_scores", frame["raw_scores"]),
        "metrics": _hash_parts(
            (
                canonical_json_bytes(dict(metrics)),
                str(files[QUERY_METRICS_FILE]["sha256"]).encode(),
            )
        ),
    }


def pack_content_sha256(
    *,
    stages: Mapping[str, str],
    model: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    provenance_sha256: str,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "stages": dict(stages),
                "model": dict(model),
                "files": {name: dict(files[name]) for name in PACK_FILES},
                "evaluation": dict(evaluation),
                "provenance_sha256": provenance_sha256,
            }
        )
    ).hexdigest()


def _is_same_or_ancestor(left: Path, right: Path) -> bool:
    return left == right or left in right.parents


def validate_new_pack_target(
    target: Path,
    *,
    protected_paths: Sequence[Path],
) -> None:
    resolved = target.expanduser().resolve(strict=False)
    broad_roots = (Path("/").resolve(), Path.home().resolve(), ROOT.resolve())
    if any(_is_same_or_ancestor(resolved, root) for root in broad_roots):
        raise EvalPackError("evaluation pack target is a protected broad directory")
    for protected in protected_paths:
        protected_resolved = Path(protected).expanduser().resolve(strict=False)
        if _is_same_or_ancestor(resolved, protected_resolved):
            raise EvalPackError("evaluation pack target contains a protected input path")
    if target.is_symlink() or target.exists():
        raise EvalPackError("evaluation pack target already exists; choose a new empty output path")


def _publish_new_directory(staging: Path, target: Path) -> None:
    if target.is_symlink():
        raise EvalPackError(
            "evaluation pack target appeared during generation; no files were replaced"
        )
    created_links: list[tuple[Path, tuple[int, int]]] = []
    target_created = False
    try:
        target.mkdir(mode=0o755, exist_ok=False)
        target_created = True
        for name in (*PACK_FILES, MANIFEST_FILE):
            source = staging / name
            destination = target / name
            os.link(source, destination, follow_symlinks=False)
            linked = destination.stat(follow_symlinks=False)
            created_links.append((destination, (linked.st_dev, linked.st_ino)))
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(target, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        for destination, expected_identity in reversed(created_links):
            try:
                current = destination.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == expected_identity:
                    destination.unlink()
            except FileNotFoundError:
                continue
        if target_created:
            try:
                target.rmdir()
            except OSError:
                pass
        if isinstance(exc, FileExistsError):
            raise EvalPackError(
                "evaluation pack target appeared during generation; no files were replaced"
            ) from exc
        raise EvalPackError("evaluation pack directory could not be published safely") from exc


def write_match_eval_pack(
    pack_dir: Path,
    *,
    model_path: Path,
    columns: Sequence[str],
    detailed_frame: DetailedFeatureFrame,
    x_test: np.ndarray,
    raw_scores: np.ndarray,
    test_users: Sequence[SyntheticUser],
    prepared_signals: Sequence[PreparedTextSignals],
    captured_sentences: Sequence[str],
    text_embeddings: np.ndarray,
    user_features: Mapping[str, PreparedUserFeatures],
    metrics: Mapping[str, float],
    policy: MatchingInputPolicy,
    encoder: TextEncoder,
    run_metadata: Mapping[str, Any],
    config_path: Path,
    corpus_path: Path | None = None,
) -> dict[str, Any]:
    """평가팩 전체를 sibling 임시 디렉터리에 쓴 뒤 원자적으로 교체한다."""

    target = Path(pack_dir)
    model_path = Path(model_path)
    config_path = Path(config_path)
    corpus_path = Path(corpus_path or DEFAULT_CORPUS_PATH)
    if not model_path.is_file() or model_path.stat().st_size > MAX_MODEL_BYTES:
        raise EvalPackError("model file missing or too large")
    if not config_path.is_file():
        raise EvalPackError("config file does not exist")
    validate_new_pack_target(
        target,
        protected_paths=(model_path, config_path, corpus_path),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.staging."))
    try:
        frame = _frame_arrays(
            detailed_frame,
            x_test=x_test,
            raw_scores=raw_scores,
            columns=columns,
        )
        text_matrix = canonical_float_array(text_embeddings, np.dtype("<f4"))
        if (
            text_matrix.ndim != 2
            or text_matrix.shape[0] != len(captured_sentences)
            or text_matrix.shape[1] == 0
        ):
            raise EvalPackError("captured text embedding shape mismatch")
        prepared_records, expected_sentences = flatten_prepared_texts(prepared_signals)
        if tuple(captured_sentences) != expected_sentences:
            raise EvalPackError("captured encoder inputs differ from prepared text order")
        user_records = _test_user_records(test_users, prepared_signals)
        pair_records, pair_trace_records = _pair_records(detailed_frame)
        query_records = _query_metric_records(detailed_frame, frame["raw_scores"])
        if not any(record["metric_included"] for record in query_records):
            raise EvalPackError("evaluation pack has no scorable queries")
        normalized_metrics = {
            "ndcg@5": float(metrics["ndcg@5"]),
            "ndcg@10": float(metrics["ndcg@10"]),
        }
        current_metrics = {
            "ndcg@5": ndcg_at_k(
                frame["y_test"],
                frame["raw_scores"],
                frame["group_sizes"].tolist(),
                5,
            ),
            "ndcg@10": ndcg_at_k(
                frame["y_test"],
                frame["raw_scores"],
                frame["group_sizes"].tolist(),
                10,
            ),
        }
        for key in normalized_metrics:
            if not np.isclose(
                normalized_metrics[key],
                current_metrics[key],
                rtol=0.0,
                atol=1e-12,
            ):
                raise EvalPackError(f"stored training metric mismatch: {key}")

        user_arrays = _user_embedding_arrays(
            test_users,
            user_features,
            dimension=text_matrix.shape[1],
        )
        _atomic_save_npz(staging / FRAME_FILE, frame, FRAME_KEYS)
        _atomic_save_npy(staging / TEXT_EMBEDDINGS_FILE, text_matrix)
        _atomic_save_npz(staging / USER_EMBEDDINGS_FILE, user_arrays, USER_EMBEDDING_KEYS)
        _atomic_write_jsonl(staging / TEST_USERS_FILE, user_records)
        _atomic_write_jsonl(staging / TEST_PAIRS_FILE, pair_records)
        _atomic_write_jsonl(staging / PAIR_TRACE_FILE, pair_trace_records)
        _atomic_write_jsonl(staging / PREPARED_TEXTS_FILE, prepared_records)
        _atomic_write_jsonl(staging / QUERY_METRICS_FILE, query_records)

        files = _file_metadata(staging)
        for name, metadata in files.items():
            if metadata["bytes"] <= 0 or metadata["bytes"] > FILE_SIZE_LIMITS[name]:
                raise EvalPackError(f"evaluation pack file size limit exceeded: {name}")
        model_sha256 = sha256_file(model_path)
        stages = stage_hashes(
            files=files,
            frame=frame,
            user_embeddings=user_arrays,
            text_embeddings=text_matrix,
            columns=columns,
            model_sha256=model_sha256,
            metrics=normalized_metrics,
        )
        evaluation = {
            "columns": list(columns),
            "rows": int(frame["x_test"].shape[0]),
            "groups": int(len(frame["group_sizes"])),
            "scored_groups": int(sum(bool(row["metric_included"]) for row in query_records)),
            "text_rows": int(text_matrix.shape[0]),
            "embedding_dimension": int(text_matrix.shape[1]),
            "metrics": normalized_metrics,
            "metric_implementation": "sklearn.metrics.ndcg_score/query-mean",
        }
        provenance = {
            "run": dict(run_metadata),
            "policy": asdict(policy),
            "config": {
                "path": (
                    config_path.relative_to(ROOT).as_posix()
                    if config_path.is_relative_to(ROOT)
                    else config_path.name
                ),
                "sha256": sha256_file(config_path),
            },
            "corpus": {
                "path": (
                    corpus_path.relative_to(ROOT).as_posix()
                    if corpus_path.is_relative_to(ROOT)
                    else corpus_path.name
                ),
                "exists": corpus_path.is_file(),
                "mode": "file" if corpus_path.is_file() else "template_fallback",
                "sha256": sha256_file(corpus_path) if corpus_path.is_file() else None,
            },
            "encoder": describe_encoder(
                encoder,
                output_dimension=text_matrix.shape[1],
                batch_size=policy.embedding_batch_size,
            ),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _runtime_versions(),
            },
            "git": _git_provenance(),
            "source_sha256": _source_hashes(),
        }
        provenance_sha256 = hashlib.sha256(canonical_json_bytes(provenance)).hexdigest()
        model = {
            "filename": model_path.name,
            "sha256": model_sha256,
            "columns": list(columns),
            "n_features": len(columns),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "provenance_sha256": provenance_sha256,
            "model": model,
            "evaluation": evaluation,
            "stages": stages,
            "files": files,
            "provenance": provenance,
        }
        manifest["pack_content_sha256"] = pack_content_sha256(
            stages=stages,
            model=model,
            files=files,
            evaluation=evaluation,
            provenance_sha256=provenance_sha256,
        )
        manifest_payload = canonical_json_bytes(manifest)
        if (
            len(manifest_payload) > MAX_MANIFEST_BYTES
            or len(manifest_payload) + sum(item["bytes"] for item in files.values())
            > MAX_PACK_BYTES
        ):
            raise EvalPackError("evaluation pack aggregate size limit exceeded")
        _atomic_write_bytes(staging / MANIFEST_FILE, manifest_payload)
        _publish_new_directory(staging, target)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
