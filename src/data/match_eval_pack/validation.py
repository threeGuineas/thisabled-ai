"""외부 MATCH v2 평가팩의 무결성·구조 검증 loader."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.data.match_eval_pack.format import (
    FILE_SIZE_LIMITS,
    FRAME_FILE,
    FRAME_KEYS,
    MANIFEST_FILE,
    MAX_FEATURES,
    MAX_FRAME_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_PACK_BYTES,
    MAX_ROWS,
    MAX_USER_EMBEDDINGS_BYTES,
    PACK_FILES,
    PAIR_TRACE_FILE,
    PREPARED_TEXTS_FILE,
    QUERY_METRICS_FILE,
    SCHEMA_VERSION,
    TEST_PAIRS_FILE,
    TEST_USERS_FILE,
    TEXT_EMBEDDINGS_FILE,
    USER_EMBEDDING_KEYS,
    USER_EMBEDDINGS_FILE,
    EvalPackIntegrityError,
    canonical_json_bytes,
    pack_content_sha256,
    stage_hashes,
    strict_json_loads,
)
from src.data.matching_input import (
    ContentSignal,
    MatchingInputPolicy,
    UserSnapshot,
    validate_user_snapshot,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_JSONL_LINE_BYTES = 2 * 1024 * 1024
MAX_JSONL_RECORDS = 200_000
NDCG_RANGE_ATOL = 1e-12


class LoadedMatchEvalPack:
    """검증을 통과한 평가팩과 재생성에 필요한 canonical record."""

    def __init__(
        self,
        *,
        path: Path,
        manifest: dict[str, Any],
        frame: dict[str, np.ndarray],
        user_embeddings: dict[str, np.ndarray],
        text_embeddings: np.ndarray,
        user_records: list[dict[str, Any]],
        pair_records: list[dict[str, Any]],
        pair_trace_records: list[dict[str, Any]],
        text_records: list[dict[str, Any]],
        query_metric_records: list[dict[str, Any]],
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.frame = frame
        self.user_embeddings = user_embeddings
        self.text_embeddings = text_embeddings
        self.user_records = user_records
        self.pair_records = pair_records
        self.pair_trace_records = pair_trace_records
        self.text_records = text_records
        self.query_metric_records = query_metric_records


def validate_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise EvalPackIntegrityError(f"invalid SHA-256: {field_name}")
    return value


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    """한 file descriptor의 bounded bytes를 hash와 parser에 함께 사용한다."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise EvalPackIntegrityError(f"cannot stat file: {path.name}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise EvalPackIntegrityError(f"pack member must be a regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvalPackIntegrityError(f"cannot open regular file: {path.name}") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise EvalPackIntegrityError(f"pack member changed type: {path.name}")
            if opened.st_size > max_bytes:
                raise EvalPackIntegrityError(f"file size limit exceeded: {path.name}")
            if expected_size is not None and opened.st_size != expected_size:
                raise EvalPackIntegrityError(f"file size mismatch: {path.name}")
            while chunk := handle.read(min(1024 * 1024, max_bytes + 1 - total)):
                chunks.append(chunk)
                digest.update(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise EvalPackIntegrityError(f"file size limit exceeded: {path.name}")
    except EvalPackIntegrityError:
        raise
    except OSError as exc:
        raise EvalPackIntegrityError(f"cannot read file: {path.name}") from exc
    if expected_size is not None and total != expected_size:
        raise EvalPackIntegrityError(f"file size changed while reading: {path.name}")
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise EvalPackIntegrityError(f"file SHA-256 mismatch: {path.name}")
    return b"".join(chunks)


def _load_json(payload: bytes, *, name: str) -> Any:
    try:
        return strict_json_loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise EvalPackIntegrityError(f"invalid UTF-8: {name}") from exc


def _load_jsonl(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(io.BytesIO(payload), start=1):
        if len(raw_line) > MAX_JSONL_LINE_BYTES:
            raise EvalPackIntegrityError(f"JSONL row too large: {name}:{line_number}")
        if not raw_line.endswith(b"\n"):
            raise EvalPackIntegrityError(f"unterminated JSONL row: {name}:{line_number}")
        try:
            value = strict_json_loads(raw_line.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise EvalPackIntegrityError(f"invalid UTF-8: {name}:{line_number}") from exc
        if not isinstance(value, dict):
            raise EvalPackIntegrityError(f"JSONL row must be an object: {name}")
        records.append(value)
        if len(records) > MAX_JSONL_RECORDS:
            raise EvalPackIntegrityError(f"too many JSONL rows: {name}")
    return records


def _safe_load_npz(
    payload: bytes,
    *,
    name: str,
    expected_keys: Sequence[str],
    max_uncompressed_bytes: int,
) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise EvalPackIntegrityError(f"duplicate NPZ member: {name}")
            if names != [f"{key}.npy" for key in expected_keys]:
                raise EvalPackIntegrityError(f"unexpected NPZ members or order: {name}")
            if sum(info.file_size for info in archive.infolist()) > max_uncompressed_bytes:
                raise EvalPackIntegrityError(f"NPZ uncompressed size limit exceeded: {name}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            return {key: np.array(loaded[key], copy=True) for key in expected_keys}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise EvalPackIntegrityError(f"cannot read NPZ: {name}") from exc


def _safe_load_npy(payload: bytes, *, name: str) -> np.ndarray:
    try:
        return np.array(np.load(io.BytesIO(payload), allow_pickle=False), copy=True)
    except (OSError, ValueError) as exc:
        raise EvalPackIntegrityError(f"cannot read NPY: {name}") from exc


def _expect_dtype(array: np.ndarray, dtype: str, *, name: str) -> None:
    if array.dtype.str != dtype:
        raise EvalPackIntegrityError(f"unexpected dtype for {name}: {array.dtype.str}")


def _validate_frame(
    manifest: Mapping[str, Any],
    frame: Mapping[str, np.ndarray],
) -> None:
    expected_dtypes = {
        "x_test": "<f4",
        "y_test": "<i4",
        "group_sizes": "<i4",
        "group_offsets": "<i8",
        "raw_scores": "<f8",
        "latent_scores": "<f8",
    }
    for name, dtype in expected_dtypes.items():
        _expect_dtype(frame[name], dtype, name=name)
    for name in ("group_query_ids", "row_query_ids", "candidate_ids"):
        if frame[name].dtype.kind != "U":
            raise EvalPackIntegrityError(f"{name} must be a Unicode array")

    columns = manifest["evaluation"]["columns"]
    x = frame["x_test"]
    y = frame["y_test"]
    groups = frame["group_sizes"]
    offsets = frame["group_offsets"]
    n_rows = x.shape[0] if x.ndim == 2 else -1
    if n_rows <= 0 or n_rows > MAX_ROWS or x.shape[1] != len(columns) or x.shape[1] > MAX_FEATURES:
        raise EvalPackIntegrityError("invalid x_test shape")
    if (
        y.shape != (n_rows,)
        or frame["raw_scores"].shape != (n_rows,)
        or frame["latent_scores"].shape != (n_rows,)
    ):
        raise EvalPackIntegrityError("row-aligned frame arrays have different lengths")
    if groups.ndim != 1 or len(groups) == 0 or np.any(groups <= 0):
        raise EvalPackIntegrityError("invalid group sizes")
    expected_offsets = np.concatenate(
        (np.asarray([0], dtype="<i8"), np.cumsum(groups, dtype=np.int64).astype("<i8"))
    )
    if not np.array_equal(offsets, expected_offsets) or int(offsets[-1]) != n_rows:
        raise EvalPackIntegrityError("invalid group offsets")
    if frame["group_query_ids"].shape != (len(groups),):
        raise EvalPackIntegrityError("group query ID count mismatch")
    if frame["row_query_ids"].shape != (n_rows,) or frame["candidate_ids"].shape != (n_rows,):
        raise EvalPackIntegrityError("row ID count mismatch")
    query_ids = frame["group_query_ids"].tolist()
    if len(query_ids) != len(set(query_ids)) or any(not value for value in query_ids):
        raise EvalPackIntegrityError("group query IDs must be non-empty and unique")
    if not np.array_equal(
        frame["row_query_ids"],
        np.repeat(frame["group_query_ids"], groups.astype(np.int64)),
    ):
        raise EvalPackIntegrityError("row query IDs do not match group boundaries")
    if np.any((y < 0) | (y > 3)):
        raise EvalPackIntegrityError("labels outside [0, 3]")
    for name in ("x_test", "raw_scores", "latent_scores"):
        if not np.isfinite(frame[name]).all():
            raise EvalPackIntegrityError(f"non-finite array: {name}")


def _validate_embeddings(
    text_embeddings: np.ndarray,
    user_embeddings: Mapping[str, np.ndarray],
) -> None:
    _expect_dtype(text_embeddings, "<f4", name="text_embeddings")
    if text_embeddings.ndim != 2 or text_embeddings.shape[0] == 0:
        raise EvalPackIntegrityError("invalid text embedding shape")
    for start in range(0, len(text_embeddings), 4096):
        if not np.isfinite(text_embeddings[start : start + 4096]).all():
            raise EvalPackIntegrityError("non-finite text embeddings")
    n_users = len(user_embeddings["user_ids"])
    dimension = text_embeddings.shape[1]
    if user_embeddings["user_ids"].dtype.kind != "U" or n_users == 0:
        raise EvalPackIntegrityError("invalid user IDs")
    for prefix in ("profile", "authored", "liked", "effective"):
        matrix = user_embeddings[f"{prefix}_embeddings"]
        available = user_embeddings[f"{prefix}_available"]
        _expect_dtype(matrix, "<f4", name=f"{prefix}_embeddings")
        if matrix.shape != (n_users, dimension) or available.shape != (n_users,):
            raise EvalPackIntegrityError(f"invalid {prefix} embedding shape")
        if available.dtype != np.dtype(np.bool_) or not np.isfinite(matrix).all():
            raise EvalPackIntegrityError(f"invalid {prefix} embedding values")
    for name in ("authored_count", "liked_count"):
        _expect_dtype(user_embeddings[name], "<i4", name=name)
        if user_embeddings[name].shape != (n_users,) or np.any(user_embeddings[name] < 0):
            raise EvalPackIntegrityError(f"invalid {name}")


def _expect_exact_keys(record: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(record) != expected:
        raise EvalPackIntegrityError(f"invalid fields for {name}")


def _is_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool) and isinstance(value, int | float) and bool(np.isfinite(value))
    )


def _parse_timestamp(value: Any, *, optional: bool = False) -> datetime | None:
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


def _validated_policy(provenance: Mapping[str, Any]) -> MatchingInputPolicy:
    raw = provenance.get("policy")
    expected_fields = {field.name for field in fields(MatchingInputPolicy)}
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise EvalPackIntegrityError("invalid policy provenance")
    values = dict(raw)
    positive_integer_fields = {
        "bio_max_chars",
        "max_tags",
        "max_tag_chars",
        "authored_lookback_days",
        "liked_lookback_days",
        "max_authored_items",
        "max_liked_items",
        "max_content_chars",
        "max_candidates",
        "embedding_batch_size",
        "rejection_cooldown_days",
        "max_reasons",
    }
    if any(
        not _is_integer(values[name]) or values[name] <= 0 for name in positive_integer_fields
    ) or any(
        not _is_finite_number(values[name]) or not 0.0 <= float(values[name]) <= 1.0
        for name in ("content_reason_min", "profile_reason_min")
    ):
        raise EvalPackIntegrityError("invalid policy limits")
    for name in ("allowed_tag_ids", "allowed_ui_modes", "allowed_content_sources"):
        value = values[name]
        if value is None and name == "allowed_tag_ids":
            continue
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))
        ):
            raise EvalPackIntegrityError(f"invalid policy set: {name}")
        values[name] = frozenset(value)
    try:
        return MatchingInputPolicy(**values)
    except (TypeError, ValueError) as exc:
        raise EvalPackIntegrityError("invalid policy values") from exc


def _validated_content(record: Any, *, policy: MatchingInputPolicy) -> ContentSignal:
    if not isinstance(record, dict):
        raise EvalPackIntegrityError("content record must be an object")
    _expect_exact_keys(
        record,
        {
            "content_id",
            "source_type",
            "text",
            "created_at",
            "is_deleted",
            "is_accessible",
            "is_blocked_author",
            "is_like_active",
        },
        name="content record",
    )
    if (
        not isinstance(record["content_id"], str)
        or not record["content_id"]
        or not isinstance(record["source_type"], str)
        or not isinstance(record["text"], str)
        or any(
            not isinstance(record[name], bool)
            for name in (
                "is_deleted",
                "is_accessible",
                "is_blocked_author",
                "is_like_active",
            )
        )
    ):
        raise EvalPackIntegrityError("invalid content record values")
    created_at = _parse_timestamp(record["created_at"])
    if record["source_type"] not in policy.allowed_content_sources:
        raise EvalPackIntegrityError("content source is outside the recorded policy")
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


def _validate_user_record(
    record: dict[str, Any],
    *,
    expected_position: int,
    policy: MatchingInputPolicy,
) -> None:
    _expect_exact_keys(
        record,
        {"user_position", "user_id", "snapshot", "latent"},
        name="test user",
    )
    if record["user_position"] != expected_position:
        raise EvalPackIntegrityError("test user positions are not contiguous")
    if not isinstance(record["user_id"], str) or not record["user_id"]:
        raise EvalPackIntegrityError("invalid test user ID")
    snapshot = record["snapshot"]
    latent = record["latent"]
    if not isinstance(snapshot, dict) or not isinstance(latent, dict):
        raise EvalPackIntegrityError("canonical test user record is incomplete")
    _expect_exact_keys(
        snapshot,
        {
            "user_id",
            "bio",
            "tag_ids",
            "age_years",
            "age_band",
            "ui_mode",
            "authored_items",
            "liked_items",
        },
        name="user snapshot",
    )
    _expect_exact_keys(
        latent,
        {"user_id", "interest_weights", "age", "social_cluster"},
        name="latent user",
    )
    if (
        snapshot["user_id"] != record["user_id"]
        or latent["user_id"] != record["user_id"]
        or not isinstance(snapshot["bio"], str)
        or not isinstance(snapshot["tag_ids"], list)
        or any(not isinstance(tag, str) or not tag for tag in snapshot["tag_ids"])
        or (snapshot["age_years"] is not None and not _is_integer(snapshot["age_years"]))
        or (snapshot["age_band"] is not None and not isinstance(snapshot["age_band"], str))
        or not isinstance(snapshot["ui_mode"], str)
        or not isinstance(snapshot["authored_items"], list)
        or not isinstance(snapshot["liked_items"], list)
    ):
        raise EvalPackIntegrityError("invalid canonical user snapshot")
    authored = tuple(_validated_content(item, policy=policy) for item in snapshot["authored_items"])
    liked = tuple(_validated_content(item, policy=policy) for item in snapshot["liked_items"])
    try:
        validate_user_snapshot(
            UserSnapshot(
                user_id=snapshot["user_id"],
                bio=snapshot["bio"],
                tag_ids=tuple(snapshot["tag_ids"]),
                age_years=snapshot["age_years"],
                age_band=snapshot["age_band"],
                ui_mode=snapshot["ui_mode"],
                authored_items=authored,
                liked_items=liked,
            ),
            policy,
        )
    except (TypeError, ValueError) as exc:
        raise EvalPackIntegrityError("canonical user snapshot violates policy") from exc

    weights = latent["interest_weights"]
    expected_dimension = len(policy.allowed_tag_ids) if policy.allowed_tag_ids is not None else None
    if (
        not isinstance(weights, list)
        or not weights
        or (expected_dimension is not None and len(weights) != expected_dimension)
        or any(not _is_finite_number(value) or float(value) < 0 for value in weights)
        or not np.isclose(sum(float(value) for value in weights), 1.0, rtol=0.0, atol=1e-9)
        or not _is_integer(latent["age"])
        or not 14 <= latent["age"] <= 120
        or not _is_integer(latent["social_cluster"])
        or latent["social_cluster"] < 0
    ):
        raise EvalPackIntegrityError("invalid latent user values")
    if snapshot["age_years"] is not None and snapshot["age_years"] != latent["age"]:
        raise EvalPackIntegrityError("observed and latent age differ")


def _validate_pair_and_trace(
    pair: dict[str, Any],
    trace: dict[str, Any],
    *,
    known_user_ids: set[str],
) -> None:
    _expect_exact_keys(
        pair,
        {
            "pair_id",
            "query_position",
            "candidate_position",
            "query_id",
            "candidate_id",
            "label",
            "latent_score",
            "relationship",
        },
        name="test pair",
    )
    _expect_exact_keys(
        trace,
        {"pair_id", "filter_status", "exclusion_reason", "feature_row_index"},
        name="pair trace",
    )
    relationship = pair["relationship"]
    if not isinstance(relationship, dict):
        raise EvalPackIntegrityError("pair relationship must be an object")
    _expect_exact_keys(
        relationship,
        {
            "candidate_id",
            "common_friend_count",
            "blocked_either_direction",
            "already_friends",
            "last_rejected_at",
        },
        name="pair relationship",
    )
    identity = {"query_id": pair["query_id"], "candidate_id": pair["candidate_id"]}
    expected_pair_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    if (
        pair["pair_id"] != expected_pair_id
        or trace["pair_id"] != expected_pair_id
        or pair["query_id"] not in known_user_ids
        or pair["candidate_id"] not in known_user_ids
        or pair["query_id"] == pair["candidate_id"]
        or relationship["candidate_id"] != pair["candidate_id"]
        or not _is_integer(pair["query_position"])
        or pair["query_position"] < 0
        or not _is_integer(pair["candidate_position"])
        or pair["candidate_position"] < 0
        or not _is_integer(pair["label"])
        or not 0 <= pair["label"] <= 3
        or not _is_finite_number(pair["latent_score"])
        or not 0.0 <= float(pair["latent_score"]) <= 1.0
        or not _is_integer(relationship["common_friend_count"])
        or relationship["common_friend_count"] < 0
        or not isinstance(relationship["blocked_either_direction"], bool)
        or not isinstance(relationship["already_friends"], bool)
    ):
        raise EvalPackIntegrityError("invalid test pair values")
    _parse_timestamp(relationship["last_rejected_at"], optional=True)
    status = trace["filter_status"]
    if status == "kept":
        if (
            trace["exclusion_reason"] is not None
            or not _is_integer(trace["feature_row_index"])
            or trace["feature_row_index"] < 0
        ):
            raise EvalPackIntegrityError("invalid kept pair trace")
    elif status == "excluded":
        if (
            not isinstance(trace["exclusion_reason"], str)
            or not trace["exclusion_reason"]
            or trace["feature_row_index"] is not None
        ):
            raise EvalPackIntegrityError("invalid excluded pair trace")
    else:
        raise EvalPackIntegrityError("invalid pair filter status")


def _validate_records(
    *,
    frame: Mapping[str, np.ndarray],
    text_embeddings: np.ndarray,
    user_embeddings: Mapping[str, np.ndarray],
    policy: MatchingInputPolicy,
    user_records: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
    pair_trace_records: list[dict[str, Any]],
    text_records: list[dict[str, Any]],
    query_records: list[dict[str, Any]],
) -> None:
    expected_user_ids = user_embeddings["user_ids"].tolist()
    if len(user_records) != len(expected_user_ids):
        raise EvalPackIntegrityError("test user and user embedding count mismatch")
    if [record.get("user_id") for record in user_records] != expected_user_ids:
        raise EvalPackIntegrityError("test user order differs from user embeddings")
    for position, record in enumerate(user_records):
        _validate_user_record(record, expected_position=position, policy=policy)

    if len(text_records) != text_embeddings.shape[0]:
        raise EvalPackIntegrityError("prepared text and embedding row count mismatch")
    known_user_ids = set(expected_user_ids)
    component_indices: dict[tuple[str, str], int] = {}
    for row, record in enumerate(text_records):
        _expect_exact_keys(
            record,
            {
                "text_row",
                "user_id",
                "component",
                "component_index",
                "text",
                "text_sha256",
            },
            name="prepared text",
        )
        text = record.get("text")
        component_key = (record.get("user_id"), record.get("component"))
        expected_component_index = component_indices.get(component_key, 0)
        if (
            record.get("text_row") != row
            or record.get("user_id") not in known_user_ids
            or record.get("component") not in {"profile", "authored", "liked"}
            or record.get("component_index") != expected_component_index
            or not isinstance(text, str)
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != record.get("text_sha256")
        ):
            raise EvalPackIntegrityError("prepared text trace is inconsistent")
        component_indices[component_key] = expected_component_index + 1

    if len(pair_records) != len(pair_trace_records) or [
        record.get("pair_id") for record in pair_records
    ] != [record.get("pair_id") for record in pair_trace_records]:
        raise EvalPackIntegrityError("pair input and trace order mismatch")
    expected_candidate_position: dict[int, int] = {}
    last_query_position = -1
    for pair, trace in zip(pair_records, pair_trace_records, strict=True):
        _validate_pair_and_trace(pair, trace, known_user_ids=known_user_ids)
        query_position = pair["query_position"]
        if query_position < last_query_position or query_position > last_query_position + 1:
            raise EvalPackIntegrityError("pair query positions are not contiguous")
        expected_position = expected_candidate_position.get(query_position, 0)
        if pair["candidate_position"] != expected_position:
            raise EvalPackIntegrityError("pair candidate positions are not contiguous")
        expected_candidate_position[query_position] = expected_position + 1
        last_query_position = query_position
    kept_pairs = [
        (pair, trace)
        for pair, trace in zip(pair_records, pair_trace_records, strict=True)
        if trace.get("filter_status") == "kept"
    ]
    n_rows = frame["x_test"].shape[0]
    if len(kept_pairs) != n_rows:
        raise EvalPackIntegrityError("kept pair and feature row count mismatch")
    if [trace.get("feature_row_index") for _pair, trace in kept_pairs] != list(range(n_rows)):
        raise EvalPackIntegrityError("kept pair feature indices are not contiguous")
    if [pair.get("query_id") for pair, _trace in kept_pairs] != frame["row_query_ids"].tolist():
        raise EvalPackIntegrityError("kept pair query order differs from frame")
    if [pair.get("candidate_id") for pair, _trace in kept_pairs] != frame["candidate_ids"].tolist():
        raise EvalPackIntegrityError("kept pair candidate order differs from frame")
    if [pair.get("label") for pair, _trace in kept_pairs] != frame["y_test"].tolist():
        raise EvalPackIntegrityError("kept pair labels differ from frame")
    if not np.array_equal(
        np.asarray(
            [pair.get("latent_score") for pair, _trace in kept_pairs],
            dtype=np.float64,
        ),
        frame["latent_scores"],
    ):
        raise EvalPackIntegrityError("kept pair latent scores differ from frame")
    pair_ids = [record.get("pair_id") for record in pair_records]
    if len(pair_ids) != len(set(pair_ids)) or any(
        not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None for value in pair_ids
    ):
        raise EvalPackIntegrityError("pair IDs must be unique SHA-256 values")

    if len(query_records) != len(frame["group_sizes"]):
        raise EvalPackIntegrityError("query metric and group count mismatch")
    if not any(record.get("metric_included") is True for record in query_records):
        raise EvalPackIntegrityError("evaluation pack has no scorable queries")
    for index, record in enumerate(query_records):
        _expect_exact_keys(
            record,
            {
                "query_id",
                "group_size",
                "metric_included",
                "ndcg@5",
                "ndcg@10",
                "pre_filter_count",
            },
            name="query metric",
        )
        size = int(frame["group_sizes"][index])
        query_id = str(frame["group_query_ids"][index])
        expected_included = (
            size >= 2
            and max(
                pair["label"]
                for pair, trace in kept_pairs
                if pair["query_id"] == query_id and trace["filter_status"] == "kept"
            )
            > 0
        )
        if (
            record.get("query_id") != query_id
            or record.get("group_size") != size
            or not isinstance(record.get("metric_included"), bool)
            or record["metric_included"] != expected_included
            or not _is_integer(record.get("pre_filter_count"))
            or record["pre_filter_count"]
            != sum(pair["query_id"] == query_id for pair in pair_records)
        ):
            raise EvalPackIntegrityError("query metric trace differs from frame boundaries")
        for k in (5, 10):
            value = record.get(f"ndcg@{k}")
            if record["metric_included"] and (
                not _is_finite_number(value)
                or not -NDCG_RANGE_ATOL <= float(value) <= 1.0 + NDCG_RANGE_ATOL
            ):
                raise EvalPackIntegrityError("invalid stored query metric")
            if not record["metric_included"] and value is not None:
                raise EvalPackIntegrityError("excluded query must not store NDCG")


def load_match_eval_pack(
    pack_dir: Path,
    *,
    expected_pack_sha256: str,
) -> LoadedMatchEvalPack:
    """외부 SHA로 고정한 pack을 단일 byte snapshot에서 검증한다."""

    trusted_pack_sha = validate_sha256(
        expected_pack_sha256,
        field_name="expected_pack_sha256",
    )
    pack_dir = Path(pack_dir)
    if pack_dir.is_symlink() or not pack_dir.is_dir():
        raise EvalPackIntegrityError("evaluation pack directory does not exist")
    expected_names = {MANIFEST_FILE, *PACK_FILES}
    try:
        members = list(pack_dir.iterdir())
    except OSError as exc:
        raise EvalPackIntegrityError("cannot enumerate evaluation pack") from exc
    if {path.name for path in members} != expected_names:
        raise EvalPackIntegrityError("evaluation pack has missing or unexpected files")
    member_stats: dict[str, os.stat_result] = {}
    for path in members:
        try:
            member_stat = path.lstat()
        except OSError as exc:
            raise EvalPackIntegrityError(f"cannot stat pack member: {path.name}") from exc
        if not stat.S_ISREG(member_stat.st_mode):
            raise EvalPackIntegrityError(f"pack member must be a regular file: {path.name}")
        member_stats[path.name] = member_stat
    if sum(item.st_size for item in member_stats.values()) > MAX_PACK_BYTES:
        raise EvalPackIntegrityError("evaluation pack aggregate size limit exceeded")

    manifest_payload = _read_bounded_regular_file(
        pack_dir / MANIFEST_FILE,
        max_bytes=MAX_MANIFEST_BYTES,
        expected_size=member_stats[MANIFEST_FILE].st_size,
    )
    manifest = _load_json(manifest_payload, name=MANIFEST_FILE)
    required_manifest_fields = {
        "schema_version",
        "provenance_sha256",
        "model",
        "evaluation",
        "stages",
        "files",
        "provenance",
        "pack_content_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != required_manifest_fields
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise EvalPackIntegrityError("unsupported evaluation pack schema")
    claimed_pack_sha = validate_sha256(
        manifest.get("pack_content_sha256"),
        field_name="pack_content_sha256",
    )
    if claimed_pack_sha != trusted_pack_sha:
        raise EvalPackIntegrityError("evaluation pack SHA-256 differs from external trust anchor")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(PACK_FILES):
        raise EvalPackIntegrityError("manifest file list mismatch")
    expected_total = len(manifest_payload)
    for name in PACK_FILES:
        entry = files[name]
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes"}:
            raise EvalPackIntegrityError(f"invalid file metadata: {name}")
        expected_size = entry["bytes"]
        if (
            not _is_integer(expected_size)
            or expected_size <= 0
            or expected_size > FILE_SIZE_LIMITS[name]
            or expected_size != member_stats[name].st_size
        ):
            raise EvalPackIntegrityError(f"file size mismatch or limit exceeded: {name}")
        validate_sha256(
            entry["sha256"],
            field_name=f"files.{name}.sha256",
        )
        expected_total += expected_size
    if expected_total > MAX_PACK_BYTES:
        raise EvalPackIntegrityError("evaluation pack aggregate size limit exceeded")

    evaluation = manifest.get("evaluation")
    model = manifest.get("model")
    provenance = manifest.get("provenance")
    if not isinstance(evaluation, dict) or not isinstance(model, dict):
        raise EvalPackIntegrityError("manifest model/evaluation section missing")
    _expect_exact_keys(
        model,
        {"filename", "sha256", "columns", "n_features"},
        name="model manifest",
    )
    _expect_exact_keys(
        evaluation,
        {
            "columns",
            "rows",
            "groups",
            "scored_groups",
            "text_rows",
            "embedding_dimension",
            "metrics",
            "metric_implementation",
        },
        name="evaluation manifest",
    )
    columns = evaluation.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or len(columns) != len(set(columns))
        or any(not isinstance(column, str) or not column for column in columns)
    ):
        raise EvalPackIntegrityError("manifest evaluation columns missing")
    if (
        not isinstance(model["filename"], str)
        or not model["filename"]
        or Path(model["filename"]).name != model["filename"]
        or model.get("columns") != columns
        or model.get("n_features") != len(columns)
    ):
        raise EvalPackIntegrityError("manifest model columns or filename differ")
    validate_sha256(model.get("sha256"), field_name="model.sha256")
    required_provenance = {
        "run",
        "policy",
        "config",
        "corpus",
        "encoder",
        "runtime",
        "git",
        "source_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise EvalPackIntegrityError("manifest provenance section missing")
    for name in ("run", "config", "corpus", "encoder", "runtime", "git", "source_sha256"):
        if not isinstance(provenance[name], dict):
            raise EvalPackIntegrityError(f"invalid provenance object: {name}")
    if "as_of" not in provenance["run"]:
        raise EvalPackIntegrityError("run provenance must include as_of")
    _parse_timestamp(provenance["run"]["as_of"])
    policy = _validated_policy(provenance)
    source_hashes = provenance["source_sha256"]
    if any(
        not isinstance(path, str)
        or not path
        or not isinstance(value, str)
        or SHA256_PATTERN.fullmatch(value) is None
        for path, value in source_hashes.items()
    ):
        raise EvalPackIntegrityError("invalid source provenance hashes")
    encoder_provenance = provenance["encoder"]
    if (
        not _is_integer(encoder_provenance.get("output_dimension"))
        or encoder_provenance["output_dimension"] <= 0
        or not _is_integer(encoder_provenance.get("batch_size"))
        or encoder_provenance["batch_size"] <= 0
    ):
        raise EvalPackIntegrityError("invalid encoder provenance")
    actual_provenance_sha = hashlib.sha256(canonical_json_bytes(provenance)).hexdigest()
    if (
        validate_sha256(
            manifest.get("provenance_sha256"),
            field_name="provenance_sha256",
        )
        != actual_provenance_sha
    ):
        raise EvalPackIntegrityError("provenance SHA-256 mismatch")

    def read_member(name: str) -> bytes:
        return _read_bounded_regular_file(
            pack_dir / name,
            max_bytes=FILE_SIZE_LIMITS[name],
            expected_size=files[name]["bytes"],
            expected_sha256=files[name]["sha256"],
        )

    frame = _safe_load_npz(
        read_member(FRAME_FILE),
        name=FRAME_FILE,
        expected_keys=FRAME_KEYS,
        max_uncompressed_bytes=MAX_FRAME_BYTES,
    )
    text_embeddings = _safe_load_npy(
        read_member(TEXT_EMBEDDINGS_FILE),
        name=TEXT_EMBEDDINGS_FILE,
    )
    user_embeddings = _safe_load_npz(
        read_member(USER_EMBEDDINGS_FILE),
        name=USER_EMBEDDINGS_FILE,
        expected_keys=USER_EMBEDDING_KEYS,
        max_uncompressed_bytes=MAX_USER_EMBEDDINGS_BYTES,
    )
    _validate_frame(manifest, frame)
    _validate_embeddings(text_embeddings, user_embeddings)

    user_records = _load_jsonl(
        read_member(TEST_USERS_FILE),
        name=TEST_USERS_FILE,
    )
    pair_records = _load_jsonl(
        read_member(TEST_PAIRS_FILE),
        name=TEST_PAIRS_FILE,
    )
    pair_trace_records = _load_jsonl(
        read_member(PAIR_TRACE_FILE),
        name=PAIR_TRACE_FILE,
    )
    text_records = _load_jsonl(
        read_member(PREPARED_TEXTS_FILE),
        name=PREPARED_TEXTS_FILE,
    )
    query_records = _load_jsonl(
        read_member(QUERY_METRICS_FILE),
        name=QUERY_METRICS_FILE,
    )
    _validate_records(
        frame=frame,
        text_embeddings=text_embeddings,
        user_embeddings=user_embeddings,
        policy=policy,
        user_records=user_records,
        pair_records=pair_records,
        pair_trace_records=pair_trace_records,
        text_records=text_records,
        query_records=query_records,
    )

    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"ndcg@5", "ndcg@10"}:
        raise EvalPackIntegrityError("manifest metrics missing")
    if any(
        not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0 for value in metrics.values()
    ):
        raise EvalPackIntegrityError("manifest metrics invalid")
    for key in ("ndcg@5", "ndcg@10"):
        per_query = [
            float(record[key]) for record in query_records if record["metric_included"] is True
        ]
        if not np.isclose(
            float(np.mean(per_query)),
            float(metrics[key]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise EvalPackIntegrityError(f"stored global and query metrics differ: {key}")
    count_fields = ("rows", "groups", "scored_groups", "text_rows", "embedding_dimension")
    if any(not _is_integer(evaluation.get(name)) or evaluation[name] <= 0 for name in count_fields):
        raise EvalPackIntegrityError("manifest evaluation counts invalid")
    if (
        evaluation.get("rows") != frame["x_test"].shape[0]
        or evaluation.get("groups") != len(frame["group_sizes"])
        or evaluation.get("scored_groups")
        != sum(record["metric_included"] is True for record in query_records)
        or evaluation.get("text_rows") != text_embeddings.shape[0]
        or evaluation.get("embedding_dimension") != text_embeddings.shape[1]
        or encoder_provenance["output_dimension"] != text_embeddings.shape[1]
        or evaluation.get("metric_implementation") != "sklearn.metrics.ndcg_score/query-mean"
    ):
        raise EvalPackIntegrityError("manifest evaluation counts or implementation differ")

    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise EvalPackIntegrityError("manifest stages missing")
    expected_stages = stage_hashes(
        files=files,
        frame=frame,
        user_embeddings=user_embeddings,
        text_embeddings=text_embeddings,
        columns=columns,
        model_sha256=model["sha256"],
        metrics={key: float(value) for key, value in metrics.items()},
    )
    if set(stages) != set(expected_stages):
        raise EvalPackIntegrityError("manifest stage list differs")
    for name, value in expected_stages.items():
        validate_sha256(stages.get(name), field_name=f"stages.{name}")
        if stages[name] != value:
            raise EvalPackIntegrityError(f"semantic stage hash mismatch: {name}")
    expected_content_sha = pack_content_sha256(
        stages=stages,
        model=model,
        files=files,
        evaluation=evaluation,
        provenance_sha256=manifest["provenance_sha256"],
    )
    if claimed_pack_sha != expected_content_sha:
        raise EvalPackIntegrityError("pack content SHA-256 mismatch")

    return LoadedMatchEvalPack(
        path=pack_dir,
        manifest=manifest,
        frame=frame,
        user_embeddings=user_embeddings,
        text_embeddings=text_embeddings,
        user_records=user_records,
        pair_records=pair_records,
        pair_trace_records=pair_trace_records,
        text_records=text_records,
        query_metric_records=query_records,
    )
