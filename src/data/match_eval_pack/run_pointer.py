"""완료된 MATCH v2 학습 run 포인터의 무결성 검증."""

from __future__ import annotations

import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.match_eval_pack.format import (
    MAX_MODEL_BYTES,
    EvalPackIntegrityError,
    strict_json_loads,
)
from src.data.match_eval_pack.validation import (
    _read_bounded_regular_file,
    validate_sha256,
)

RUN_POINTER_SCHEMA = "match-v2-run-pointer/v1"
RUN_MANIFEST_SCHEMA = "match-v2-run/v1"
MAX_RUN_POINTER_BYTES = 64 * 1024
MAX_RUN_MANIFEST_BYTES = 256 * 1024
MAX_RUN_METRICS_BYTES = 16 * 1024 * 1024
RUN_DIRECTORY_PATTERN = re.compile(r"^match_v2_run_[A-Za-z0-9_-]+$")


class MatchRunIntegrityError(EvalPackIntegrityError):
    """학습 run 포인터, manifest 또는 참조 산출물의 무결성 오류."""


@dataclass(frozen=True)
class CurrentMatchRun:
    """검증된 current 포인터가 가리키는 완료 run."""

    pointer_path: Path
    run_dir: Path
    run_manifest_path: Path
    model_path: Path
    model_sha256: str
    metrics_path: Path
    metrics_sha256: str
    evaluation_pack_path: Path | None
    evaluation_pack_sha256: str | None


def _load_json_object(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, EvalPackIntegrityError) as exc:
        raise MatchRunIntegrityError(f"invalid JSON: {name}") from exc
    if not isinstance(value, dict):
        raise MatchRunIntegrityError(f"{name} must be a JSON object")
    return value


def _expect_exact_keys(value: dict[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MatchRunIntegrityError(f"invalid fields for {name}")


def _artifact_reference(
    value: Any,
    *,
    name: str,
    expected_path: str,
) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise MatchRunIntegrityError(f"{name} must be an object")
    _expect_exact_keys(value, {"path", "sha256"}, name=name)
    path = value["path"]
    if not isinstance(path, str) or path != expected_path:
        raise MatchRunIntegrityError(f"invalid path for {name}")
    digest = validate_sha256(value["sha256"], field_name=f"{name}.sha256")
    return path, digest


def _require_directory(path: Path, *, name: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MatchRunIntegrityError(f"cannot stat {name}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise MatchRunIntegrityError(f"{name} must be a real directory")


def load_current_match_run(pointer_path: Path) -> CurrentMatchRun:
    """current 포인터와 완료 manifest 및 참조 파일 SHA를 함께 검증한다."""

    pointer_path = Path(pointer_path)
    pointer_payload = _read_bounded_regular_file(
        pointer_path,
        max_bytes=MAX_RUN_POINTER_BYTES,
    )
    pointer = _load_json_object(pointer_payload, name=pointer_path.name)
    _expect_exact_keys(
        pointer,
        {
            "schema_version",
            "run_dir",
            "run_manifest",
            "model",
            "metrics",
            "evaluation_pack",
        },
        name="current pointer",
    )
    if pointer["schema_version"] != RUN_POINTER_SCHEMA:
        raise MatchRunIntegrityError("unsupported current pointer schema")

    run_name = pointer["run_dir"]
    if not isinstance(run_name, str) or RUN_DIRECTORY_PATTERN.fullmatch(run_name) is None:
        raise MatchRunIntegrityError("invalid current run directory")
    run_dir = pointer_path.parent / run_name
    _require_directory(run_dir, name="current run directory")

    manifest_rel, manifest_sha = _artifact_reference(
        pointer["run_manifest"],
        name="current pointer run_manifest",
        expected_path=f"{run_name}/run_manifest.json",
    )
    model_rel, model_sha = _artifact_reference(
        pointer["model"],
        name="current pointer model",
        expected_path=f"{run_name}/module2_lambdamart_v2.pkl",
    )
    metrics_rel, metrics_sha = _artifact_reference(
        pointer["metrics"],
        name="current pointer metrics",
        expected_path=f"{run_name}/metrics_v2.json",
    )

    pointer_pack = pointer["evaluation_pack"]
    pack_rel: str | None
    pack_sha: str | None
    if pointer_pack is None:
        pack_rel, pack_sha = None, None
    else:
        pack_rel, pack_sha = _artifact_reference(
            pointer_pack,
            name="current pointer evaluation_pack",
            expected_path=f"{run_name}/match_v2_eval_pack",
        )

    manifest_path = pointer_path.parent / manifest_rel
    manifest_payload = _read_bounded_regular_file(
        manifest_path,
        max_bytes=MAX_RUN_MANIFEST_BYTES,
        expected_sha256=manifest_sha,
    )
    manifest = _load_json_object(manifest_payload, name=manifest_path.name)
    _expect_exact_keys(
        manifest,
        {"schema_version", "status", "model", "metrics", "evaluation_pack"},
        name="run manifest",
    )
    if manifest["schema_version"] != RUN_MANIFEST_SCHEMA or manifest["status"] != "complete":
        raise MatchRunIntegrityError("run manifest is not a supported complete run")

    _manifest_model_path, manifest_model_sha = _artifact_reference(
        manifest["model"],
        name="run manifest model",
        expected_path="module2_lambdamart_v2.pkl",
    )
    _manifest_metrics_path, manifest_metrics_sha = _artifact_reference(
        manifest["metrics"],
        name="run manifest metrics",
        expected_path="metrics_v2.json",
    )
    if manifest_model_sha != model_sha or manifest_metrics_sha != metrics_sha:
        raise MatchRunIntegrityError("current pointer and run manifest disagree")

    manifest_pack = manifest["evaluation_pack"]
    if pack_rel is None:
        if manifest_pack is not None:
            raise MatchRunIntegrityError("current pointer and run manifest disagree on eval pack")
    else:
        _manifest_pack_path, manifest_pack_sha = _artifact_reference(
            manifest_pack,
            name="run manifest evaluation_pack",
            expected_path="match_v2_eval_pack",
        )
        if manifest_pack_sha != pack_sha:
            raise MatchRunIntegrityError("current pointer and run manifest disagree on eval pack")

    model_path = pointer_path.parent / model_rel
    metrics_path = pointer_path.parent / metrics_rel
    _read_bounded_regular_file(
        model_path,
        max_bytes=MAX_MODEL_BYTES,
        expected_sha256=model_sha,
    )
    _read_bounded_regular_file(
        metrics_path,
        max_bytes=MAX_RUN_METRICS_BYTES,
        expected_sha256=metrics_sha,
    )

    evaluation_pack_path = pointer_path.parent / pack_rel if pack_rel is not None else None
    if evaluation_pack_path is not None:
        _require_directory(evaluation_pack_path, name="evaluation pack")

    return CurrentMatchRun(
        pointer_path=pointer_path,
        run_dir=run_dir,
        run_manifest_path=manifest_path,
        model_path=model_path,
        model_sha256=model_sha,
        metrics_path=metrics_path,
        metrics_sha256=metrics_sha,
        evaluation_pack_path=evaluation_pack_path,
        evaluation_pack_sha256=pack_sha,
    )
