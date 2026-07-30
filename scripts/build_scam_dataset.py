"""SAFE v7 사기 학습 데이터 생성 파이프라인 — 합성(A3) → LLM 검수(B2) → 저장.

실행: GEMINI_API_KEY가 .env 또는 환경변수에 있어야 한다.
    python scripts/build_scam_dataset.py --per-subtype 40 \
      --forbidden tests/fixtures/safe_blind_v1.jsonl
기본 결과: outputs/10_thisabled-ai/데이터/YYYYMMDD_thisabled_사기학습후보.jsonl.
canonical data/synthetic/scam/train.jsonl 승격은 사람 승인 단계에서만 별도로 수행한다.

합성/검수 로직은 src/data/scam_synthesis.py에 있고 여기서는 실제 Gemini를 주입해 실행만 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.llm_client import (  # noqa: E402
    GeminiAPIError,
    GeminiClient,
    GeminiClientError,
    RequestPacer,
)
from src.data.scam_synthesis import (  # noqa: E402
    BENIGN_SUBTYPES,
    LABEL_REVIEW_REVISION,
    QUALITY_REVIEW_REVISION,
    SCAM_SUBTYPES,
    SOURCE,
    LLMOutputError,
    contains_sensitive_identifier,
    duplicate_text_key,
    filter_forbidden,
    normalize_generated_text,
    review_quality,
    synthesize_scam,
    verify_labels,
)

_RUN_DATE = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
DEFAULT_OUT = (
    ROOT / "outputs" / "10_thisabled-ai" / "데이터" / f"{_RUN_DATE}_thisabled_사기학습후보.jsonl"
)
CANONICAL_TRAIN_OUT = ROOT / "data" / "synthetic" / "scam" / "train.jsonl"
SYNTHESIS_CACHE_SCHEMA_VERSION = 2
VERIFICATION_CHECKPOINT_SCHEMA_VERSION = 3
SYNTHESIS_PROMPT_REVISION = "scam-v7-subtype-logic-abstain-v4"
MODELS_WITHOUT_SAMPLING_PARAMETERS = {"gemini-3.5-flash-lite", "gemini-3.6-flash"}


class DatasetBuildError(RuntimeError):
    """불완전 데이터 저장을 막기 위한 빌드 실패."""


def _load_api_key() -> str:
    if not os.getenv("GEMINI_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise DatasetBuildError("GEMINI_API_KEY 없음 — .env 또는 환경변수 설정 필요")
    return key


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("양수여야 합니다")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("유한한 0 이상 값이어야 합니다")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("0 초과 1 이하여야 합니다")
    return parsed


def _paths_refer_to_same_file(left: Path, right: Path) -> bool:
    try:
        if left.resolve(strict=False) == right.resolve(strict=False):
            return True
        if left.exists() and right.exists():
            return os.path.samefile(left, right)
    except OSError:
        return os.path.abspath(left) == os.path.abspath(right)
    return False


def _paths_overlap(left: Path, right: Path) -> bool:
    if _paths_refer_to_same_file(left, right):
        return True
    try:
        left_resolved = left.resolve(strict=False)
        right_resolved = right.resolve(strict=False)
    except OSError:
        left_resolved = Path(os.path.abspath(left))
        right_resolved = Path(os.path.abspath(right))
    return left_resolved in right_resolved.parents or right_resolved in left_resolved.parents


def _validate_path_separation(
    *,
    out: Path,
    synthesis_cache: Path | None,
    verification_report: Path | None,
    forbidden: Sequence[str],
) -> None:
    outputs = [("out", out)]
    if synthesis_cache is not None:
        outputs.append(("synthesis-cache", synthesis_cache))
    if verification_report is not None:
        outputs.append(("verification-report", verification_report))

    for index, (left_name, left_path) in enumerate(outputs):
        if left_path.exists() and left_path.is_dir():
            raise DatasetBuildError(f"산출물 경로가 디렉터리임: --{left_name}={left_path}")
        for right_name, right_path in outputs[index + 1 :]:
            if _paths_overlap(left_path, right_path):
                raise DatasetBuildError(f"산출물 경로 충돌: --{left_name}와 --{right_name}")
        for forbidden_path in map(Path, forbidden):
            if _paths_overlap(left_path, forbidden_path):
                raise DatasetBuildError(
                    f"산출물과 forbidden 입력 경로 충돌: --{left_name}={left_path}"
                )


def _load_forbidden_texts(
    paths: Sequence[str],
) -> tuple[list[str], list[dict[str, object]]]:
    """비싼 LLM 호출 전에 forbidden JSONL을 검증하고 텍스트를 읽는다."""

    texts: list[str] = []
    manifest: list[dict[str, object]] = []
    for raw_path in paths:
        path = Path(raw_path)
        rows_before = len(texts)
        try:
            raw_bytes = path.read_bytes()
            lines = raw_bytes.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise DatasetBuildError(f"forbidden UTF-8 오류: {path}") from exc
        except OSError as exc:
            raise DatasetBuildError(f"forbidden 파일을 읽을 수 없음: {path}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetBuildError(
                    f"forbidden JSON 오류: {path}:{line_number}: {exc.msg}"
                ) from exc
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise DatasetBuildError(f"forbidden text 누락: {path}:{line_number}")
            texts.append(text.strip())
        if len(texts) == rows_before:
            raise DatasetBuildError(f"forbidden 파일이 비어 있음: {path}")
        manifest.append(
            {
                "path": str(path.resolve(strict=False)),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "rows": len(texts) - rows_before,
            }
        )
    return texts, manifest


def _require_subtype_counts(
    examples: Sequence[dict[str, object]],
    *,
    include_benign: bool,
    minimum: int,
    stage: str,
    exact: bool = False,
) -> None:
    expected = set(SCAM_SUBTYPES)
    if include_benign:
        expected.update(BENIGN_SUBTYPES)
    counts = Counter(
        subtype for example in examples if isinstance((subtype := example.get("subtype")), str)
    )
    failures = []
    for subtype in sorted(expected):
        count = counts[subtype]
        if (exact and count != minimum) or (not exact and count < minimum):
            comparator = "=" if exact else ">="
            failures.append(f"{subtype}={count} (필요 {comparator}{minimum})")
    if failures:
        raise DatasetBuildError(f"{stage} 하위유형 수량 미달: {', '.join(failures)}")


def _write_jsonl_atomic(path: Path, examples: Sequence[dict[str, object]]) -> None:
    """완성된 데이터만 원자적으로 교체해 기존 정상 파일을 보존한다."""

    temp_path = _stage_jsonl(path, examples)
    try:
        os.replace(temp_path, path)
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise DatasetBuildError(f"출력 저장 실패: {path}: {exc}") from exc


def _stage_jsonl(path: Path, examples: Sequence[dict[str, object]]) -> Path:
    temp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except (OSError, TypeError, ValueError) as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise DatasetBuildError(f"출력 저장 실패: {path}: {exc}") from exc


def _stage_json(path: Path, payload: dict[str, object]) -> Path:
    temp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except (OSError, TypeError, ValueError) as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise DatasetBuildError(f"JSON 준비 실패: {path}: {exc}") from exc


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temp_path = _stage_json(path, payload)
    try:
        os.replace(temp_path, path)
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise DatasetBuildError(f"캐시 저장 실패: {path}: {exc}") from exc


def _commit_output_and_report(
    *,
    output_path: Path,
    output_temp: Path,
    report_path: Path | None,
    report_temp: Path | None,
) -> None:
    """미리 fsync한 output/report를 교체하고 report 실패 시 기존 output을 복구한다."""

    if (report_path is None) != (report_temp is None):
        for staged in (output_temp, report_temp):
            if staged is not None:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    pass
        raise DatasetBuildError("report 경로와 준비 파일은 함께 지정해야 함")

    backup_path: Path | None = None
    output_replaced = False
    preserve_backup = False
    try:
        if output_path.exists():
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.name}.backup.",
                delete=False,
            ) as backup:
                backup_path = Path(backup.name)
                with output_path.open("rb") as existing:
                    shutil.copyfileobj(existing, backup)
                backup.flush()
                os.fsync(backup.fileno())
        os.replace(output_temp, output_path)
        output_replaced = True
        if report_path is not None and report_temp is not None:
            os.replace(report_temp, report_path)
    except OSError as exc:
        rollback_error: OSError | None = None
        if output_replaced:
            try:
                if backup_path is not None:
                    os.replace(backup_path, output_path)
                    backup_path = None
                else:
                    output_path.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_error = rollback_exc
                preserve_backup = backup_path is not None
        for staged in (output_temp, report_temp):
            if staged is not None:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    pass
        if rollback_error is not None:
            backup_detail = (
                f"; 기존 출력 백업 보존: {backup_path}"
                if preserve_backup and backup_path is not None
                else ""
            )
            raise DatasetBuildError(
                f"출력/report 저장 및 기존 출력 복구 실패: " f"{rollback_error}{backup_detail}"
            ) from exc
        raise DatasetBuildError(f"출력/report 원자 교체 실패: {exc}") from exc
    finally:
        if backup_path is not None and not preserve_backup:
            try:
                backup_path.unlink(missing_ok=True)
            except OSError:
                pass


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_verification_checkpoint(
    path: Path,
    *,
    expected_metadata: dict[str, object],
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetBuildError(f"검수 체크포인트를 읽을 수 없음: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetBuildError(f"검수 체크포인트 JSON 오류: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        raise DatasetBuildError(
            f"검수 체크포인트 조건 불일치: {path} "
            "(입력·모델·프롬프트·forbidden 조건이 같은 보고서를 사용하세요)"
        )
    allowed_statuses = {
        "pending",
        "label_in_progress",
        "label_complete",
        "quality_in_progress",
        "quality_complete",
        "leakage_complete",
        "failed_label_retention",
        "failed_quality_retention",
        "failed_leakage_retention",
        "complete",
    }
    status = payload.get("status")
    if status not in allowed_statuses:
        raise DatasetBuildError(f"검수 체크포인트 status 오류: {path}")

    input_count = expected_metadata.get("input_count")
    if isinstance(input_count, bool) or not isinstance(input_count, int) or input_count < 0:
        raise DatasetBuildError("검수 체크포인트 expected metadata input_count 오류")

    label_verdicts = payload.get("label_verdicts")
    quality_reviews = payload.get("quality_reviews")
    if not isinstance(label_verdicts, list) or not isinstance(quality_reviews, list):
        raise DatasetBuildError(f"검수 체크포인트 배열 형식 오류: {path}")
    if len(label_verdicts) > input_count or any(
        isinstance(verdict, bool) or not isinstance(verdict, int) or verdict not in (-1, 0, 1)
        for verdict in label_verdicts
    ):
        raise DatasetBuildError(f"검수 체크포인트 label verdict 오류: {path}")
    if len(label_verdicts) < input_count and len(label_verdicts) % 20:
        raise DatasetBuildError(f"검수 체크포인트 label batch 경계 오류: {path}")

    label_complete_statuses = allowed_statuses - {"pending", "label_in_progress"}
    if status in label_complete_statuses and len(label_verdicts) != input_count:
        raise DatasetBuildError(f"검수 체크포인트 label 완료 수량 오류: {path}")
    if status == "pending" and label_verdicts:
        raise DatasetBuildError(f"검수 체크포인트 pending label 오류: {path}")

    quality_input_count = payload.get("quality_input_count")
    if quality_input_count is not None and (
        isinstance(quality_input_count, bool)
        or not isinstance(quality_input_count, int)
        or quality_input_count < 0
        or quality_input_count > input_count
    ):
        raise DatasetBuildError(f"검수 체크포인트 quality 입력 수량 오류: {path}")
    if quality_input_count is None and quality_reviews:
        raise DatasetBuildError(f"검수 체크포인트 quality 입력 누락: {path}")
    if quality_input_count is not None and len(quality_reviews) > quality_input_count:
        raise DatasetBuildError(f"검수 체크포인트 quality review 수량 오류: {path}")
    if (
        quality_input_count is not None
        and len(quality_reviews) < quality_input_count
        and len(quality_reviews) % 20
    ):
        raise DatasetBuildError(f"검수 체크포인트 quality batch 경계 오류: {path}")
    for index, review in enumerate(quality_reviews):
        if (
            not isinstance(review, dict)
            or review.get("index") != index
            or not isinstance(review.get("text"), str)
            or not isinstance(review.get("subtype"), str)
            or not isinstance(review.get("accepted"), bool)
            or not isinstance(review.get("reason"), str)
            or not review["reason"].strip()
            or len(review["reason"]) > 500
            or contains_sensitive_identifier(review["reason"])
        ):
            raise DatasetBuildError(f"검수 체크포인트 quality review 오류: {path}")

    quality_started_statuses = {
        "quality_in_progress",
        "quality_complete",
        "leakage_complete",
        "failed_quality_retention",
        "failed_leakage_retention",
        "complete",
    }
    quality_complete_statuses = quality_started_statuses - {"quality_in_progress"}
    if status in quality_started_statuses and quality_input_count is None:
        raise DatasetBuildError(f"검수 체크포인트 quality 입력 수량 누락: {path}")
    if status in quality_complete_statuses and len(quality_reviews) != quality_input_count:
        raise DatasetBuildError(f"검수 체크포인트 quality 완료 수량 오류: {path}")
    if status == "complete":
        final_output = payload.get("final_output")
        if (
            not isinstance(final_output, dict)
            or not isinstance(final_output.get("path"), str)
            or not isinstance(final_output.get("sha256"), str)
            or not isinstance(final_output.get("count"), int)
        ):
            raise DatasetBuildError(f"검수 체크포인트 최종 산출물 오류: {path}")
    return payload


def _quality_audit_prefix(
    examples: Sequence[dict[str, object]],
    reviews: Sequence[tuple[bool, str]],
) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "text": examples[index]["text"],
            "subtype": examples[index].get("subtype", ""),
            "accepted": accepted,
            "reason": reason,
        }
        for index, (accepted, reason) in enumerate(reviews)
    ]


def _quality_review_pairs(
    examples: Sequence[dict[str, object]],
    audit: object,
    *,
    require_complete: bool,
) -> list[tuple[bool, str]]:
    if not isinstance(audit, list):
        raise DatasetBuildError("품질 검수 체크포인트 형식 오류")
    if len(audit) > len(examples) or (require_complete and len(audit) != len(examples)):
        raise DatasetBuildError("품질 검수 체크포인트 수량 불일치")

    reviews: list[tuple[bool, str]] = []
    for index, row in enumerate(audit):
        if (
            not isinstance(row, dict)
            or row.get("index") != index
            or row.get("text") != examples[index].get("text")
            or row.get("subtype") != examples[index].get("subtype")
        ):
            raise DatasetBuildError("품질 검수 체크포인트 행 불일치")
        accepted = row.get("accepted")
        reason = row.get("reason")
        if (
            not isinstance(accepted, bool)
            or not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > 500
            or contains_sensitive_identifier(reason)
        ):
            raise DatasetBuildError("품질 검수 체크포인트 판정 형식 오류")
        reviews.append((accepted, reason.strip()))
    return reviews


def _cache_metadata(
    *,
    model: str,
    per_subtype: int,
    include_benign: bool,
) -> dict[str, object]:
    return {
        "schema_version": SYNTHESIS_CACHE_SCHEMA_VERSION,
        "prompt_revision": SYNTHESIS_PROMPT_REVISION,
        "model": model,
        "per_subtype": per_subtype,
        "include_benign": include_benign,
        "source": SOURCE,
    }


def _temperature_for_model(model: str, value: float) -> float | None:
    if model in MODELS_WITHOUT_SAMPLING_PARAMETERS:
        return None
    return value


def _validate_cached_examples(
    value: object,
    *,
    include_benign: bool,
    per_subtype: int,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise DatasetBuildError("합성 캐시 examples가 배열이 아님")
    subtype_labels = {subtype: 1 for subtype in SCAM_SUBTYPES}
    if include_benign:
        subtype_labels.update({subtype: 0 for subtype in BENIGN_SUBTYPES})

    examples: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    seen_texts: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise DatasetBuildError(f"합성 캐시 행 형식 오류: index={index}")
        text = row.get("text")
        label = row.get("label")
        subtype = row.get("subtype")
        expected_label = subtype_labels.get(subtype) if isinstance(subtype, str) else None
        expected_slice = (
            "scam" if expected_label == 1 else "scam_boundary" if expected_label == 0 else None
        )
        if (
            not isinstance(text, str)
            or not text.strip()
            or contains_sensitive_identifier(text)
            or isinstance(label, bool)
            or not isinstance(label, int)
            or expected_label is None
            or label != expected_label
            or row.get("source") != SOURCE
            or row.get("slice") != expected_slice
        ):
            raise DatasetBuildError(f"합성 캐시 행 검증 실패: index={index}")
        normalized_text = normalize_generated_text(text)
        text_key = duplicate_text_key(normalized_text)
        if not text_key or text_key in seen_texts:
            raise DatasetBuildError(f"합성 캐시 중복 문장: index={index}")
        seen_texts.add(text_key)
        counts[subtype] += 1
        if counts[subtype] > per_subtype:
            raise DatasetBuildError(
                f"합성 캐시 subtype 목표 개수 초과: {subtype}={counts[subtype]}"
            )
        examples.append(
            {
                "text": normalized_text,
                "label": label,
                "slice": expected_slice,
                "subtype": subtype,
                "source": SOURCE,
            }
        )
    return examples


def _load_synthesis_cache(
    path: Path,
    *,
    expected_metadata: dict[str, object],
    include_benign: bool,
    per_subtype: int,
) -> tuple[list[dict[str, object]], bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetBuildError(f"합성 캐시를 읽을 수 없음: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetBuildError(f"합성 캐시 JSON 오류: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise DatasetBuildError(f"합성 캐시 루트가 객체가 아님: {path}")
    metadata = payload.get("metadata")
    legacy_complete_cache = False
    if isinstance(metadata, dict):
        legacy_expected = dict(expected_metadata)
        legacy_expected["schema_version"] = 1
        legacy_complete_cache = metadata == legacy_expected
    if metadata != expected_metadata and not legacy_complete_cache:
        raise DatasetBuildError(
            f"합성 캐시 조건 불일치: {path} " "(모델·개수·프롬프트 조건이 같은 캐시를 사용하세요)"
        )
    complete = True if legacy_complete_cache else payload.get("complete")
    if not isinstance(complete, bool):
        raise DatasetBuildError(f"합성 캐시 complete 플래그 오류: {path}")
    examples = _validate_cached_examples(
        payload.get("examples"),
        include_benign=include_benign,
        per_subtype=per_subtype,
    )
    if complete:
        _require_subtype_counts(
            examples,
            include_benign=include_benign,
            minimum=per_subtype,
            stage="합성 캐시",
            exact=True,
        )
    return examples, complete


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-subtype", type=_positive_int, default=40, help="유형별 합성 개수")
    parser.add_argument("--no-benign", action="store_true", help="경계 반례(정상) 미생성")
    parser.add_argument("--no-verify", action="store_true", help="LLM 검수 생략(디버그)")
    parser.add_argument(
        "--skip-label-verify",
        action="store_true",
        help="라벨 검수만 건너뛰고 subtype·문장논리 품질 검수는 수행",
    )
    parser.add_argument(
        "--no-quality-review",
        action="store_true",
        help="subtype·문장논리 품질 검수 생략(디버그)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synth-model", default="gemini-3.5-flash")
    parser.add_argument(
        "--verify-model",
        default=None,
        help="검수 모델(기본: --synth-model과 동일)",
    )
    parser.add_argument(
        "--synthesis-cache",
        type=Path,
        default=None,
        help="하위유형 완료 단위 합성 체크포인트 JSON. 같은 조건이면 중단 지점부터 재개",
    )
    parser.add_argument(
        "--verification-report",
        type=Path,
        default=None,
        help="라벨·품질 검수 통계와 행별 품질 판정 JSON 저장 경로",
    )
    parser.add_argument(
        "--request-interval",
        type=_nonnegative_float,
        default=15.0,
        help="Gemini 요청 시작 간 최소 초(429 방지, 0이면 비활성)",
    )
    parser.add_argument(
        "--min-retention-ratio",
        type=_unit_interval,
        default=0.8,
        help="라벨·품질·누수 검수 후 각 subtype 최소 보존 비율(기본 0.8)",
    )
    parser.add_argument(
        "--forbidden",
        action="append",
        default=[],
        help="holdout/blind jsonl 경로(반복). 근사중복 합성분을 학습에서 제거(누수 방지)",
    )
    parser.add_argument(
        "--allow-no-forbidden",
        action="store_true",
        help="forbidden 누수 검사를 생략(테스트·디버그 전용)",
    )
    args = parser.parse_args(argv)
    if args.no_verify and args.skip_label_verify:
        parser.error("--no-verify와 --skip-label-verify는 함께 사용할 수 없습니다.")
    if args.no_verify and args.verification_report is not None:
        parser.error("--no-verify 사용 시 --verification-report를 지정할 수 없습니다.")
    if args.no_quality_review and args.verification_report is not None:
        parser.error("--no-quality-review 사용 시 --verification-report를 지정할 수 없습니다.")
    if not args.forbidden and not args.allow_no_forbidden:
        parser.error(
            "최소 1개 --forbidden이 필요합니다. 테스트·디버그만 "
            "--allow-no-forbidden을 사용하세요."
        )
    if args.forbidden and args.allow_no_forbidden:
        parser.error("--forbidden과 --allow-no-forbidden은 함께 사용할 수 없습니다.")
    if _paths_refer_to_same_file(args.out, CANONICAL_TRAIN_OUT):
        parser.error(
            "승인 전 빌드는 canonical data/synthetic/scam/train.jsonl에 쓸 수 없습니다. "
            "outputs/... 후보 경로를 사용하고 사람 승인 뒤 승격하세요."
        )
    try:
        _validate_path_separation(
            out=args.out,
            synthesis_cache=args.synthesis_cache,
            verification_report=args.verification_report,
            forbidden=args.forbidden,
        )
    except DatasetBuildError as exc:
        parser.error(str(exc))

    try:
        forbidden_texts, forbidden_manifest = _load_forbidden_texts(args.forbidden)
        api_key = _load_api_key()
        pacer = RequestPacer(args.request_interval)
        synth_llm = GeminiClient(
            api_key,
            model=args.synth_model,
            temperature=_temperature_for_model(args.synth_model, 0.95),
            pacer=pacer,
        )
        include_benign = not args.no_benign
        cache_metadata = _cache_metadata(
            model=args.synth_model,
            per_subtype=args.per_subtype,
            include_benign=include_benign,
        )
        expected_subtype_total = len(SCAM_SUBTYPES) + (
            len(BENIGN_SUBTYPES) if include_benign else 0
        )
        expected_example_total = expected_subtype_total * args.per_subtype
        minimum_retained = max(
            1,
            math.ceil(args.per_subtype * args.min_retention_ratio),
        )

        if args.synthesis_cache is not None and args.synthesis_cache.exists():
            examples, cache_complete = _load_synthesis_cache(
                args.synthesis_cache,
                expected_metadata=cache_metadata,
                include_benign=include_benign,
                per_subtype=args.per_subtype,
            )
            cache_complete = cache_complete or len(examples) == expected_example_total
            if cache_complete:
                print(f"1. 완성 합성 캐시 재사용: {args.synthesis_cache}")
            else:
                print(
                    f"1. 부분 합성 캐시에서 재개: {args.synthesis_cache} "
                    f"({len(examples)}/{expected_example_total}건)"
                )
        else:
            examples = []
            cache_complete = False

        complete_checkpoint_written = cache_complete

        def save_synthesis_checkpoint(
            checkpoint_examples: list[dict[str, object]],
        ) -> None:
            nonlocal complete_checkpoint_written
            if args.synthesis_cache is None:
                return
            is_complete = len(checkpoint_examples) == expected_example_total
            _write_json_atomic(
                args.synthesis_cache,
                {
                    "metadata": cache_metadata,
                    "complete": is_complete,
                    "examples": checkpoint_examples,
                },
            )
            complete_checkpoint_written = complete_checkpoint_written or is_complete
            print(
                f"   합성 체크포인트: {args.synthesis_cache} "
                f"({len(checkpoint_examples)}/{expected_example_total}건)"
            )

        if not cache_complete:
            print(f"1. 합성 (모델 {args.synth_model}, 유형별 {args.per_subtype}개)...")
            examples = synthesize_scam(
                synth_llm,
                per_subtype=args.per_subtype,
                include_benign=include_benign,
                initial_examples=examples,
                on_subtype_complete=save_synthesis_checkpoint,
            )
        _require_subtype_counts(
            examples,
            include_benign=include_benign,
            minimum=args.per_subtype,
            stage="합성",
            exact=True,
        )
        if args.synthesis_cache is not None and not complete_checkpoint_written:
            save_synthesis_checkpoint(examples)
        print(
            f"   합성 {len(examples)}건 (사기 {sum(e['label'] == 1 for e in examples)} / "
            f"정상 {sum(e['label'] == 0 for e in examples)})"
        )

        label_stats: dict[str, int]
        quality_stats: dict[str, int] | None = None
        quality_audit: list[dict[str, object]] = []
        verification_state: dict[str, object] | None = None
        preserve_completed_report = False

        def save_verification_state() -> None:
            if (
                args.verification_report is not None
                and verification_state is not None
                and not preserve_completed_report
            ):
                _write_json_atomic(args.verification_report, verification_state)

        if args.no_verify:
            kept = examples
            label_stats = {
                "total": len(examples),
                "kept": len(examples),
                "dropped": 0,
                "unparsed": 0,
            }
        else:
            verify_model = args.verify_model or args.synth_model
            verification_metadata: dict[str, object] = {
                "schema_version": VERIFICATION_CHECKPOINT_SCHEMA_VERSION,
                "input_sha256": _sha256_json(examples),
                "input_count": len(examples),
                "synth_model": args.synth_model,
                "verify_model": verify_model,
                "source": SOURCE,
                "synthesis_prompt_revision": SYNTHESIS_PROMPT_REVISION,
                "label_review_revision": LABEL_REVIEW_REVISION,
                "quality_review_revision": QUALITY_REVIEW_REVISION,
                "label_review_skipped": args.skip_label_verify,
                "per_subtype": args.per_subtype,
                "include_benign": include_benign,
                "min_retention_ratio": args.min_retention_ratio,
                "forbidden": forbidden_manifest,
                "output_path": str(args.out.resolve(strict=False)),
            }
            verification_state = {
                "metadata": verification_metadata,
                "status": "pending",
                "label_verdicts": [],
                "label_stats": None,
                "quality_input_sha256": None,
                "quality_input_count": None,
                "quality_stats": None,
                "quality_reviews": [],
                "leakage_stats": None,
                "final_output": None,
            }
            if args.verification_report is not None and args.verification_report.exists():
                verification_state = _load_verification_checkpoint(
                    args.verification_report,
                    expected_metadata=verification_metadata,
                )
                preserve_completed_report = verification_state.get("status") == "complete"
            verify_llm = GeminiClient(
                api_key,
                model=verify_model,
                temperature=_temperature_for_model(verify_model, 0.0),
                pacer=pacer,
            )
            if args.skip_label_verify:
                kept = examples
                expected_verdicts = [int(example["label"]) for example in examples]
                cached_verdicts = verification_state.get("label_verdicts", [])
                if cached_verdicts not in ([], expected_verdicts):
                    raise DatasetBuildError("라벨 생략 체크포인트 verdict 불일치")
                verification_state["label_verdicts"] = expected_verdicts
                label_stats = {
                    "total": len(examples),
                    "kept": len(examples),
                    "dropped": 0,
                    "unparsed": 0,
                }
                print("2. LLM 라벨 검수 건너뜀 (--skip-label-verify)")
            else:
                print(f"2. LLM 라벨 검수 (B2, 모델 {verify_model})...")
                initial_label_verdicts = verification_state.get("label_verdicts", [])
                if not isinstance(initial_label_verdicts, list):
                    raise DatasetBuildError("라벨 검수 체크포인트 형식 오류")

                def save_label_checkpoint(verdicts: list[int]) -> None:
                    if verification_state is None:
                        return
                    verification_state["status"] = "label_in_progress"
                    verification_state["label_verdicts"] = verdicts
                    save_verification_state()

                try:
                    kept, label_stats = verify_labels(
                        verify_llm,
                        examples,
                        initial_verdicts=initial_label_verdicts,
                        on_batch_complete=save_label_checkpoint,
                    )
                except ValueError as exc:
                    raise DatasetBuildError(f"라벨 검수 체크포인트 적용 실패: {exc}") from exc
            verification_state["status"] = "label_complete"
            verification_state["label_stats"] = label_stats
            save_verification_state()
            print(
                f"   통과 {label_stats['kept']} / 탈락 {label_stats['dropped']} / "
                f"미파싱·보류 {label_stats['unparsed']}"
            )
            try:
                _require_subtype_counts(
                    kept,
                    include_benign=include_benign,
                    minimum=minimum_retained,
                    stage="라벨 검수",
                )
            except DatasetBuildError as exc:
                verification_state["status"] = "failed_label_retention"
                verification_state["failure"] = str(exc)
                save_verification_state()
                raise
            if not args.no_quality_review:
                print("3. subtype·문장논리 품질 검수...")
                quality_candidates = kept
                quality_input_sha256 = _sha256_json(quality_candidates)
                cached_quality_sha = verification_state.get("quality_input_sha256")
                if cached_quality_sha not in (None, quality_input_sha256):
                    raise DatasetBuildError("품질 검수 체크포인트 입력 SHA 불일치")
                verification_state["quality_input_sha256"] = quality_input_sha256
                cached_quality_count = verification_state.get("quality_input_count")
                if cached_quality_count not in (None, len(quality_candidates)):
                    raise DatasetBuildError("품질 검수 체크포인트 입력 수량 불일치")
                verification_state["quality_input_count"] = len(quality_candidates)
                cached_quality_audit = verification_state.get("quality_reviews", [])
                initial_quality_reviews = _quality_review_pairs(
                    quality_candidates,
                    cached_quality_audit,
                    require_complete=False,
                )

                def save_quality_checkpoint(reviews: list[tuple[bool, str]]) -> None:
                    if verification_state is None:
                        return
                    verification_state["status"] = "quality_in_progress"
                    verification_state["quality_reviews"] = _quality_audit_prefix(
                        quality_candidates,
                        reviews,
                    )
                    save_verification_state()

                try:
                    kept, quality_stats, quality_audit = review_quality(
                        verify_llm,
                        quality_candidates,
                        initial_reviews=initial_quality_reviews,
                        on_batch_complete=save_quality_checkpoint,
                    )
                except ValueError as exc:
                    raise DatasetBuildError(f"품질 검수 체크포인트 적용 실패: {exc}") from exc
                _quality_review_pairs(
                    quality_candidates,
                    quality_audit,
                    require_complete=True,
                )
                verification_state["status"] = "quality_complete"
                verification_state["quality_stats"] = quality_stats
                verification_state["quality_reviews"] = quality_audit
                save_verification_state()
                print(
                    f"   통과 {quality_stats['kept']} / 거절 {quality_stats['rejected']} / "
                    f"미파싱 {quality_stats['unparsed']}"
                )
                try:
                    _require_subtype_counts(
                        kept,
                        include_benign=include_benign,
                        minimum=minimum_retained,
                        stage="품질 검수",
                    )
                except DatasetBuildError as exc:
                    verification_state["status"] = "failed_quality_retention"
                    verification_state["failure"] = str(exc)
                    save_verification_state()
                    raise

        removed = 0
        if forbidden_texts:
            kept, removed = filter_forbidden(kept, forbidden_texts)
            if verification_state is not None:
                verification_state["leakage_stats"] = {
                    "forbidden_rows": len(forbidden_texts),
                    "removed": removed,
                    "kept": len(kept),
                }
                verification_state["status"] = "leakage_complete"
                save_verification_state()
            print(f"4. 누수 제거(holdout/blind 근사중복): {removed}건 제거 → {len(kept)}건")
        try:
            _require_subtype_counts(
                kept,
                include_benign=include_benign,
                minimum=minimum_retained,
                stage="누수 제거",
            )
        except DatasetBuildError as exc:
            if verification_state is not None:
                verification_state["status"] = "failed_leakage_retention"
                verification_state["failure"] = str(exc)
                verification_state["leakage_stats"] = {
                    "forbidden_rows": len(forbidden_texts),
                    "removed": removed,
                    "kept": len(kept),
                }
                save_verification_state()
            raise

        output_temp = _stage_jsonl(args.out, kept)
        try:
            output_sha256 = hashlib.sha256(output_temp.read_bytes()).hexdigest()
        except OSError as exc:
            try:
                output_temp.unlink(missing_ok=True)
            except OSError:
                pass
            raise DatasetBuildError(f"출력 해시 계산 실패: {args.out}: {exc}") from exc
        report_temp: Path | None = None
        if verification_state is not None:
            verification_state.pop("failure", None)
            verification_state["status"] = "complete"
            verification_state["final_output"] = {
                "path": str(args.out.resolve(strict=False)),
                "sha256": output_sha256,
                "count": len(kept),
            }
            if args.verification_report is not None:
                try:
                    report_temp = _stage_json(
                        args.verification_report,
                        verification_state,
                    )
                except DatasetBuildError:
                    try:
                        output_temp.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
        _commit_output_and_report(
            output_path=args.out,
            output_temp=output_temp,
            report_path=args.verification_report,
            report_temp=report_temp,
        )
        if args.verification_report is not None:
            print(f"   검수 보고서 저장: {args.verification_report}")
        print(f"저장: {args.out} ({len(kept)}건)")
        return 0
    except (DatasetBuildError, GeminiClientError, LLMOutputError) as exc:
        print(f"[scam] 데이터셋 빌드 실패: {exc}", file=sys.stderr)
        if isinstance(exc, GeminiAPIError) and exc.status_code == 429:
            if exc.retry_after is not None:
                print(
                    f"[scam] 서버 재시도 최소 대기: {exc.retry_after:g}초",
                    file=sys.stderr,
                )
            print(
                "[scam] 429가 반복되면 quota 확인 후 --request-interval 값을 늘려 재실행하세요.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
