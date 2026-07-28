"""사기 데이터셋 CLI의 fail-closed 저장 동작 검증."""

from __future__ import annotations

import json
import re

import pytest

from scripts import build_scam_dataset as builder
from src.data.llm_client import GeminiAPIError, RateLimitedError


def _all_subtype_rows(per_subtype: int = 1) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subtype in builder.SCAM_SUBTYPES:
        for variant in range(per_subtype):
            rows.append(
                {
                    "text": f"{subtype} 사기 예시 {variant}",
                    "label": 1,
                    "slice": "scam",
                    "subtype": subtype,
                    "source": builder.SOURCE,
                }
            )
    for subtype in builder.BENIGN_SUBTYPES:
        for variant in range(per_subtype):
            rows.append(
                {
                    "text": f"{subtype} 정상 예시 {variant}",
                    "label": 0,
                    "slice": "scam_boundary",
                    "subtype": subtype,
                    "source": builder.SOURCE,
                }
            )
    return rows


def test_all_synthesis_failures_return_nonzero_and_preserve_existing_output(
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "train.jsonl"
    original = b'{"text":"existing"}\n'
    output.write_bytes(original)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", lambda *args, **kwargs: [])

    result = builder.main(
        [
            "--per-subtype",
            "1",
            "--no-verify",
            "--allow-no-forbidden",
            "--out",
            str(output),
        ]
    )

    assert result == 1
    assert output.read_bytes() == original


def test_successful_build_atomically_writes_complete_coverage(monkeypatch, tmp_path):
    output = tmp_path / "nested" / "train.jsonl"
    rows = _all_subtype_rows()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", lambda *args, **kwargs: rows)
    monkeypatch.setattr(
        builder,
        "verify_labels",
        lambda _client, examples, **kwargs: (
            list(examples),
            {"kept": len(examples), "dropped": 0, "unparsed": 0},
        ),
    )

    result = builder.main(
        [
            "--per-subtype",
            "1",
            "--no-quality-review",
            "--allow-no-forbidden",
            "--out",
            str(output),
        ]
    )

    assert result == 0
    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == len(builder.SCAM_SUBTYPES) + len(builder.BENIGN_SUBTYPES)
    assert {row["subtype"] for row in saved} == {
        *builder.SCAM_SUBTYPES,
        *builder.BENIGN_SUBTYPES,
    }
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_invalid_forbidden_file_fails_before_llm_call(monkeypatch, tmp_path):
    called = False

    def fake_synthesize(*args, **kwargs):
        nonlocal called
        called = True
        return _all_subtype_rows()

    monkeypatch.setattr(builder, "synthesize_scam", fake_synthesize)

    result = builder.main(
        [
            "--forbidden",
            str(tmp_path / "missing.jsonl"),
            "--out",
            str(tmp_path / "train.jsonl"),
        ]
    )

    assert result == 1
    assert called is False


def test_synthesis_cache_survives_verify_failure_and_is_reused(monkeypatch, tmp_path):
    output = tmp_path / "train.jsonl"
    cache = tmp_path / "synthesis_cache.json"
    rows = _all_subtype_rows()
    synth_calls = 0

    def fake_synthesize(*args, **kwargs):
        nonlocal synth_calls
        synth_calls += 1
        return rows

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", fake_synthesize)
    monkeypatch.setattr(
        builder,
        "verify_labels",
        lambda _client, examples, **kwargs: (
            [],
            {"kept": 0, "dropped": 0, "unparsed": len(examples)},
        ),
    )

    args = [
        "--per-subtype",
        "1",
        "--out",
        str(output),
        "--no-quality-review",
        "--allow-no-forbidden",
        "--synthesis-cache",
        str(cache),
    ]
    assert builder.main(args) == 1
    assert cache.exists()
    assert not output.exists()
    assert synth_calls == 1

    def must_not_synthesize(*args, **kwargs):
        raise AssertionError("cache가 있으면 합성을 다시 호출하면 안 됨")

    monkeypatch.setattr(builder, "synthesize_scam", must_not_synthesize)
    monkeypatch.setattr(
        builder,
        "verify_labels",
        lambda _client, examples, **kwargs: (
            list(examples),
            {"kept": len(examples), "dropped": 0, "unparsed": 0},
        ),
    )

    assert builder.main(args) == 0
    assert output.exists()
    assert synth_calls == 1


def test_skip_label_verify_runs_quality_review_and_writes_report(monkeypatch, tmp_path):
    output = tmp_path / "train.jsonl"
    report = tmp_path / "report.json"
    rows = _all_subtype_rows()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", lambda *args, **kwargs: rows)

    def must_not_verify_labels(*args, **kwargs):
        raise AssertionError("skip flag가 있으면 라벨 검수를 호출하면 안 됨")

    monkeypatch.setattr(builder, "verify_labels", must_not_verify_labels)
    monkeypatch.setattr(
        builder,
        "review_quality",
        lambda _client, examples, **kwargs: (
            list(examples),
            {"total": len(examples), "kept": len(examples), "rejected": 0, "unparsed": 0},
            [
                {
                    "index": index,
                    "text": row["text"],
                    "subtype": row["subtype"],
                    "accepted": True,
                    "reason": "품질 통과",
                }
                for index, row in enumerate(examples)
            ],
        ),
    )

    result = builder.main(
        [
            "--per-subtype",
            "1",
            "--skip-label-verify",
            "--allow-no-forbidden",
            "--out",
            str(output),
            "--verification-report",
            str(report),
        ]
    )

    assert result == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["metadata"]["label_review_skipped"] is True
    assert payload["metadata"]["quality_review_revision"] == builder.QUALITY_REVIEW_REVISION


def test_synthesis_cache_rejects_wrong_slice():
    rows = _all_subtype_rows()
    rows[0]["slice"] = "scam_boundary"

    with pytest.raises(builder.DatasetBuildError, match="행 검증 실패"):
        builder._validate_cached_examples(
            rows,
            include_benign=True,
            per_subtype=1,
        )


def test_synthesis_cache_rejects_normalized_duplicate_across_subtypes():
    rows = _all_subtype_rows()
    rows[1]["text"] = f"{rows[0]['text']}!!!"

    with pytest.raises(builder.DatasetBuildError, match="중복 문장"):
        builder._validate_cached_examples(
            rows,
            include_benign=True,
            per_subtype=1,
        )


def test_legacy_complete_cache_is_read_without_new_api_calls(tmp_path):
    path = tmp_path / "legacy.json"
    metadata = builder._cache_metadata(
        model="test-model",
        per_subtype=1,
        include_benign=True,
    )
    metadata["schema_version"] = 1
    path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "examples": _all_subtype_rows(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    expected = dict(metadata)
    expected["schema_version"] = builder.SYNTHESIS_CACHE_SCHEMA_VERSION

    rows, complete = builder._load_synthesis_cache(
        path,
        expected_metadata=expected,
        include_benign=True,
        per_subtype=1,
    )

    assert complete is True
    assert len(rows) == len(builder.SCAM_SUBTYPES) + len(builder.BENIGN_SUBTYPES)


def test_no_verify_rejects_verification_report_argument(tmp_path):
    with pytest.raises(SystemExit):
        builder.main(
            [
                "--no-verify",
                "--verification-report",
                str(tmp_path / "report.json"),
            ]
        )


def test_output_paths_must_not_collide_with_each_other_or_forbidden(tmp_path):
    shared = tmp_path / "shared.json"
    forbidden = tmp_path / "blind.jsonl"
    forbidden.write_text('{"text":"fresh blind"}\n', encoding="utf-8")

    with pytest.raises(SystemExit):
        builder.main(
            [
                "--out",
                str(shared),
                "--synthesis-cache",
                str(shared),
                "--allow-no-forbidden",
            ]
        )
    with pytest.raises(SystemExit):
        builder.main(
            [
                "--out",
                str(forbidden),
                "--forbidden",
                str(forbidden),
            ]
        )

    assert forbidden.read_text(encoding="utf-8") == '{"text":"fresh blind"}\n'


def test_verification_report_requires_quality_review(tmp_path):
    with pytest.raises(SystemExit):
        builder.main(
            [
                "--no-quality-review",
                "--verification-report",
                str(tmp_path / "report.json"),
            ]
        )


def test_partial_synthesis_cache_resumes_after_api_failure(monkeypatch, tmp_path):
    output = tmp_path / "train.jsonl"
    cache = tmp_path / "cache.json"

    class FailAfterFirstSubtype:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def __call__(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 2:
                raise GeminiAPIError(429, "quota", status="RESOURCE_EXHAUSTED")
            subtype = re.search(r"# 과제\n'([^']+)'", prompt).group(1)
            return json.dumps(
                [{"text": f"{subtype} 첫 실행", "label": 1, "subtype": subtype}],
                ensure_ascii=False,
            )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "GeminiClient", FailAfterFirstSubtype)
    args = [
        "--per-subtype",
        "1",
        "--no-benign",
        "--no-verify",
        "--allow-no-forbidden",
        "--synthesis-cache",
        str(cache),
        "--out",
        str(output),
    ]

    assert builder.main(args) == 1
    checkpoint = json.loads(cache.read_text(encoding="utf-8"))
    assert checkpoint["complete"] is False
    assert len(checkpoint["examples"]) == 1
    assert not output.exists()

    generated_on_resume: list[str] = []

    class ResumeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, prompt: str) -> str:
            subtype = re.search(r"# 과제\n'([^']+)'", prompt).group(1)
            generated_on_resume.append(subtype)
            return json.dumps(
                [{"text": f"{subtype} 재개 실행", "label": 1, "subtype": subtype}],
                ensure_ascii=False,
            )

    monkeypatch.setattr(builder, "GeminiClient", ResumeClient)
    assert builder.main(args) == 0

    first_subtype = next(iter(builder.SCAM_SUBTYPES))
    assert first_subtype not in generated_on_resume
    assert len(output.read_text(encoding="utf-8").splitlines()) == len(builder.SCAM_SUBTYPES)


def test_partial_synthesis_count_fails_without_replacing_output(monkeypatch, tmp_path):
    output = tmp_path / "train.jsonl"
    original = b'{"text":"existing"}\n'
    output.write_bytes(original)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", lambda *args, **kwargs: _all_subtype_rows())

    result = builder.main(
        [
            "--per-subtype",
            "2",
            "--no-verify",
            "--allow-no-forbidden",
            "--out",
            str(output),
        ]
    )

    assert result == 1
    assert output.read_bytes() == original


def test_forbidden_is_required_and_empty_file_fails_before_llm(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        builder.main(["--out", str(tmp_path / "candidate.jsonl")])

    called = False

    def fake_synthesize(*args, **kwargs):
        nonlocal called
        called = True
        return _all_subtype_rows()

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setattr(builder, "synthesize_scam", fake_synthesize)

    assert (
        builder.main(
            [
                "--forbidden",
                str(empty),
                "--out",
                str(tmp_path / "candidate.jsonl"),
            ]
        )
        == 1
    )
    assert called is False


def test_canonical_output_and_ancestor_collisions_are_rejected(tmp_path):
    with pytest.raises(SystemExit):
        builder.main(
            [
                "--out",
                str(builder.CANONICAL_TRAIN_OUT),
                "--allow-no-forbidden",
            ]
        )

    parent_output = tmp_path / "result"
    with pytest.raises(SystemExit):
        builder.main(
            [
                "--out",
                str(parent_output),
                "--synthesis-cache",
                str(parent_output / "cache.json"),
                "--allow-no-forbidden",
            ]
        )


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_request_interval_is_rejected(value):
    with pytest.raises(SystemExit):
        builder.main(
            [
                "--request-interval",
                value,
                "--allow-no-forbidden",
            ]
        )


def test_429_stderr_includes_server_retry_delay(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fail(*args, **kwargs):
        raise RateLimitedError(
            429,
            "quota exhausted",
            status="RESOURCE_EXHAUSTED",
            retry_after=300,
        )

    monkeypatch.setattr(builder, "synthesize_scam", fail)

    result = builder.main(
        [
            "--per-subtype",
            "1",
            "--no-verify",
            "--allow-no-forbidden",
            "--out",
            str(tmp_path / "candidate.jsonl"),
        ]
    )

    assert result == 1
    assert "서버 재시도 최소 대기: 300초" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("kept_in_first_subtype", "expected_result"),
    [(3, 1), (4, 0)],
)
def test_label_retention_ratio_is_enforced_per_subtype(
    monkeypatch,
    tmp_path,
    kept_in_first_subtype,
    expected_result,
):
    rows = _all_subtype_rows(5)
    first_subtype = next(iter(builder.SCAM_SUBTYPES))
    output = tmp_path / f"candidate-{kept_in_first_subtype}.jsonl"
    original = b'{"text":"existing"}\n'
    output.write_bytes(original)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", lambda *args, **kwargs: rows)

    def fake_verify(_client, examples, **kwargs):
        first_rows = [row for row in examples if row["subtype"] == first_subtype]
        kept = [
            row
            for row in examples
            if row["subtype"] != first_subtype or row in first_rows[:kept_in_first_subtype]
        ]
        return kept, {
            "total": len(examples),
            "kept": len(kept),
            "dropped": len(examples) - len(kept),
            "unparsed": 0,
        }

    monkeypatch.setattr(builder, "verify_labels", fake_verify)
    result = builder.main(
        [
            "--per-subtype",
            "5",
            "--no-quality-review",
            "--allow-no-forbidden",
            "--out",
            str(output),
        ]
    )

    assert result == expected_result
    if expected_result:
        assert output.read_bytes() == original
    else:
        assert len(output.read_text(encoding="utf-8").splitlines()) == 44


def test_quality_and_forbidden_retention_failures_preserve_output_and_report(
    monkeypatch,
    tmp_path,
):
    rows = _all_subtype_rows(5)
    first_subtype = next(iter(builder.SCAM_SUBTYPES))
    first_rows = [row for row in rows if row["subtype"] == first_subtype]
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", lambda *args, **kwargs: rows)
    monkeypatch.setattr(
        builder,
        "verify_labels",
        lambda _client, examples, **kwargs: (
            list(examples),
            {"total": len(examples), "kept": len(examples), "dropped": 0, "unparsed": 0},
        ),
    )

    def reject_quality(_client, examples, **kwargs):
        kept = [row for row in examples if row["subtype"] != first_subtype or row in first_rows[:3]]
        audit = [
            {
                "index": index,
                "text": row["text"],
                "subtype": row["subtype"],
                "accepted": row in kept,
                "reason": "통과" if row in kept else "품질 거절",
            }
            for index, row in enumerate(examples)
        ]
        return (
            kept,
            {
                "total": len(examples),
                "kept": len(kept),
                "rejected": len(examples) - len(kept),
                "unparsed": 0,
            },
            audit,
        )

    monkeypatch.setattr(builder, "review_quality", reject_quality)
    output = tmp_path / "quality-candidate.jsonl"
    report = tmp_path / "quality-report.json"
    original = b'{"text":"existing"}\n'
    output.write_bytes(original)
    assert (
        builder.main(
            [
                "--per-subtype",
                "5",
                "--skip-label-verify",
                "--allow-no-forbidden",
                "--out",
                str(output),
                "--verification-report",
                str(report),
            ]
        )
        == 1
    )
    assert output.read_bytes() == original
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == ("failed_quality_retention")

    forbidden = tmp_path / "forbidden.jsonl"
    forbidden.write_text(
        "".join(
            json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n" for row in first_rows[:2]
        ),
        encoding="utf-8",
    )
    leak_output = tmp_path / "leak-candidate.jsonl"
    leak_output.write_bytes(original)
    assert (
        builder.main(
            [
                "--per-subtype",
                "5",
                "--no-verify",
                "--forbidden",
                str(forbidden),
                "--out",
                str(leak_output),
            ]
        )
        == 1
    )
    assert leak_output.read_bytes() == original


def test_verification_batches_resume_with_changed_request_interval(
    monkeypatch,
    tmp_path,
):
    rows = _all_subtype_rows(5)
    output = tmp_path / "candidate.jsonl"
    report = tmp_path / "verification.json"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", lambda *args, **kwargs: rows)

    class FirstRunClient:
        label_calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt: str) -> str:
            if "# 입력(JSON 배열, 각 메시지)" in prompt:
                type(self).label_calls += 1
                if type(self).label_calls == 2:
                    raise RateLimitedError(
                        429,
                        "quota",
                        status="RESOURCE_EXHAUSTED",
                        retry_after=300,
                    )
                texts = json.loads(
                    prompt.split("# 입력(JSON 배열, 각 메시지)\n", 1)[1].split("\n\n# 출력", 1)[0]
                )
                return json.dumps([{"label": 0 if "정상" in text else 1} for text in texts])
            raise AssertionError("첫 실행은 quality 단계에 도달하면 안 됨")

    monkeypatch.setattr(builder, "GeminiClient", FirstRunClient)
    args = [
        "--per-subtype",
        "5",
        "--allow-no-forbidden",
        "--out",
        str(output),
        "--verification-report",
        str(report),
    ]
    assert builder.main(args) == 1
    partial = json.loads(report.read_text(encoding="utf-8"))
    assert partial["status"] == "label_in_progress"
    assert len(partial["label_verdicts"]) == 20
    assert not output.exists()

    class ResumeClient:
        label_batch_sizes: list[int] = []
        quality_batch_sizes: list[int] = []

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt: str) -> str:
            if "# 입력(JSON 배열, 각 메시지)" in prompt:
                texts = json.loads(
                    prompt.split("# 입력(JSON 배열, 각 메시지)\n", 1)[1].split("\n\n# 출력", 1)[0]
                )
                type(self).label_batch_sizes.append(len(texts))
                return json.dumps([{"label": 0 if "정상" in text else 1} for text in texts])
            payload = json.loads(
                prompt.split("# 입력(JSON 배열)\n", 1)[1].split("\n\n# 출력", 1)[0]
            )
            type(self).quality_batch_sizes.append(len(payload))
            return json.dumps(
                [{"id": row["id"], "accept": True, "reason": "품질 통과"} for row in payload],
                ensure_ascii=False,
            )

    monkeypatch.setattr(builder, "GeminiClient", ResumeClient)
    assert builder.main([*args, "--request-interval", "30"]) == 0
    assert ResumeClient.label_batch_sizes == [20, 5]
    assert ResumeClient.quality_batch_sizes == [20, 20, 5]
    completed = json.loads(report.read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert completed["final_output"]["count"] == 45


def test_corrupt_verification_checkpoint_fails_closed(monkeypatch, tmp_path):
    rows = _all_subtype_rows()
    output = tmp_path / "candidate.jsonl"
    report = tmp_path / "verification.json"
    original = b'{"text":"existing"}\n'
    output.write_bytes(original)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", lambda *args, **kwargs: rows)

    def write_corrupt_prefix(_client, _examples, **kwargs):
        kwargs["on_batch_complete"]([2])
        raise RateLimitedError(
            429,
            "quota",
            status="RESOURCE_EXHAUSTED",
            retry_after=300,
        )

    monkeypatch.setattr(builder, "verify_labels", write_corrupt_prefix)
    args = [
        "--per-subtype",
        "1",
        "--allow-no-forbidden",
        "--out",
        str(output),
        "--verification-report",
        str(report),
    ]
    assert builder.main(args) == 1
    assert output.read_bytes() == original

    def must_not_verify(*args, **kwargs):
        raise AssertionError("손상 체크포인트는 검수 호출 전에 거부되어야 함")

    monkeypatch.setattr(builder, "verify_labels", must_not_verify)
    assert builder.main(args) == 1
    assert output.read_bytes() == original
    assert json.loads(report.read_text(encoding="utf-8"))["label_verdicts"] == [2]


def test_final_report_replace_failure_rolls_back_output_and_preserves_complete_report(
    monkeypatch,
    tmp_path,
):
    rows = _all_subtype_rows()
    output = tmp_path / "candidate.jsonl"
    report = tmp_path / "verification.json"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(builder, "synthesize_scam", lambda *args, **kwargs: rows)

    def accept_quality(_client, examples, **kwargs):
        audit = [
            {
                "index": index,
                "text": row["text"],
                "subtype": row["subtype"],
                "accepted": True,
                "reason": "품질 통과",
            }
            for index, row in enumerate(examples)
        ]
        return (
            list(examples),
            {"total": len(examples), "kept": len(examples), "rejected": 0, "unparsed": 0},
            audit,
        )

    monkeypatch.setattr(builder, "review_quality", accept_quality)
    args = [
        "--per-subtype",
        "1",
        "--skip-label-verify",
        "--allow-no-forbidden",
        "--out",
        str(output),
        "--verification-report",
        str(report),
    ]
    assert builder.main(args) == 0
    original_output = output.read_bytes()
    original_report = report.read_bytes()
    real_replace = builder.os.replace

    def fail_report_replace(source, destination):
        if builder._paths_refer_to_same_file(builder.Path(destination), report):
            raise OSError("simulated report replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", fail_report_replace)
    assert builder.main(args) == 1
    assert output.read_bytes() == original_output
    assert report.read_bytes() == original_report
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.backup.*")) == []


def test_rollback_failure_preserves_recovery_backup(monkeypatch, tmp_path):
    output = tmp_path / "candidate.jsonl"
    report = tmp_path / "verification.json"
    original_output = b'{"text":"existing"}\n'
    original_report = b'{"status":"complete"}\n'
    output.write_bytes(original_output)
    report.write_bytes(original_report)
    output_temp = builder._stage_jsonl(output, [{"text": "replacement"}])
    report_temp = builder._stage_json(report, {"status": "replacement"})
    real_replace = builder.os.replace

    def fail_report_and_rollback(source, destination):
        source_path = builder.Path(source)
        destination_path = builder.Path(destination)
        if builder._paths_refer_to_same_file(destination_path, report):
            raise OSError("simulated report replace failure")
        if ".backup." in source_path.name:
            raise OSError("simulated rollback failure")
        return real_replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", fail_report_and_rollback)
    with pytest.raises(builder.DatasetBuildError, match="기존 출력 백업 보존"):
        builder._commit_output_and_report(
            output_path=output,
            output_temp=output_temp,
            report_path=report,
            report_temp=report_temp,
        )

    backups = list(tmp_path.glob(".*.backup.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original_output
    assert report.read_bytes() == original_report
