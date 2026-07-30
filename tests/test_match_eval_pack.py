"""MATCH v2 재현성 평가팩 무결성·단계 비교 테스트."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.train_match_v2 as train_module
import src.data.match_eval_pack.compare as compare_module
import src.data.match_eval_pack.format as format_module
import src.data.match_eval_pack.validation as validation_module
from scripts.train_match_v2 import run
from src.data.match_eval_pack import (
    EvalPackError,
    EvalPackIntegrityError,
    compare_match_eval_packs,
    evaluate_model_against_pack,
    load_current_match_run,
    load_match_eval_pack,
    replay_match_eval_pack,
    sha256_file,
)
from src.data.match_eval_pack.format import (
    QUERY_METRICS_FILE,
    TEST_USERS_FILE,
    canonical_json_bytes,
    pack_content_sha256,
    stage_hashes,
)
from src.data.matching_trainset import ndcg_at_k


class StableFakeEncoder:
    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def encode(self, sentences, *, batch_size, show_progress_bar):
        rows = []
        for text in sentences:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            rows.append(rng.normal(size=24) + self.offset)
        return np.asarray(rows, dtype=np.float32)

    def reproducibility_metadata(self):
        return {
            "model_name": "stable-test-encoder",
            "resolved_revision": "test-v1",
            "device": "cpu",
            "parameter_dtype": "float32",
        }


def _train_small(out_dir, *, encoder=None):
    return run(
        encoder=encoder or StableFakeEncoder(),
        n_users=240,
        n_train_queries=60,
        n_test_queries=24,
        n_candidates=8,
        seed=19,
        out_dir=out_dir,
        ablations=False,
        n_estimators=20,
    )


def _pack_path(result) -> Path:
    return Path(result["eval_pack_path"])


def _model_path(result) -> Path:
    return Path(result["model_path"])


def test_eval_pack_round_trip_and_model_replay(tmp_path):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)

    loaded = load_match_eval_pack(
        pack_path,
        expected_pack_sha256=result["eval_pack_sha256"],
    )
    assert loaded.frame["x_test"].shape[1] == len(result["columns"])
    for key in ("ndcg@5", "ndcg@10"):
        assert loaded.manifest["evaluation"]["metrics"][key] == result["metrics"]["v2_full"][key]

    model_path = _model_path(result)
    model_sha256 = sha256_file(model_path)
    replay = evaluate_model_against_pack(
        model_path,
        pack_path,
        expected_model_sha256=model_sha256,
        expected_pack_sha256=result["eval_pack_sha256"],
    )
    assert replay["status"] == "match"
    assert replay["score_max_abs"] <= 1e-10
    assert replay["metric_deltas"]["ndcg@5"] == pytest.approx(0.0, abs=1e-12)
    assert replay["metric_deltas"]["ndcg@10"] == pytest.approx(0.0, abs=1e-12)


def test_eval_pack_replays_same_encoder_without_retraining_model(tmp_path):
    reference_result = _train_small(tmp_path / "reference")
    reference_pack = _pack_path(reference_result)
    model_path = _model_path(reference_result)
    report = replay_match_eval_pack(
        reference_pack,
        model_path=model_path,
        expected_model_sha256=sha256_file(model_path),
        expected_reference_sha256=reference_result["eval_pack_sha256"],
        encoder=StableFakeEncoder(),
        output_dir=tmp_path / "candidate_pack",
        config_path=Path("configs/module2_matching.yaml"),
    )
    assert report["status"] == "match"
    assert report["first_mismatch"] is None
    assert report["replayed_metrics"]["ndcg@5"] == pytest.approx(
        load_match_eval_pack(
            reference_pack,
            expected_pack_sha256=reference_result["eval_pack_sha256"],
        ).manifest["evaluation"]["metrics"]["ndcg@5"],
        abs=1e-12,
    )


def test_eval_pack_reports_first_embedding_mismatch(tmp_path):
    reference_result = _train_small(
        tmp_path / "reference",
        encoder=StableFakeEncoder(),
    )
    candidate_result = _train_small(
        tmp_path / "candidate",
        encoder=StableFakeEncoder(offset=0.05),
    )

    report = compare_match_eval_packs(
        _pack_path(reference_result),
        _pack_path(candidate_result),
        expected_reference_sha256=reference_result["eval_pack_sha256"],
        expected_candidate_sha256=candidate_result["eval_pack_sha256"],
    )
    assert report["status"] == "drift"
    assert report["first_mismatch"] == "text_embeddings"
    assert report["details"]["max_abs"] > 0


def test_eval_pack_rejects_file_corruption(tmp_path):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)
    frame_path = pack_path / "frame.npz"
    payload = bytearray(frame_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    frame_path.write_bytes(payload)

    with pytest.raises(EvalPackIntegrityError, match="SHA-256 mismatch"):
        load_match_eval_pack(
            pack_path,
            expected_pack_sha256=result["eval_pack_sha256"],
        )


def test_eval_pack_rejects_provenance_tampering(tmp_path):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)
    manifest_path = pack_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["encoder"]["device"] = "tampered"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(EvalPackIntegrityError, match="provenance SHA-256 mismatch"):
        load_match_eval_pack(
            pack_path,
            expected_pack_sha256=result["eval_pack_sha256"],
        )


@pytest.mark.parametrize(
    ("stored_ndcg", "accepted"),
    [
        (1.0000000000000002, True),
        (1.0 + 2e-12, False),
    ],
)
def test_eval_pack_allows_only_roundoff_past_ndcg_bounds(tmp_path, stored_ndcg, accepted):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)
    loaded = load_match_eval_pack(
        pack_path,
        expected_pack_sha256=result["eval_pack_sha256"],
    )
    query_records = list(loaded.query_metric_records)
    record = next(row for row in query_records if row["metric_included"])
    record["ndcg@10"] = stored_ndcg
    query_payload = b"".join(canonical_json_bytes(row) for row in query_records)
    (pack_path / QUERY_METRICS_FILE).write_bytes(query_payload)

    manifest_path = pack_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][QUERY_METRICS_FILE] = {
        "sha256": hashlib.sha256(query_payload).hexdigest(),
        "bytes": len(query_payload),
    }
    manifest["evaluation"]["metrics"] = {
        key: float(
            np.mean([row[key] for row in query_records if row["metric_included"]])
        )
        for key in ("ndcg@5", "ndcg@10")
    }
    manifest["stages"] = stage_hashes(
        files=manifest["files"],
        frame=loaded.frame,
        user_embeddings=loaded.user_embeddings,
        text_embeddings=loaded.text_embeddings,
        columns=manifest["evaluation"]["columns"],
        model_sha256=manifest["model"]["sha256"],
        metrics=manifest["evaluation"]["metrics"],
    )
    manifest["pack_content_sha256"] = pack_content_sha256(
        stages=manifest["stages"],
        model=manifest["model"],
        files=manifest["files"],
        evaluation=manifest["evaluation"],
        provenance_sha256=manifest["provenance_sha256"],
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    if accepted:
        assert (
            load_match_eval_pack(
                pack_path,
                expected_pack_sha256=manifest["pack_content_sha256"],
            ).query_metric_records[query_records.index(record)]["ndcg@10"]
            == stored_ndcg
        )
    else:
        with pytest.raises(EvalPackIntegrityError, match="invalid stored query metric"):
            load_match_eval_pack(
                pack_path,
                expected_pack_sha256=manifest["pack_content_sha256"],
            )


def test_external_pack_sha_rejects_self_consistent_forgery(tmp_path):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)
    manifest_path = pack_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["encoder"]["device"] = "forged-device"
    manifest["provenance_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest["provenance"])
    ).hexdigest()
    manifest["pack_content_sha256"] = pack_content_sha256(
        stages=manifest["stages"],
        model=manifest["model"],
        files=manifest["files"],
        evaluation=manifest["evaluation"],
        provenance_sha256=manifest["provenance_sha256"],
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    forged_sha = manifest["pack_content_sha256"]
    assert (
        load_match_eval_pack(
            pack_path,
            expected_pack_sha256=forged_sha,
        ).manifest["provenance"]["encoder"]["device"]
        == "forged-device"
    )
    with pytest.raises(EvalPackIntegrityError, match="external trust anchor"):
        load_match_eval_pack(
            pack_path,
            expected_pack_sha256=result["eval_pack_sha256"],
        )


def test_loader_rejects_self_consistent_malformed_raw_snapshot(tmp_path):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)
    loaded = load_match_eval_pack(
        pack_path,
        expected_pack_sha256=result["eval_pack_sha256"],
    )
    user_records = list(loaded.user_records)
    user_records[0] = json.loads(json.dumps(user_records[0], ensure_ascii=False))
    user_records[0]["snapshot"]["bio"] = 123
    user_payload = b"".join(canonical_json_bytes(record) for record in user_records)
    user_path = pack_path / TEST_USERS_FILE
    user_path.write_bytes(user_payload)

    manifest_path = pack_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][TEST_USERS_FILE] = {
        "sha256": hashlib.sha256(user_payload).hexdigest(),
        "bytes": len(user_payload),
    }
    manifest["stages"] = stage_hashes(
        files=manifest["files"],
        frame=loaded.frame,
        user_embeddings=loaded.user_embeddings,
        text_embeddings=loaded.text_embeddings,
        columns=manifest["evaluation"]["columns"],
        model_sha256=manifest["model"]["sha256"],
        metrics=manifest["evaluation"]["metrics"],
    )
    manifest["pack_content_sha256"] = pack_content_sha256(
        stages=manifest["stages"],
        model=manifest["model"],
        files=manifest["files"],
        evaluation=manifest["evaluation"],
        provenance_sha256=manifest["provenance_sha256"],
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(EvalPackIntegrityError, match="canonical user snapshot"):
        load_match_eval_pack(
            pack_path,
            expected_pack_sha256=manifest["pack_content_sha256"],
        )


def test_eval_pack_rejects_wrong_external_model_sha_before_pickle(tmp_path, monkeypatch):
    result = _train_small(tmp_path)
    model_path = _model_path(result)
    pack_path = _pack_path(result)

    def fail_if_unpickled(_payload):
        pytest.fail("untrusted model bytes must not be deserialized")

    monkeypatch.setattr(compare_module.pickle, "loads", fail_if_unpickled)
    with pytest.raises(EvalPackIntegrityError, match="external trust anchor"):
        evaluate_model_against_pack(
            model_path,
            pack_path,
            expected_model_sha256="0" * 64,
            expected_pack_sha256=result["eval_pack_sha256"],
        )


def test_eval_pack_checks_declared_size_limit_before_hashing(tmp_path, monkeypatch):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)
    frame_path = pack_path / "frame.npz"
    monkeypatch.setitem(
        validation_module.FILE_SIZE_LIMITS,
        frame_path.name,
        frame_path.stat().st_size - 1,
    )

    original_reader = validation_module._read_bounded_regular_file

    def fail_if_oversized_member_is_read(path, **kwargs):
        if path.name == frame_path.name:
            pytest.fail("oversized files must be rejected before hashing or parsing")
        return original_reader(path, **kwargs)

    monkeypatch.setattr(
        validation_module,
        "_read_bounded_regular_file",
        fail_if_oversized_member_is_read,
    )
    with pytest.raises(EvalPackIntegrityError, match="size mismatch or limit exceeded"):
        load_match_eval_pack(
            pack_path,
            expected_pack_sha256=result["eval_pack_sha256"],
        )


def test_eval_pack_checks_aggregate_limit_before_reading_members(tmp_path, monkeypatch):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)
    total_bytes = sum(path.lstat().st_size for path in pack_path.iterdir())
    monkeypatch.setattr(validation_module, "MAX_PACK_BYTES", total_bytes - 1)

    def fail_if_any_member_is_read(*_args, **_kwargs):
        pytest.fail("aggregate size must be rejected before hashing or parsing")

    monkeypatch.setattr(
        validation_module,
        "_read_bounded_regular_file",
        fail_if_any_member_is_read,
    )
    with pytest.raises(EvalPackIntegrityError, match="aggregate size limit"):
        load_match_eval_pack(
            pack_path,
            expected_pack_sha256=result["eval_pack_sha256"],
        )


def test_metric_runtime_drift_is_reported_at_metric_stage(tmp_path, monkeypatch):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)
    monkeypatch.setattr(
        compare_module,
        "_runtime_metric_deltas",
        lambda _pack: {"ndcg@5": 1e-4, "ndcg@10": 0.0},
    )

    report = compare_match_eval_packs(
        pack_path,
        pack_path,
        expected_reference_sha256=result["eval_pack_sha256"],
        expected_candidate_sha256=result["eval_pack_sha256"],
        metric_atol=1e-6,
    )
    assert report["status"] == "drift"
    assert report["first_mismatch"] == "metrics"
    assert report["details"]["reference_runtime_deltas"]["ndcg@5"] == pytest.approx(1e-4)


def test_training_publishes_versioned_runs_without_overwriting(tmp_path):
    first = _train_small(tmp_path)
    first_model_sha = sha256_file(_model_path(first))
    first_pack_sha = first["eval_pack_sha256"]

    second = _train_small(tmp_path)

    assert first["run_dir"] != second["run_dir"]
    assert sha256_file(_model_path(first)) == first_model_sha
    assert first["eval_pack_sha256"] == first_pack_sha
    pointer = json.loads(Path(second["current_pointer_path"]).read_text(encoding="utf-8"))
    assert pointer["run_dir"] == Path(second["run_dir"]).name
    assert pointer["model"]["sha256"] == second["model_sha256"]
    assert pointer["evaluation_pack"]["sha256"] == second["eval_pack_sha256"]
    current = load_current_match_run(Path(second["current_pointer_path"]))
    assert current.model_path == _model_path(second)
    assert current.evaluation_pack_path == _pack_path(second)
    assert current.evaluation_pack_sha256 == second["eval_pack_sha256"]


def test_pack_publish_race_preserves_appearing_target(tmp_path, monkeypatch):
    original_publish = format_module._publish_new_directory

    def inject_competing_target(staging, target):
        target.mkdir()
        (target / "sentinel.txt").write_text("keep", encoding="utf-8")
        return original_publish(staging, target)

    monkeypatch.setattr(
        format_module,
        "_publish_new_directory",
        inject_competing_target,
    )
    with pytest.raises(EvalPackError, match="appeared during generation"):
        _train_small(tmp_path)

    sentinels = list(tmp_path.glob("match_v2_run_*/match_v2_eval_pack/sentinel.txt"))
    assert len(sentinels) == 1
    assert sentinels[0].read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "match_v2_current.json").exists()


def test_failed_new_run_leaves_previous_current_pointer_unchanged(tmp_path, monkeypatch):
    first = _train_small(tmp_path)
    pointer_path = Path(first["current_pointer_path"])
    pointer_before = pointer_path.read_bytes()
    original_write = train_module._write_new_json

    def fail_metrics_write(path, payload):
        if path.name == "metrics_v2.json":
            raise OSError("injected metrics failure")
        return original_write(path, payload)

    monkeypatch.setattr(train_module, "_write_new_json", fail_metrics_write)
    with pytest.raises(OSError, match="injected metrics failure"):
        _train_small(tmp_path)

    assert pointer_path.read_bytes() == pointer_before
    assert sha256_file(_model_path(first)) == first["model_sha256"]


def test_replay_output_must_not_overlap_reference_inputs(tmp_path):
    result = _train_small(tmp_path / "reference")
    reference = _pack_path(result)
    model_path = _model_path(result)

    with pytest.raises(EvalPackError, match="must not overlap"):
        replay_match_eval_pack(
            reference,
            model_path=model_path,
            expected_model_sha256=sha256_file(model_path),
            expected_reference_sha256=result["eval_pack_sha256"],
            encoder=StableFakeEncoder(),
            output_dir=reference / "candidate",
            config_path=Path("configs/module2_matching.yaml"),
        )


def test_eval_and_parity_clis_use_external_hashes_and_protect_pack(tmp_path):
    result = _train_small(tmp_path)
    model_path = _model_path(result)
    pack_path = _pack_path(result)
    model_sha = sha256_file(model_path)
    pack_sha = result["eval_pack_sha256"]

    evaluate_command = [
        sys.executable,
        "scripts/evaluate_match_v2_pack.py",
        "--model",
        str(model_path),
        "--eval-pack",
        str(pack_path),
        "--expected-model-sha256",
        model_sha,
        "--expected-pack-sha256",
        pack_sha,
    ]
    evaluated = subprocess.run(
        evaluate_command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert json.loads(evaluated.stdout)["status"] == "match"

    protected_report = pack_path / "report.json"
    rejected = subprocess.run(
        [*evaluate_command, "--out", str(protected_report)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert not protected_report.exists()

    parity = subprocess.run(
        [
            sys.executable,
            "scripts/verify_match_v2_parity.py",
            "--reference",
            str(pack_path),
            "--expected-reference-sha256",
            pack_sha,
            "--candidate",
            str(pack_path),
            "--expected-candidate-sha256",
            pack_sha,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert parity.returncode == 0, parity.stderr
    assert json.loads(parity.stdout)["status"] == "match"

    mutable_encoder = subprocess.run(
        [
            sys.executable,
            "scripts/replay_match_v2_pack.py",
            "--reference",
            str(pack_path),
            "--expected-reference-sha256",
            pack_sha,
            "--model",
            str(model_path),
            "--expected-model-sha256",
            model_sha,
            "--out",
            str(tmp_path / "replayed"),
            "--encoder-model",
            "stable-test-encoder",
            "--encoder-revision",
            "main",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert mutable_encoder.returncode == 2
    assert "immutable" in mutable_encoder.stderr
    assert not (tmp_path / "replayed").exists()


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ([2], "cover every prediction row"),
        ([1, 0, 2], "positive integers"),
    ],
)
def test_ndcg_rejects_invalid_query_boundaries(groups, message):
    y_true = np.asarray([3, 2, 1], dtype=np.int32)
    y_pred = np.asarray([0.8, 0.5, 0.2], dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        ndcg_at_k(y_true, y_pred, groups, 5)


def test_ndcg_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        ndcg_at_k(
            np.asarray([3, 1], dtype=np.int32),
            np.asarray([0.8, np.nan], dtype=np.float64),
            [2],
            5,
        )


def test_pack_comparison_rejects_infinite_tolerance(tmp_path):
    result = _train_small(tmp_path)
    pack_path = _pack_path(result)
    with pytest.raises(ValueError, match="finite"):
        compare_match_eval_packs(
            pack_path,
            pack_path,
            expected_reference_sha256=result["eval_pack_sha256"],
            expected_candidate_sha256=result["eval_pack_sha256"],
            score_atol=float("inf"),
        )
