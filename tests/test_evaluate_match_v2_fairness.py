"""MATCH v2 공정성 평가기의 versioned run 연동 회귀 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.evaluate_match_v2_fairness as fairness_module
import scripts.train_match_v2 as train_module
import src.data.match_eval_pack.compare as compare_module
from src.data.match_eval_pack import (
    MatchRunIntegrityError,
    load_current_match_run,
)
from src.data.match_eval_pack.format import canonical_json_bytes


class StableFakeEncoder:
    def encode(self, sentences, *, batch_size, show_progress_bar):
        rows = []
        for text in sentences:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rows.append(np.random.default_rng(seed).normal(size=24))
        return np.asarray(rows, dtype=np.float32)


def _train_small(out_dir: Path) -> dict:
    return train_module.run(
        encoder=StableFakeEncoder(),
        n_users=160,
        n_train_queries=40,
        n_test_queries=16,
        n_candidates=6,
        seed=31,
        out_dir=out_dir,
        ablations=False,
        n_estimators=12,
        write_eval_pack=False,
    )


def test_default_current_pointer_resolves_versioned_model_and_runs_fairness(
    tmp_path,
    monkeypatch,
):
    trained = _train_small(tmp_path / "artifacts")
    pointer_path = Path(trained["current_pointer_path"])
    current = load_current_match_run(pointer_path)
    assert current.model_path == Path(trained["model_path"])
    assert current.model_sha256 == trained["model_sha256"]

    monkeypatch.setattr(train_module, "_build_sbert", lambda _config: StableFakeEncoder())
    report_path = tmp_path / "fairness.json"
    exit_code = fairness_module.main(
        [
            "--current-pointer",
            str(pointer_path),
            "--n-users",
            "160",
            "--test-queries",
            "16",
            "--candidates",
            "6",
            "--k",
            "3",
            "--out",
            str(report_path),
        ]
    )

    assert exit_code in {0, 1}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model"]["path"] == trained["model_path"]
    assert report["model"]["sha256"] == trained["model_sha256"]
    assert report["n_test_slates"] > 0


def test_current_pointer_rejects_path_escape_and_incomplete_manifest(tmp_path):
    trained = _train_small(tmp_path / "artifacts")
    pointer_path = Path(trained["current_pointer_path"])
    original_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))

    escaped = json.loads(json.dumps(original_pointer))
    escaped["model"]["path"] = "../outside.pkl"
    pointer_path.write_bytes(canonical_json_bytes(escaped))
    with pytest.raises(MatchRunIntegrityError, match="invalid path"):
        load_current_match_run(pointer_path)

    pointer_path.write_bytes(canonical_json_bytes(original_pointer))
    manifest_path = Path(trained["run_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "incomplete"
    manifest_payload = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_payload)
    original_pointer["run_manifest"]["sha256"] = hashlib.sha256(manifest_payload).hexdigest()
    pointer_path.write_bytes(canonical_json_bytes(original_pointer))
    with pytest.raises(MatchRunIntegrityError, match="not a supported complete run"):
        load_current_match_run(pointer_path)


def test_current_pointer_rejects_duplicate_keys(tmp_path):
    trained = _train_small(tmp_path / "artifacts")
    pointer_path = Path(trained["current_pointer_path"])
    payload = pointer_path.read_text(encoding="utf-8").replace(
        '"schema_version":',
        '"schema_version":"duplicate","schema_version":',
        1,
    )
    pointer_path.write_text(payload, encoding="utf-8")

    with pytest.raises(MatchRunIntegrityError, match="invalid JSON"):
        load_current_match_run(pointer_path)


def test_default_cli_rejects_model_corruption_before_pickle(tmp_path, monkeypatch):
    trained = _train_small(tmp_path / "artifacts")
    Path(trained["model_path"]).write_bytes(b"not-a-model")

    def fail_if_unpickled(_payload):
        pytest.fail("model with a mismatched pointer SHA must not be deserialized")

    monkeypatch.setattr(compare_module.pickle, "loads", fail_if_unpickled)
    with pytest.raises(SystemExit) as exc_info:
        fairness_module.main(
            [
                "--current-pointer",
                trained["current_pointer_path"],
                "--out",
                str(tmp_path / "fairness.json"),
            ]
        )
    assert exc_info.value.code == 2


def test_cli_requires_sha_for_explicit_model_and_protects_current_run(tmp_path):
    trained = _train_small(tmp_path / "artifacts")
    with pytest.raises(SystemExit) as exc_info:
        fairness_module.main(["--model", trained["model_path"]])
    assert exc_info.value.code == 2

    with pytest.raises(SystemExit) as exc_info:
        fairness_module.main(
            [
                "--current-pointer",
                trained["current_pointer_path"],
                "--out",
                str(Path(trained["run_dir"]) / "fairness.json"),
            ]
        )
    assert exc_info.value.code == 2
