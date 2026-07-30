"""P3 학습 파이프라인 로컬 스모크 (fake encoder + 실제 LightGBM).

SBERT만 Colab에서 돌리고, 나머지 전체 경로(생성→분할→임베딩 집계→페어 특성→
LambdaMART 학습→NDCG)는 로컬에서 검증한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.train_match_v2 import run
from src.data.match_eval_pack import compare_match_eval_packs, load_match_eval_pack
from src.data.matching_trainset import load_v2_feature_columns


class FakeEncoder:
    """텍스트 해시 기반 결정적 벡터(SBERT 대역)."""

    def encode(self, sentences, *, batch_size, show_progress_bar):
        rows = []
        for text in sentences:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            rows.append(rng.normal(size=32))
        return np.asarray(rows, dtype=np.float32)


def test_v2_training_pipeline_runs_end_to_end(tmp_path):
    result = run(
        encoder=FakeEncoder(),
        n_users=400,
        n_train_queries=120,
        n_test_queries=40,
        n_candidates=12,
        seed=13,
        out_dir=tmp_path,
        ablations=True,
        n_estimators=40,
    )
    # 열이 config v2_pair_features와 일치.
    assert result["columns"] == load_v2_feature_columns()
    # 학습/평가 페어가 생성됨.
    assert result["train_pairs"] > 0 and result["test_pairs"] > 0
    # 세 모델 모두 유한한 NDCG를 낸다.
    for key in ("v2_full", "v2_no_content", "legacy_embedding"):
        m = result["metrics"][key]
        assert 0.0 <= m["ndcg@5"] <= 1.0
        assert 0.0 <= m["ndcg@10"] <= 1.0
    # 산출물이 저장됨.
    model_path = Path(result["model_path"])
    metrics_path = Path(result["metrics_path"])
    pack_path = Path(result["eval_pack_path"])
    assert model_path.exists()
    assert metrics_path.exists()
    assert pack_path == Path(result["run_dir"]) / "match_v2_eval_pack"
    assert Path(result["current_pointer_path"]).exists()
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved_metrics["eval_pack_sha256"] == result["eval_pack_sha256"]
    pack = load_match_eval_pack(
        pack_path,
        expected_pack_sha256=result["eval_pack_sha256"],
    )
    for key in ("ndcg@5", "ndcg@10"):
        assert pack.manifest["evaluation"]["metrics"][key] == result["metrics"]["v2_full"][key]
    # gain 중요도가 열별로 채워짐.
    assert set(result["gain_importance"]) == set(result["columns"])


def test_v2_training_is_deterministic(tmp_path):
    common = dict(
        encoder=FakeEncoder(),
        n_users=300,
        n_train_queries=80,
        n_test_queries=30,
        n_candidates=10,
        seed=7,
        ablations=False,
        n_estimators=30,
    )
    a = run(out_dir=tmp_path / "a", **common)
    b = run(out_dir=tmp_path / "b", **common)
    assert a["metrics"]["v2_full"] == b["metrics"]["v2_full"]
    parity = compare_match_eval_packs(
        Path(a["eval_pack_path"]),
        Path(b["eval_pack_path"]),
        expected_reference_sha256=a["eval_pack_sha256"],
        expected_candidate_sha256=b["eval_pack_sha256"],
    )
    assert parity["status"] == "match"


def test_v2_training_can_skip_eval_pack(tmp_path):
    result = run(
        encoder=FakeEncoder(),
        n_users=120,
        n_train_queries=30,
        n_test_queries=12,
        n_candidates=6,
        seed=29,
        out_dir=tmp_path,
        ablations=False,
        n_estimators=10,
        write_eval_pack=False,
    )
    assert "eval_pack_path" not in result
    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert Path(result["current_pointer_path"]).exists()
