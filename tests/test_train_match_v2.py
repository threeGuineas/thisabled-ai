"""P3 학습 파이프라인 로컬 스모크 (fake encoder + 실제 LightGBM).

SBERT만 Colab에서 돌리고, 나머지 전체 경로(생성→분할→임베딩 집계→페어 특성→
LambdaMART 학습→NDCG)는 로컬에서 검증한다.
"""

from __future__ import annotations

import numpy as np

from scripts.train_match_v2 import run
from src.data.matching_trainset import load_v2_feature_columns


class FakeEncoder:
    """텍스트 해시 기반 결정적 벡터(SBERT 대역)."""

    def encode(self, sentences, *, batch_size, show_progress_bar):
        rows = []
        for text in sentences:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
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
    assert (tmp_path / "module2_lambdamart_v2.pkl").exists()
    assert (tmp_path / "metrics_v2.json").exists()
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
