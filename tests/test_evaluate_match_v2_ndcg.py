"""NDCG 독립 재측정 로컬 스모크 (fake encoder + 실제 LightGBM).

SBERT만 대역을 쓰고 번들 검증 -> 홀드아웃 프레임 -> 채점 -> 게이트 판정까지 검증한다.
"""

from __future__ import annotations

import hashlib
import json
import pickle

import lightgbm as lgb
import numpy as np
import pytest

from scripts.evaluate_match_v2_ndcg import NDCG_GATE, evaluate
from src.data.matching_trainset import load_serving_policy, load_v2_feature_columns


class FakeEncoder:
    """텍스트 해시 기반 결정적 벡터(SBERT 대역)."""

    def encode(self, sentences, *, batch_size, show_progress_bar):
        rows = []
        for text in sentences:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            rows.append(rng.normal(size=32))
        return np.asarray(rows, dtype=np.float32)


def _make_bundle(tmp_path, columns, *, metrics=None):
    rng = np.random.default_rng(0)
    n = 240
    x = rng.normal(size=(n, len(columns)))
    y = rng.integers(0, 4, size=n)
    dataset = lgb.Dataset(x, label=y, group=[12] * (n // 12))
    booster = lgb.train(
        {"objective": "lambdarank", "num_leaves": 7, "verbose": -1},
        dataset,
        num_boost_round=15,
    )
    path = tmp_path / "module2_lambdamart_v2.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {"model": booster, "columns": list(columns), "params": {}, "metrics": metrics or {}},
            handle,
        )
    return path


def _run(tmp_path, model_path, **kwargs):
    return evaluate(
        encoder=FakeEncoder(),
        model_path=model_path,
        out_path=tmp_path / "ndcg.json",
        n_users=300,
        n_queries=40,
        n_candidates=10,
        seed=7,
        **kwargs,
    )


def test_rejects_bundle_whose_columns_drifted_from_config(tmp_path):
    path = _make_bundle(tmp_path, ["f_effective_cosine", "f_tag_overlap"])
    with pytest.raises(ValueError, match="config"):
        _run(tmp_path, path)


def test_rejects_legacy_bare_pickle(tmp_path):
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump(object(), handle)
    with pytest.raises(ValueError, match="v2 번들이 아닙니다"):
        _run(tmp_path, path)


def test_measures_ndcg_and_records_provenance(tmp_path):
    columns = load_v2_feature_columns()
    model_path = _make_bundle(tmp_path, columns)

    payload = _run(tmp_path, model_path)

    assert payload["sample"]["n_pairs"] > 0
    assert payload["sample"]["n_scored_queries"] > 0
    for key in ("ndcg@5", "ndcg@10"):
        assert 0.0 <= payload["measured"][key] <= 1.0
    assert payload["gate"]["threshold"] == NDCG_GATE
    assert payload["gate"]["pass"] == (payload["measured"]["ndcg@10"] >= NDCG_GATE)
    assert payload["model"]["sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()

    written = json.loads((tmp_path / "ndcg.json").read_text(encoding="utf-8"))
    assert written["measured"] == payload["measured"]


def test_reports_diff_against_metrics_recorded_in_bundle(tmp_path):
    """번들 기록값과 재측정값의 차이를 드러내야 배포 파일 불일치를 잡을 수 있다."""

    columns = load_v2_feature_columns()
    model_path = _make_bundle(
        tmp_path, columns, metrics={"v2_full": {"ndcg@5": 0.0, "ndcg@10": 0.0}}
    )

    payload = _run(tmp_path, model_path)

    assert payload["recorded_in_bundle"] == {"ndcg@5": 0.0, "ndcg@10": 0.0}
    assert payload["recorded_vs_measured_diff"]["ndcg@10"] == pytest.approx(
        payload["measured"]["ndcg@10"]
    )


def test_is_deterministic_for_same_seed(tmp_path):
    columns = load_v2_feature_columns()
    model_path = _make_bundle(tmp_path, columns)

    first = _run(tmp_path, model_path)["measured"]
    second = _run(tmp_path, model_path)["measured"]

    assert first == second


def test_serving_policy_comes_from_config_not_dataclass_defaults():
    """배포 임계값으로 평가해야 하므로 config 로더를 거치는지 고정한다."""

    policy = load_serving_policy()

    assert policy.tag_reason_min_overlap == 3
    assert policy.age_reason_max_diff == 5
    assert policy.allowed_tag_ids
