"""SHAP v2 근거 생성 로컬 스모크 (fake encoder + 실제 LightGBM/SHAP).

SBERT만 무거우므로 대역을 쓰고, 번들 검증 -> 특성 프레임 -> SHAP -> 사유 정합성까지
전 경로를 로컬에서 검증한다.
"""

from __future__ import annotations

import hashlib
import json
import pickle

import lightgbm as lgb
import numpy as np
import pytest

from scripts.generate_shap_match_v2 import (
    FEATURE_REASON,
    analyze_reason_alignment,
    compute_global_importance,
    generate,
    load_v2_bundle,
)
from src.data.matching_input import ALLOWED_RECOMMENDATION_REASONS, MatchingInputPolicy
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


def _make_bundle(tmp_path, columns):
    """v2 번들 형태로 저장된 소형 LambdaMART."""

    rng = np.random.default_rng(0)
    n = 240
    x = rng.normal(size=(n, len(columns)))
    y = rng.integers(0, 4, size=n)
    groups = [12] * (n // 12)
    dataset = lgb.Dataset(x, label=y, group=groups)
    booster = lgb.train(
        {"objective": "lambdarank", "num_leaves": 7, "verbose": -1},
        dataset,
        num_boost_round=15,
    )
    path = tmp_path / "module2_lambdamart_v2.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {"model": booster, "columns": list(columns), "params": {}, "metrics": {}}, handle
        )
    return path


def test_reason_vocabulary_stays_inside_allowlist():
    """SHAP 매핑이 화면 문구를 새로 만들어내지 않아야 한다."""

    assert set(FEATURE_REASON.values()) <= set(ALLOWED_RECOMMENDATION_REASONS)


def test_load_v2_bundle_rejects_legacy_bare_pickle(tmp_path):
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump(object(), handle)
    with pytest.raises(ValueError, match="v2 번들이 아닙니다"):
        load_v2_bundle(path)


def test_load_v2_bundle_rejects_column_mismatch(tmp_path):
    path = _make_bundle(tmp_path, ["f_effective_cosine", "f_tag_overlap"])
    with pytest.raises(ValueError, match="config v2_pair_features"):
        load_v2_bundle(path)


def test_global_importance_shares_sum_to_one():
    columns = ["f_tag_overlap", "f_age_diff", "f_ui_mode_match"]
    shap_values = np.array([[1.0, -2.0, 0.0], [3.0, 0.0, 1.0]])
    rows = compute_global_importance(shap_values, columns)

    assert [row["feature"] for row in rows] == ["f_tag_overlap", "f_age_diff", "f_ui_mode_match"]
    assert sum(row["share"] for row in rows) == pytest.approx(1.0)
    # 노출 문구가 없는 특성은 None 으로 표시된다.
    assert rows[0]["user_facing_reason"] == "관심사가 비슷해요"
    assert rows[-1]["user_facing_reason"] is None


def test_alignment_counts_displayed_reason_backed_by_positive_shap():
    v2_columns = ["f_tag_overlap", "f_common_friend_count"]
    frame_columns = [*v2_columns, "f_profile_available"]
    # 두 신호 모두 켜진 페어 하나.
    x_full = np.array([[2.0, 1.0, 0.0]])
    # 태그는 양(+) 기여, 공통 친구는 음(-) 기여.
    shap_values = np.array([[0.9, -0.4]])

    summary, samples = analyze_reason_alignment(
        shap_values=shap_values,
        x_full=x_full,
        frame_columns=frame_columns,
        v2_columns=v2_columns,
        policy=MatchingInputPolicy(),
        top_k=2,
        n_samples=1,
    )

    assert samples[0]["displayed_reasons"] == ["관심사가 비슷해요", "공통 친구가 있어요"]
    # 양의 기여를 낸 태그만 뒷받침된다.
    assert samples[0]["shap_supported_reasons"] == ["관심사가 비슷해요"]
    assert summary["displayed_reason_count"] == 2
    assert summary["displayed_backed_by_positive_shap"] == 1
    assert summary["displayed_support_rate"] == pytest.approx(0.5)
    assert summary["displayed_without_shap_support"] == {"공통 친구가 있어요": 1}
    assert summary["top_driver_is_user_facing_rate"] == pytest.approx(1.0)


def test_generate_end_to_end_writes_evidence(tmp_path):
    columns = load_v2_feature_columns()
    model_path = _make_bundle(tmp_path, columns)
    out_path = tmp_path / "match_v2_shap.json"

    payload = generate(
        encoder=FakeEncoder(),
        model_path=model_path,
        out_path=out_path,
        n_users=300,
        n_queries=40,
        n_candidates=10,
        seed=7,
        top_k=3,
        n_samples=5,
    )

    assert payload["model"]["columns"] == columns
    assert payload["sample"]["n_pairs"] > 0
    # 전역 기여도는 15열 전부를 덮는다.
    assert {row["feature"] for row in payload["global_importance"]} == set(columns)
    assert len(payload["samples"]) == 5
    for sample in payload["samples"]:
        assert len(sample["top_features"]) == 3
        assert set(sample["displayed_reasons"]) <= set(ALLOWED_RECOMMENDATION_REASONS)
        assert set(sample["shap_supported_reasons"]) <= set(ALLOWED_RECOMMENDATION_REASONS)

    # MATCH-04: UI 모드는 화면 문구로 이어지지 않아야 한다.
    assert "f_ui_mode_match" in payload["unexposed_feature_share"]

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["model"]["sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()


def test_generate_is_deterministic_for_same_seed(tmp_path):
    columns = load_v2_feature_columns()
    model_path = _make_bundle(tmp_path, columns)

    runs = []
    for name in ("a", "b"):
        payload = generate(
            encoder=FakeEncoder(),
            model_path=model_path,
            out_path=tmp_path / f"{name}.json",
            n_users=300,
            n_queries=40,
            n_candidates=10,
            seed=7,
            top_k=3,
            n_samples=3,
        )
        runs.append(payload["global_importance"])

    assert runs[0] == runs[1]
