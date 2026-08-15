"""MATCH 모듈② v2 랭커의 SHAP 근거 생성 (W3 DoD "SHAP 근거").

구형 scripts/generate_shap_module2.py 는 legacy 3열 모델(f_cosine/f_l2/f_dis_match)에
np.random 으로 만든 입력을 먹여 SHAP 을 그렸다. 서빙 중인 v2 모델(15열)과도, 실제
사용자 분포와도 무관한 그림이라 근거로 쓸 수 없어 이 스크립트로 대체한다.

여기서는 서빙과 같은 v2 번들과 같은 build_pair_features 경로를 써서 세 가지를 만든다.

  1. 전역 기여도 — 특성별 mean |SHAP| 과 점유율
  2. 페어별 상위 기여 특성
  3. 정합성 — 화면에 보여주는 추천 사유(build_recommendation_reasons)가 실제로
     모델이 점수를 올리는 데 쓴 신호인지

3번이 이 스크립트의 핵심이다. 사유가 모델 기여와 무관하면 화면 문구는 장식이 된다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_profiles_v2 import DEFAULT_CONFIG_PATH  # noqa: E402
from src.data.matching_input import (  # noqa: E402
    ALLOWED_RECOMMENDATION_REASONS,
    MatchingInputPolicy,
    TextEncoder,
    build_recommendation_reasons,
)
from src.data.matching_trainset import (  # noqa: E402
    build_holdout_frame,
    load_serving_policy,
    load_v2_feature_columns,
)

SCHEMA_VERSION = 1

# 특성 -> 화면 문구. build_recommendation_reasons 가 각 문구를 켤 때 보는 신호와 1:1로 맞춘다.
# 여기에 없는 v2 특성은 사용자에게 노출할 문구가 없는 것으로 취급한다.
FEATURE_REASON: dict[str, str] = {
    "f_tag_overlap": "관심사가 비슷해요",
    "f_tag_jaccard": "관심사가 비슷해요",
    "f_liked_cosine": "관심 있는 콘텐츠가 비슷해요",
    "f_authored_cosine": "관심 있는 콘텐츠가 비슷해요",
    "f_liked_authored_cosine": "관심 있는 콘텐츠가 비슷해요",
    "f_common_friend_count": "공통 친구가 있어요",
    # 연령 문구는 f_age_band_match 가 아니라 f_age_diff 에 걸려 있다
    # (build_recommendation_reasons 의 age_reason_max_diff 판정과 일치).
    "f_age_diff": "비슷한 연령대예요",
}

# build_recommendation_reasons 판정에 필요하지만 v2 15열에는 없는 가용성 플래그.
# 특성 프레임에는 넣고 SHAP 입력에서는 뺀다.
REASON_SUPPORT_COLUMNS = ("f_liked_authored_available",)


def _assert_reason_vocabulary() -> None:
    """화면 문구를 새로 만들지 않고 기존 allowlist만 쓰는지 고정한다."""

    unknown = set(FEATURE_REASON.values()) - set(ALLOWED_RECOMMENDATION_REASONS)
    if unknown:
        raise ValueError(f"허용되지 않은 추천 사유 문구: {sorted(unknown)}")


def load_v2_bundle(model_path: Path) -> dict[str, Any]:
    """train_match_v2 가 저장하는 {model, columns, params, metrics} 번들을 읽는다."""

    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or "model" not in bundle or "columns" not in bundle:
        raise ValueError(
            f"{model_path} 는 v2 번들이 아닙니다. "
            "{'model','columns',...} 형태여야 하며 구형 bare pickle은 쓸 수 없습니다."
        )
    columns = list(bundle["columns"])
    expected = load_v2_feature_columns()
    if columns != expected:
        raise ValueError(
            "번들 columns 가 config v2_pair_features 와 다릅니다.\n"
            f"  번들 : {columns}\n  config: {expected}"
        )
    return bundle


def resolve_model_path(
    *,
    model_path: Path | None,
    hf_repo: str | None,
    hf_file: str,
    hf_revision: str | None,
) -> Path:
    """로컬 경로를 우선하고, 없으면 revision 고정으로 HF에서 받는다."""

    if model_path is not None:
        if not model_path.is_file():
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
        return model_path
    if hf_repo is None:
        raise ValueError("--model 또는 --hf-repo 중 하나는 필요합니다.")
    if not hf_revision:
        raise ValueError(
            "--hf-revision 이 필요합니다. 재현 가능한 근거를 만들려면 revision을 고정해야 합니다."
        )
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=hf_repo, filename=hf_file, revision=hf_revision))


def build_evaluation_frame(
    *,
    encoder: TextEncoder,
    v2_columns: list[str],
    n_users: int,
    n_queries: int,
    n_candidates: int,
    seed: int,
    config_path: Path,
) -> tuple[np.ndarray, list[str], Any]:
    """공유 홀드아웃 프레임에 사유 판정용 가용성 플래그를 덧붙여 만든다."""

    frame_columns = list(dict.fromkeys([*v2_columns, *REASON_SUPPORT_COLUMNS]))
    frame = build_holdout_frame(
        encoder=encoder,
        columns=frame_columns,
        n_users=n_users,
        n_queries=n_queries,
        n_candidates=n_candidates,
        seed=seed,
        config_path=config_path,
    )
    return frame.x, frame_columns, frame


def compute_global_importance(
    shap_values: np.ndarray, v2_columns: list[str]
) -> list[dict[str, Any]]:
    """특성별 mean |SHAP| 과 전체 대비 점유율."""

    mean_abs = np.abs(shap_values).mean(axis=0)
    total = float(mean_abs.sum())
    rows = [
        {
            "feature": column,
            "mean_abs_shap": float(value),
            "share": (float(value) / total if total > 0 else 0.0),
            "user_facing_reason": FEATURE_REASON.get(column),
        }
        for column, value in zip(v2_columns, mean_abs, strict=True)
    ]
    rows.sort(key=lambda row: row["mean_abs_shap"], reverse=True)
    return rows


def _row_features(row: np.ndarray, frame_columns: list[str]) -> dict[str, float]:
    return {column: float(value) for column, value in zip(frame_columns, row, strict=True)}


def analyze_reason_alignment(
    *,
    shap_values: np.ndarray,
    x_full: np.ndarray,
    frame_columns: list[str],
    v2_columns: list[str],
    policy: MatchingInputPolicy,
    top_k: int,
    n_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """화면 사유와 모델 기여가 맞물리는지 센다.

    - displayed  : build_recommendation_reasons 가 실제로 화면에 내보내는 문구
    - supported  : SHAP 이 양(+)으로 기여했다고 말하는 신호에서 유도되는 문구
    두 집합을 비교해 '보여주는 이유가 모델이 실제로 쓴 이유인지'를 본다.
    """

    n_rows = shap_values.shape[0]
    displayed_total = 0
    displayed_supported = 0
    unsupported_examples: dict[str, int] = {}
    missing_from_display: dict[str, int] = {}
    top_driver_exposed = 0
    rows_with_display = 0

    samples: list[dict[str, Any]] = []
    for index in range(n_rows):
        feats = _row_features(x_full[index], frame_columns)
        displayed = build_recommendation_reasons(feats, policy=policy)

        contributions = sorted(
            zip(v2_columns, shap_values[index], strict=True),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        if FEATURE_REASON.get(contributions[0][0]) is not None:
            top_driver_exposed += 1

        supported: list[str] = []
        for column, value in contributions:
            if value <= 0:
                continue
            reason = FEATURE_REASON.get(column)
            if reason is not None and reason not in supported:
                supported.append(reason)

        displayed_total += len(displayed)
        for reason in displayed:
            if reason in supported:
                displayed_supported += 1
            else:
                unsupported_examples[reason] = unsupported_examples.get(reason, 0) + 1
        if displayed:
            rows_with_display += 1
        for reason in supported:
            if reason not in displayed:
                missing_from_display[reason] = missing_from_display.get(reason, 0) + 1

        if len(samples) < n_samples:
            samples.append(
                {
                    "row": index,
                    "displayed_reasons": displayed,
                    "shap_supported_reasons": supported,
                    "top_features": [
                        {
                            "feature": column,
                            "shap": float(value),
                            "value": feats[column],
                            "user_facing_reason": FEATURE_REASON.get(column),
                        }
                        for column, value in contributions[:top_k]
                    ],
                }
            )

    summary = {
        "n_rows": int(n_rows),
        "rows_with_displayed_reason": int(rows_with_display),
        "displayed_reason_count": int(displayed_total),
        "displayed_backed_by_positive_shap": int(displayed_supported),
        "displayed_support_rate": (
            float(displayed_supported / displayed_total) if displayed_total else 0.0
        ),
        "top_driver_is_user_facing_rate": float(top_driver_exposed / n_rows) if n_rows else 0.0,
        "displayed_without_shap_support": dict(
            sorted(unsupported_examples.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "shap_supported_but_not_displayed": dict(
            sorted(missing_from_display.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }
    return summary, samples


def generate(
    *,
    encoder: TextEncoder,
    model_path: Path,
    out_path: Path,
    n_users: int,
    n_queries: int,
    n_candidates: int,
    seed: int,
    top_k: int,
    n_samples: int,
    config_path: Path | None = None,
) -> dict[str, Any]:
    _assert_reason_vocabulary()
    config_path = config_path or DEFAULT_CONFIG_PATH

    bundle = load_v2_bundle(model_path)
    booster = bundle["model"]
    v2_columns = list(bundle["columns"])

    x_full, frame_columns, _frame = build_evaluation_frame(
        encoder=encoder,
        v2_columns=v2_columns,
        n_users=n_users,
        n_queries=n_queries,
        n_candidates=n_candidates,
        seed=seed,
        config_path=config_path,
    )
    if x_full.shape[0] == 0:
        raise RuntimeError("설명할 페어가 없습니다. n_users/n_queries를 늘리세요.")

    v2_index = [frame_columns.index(column) for column in v2_columns]
    x_v2 = x_full[:, v2_index]

    import shap

    explainer = shap.TreeExplainer(booster)
    shap_values = np.asarray(explainer.shap_values(x_v2), dtype=np.float64)
    if shap_values.shape != x_v2.shape:
        raise RuntimeError(f"예상치 못한 SHAP 형태: {shap_values.shape} != {x_v2.shape}")

    policy = load_serving_policy(config_path)
    alignment, samples = analyze_reason_alignment(
        shap_values=shap_values,
        x_full=x_full,
        frame_columns=frame_columns,
        v2_columns=v2_columns,
        policy=policy,
        top_k=top_k,
        n_samples=n_samples,
    )
    global_importance = compute_global_importance(shap_values, v2_columns)
    unexposed = {
        row["feature"]: row["share"]
        for row in global_importance
        if row["user_facing_reason"] is None
    }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "path": str(model_path),
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "columns": v2_columns,
            "metrics": bundle.get("metrics"),
        },
        "sample": {
            "n_users": n_users,
            "n_queries": n_queries,
            "n_candidates": n_candidates,
            "seed": seed,
            "n_pairs": int(x_v2.shape[0]),
            "encoder": (
                encoder.reproducibility_metadata()
                if hasattr(encoder, "reproducibility_metadata")
                else {"type": type(encoder).__name__}
            ),
        },
        "reason_policy": {
            "content_reason_min": policy.content_reason_min,
            "tag_reason_min_overlap": policy.tag_reason_min_overlap,
            "age_reason_max_diff": policy.age_reason_max_diff,
            "max_reasons": policy.max_reasons,
        },
        "global_importance": global_importance,
        "unexposed_feature_share": unexposed,
        "reason_alignment": alignment,
        "samples": samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="로컬 v2 번들 pickle 경로")
    parser.add_argument("--hf-repo", default=None, help="예: soyuncj/module2")
    parser.add_argument("--hf-file", default="module2_lambdamart_v2.pkl")
    parser.add_argument("--hf-revision", default=None, help="고정 commit sha (필수)")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "match_v2_shap.json")
    # 기본값은 train_match_v2.py 학습 기본값과 일치시킨다. 그래야 test 분할이 실제 holdout 이다.
    parser.add_argument("--n-users", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    from scripts.train_match_v2 import _build_sbert

    model_path = resolve_model_path(
        model_path=args.model,
        hf_repo=args.hf_repo,
        hf_file=args.hf_file,
        hf_revision=args.hf_revision,
    )
    payload = generate(
        encoder=_build_sbert(DEFAULT_CONFIG_PATH),
        model_path=model_path,
        out_path=args.out,
        n_users=args.n_users,
        n_queries=args.queries,
        n_candidates=args.candidates,
        seed=args.seed,
        top_k=args.top_k,
        n_samples=args.samples,
    )

    print(f"pairs: {payload['sample']['n_pairs']:,}")
    print("\n=== 전역 기여도 (mean |SHAP|) ===")
    for row in payload["global_importance"]:
        reason = row["user_facing_reason"] or "-"
        print(f"  {row['feature']:<26} {row['mean_abs_shap']:.4f}  {row['share']:6.1%}  {reason}")
    alignment = payload["reason_alignment"]
    print("\n=== 사유 정합성 ===")
    print(
        f"  화면 사유 {alignment['displayed_reason_count']}건 중 "
        f"SHAP 양(+) 기여로 뒷받침 {alignment['displayed_backed_by_positive_shap']}건 "
        f"({alignment['displayed_support_rate']:.1%})"
    )
    print(
        f"  최상위 기여 특성이 화면 문구로 이어지는 비율: "
        f"{alignment['top_driver_is_user_facing_rate']:.1%}"
    )
    if alignment["displayed_without_shap_support"]:
        print(f"  근거 없이 노출된 문구: {alignment['displayed_without_shap_support']}")
    if alignment["shap_supported_but_not_displayed"]:
        print(f"  기여했으나 미노출: {alignment['shap_supported_but_not_displayed']}")
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
