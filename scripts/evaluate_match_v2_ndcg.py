"""배포된 MATCH v2 랭커의 NDCG를 독립 재측정한다 (W1 DoD 증빙).

번들에 기록된 metrics 는 학습 시점에 학습 스크립트가 스스로 남긴 값이다. 이 스크립트는
그 값을 믿지 않고, 같은 설정으로 test 분할을 다시 만들어 배포 pickle 로 직접 채점한다.
두 값이 어긋나면 배포된 파일이 그 metrics 를 낸 모델이 아니라는 뜻이다.

프레임 구성은 matching_trainset.build_holdout_frame 을 그대로 쓴다(학습·SHAP 과 공유).
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
from src.data.matching_input import TextEncoder  # noqa: E402
from src.data.matching_trainset import (  # noqa: E402
    build_holdout_frame,
    load_v2_feature_columns,
    ndcg_at_k,
)

SCHEMA_VERSION = 1
# W1 DoD: 오프라인 평가 NDCG@10 >= 0.70
NDCG_GATE = 0.70


def evaluate(
    *,
    encoder: TextEncoder,
    model_path: Path,
    out_path: Path,
    n_users: int,
    n_queries: int,
    n_candidates: int,
    seed: int,
    config_path: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path or DEFAULT_CONFIG_PATH
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or "model" not in bundle or "columns" not in bundle:
        raise ValueError(f"{model_path} 는 v2 번들이 아닙니다.")

    columns = list(bundle["columns"])
    expected = load_v2_feature_columns(config_path)
    if columns != expected:
        raise ValueError(
            f"번들 columns 가 config 와 다릅니다.\n  번들: {columns}\n  config: {expected}"
        )

    frame = build_holdout_frame(
        encoder=encoder,
        columns=columns,
        n_users=n_users,
        n_queries=n_queries,
        n_candidates=n_candidates,
        seed=seed,
        config_path=config_path,
    )
    if frame.x.shape[0] == 0:
        raise RuntimeError("채점할 페어가 없습니다.")

    scores = np.asarray(bundle["model"].predict(frame.x), dtype=np.float64)
    measured = {
        "ndcg@5": ndcg_at_k(frame.y, scores, frame.group_sizes, 5),
        "ndcg@10": ndcg_at_k(frame.y, scores, frame.group_sizes, 10),
    }
    recorded = (bundle.get("metrics") or {}).get("v2_full") or {}

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "path": str(model_path),
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "columns": columns,
        },
        "sample": {
            "n_users": n_users,
            "n_queries": n_queries,
            "n_candidates": n_candidates,
            "seed": seed,
            "n_pairs": int(frame.x.shape[0]),
            "n_scored_queries": len(frame.group_sizes),
            "encoder": (
                encoder.reproducibility_metadata()
                if hasattr(encoder, "reproducibility_metadata")
                else {"type": type(encoder).__name__}
            ),
        },
        "measured": measured,
        "recorded_in_bundle": {k: recorded[k] for k in ("ndcg@5", "ndcg@10") if k in recorded},
        "gate": {
            "metric": "ndcg@10",
            "threshold": NDCG_GATE,
            "pass": measured["ndcg@10"] >= NDCG_GATE,
        },
    }
    if "ndcg@10" in recorded:
        payload["recorded_vs_measured_diff"] = {
            key: float(measured[key] - recorded[key]) for key in measured if key in recorded
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "match_v2_ndcg.json")
    # 학습 기본값과 일치시켜야 test 분할이 실제 holdout 이 된다.
    parser.add_argument("--n-users", type=int, default=10_000)
    parser.add_argument("--test-queries", type=int, default=1_000)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.expected_model_sha256:
        actual = hashlib.sha256(args.model.read_bytes()).hexdigest()
        if actual != args.expected_model_sha256:
            print(f"모델 sha256 불일치\n  기대: {args.expected_model_sha256}\n  실제: {actual}")
            return 2

    from scripts.train_match_v2 import _build_sbert

    payload = evaluate(
        encoder=_build_sbert(DEFAULT_CONFIG_PATH),
        model_path=args.model,
        out_path=args.out,
        n_users=args.n_users,
        n_queries=args.test_queries,
        n_candidates=args.candidates,
        seed=args.seed,
    )

    print(f"페어 {payload['sample']['n_pairs']:,} / 쿼리 {payload['sample']['n_scored_queries']:,}")
    print("\n=== 재측정 ===")
    for key, value in payload["measured"].items():
        print(f"  {key:<10} {value:.4f}")
    if payload["recorded_in_bundle"]:
        print("\n=== 번들 기록값 대비 ===")
        for key, value in payload["recorded_in_bundle"].items():
            diff = payload["recorded_vs_measured_diff"][key]
            print(f"  {key:<10} 기록 {value:.4f}  차이 {diff:+.4f}")
    gate = payload["gate"]
    print(
        f"\n게이트 {gate['metric']} >= {gate['threshold']} → "
        f"{'PASS' if gate['pass'] else 'FAIL'}"
    )
    print(f"saved: {args.out}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
