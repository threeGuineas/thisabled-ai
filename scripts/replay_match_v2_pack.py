"""고정 MATCH 모델·raw snapshot을 target SBERT 환경에서 끝까지 재생성한다."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_profiles_v2 import DEFAULT_CONFIG_PATH  # noqa: E402
from src.data.match_eval_pack import (  # noqa: E402
    EvalPackError,
    load_match_eval_pack,
    replay_match_eval_pack,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-atol", type=float, default=1e-10)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    args = parser.parse_args()

    try:
        reference = load_match_eval_pack(
            args.reference,
            expected_pack_sha256=args.expected_reference_sha256,
        )
        reference_encoder = reference.manifest["provenance"]["encoder"]
        model_name = args.encoder_model
        revision = args.encoder_revision.lower()
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision) is None:
            raise EvalPackError("encoder revision must be an immutable 40- or 64-digit commit SHA")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, revision=revision, device=args.device)
        max_seq_length = reference_encoder.get("actual_max_seq_length")
        if isinstance(max_seq_length, int) and max_seq_length > 0:
            model.max_seq_length = max_seq_length
        first_module = model._first_module()
        actual_revision = getattr(
            getattr(getattr(first_module, "auto_model", None), "config", None),
            "_commit_hash",
            None,
        )
        if not isinstance(actual_revision, str) or actual_revision.lower() != revision:
            raise EvalPackError("loaded encoder revision differs from the trusted commit SHA")

        class Adapter:
            def encode(self, sentences, *, batch_size, show_progress_bar):
                return np.asarray(
                    model.encode(
                        sentences,
                        batch_size=batch_size,
                        show_progress_bar=show_progress_bar,
                    ),
                    dtype=np.float32,
                )

            def reproducibility_metadata(self):
                return {
                    "model_name": model_name,
                    "requested_revision": revision,
                    "resolved_revision": actual_revision,
                    "device": str(model.device),
                    "parameter_dtype": str(next(model.parameters()).dtype),
                    "actual_max_seq_length": int(model.max_seq_length),
                    "normalize_embeddings": False,
                }

        report = replay_match_eval_pack(
            args.reference,
            model_path=args.model,
            expected_model_sha256=args.expected_model_sha256,
            expected_reference_sha256=args.expected_reference_sha256,
            encoder=Adapter(),
            output_dir=args.out,
            config_path=args.config,
            score_atol=args.score_atol,
            metric_atol=args.metric_atol,
        )
    except (EvalPackError, OSError, ValueError) as exc:
        print(f"[match-replay] invalid input: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
