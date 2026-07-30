"""직렬화된 MATCH v2 모델을 frozen 평가팩에서 다시 채점한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.match_eval_pack import (  # noqa: E402
    EvalPackError,
    evaluate_model_against_pack,
)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _is_inside(path: Path, directory: Path) -> bool:
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_directory = directory.expanduser().resolve(strict=False)
    return resolved_path == resolved_directory or resolved_directory in resolved_path.parents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-pack", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-pack-sha256", required=True)
    parser.add_argument("--score-atol", type=float, default=1e-10)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        if args.out is not None and (
            _is_inside(args.out, args.eval_pack)
            or args.out.expanduser().resolve(strict=False)
            == args.model.expanduser().resolve(strict=False)
        ):
            raise EvalPackError("report output must be outside the pack and model paths")
        report = evaluate_model_against_pack(
            args.model,
            args.eval_pack,
            expected_model_sha256=args.expected_model_sha256,
            expected_pack_sha256=args.expected_pack_sha256,
            score_atol=args.score_atol,
            metric_atol=args.metric_atol,
        )
    except (EvalPackError, OSError) as exc:
        print(f"[match-eval-pack] invalid input: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    print(output, end="")
    if args.out is not None:
        _write_atomic(args.out, output)
    return 0 if report["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
