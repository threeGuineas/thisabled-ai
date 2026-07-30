"""두 MATCH v2 평가팩의 최초 재현성 불일치 단계를 찾는다."""

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
    compare_match_eval_packs,
)


def _render(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--score-atol", type=float, default=1e-10)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        if args.out is not None and (
            _is_inside(args.out, args.reference) or _is_inside(args.out, args.candidate)
        ):
            raise EvalPackError("report output must be outside both evaluation packs")
        report = compare_match_eval_packs(
            args.reference,
            args.candidate,
            expected_reference_sha256=args.expected_reference_sha256,
            expected_candidate_sha256=args.expected_candidate_sha256,
            score_atol=args.score_atol,
            metric_atol=args.metric_atol,
        )
    except (EvalPackError, OSError) as exc:
        print(f"[match-parity] invalid evaluation pack: {exc}", file=sys.stderr)
        return 2

    output = _render(report)
    print(output, end="")
    if args.out is not None:
        _write_atomic(args.out, output)
    return 0 if report["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
