#!/usr/bin/env python3
"""Frozen SAFE blind-set evaluation against the deployed /analyze contract."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_cases(path: Path) -> list[dict]:
    cases = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    required = {"id", "label", "slice", "receiver_is_minor", "text"}
    if not cases or any(set(case) != required for case in cases):
        raise ValueError(f"Each JSONL row must contain exactly {sorted(required)}")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case id")
    if any(case["label"] not in (0, 1) for case in cases):
        raise ValueError("label must be 0 or 1")
    return cases


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if body.get("verdict") not in ("safe", "flagged"):
        raise ValueError(f"Invalid /analyze response: {body}")
    return body


def build_local_predictor(
    model_path: Path,
    adult_threshold: float,
    minor_threshold: float,
    rule_assist: bool,
) -> Callable[[dict], dict]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from serving.safety_server.app import _rule_hit

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    if int(model.config.num_labels) != 2:
        raise ValueError(f"Expected binary model, got num_labels={model.config.num_labels}")

    def predict(case: dict) -> dict:
        encoded = tokenizer(case["text"], truncation=True, max_length=128, return_tensors="pt")
        with torch.inference_mode():
            risk_prob = functional.softmax(model(**encoded).logits[0], dim=-1)[1].item()
        threshold = minor_threshold if case["receiver_is_minor"] else adult_threshold
        hit = bool(rule_assist and _rule_hit(case["text"]))
        return {
            "verdict": "flagged" if risk_prob >= threshold or hit else "safe",
            "risk_prob": risk_prob,
            "rule_assist": hit,
        }

    return predict


def metrics(rows: list[dict]) -> dict:
    tn = sum(row["label"] == 0 and row["prediction"] == 0 for row in rows)
    fp = sum(row["label"] == 0 and row["prediction"] == 1 for row in rows)
    fn = sum(row["label"] == 1 and row["prediction"] == 0 for row in rows)
    tp = sum(row["label"] == 1 and row["prediction"] == 1 for row in rows)
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    npv = tn / (tn + fn) if tn + fn else None

    def f1(precision_value: float | None, recall_value: float | None) -> float | None:
        if precision_value is None or recall_value is None:
            return None
        return (
            2 * precision_value * recall_value / (precision_value + recall_value)
            if precision_value + recall_value
            else 0.0
        )

    f1_pos = f1(precision, recall)
    f1_neg = f1(npv, specificity)
    return {
        "n": len(rows),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "risk_recall": recall,
        "specificity": specificity,
        "fpr": 1.0 - specificity if specificity is not None else None,
        "risk_precision": precision,
        "macro_f1": (f1_neg + f1_pos) / 2 if f1_neg is not None and f1_pos is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--endpoint", help="Base URL or full /analyze URL")
    target.add_argument("--model", type=Path, help="Local binary Hugging Face checkpoint")
    parser.add_argument("--data", type=Path, default=Path("tests/fixtures/safe_blind_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/safe_blind_v1_results.json"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--adult-threshold", type=float, default=0.66)
    parser.add_argument("--minor-threshold", type=float, default=0.50)
    parser.add_argument("--no-rule-assist", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.data)
    if args.endpoint:
        analyze_url = args.endpoint.rstrip("/")
        if not analyze_url.endswith("/analyze"):
            analyze_url += "/analyze"

        def predict(case: dict) -> dict:
            return post_json(
                analyze_url,
                {"text": case["text"], "receiver_is_minor": case["receiver_is_minor"]},
                args.timeout,
            )

        target_name = args.endpoint
    else:
        predict = build_local_predictor(
            args.model,
            adult_threshold=args.adult_threshold,
            minor_threshold=args.minor_threshold,
            rule_assist=not args.no_rule_assist,
        )
        target_name = str(args.model)

    rows = []
    try:
        for index, case in enumerate(cases, 1):
            body = predict(case)
            rows.append(
                {
                    **case,
                    "prediction": int(body["verdict"] == "flagged"),
                    "risk_prob": body.get("risk_prob"),
                    "rule_assist": body.get("rule_assist"),
                }
            )
            print(f"[{index:02d}/{len(cases)}] {case['id']} -> {body['verdict']}", flush=True)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"Evaluation aborted: {exc}", file=sys.stderr)
        return 2

    by_slice: dict[str, list[dict]] = defaultdict(list)
    by_audience: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_slice[row["slice"]].append(row)
        by_audience["minor" if row["receiver_is_minor"] else "adult"].append(row)

    report = {
        "dataset": str(args.data),
        "target": target_name,
        "thresholds": {"adult": args.adult_threshold, "minor": args.minor_threshold},
        "rule_assist": not args.no_rule_assist,
        "overall": metrics(rows),
        "by_audience": {key: metrics(value) for key, value in sorted(by_audience.items())},
        "by_slice": {key: metrics(value) for key, value in sorted(by_slice.items())},
        "errors": [row for row in rows if row["label"] != row["prediction"]],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"RESULT: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
