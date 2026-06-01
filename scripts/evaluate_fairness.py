"""모듈 ① 공정성 평가 CLI — 모델 + test → 보호집단별 F1.

EXP-3 단계 1. src.evaluation.fairness 모듈 기반.

Usage:
    python scripts/evaluate_fairness.py --model-dir models/checkpoints/module1_final
    python scripts/evaluate_fairness.py --model-dir models/checkpoints/module1_final \\
        --output-json reports/validation_reports/module1/fairness_after_oversample.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.fairness import run_full_fairness_evaluation  # noqa: E402
from src.training.dataset import RiskTextDataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument(
        "--test-parquet",
        type=str,
        default=str(ROOT / "data/processed/test.parquet"),
    )
    parser.add_argument(
        "--unsmile-train",
        type=str,
        default=str(ROOT / "data/raw/unsmile/unsmile_train_v1.0.tsv"),
    )
    parser.add_argument(
        "--unsmile-valid",
        type=str,
        default=str(ROOT / "data/raw/unsmile/unsmile_valid_v1.0.tsv"),
    )
    parser.add_argument(
        "--kold-json",
        type=str,
        default=str(ROOT / "data/raw/kold/kold_v1.json"),
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장비: {device}")

    print(f"1. 모델 로딩: {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    print(f"2. Test 데이터 로딩: {args.test_parquet}")
    test_df = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    ds = RiskTextDataset(args.test_parquet, tok, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=DataCollatorWithPadding(tok))

    print("3. 예측 추출...")
    logits_list = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_list.append(model(**batch).logits.cpu().numpy())
    logits = np.concatenate(logits_list)
    y_pred = logits.argmax(axis=-1)

    print("4. 보호집단별 공정성 평가...")
    result = run_full_fairness_evaluation(
        test_df=test_df,
        y_pred=y_pred,
        unsmile_raw_train_path=Path(args.unsmile_train),
        unsmile_raw_valid_path=Path(args.unsmile_valid),
        kold_raw_path=Path(args.kold_json),
    )

    out_path = (
        Path(args.output_json)
        if args.output_json
        else ROOT / "reports" / "validation_reports" / "module1" / "fairness.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n✓ 결과 저장: {out_path.relative_to(ROOT)}")
    print("\n=== 요약 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
