"""합성 데이터 (긴급 클래스) Hold-out 별도 평가 스크립트.

Train에 섞이지 않고 분리된 synthetic_holdout.parquet 데이터를 평가하여
긴급(3) 클래스의 Recall 및 Boundary 케이스의 오분류(FPR)를 계산합니다.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.tracking import log_metrics, mlflow_run  # noqa: E402


def evaluate_synthetic(
    model_dir: Path,
    data_path: Path,
    out_path: Path,
) -> None:
    print("=== [모듈 ①] 합성 Hold-out (긴급) 평가 시작 ===")

    if not data_path.exists():
        print(f"❌ 합성 데이터를 찾을 수 없습니다: {data_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1. 모델 로딩 ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    print(f"2. 합성 데이터 로딩 ({data_path.name})...")
    df = pd.read_parquet(data_path)

    print("3. 예측 진행...")
    texts = df["text"].tolist()
    preds = pipe(texts, batch_size=64)
    pred_labels = [int(p["label"].split("_")[-1]) for p in preds]
    df["pred"] = pred_labels

    # Subcategory 별 평가 (source_id 에서 추출)
    # source_id 포맷: synth_{subcategory}_{split}_{idx}
    if "source_id" in df.columns:
        df["subcategory"] = df["source_id"].apply(
            lambda x: x.split("_")[1] if len(x.split("_")) > 1 else "unknown"
        )
    else:
        df["subcategory"] = "unknown"

    # 평가 수행
    # 긴급(3) 라벨 타겟
    df_emergency = df[df["label"] == 3]
    overall_recall = 0.0
    if not df_emergency.empty:
        overall_recall = (df_emergency["pred"] == 3).mean()

    per_subcat = {}
    for subcat, grp in df_emergency.groupby("subcategory"):
        recall = (grp["pred"] == 3).mean()
        per_subcat[subcat] = {"count": len(grp), "recall": recall}

    # Boundary 케이스 (라벨이 3이 아닌데 긴급 경계선에 있는 시나리오)
    # 합성 데이터 중 label이 3이 아닌 데이터
    df_boundary = df[df["label"] != 3]
    fpr = 0.0
    if not df_boundary.empty:
        # FPR: 모델이 3으로 잘못 예측한 비율
        fpr = (df_boundary["pred"] == 3).mean()

    result = {
        "n_total": len(df),
        "overall_recall": float(overall_recall),
        "per_subcategory": {
            k: {"count": int(v["count"]), "recall": float(v["recall"])}
            for k, v in per_subcat.items()
        },
        "boundary_fpr": float(fpr),
    }

    print("\n[합성 Hold-out 평가 결과]")
    print(f"  - 총 샘플 수: {len(df)}")
    print(f"  - 긴급(3) Recall: {overall_recall:.4f}")
    print(f"  - Boundary FPR: {fpr:.4f}")
    print(
        "  ⚠ 주의: 이 Recall은 합성 hold-out 기준입니다. 합성으로 학습 후 같은 분포의 합성으로 "
        "평가하는 순환 구조이므로 실데이터 긴급 일반화를 보장하지 않습니다."
    )

    with mlflow_run(
        "thisabled-module1",
        run_name=f"synthetic-holdout-eval/{model_dir.name}",
        params={"eval/model_dir": str(model_dir), "eval/data": data_path.name},
    ):
        log_metrics(
            {
                "holdout_emergency_recall": overall_recall,
                "holdout_boundary_fpr": fpr,
                "n_total": len(df),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✔ 합성 평가 리포트 저장 완료: {out_path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=str, default=str(ROOT / "models" / "checkpoints" / "module1_ce")
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(ROOT / "data" / "processed" / "synthetic_holdout.parquet"),
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(
            ROOT / "reports" / "validation_reports" / "module1" / "emergency_synthetic_eval.json"
        ),
    )
    args = parser.parse_args()

    evaluate_synthetic(Path(args.model_dir), Path(args.data_path), Path(args.out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
