"""D4: SHAP XAI 해석 스크립트.

Test 세트에서 오분류된 Top-10 케이스(주로 긴급 라벨 관련)를 추출하고
shap.Explainer를 사용하여 각 토큰이 예측에 미친 영향도를 시각화합니다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate_shap(
    model_dir: Path,
    parquet_path: Path,
    out_dir: Path,
    top_k: int = 10,
) -> None:
    print("=== [모듈 ①] SHAP XAI 해석 시작 ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1. 모델 로딩 ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
    )

    print(f"2. 데이터 로딩 ({parquet_path.name})...")
    df = pd.read_parquet(parquet_path)

    # 빠른 추론을 위해 500개 정도만 샘플링 (시간 관계상)
    df_sample = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)
    texts = df_sample["text"].tolist()
    true_labels = df_sample["label"].tolist()

    print("3. 예측 진행 및 오분류 탐색...")
    preds = pipe(texts)

    pred_labels = [int(max(p, key=lambda x: x["score"])["label"].split("_")[-1]) for p in preds]

    errors = []
    for i, (true_y, pred_y) in enumerate(zip(true_labels, pred_labels, strict=False)):
        if true_y != pred_y:
            # 타겟 클래스 예측 확률
            pred_score = next(
                p["score"] for p in preds[i] if int(p["label"].split("_")[-1]) == pred_y
            )
            errors.append(
                {"idx": i, "text": texts[i], "true": true_y, "pred": pred_y, "conf": pred_score}
            )

    # 확신도(Confidence)가 높은데 틀린 Top-K 개
    errors = sorted(errors, key=lambda x: x["conf"], reverse=True)[:top_k]
    print(f"  → 확신도 높은 오분류 Top-{len(errors)} 추출 완료.")

    if not errors:
        print("오분류 케이스가 없습니다.")
        return

    print("4. SHAP Explainer 실행 중 (텍스트 마스킹)...")

    # SHAP Explainer 설정 (함수가 각 클래스의 확률 배열을 반환하도록 wrapping)
    def f(x):
        res = pipe(x)
        # res = [[{'label':'LABEL_0', 'score':0.1}, ...], ...]
        out = []
        for r in res:
            scores = [0.0] * 4
            for d in r:
                lbl_idx = int(d["label"].split("_")[-1])
                scores[lbl_idx] = d["score"]
            out.append(scores)
        return np.array(out)

    explainer = shap.Explainer(f, tokenizer)
    error_texts = [e["text"] for e in errors]
    shap_values = explainer(error_texts)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("5. HTML 시각화 리포트 생성 중...")
    html_path = out_dir / "shap_misclassifications_top10.html"

    # SHAP HTML 렌더링
    html_content = "<html><head><title>SHAP XAI Report</title></head><body><h1>Top 10 Misclassifications SHAP Analysis</h1>"
    shap_html = shap.plots.text(shap_values, display=False)
    html_content += shap_html
    html_content += "</body></html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✔ SHAP XAI 리포트 생성 완료: {html_path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=str, default=str(ROOT / "models" / "checkpoints" / "module1_ce")
    )
    parser.add_argument(
        "--data-path", type=str, default=str(ROOT / "data" / "processed" / "test.parquet")
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "reports" / "validation_reports" / "module1" / "shap"),
    )
    args = parser.parse_args()

    evaluate_shap(Path(args.model_dir), Path(args.data_path), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
