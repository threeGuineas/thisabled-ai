"""D4: Fairlearn 기반 공정성 평가 스크립트.

Test 셋에 대해 UnSmile 7집단, KOLD GRP top-7 집단, 그리고 장애 키워드 포함 여부별로
모델의 Macro-F1 성능 격차를 계산하여 보고합니다.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fairlearn.metrics import MetricFrame
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.stacker import DISABILITY_KEYWORDS, LightGBMStacker

UNSMILE_GROUPS = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교"]


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = data_dir.parent / "raw"
    us_tr = pd.read_csv(raw_dir / "unsmile/unsmile_train_v1.0.tsv", sep="\t")
    us_va = pd.read_csv(raw_dir / "unsmile/unsmile_valid_v1.0.tsv", sep="\t")
    unsmile = pd.concat([us_tr, us_va], ignore_index=True)

    kold = pd.read_json(raw_dir / "kold/kold_v1.json")
    return unsmile, kold


def compute_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def evaluate_fairness(
    model_dir: Path,
    stacker_path: Path,
    data_dir: Path,
) -> None:
    print("=== [모듈 ①] Fairlearn 공정성 평가 시작 ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1. 모델 로딩 ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    num_labels = model.config.num_labels
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=num_labels,
    )

    with open(stacker_path, "rb") as f:
        stacker: LightGBMStacker = pickle.load(f)

    print("2. 데이터 로딩...")
    df_test = pd.read_parquet(data_dir / "test.parquet")
    unsmile_raw, kold_raw = load_raw_data(data_dir)

    print("3. Stacking 메타 피처 및 예측 추출...")
    # Base Logits 추출
    texts = df_test["text"].tolist()
    preds = pipe(texts, batch_size=64)
    logits = []
    for r in preds:
        # r is a list of dicts: [{"label": "LABEL_0", "score": ...}, ...]
        scores = [0.0] * num_labels
        for d in r:
            label_str = d["label"]
            # "LABEL_0" -> 0, "LABEL_1" -> 1, etc.
            lbl_idx = int(label_str.split("_")[-1])
            if lbl_idx < num_labels:
                scores[lbl_idx] = d["score"]
        logits.append(scores)
    logits = np.array(logits)

    X_test, y_true = stacker.build_meta_features(df_test, logits)
    y_pred_probs = stacker.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("4. 보호집단 라벨(Sensitive Attributes) 매핑...")
    # UnSmile 매핑
    unsmile_idx = df_test[df_test["source"] == "unsmile"].index
    unsmile_source_ids = df_test.loc[unsmile_idx, "source_id"].astype(int)

    # KOLD 매핑
    kold_idx = df_test[df_test["source"] == "kold"].index
    kold_source_ids = df_test.loc[kold_idx, "source_id"].astype(int)

    # 4.1 UnSmile 7집단
    for group in UNSMILE_GROUPS:
        df_test[f"unsmile_{group}"] = 0
        group_raw_idx = unsmile_raw[unsmile_raw[group] == 1].index
        mask = unsmile_source_ids.isin(group_raw_idx)
        df_test.loc[unsmile_idx[mask], f"unsmile_{group}"] = 1

    # 4.2 KOLD GRP top-7
    if "GRP" in kold_raw.columns:
        top_kold_groups = kold_raw["GRP"].value_counts().head(7).index.tolist()
        for group in top_kold_groups:
            if not group:
                continue
            df_test[f"kold_{group}"] = 0
            group_raw_idx = kold_raw[kold_raw["GRP"] == group].index
            mask = kold_source_ids.isin(group_raw_idx)
            df_test.loc[kold_idx[mask], f"kold_{group}"] = 1

    # 4.3 장애 도메인
    df_test["disability_domain"] = df_test["text"].apply(
        lambda t: "disability"
        if any(kw in str(t) for kw in DISABILITY_KEYWORDS)
        else "non_disability"
    )

    print("5. 공정성(Fairlearn) 평가 지표 도출...")

    def print_fairness(sensitive_col, mask_idx=None):
        sub_y_true = y_true[mask_idx] if mask_idx is not None else y_true
        sub_y_pred = y_pred[mask_idx] if mask_idx is not None else y_pred
        sub_sensitive = (
            df_test.loc[mask_idx, sensitive_col] if mask_idx is not None else df_test[sensitive_col]
        )

        mf = MetricFrame(
            metrics=compute_macro_f1,
            y_true=sub_y_true,
            y_pred=sub_y_pred,
            sensitive_features=sub_sensitive,
        )
        print(f"\n[공정성: {sensitive_col}]")
        for k, v in mf.by_group.items():
            print(f"  - {k}: {v:.4f}")
        gap = mf.difference()
        print(f"  => 최대 격차: {gap:.4f} (목표 ≤ 0.10)")
        return gap

    print_fairness("source")
    print_fairness("disability_domain")

    print("\n[UnSmile 보호집단별 (1: 포함, 0: 미포함)]")
    for group in UNSMILE_GROUPS:
        print_fairness(f"unsmile_{group}", mask_idx=unsmile_idx)

    if "GRP" in kold_raw.columns:
        print("\n[KOLD 상위 7개 보호집단별]")
        for group in top_kold_groups:
            if group:
                print_fairness(f"kold_{group}", mask_idx=kold_idx)

    print("\n✔ Fairlearn 평가 리포트 생성 완료.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=str, default=str(ROOT / "models" / "checkpoints" / "module1_ce")
    )
    parser.add_argument(
        "--stacker-path",
        type=str,
        default=str(ROOT / "models" / "checkpoints" / "module1_stacking.pkl"),
    )
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data" / "processed"))
    args = parser.parse_args()

    evaluate_fairness(Path(args.model_dir), Path(args.stacker_path), Path(args.data_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
