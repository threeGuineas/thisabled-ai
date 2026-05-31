"""D3: LightGBM Stacking 메타 학습기 훈련.

KcELECTRA base 모델의 logits + 메타 피처(source, length, 장애 키워드)를 입력받아
최종 4단계 위험도 분류를 수행하는 LightGBM Stacking 메타 모델을 훈련합니다.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_classification_metrics  # noqa: E402
from src.training.dataset import RiskTextDataset  # noqa: E402

DISABILITY_KEYWORDS = [
    "장애",
    "발달장애",
    "시각장애",
    "청각장애",
    "지체장애",
    "휠체어",
    "정신장애",
    "지적장애",
    "치료사",
    "활동지원",
]


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_logits(
    model_dir: Path,
    parquet_path: Path,
    batch_size: int = 64,
    max_length: int = 128,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """KcELECTRA base 모델을 사용하여 데이터셋의 logits과 labels를 배치 단위로 추출."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  - 모델 로딩 중: {model_dir.name}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    ds = RiskTextDataset(parquet_path, tokenizer, max_length)
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer)
    )

    all_logits = []
    all_labels = []

    print(f"  - Logits 추출 중 ({parquet_path.name})...")
    with torch.no_grad():
        for batch in tqdm(dl, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def build_meta_features(df: pd.DataFrame, logits: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """logits와 메타 피처들을 결합하여 Stacking용 Feature Matrix 및 Target 구축."""
    meta_df = pd.DataFrame()

    # 1. KcELECTRA Logits (4차원)
    for i in range(4):
        meta_df[f"logit_{i}"] = logits[:, i]

    # 2. Source 인코딩 (Categorical)
    # unsmile: 0, kold: 1, synthetic_emergency_v1: 2, 그 외: 3
    source_map = {"unsmile": 0, "kold": 1, "synthetic_emergency_v1": 2}
    meta_df["source"] = df["source"].map(source_map).fillna(3).astype("category")

    # 3. 텍스트 길이 (기본 정규화 없이 사용, 나무 기반 모델이므로 무방)
    meta_df["length"] = df["text"].str.len().fillna(0).astype(np.float32)

    # 4. 장애 관련 맥락 키워드 여부 (바이너리)
    has_disability = (
        df["text"].apply(lambda t: any(kw in str(t) for kw in DISABILITY_KEYWORDS)).astype(np.int8)
    )
    meta_df["has_disability"] = has_disability

    targets = df["label"].values.astype(np.int8)
    return meta_df, targets


def train_stacking_meta_learner(
    config_path: Path,
    model_dir: Path,
    data_dir: Path,
    out_model_path: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    cfg = load_yaml_config(config_path)
    stack_cfg = cfg.get("stacking", {})
    lgbm_params = stack_cfg.get("lgbm_params", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장비: {device}")

    # 1. Logits 추출 및 메타 피처 구축
    splits = ["train", "val", "test"]
    data_features = {}
    data_targets = {}
    original_dfs = {}

    for split in splits:
        parquet_path = data_dir / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet 파일이 없습니다: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        original_dfs[split] = df

        logits, _ = extract_logits(
            model_dir=model_dir,
            parquet_path=parquet_path,
            batch_size=cfg["training"].get("eval_batch_size", 64),
            max_length=cfg["model"].get("max_length", 128),
            device=device,
        )

        X, y = build_meta_features(df, logits)  # noqa: N806
        data_features[split] = X
        data_targets[split] = y

    X_train, y_train = data_features["train"], data_targets["train"]  # noqa: N806
    X_val, y_val = data_features["val"], data_targets["val"]  # noqa: N806

    print("\n=== LightGBM 메타 학습기 학습 ===")
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

    # LightGBM Classifier 학습
    # multiclass 일때 class_weight='balanced'를 적용하여 불균형 해소
    model = lgb.LGBMClassifier(
        objective=lgbm_params.get("objective", "multiclass"),
        num_class=lgbm_params.get("num_class", 4),
        n_estimators=lgbm_params.get("n_estimators", 500),
        learning_rate=lgbm_params.get("learning_rate", 0.05),
        num_leaves=lgbm_params.get("num_leaves", 31),
        max_depth=lgbm_params.get("max_depth", -1),
        random_state=lgbm_params.get("random_state", 42),
        class_weight="balanced",
    )

    # 학습
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True)],
    )

    # 2. 평가 진행
    print("\n=== 최종 Stacking 모델 평가 ===")
    metrics_result = {}
    for split in ["val", "test"]:
        X_eval, y_eval = data_features[split], data_targets[split]  # noqa: N806
        preds = model.predict(X_eval)
        proba = model.predict_proba(X_eval)

        m = compute_classification_metrics(y_eval, preds, proba)
        metrics_result[split] = m

        print(f"\n[{split.upper()} 세트 평가 결과]")
        print(f"  - Macro-F1: {m['macro_f1']:.4f}")
        print(f"  - 긴급 Recall: {m['emergency_recall']:.4f}")
        if "auc_pr" in m:
            print(f"  - AUC-PR: {m['auc_pr']:.4f}")

        for cls, stats in m["per_class"].items():
            print(
                f"    * Class {cls} ({stats['name']}): F1 {stats['f1']:.4f} (Recall {stats['recall']:.4f})"
            )

    # 3. 모델 저장
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    with out_model_path.open("wb") as f:
        pickle.dump(model, f)
    print(f"\n✔ Stacking 메타 모델 저장 완료: {out_model_path.relative_to(ROOT)}")

    return metrics_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "module1_kcelectra_ce.yaml"),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(ROOT / "models" / "checkpoints" / "module1_ce"),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "data" / "processed"),
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(ROOT / "models" / "checkpoints" / "module1_stacking.pkl"),
    )
    args = parser.parse_args()

    train_stacking_meta_learner(
        config_path=Path(args.config),
        model_dir=Path(args.model_dir),
        data_dir=Path(args.data_dir),
        out_model_path=Path(args.output_model),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
