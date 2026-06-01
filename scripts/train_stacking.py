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
from src.models.stacker import DISABILITY_KEYWORDS, LightGBMStacker  # noqa: E402
from src.training.dataset import RiskTextDataset  # noqa: E402

SOURCE_MAP = {"unsmile": 0, "kold": 1}


def _encode_source(src: str) -> int:
    """unsmile=0, kold=1, synthetic_*=2, 그 외 3."""
    if src in SOURCE_MAP:
        return SOURCE_MAP[src]
    if isinstance(src, str) and src.startswith("synthetic"):
        return 2
    return 3


def build_meta_features(df: pd.DataFrame, logits: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Stacking 메타 피처 빌드 (standalone, 단위 테스트 가능).

    Returns:
        X: ``[logit_0..3, source(category), length(float), has_disability(int8)]`` (7-dim).
        y: shape ``(N,)`` int8 라벨 배열.
    """
    if logits.shape[1] != 4:
        raise ValueError(f"Logits shape must be (N, 4), got {logits.shape}")
    if len(df) != logits.shape[0]:
        raise ValueError(f"df({len(df)}) and logits({logits.shape[0]}) length mismatch")

    cols = {f"logit_{i}": logits[:, i] for i in range(4)}
    cols["source"] = pd.Categorical(df["source"].map(_encode_source).astype(int))
    cols["length"] = df["text"].astype(str).str.len().astype(float).values
    cols["has_disability"] = (
        df["text"]
        .astype(str)
        .apply(lambda t: int(any(kw in t for kw in DISABILITY_KEYWORDS)))
        .astype(np.int8)
        .values
    )
    X = pd.DataFrame(  # noqa: N806
        cols,
        columns=[
            "logit_0",
            "logit_1",
            "logit_2",
            "logit_3",
            "source",
            "length",
            "has_disability",
        ],
    )
    y = (
        df["label"].astype(np.int8).values
        if "label" in df.columns
        else np.zeros(len(df), dtype=np.int8)
    )
    return X, y


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
        ds, batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer)
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


def train_stacking_meta_learner(
    config_path: Path,
    model_dir: Path,
    data_dir: Path,
    out_model_path: Path,
) -> dict[str, Any]:
    cfg = load_yaml_config(config_path)
    stack_cfg = cfg.get("stacking", {})
    lgbm_params = stack_cfg.get("lgbm_params", {})

    stacker = LightGBMStacker(lgbm_params=lgbm_params)

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
    # 2. LightGBM 모델 훈련
    print("\n  - LightGBM Stacking 모델 훈련 시작...")
    stacker.fit(
        X_train=data_features["train"],
        y_train=data_targets["train"],
        X_val=data_features["val"],
        y_val=data_targets["val"],
        num_boost_round=stack_cfg.get("num_boost_round", 100),
    )

    # 3. 테스트셋 평가
    print("\n  - 테스트셋 평가 중...")
    y_pred_probs = stacker.predict(data_features["test"])
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = compute_classification_metrics(data_targets["test"], y_pred, y_pred_probs)
    print("\n=== 최종 Stacking 평가 메트릭 ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: \n{v}")

    # 4. 저장
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    with out_model_path.open("wb") as f:
        pickle.dump(stacker, f)
    print(f"\n✔ Stacking 메타 러너 저장 완료: {out_model_path.relative_to(ROOT)}")

    return metrics


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
