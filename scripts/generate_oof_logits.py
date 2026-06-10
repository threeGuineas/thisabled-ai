"""Stacking 누수 차단: train split의 OOF(out-of-fold) KcELECTRA logits 생성.

문제: 기존 stacking은 train으로 fine-tune한 base 모델로 다시 train의 logits을 뽑아
메타러너를 학습했다(= in-fold/in-sample 예측). base가 이미 본 라벨을 입력으로 쓰므로
메타러너가 누수로 부풀려진다.

해결: train을 Stratified K-fold로 나눠 각 fold마다 나머지 K-1 fold로 base를 새로
fine-tune하고, held-out fold의 logits만 모은다. 결과 ``oof_train_logits.npy``는
train.parquet의 행 순서와 정확히 정렬되며, 메타러너 학습에는 이 OOF logits을 쓴다.

재현성: ``seed=42`` 고정, ``StratifiedKFold(shuffle=True, random_state=seed)``.

실행 (GPU 권장 — Colab A100):
    python scripts/generate_oof_logits.py --config configs/module1_kcelectra_ce.yaml --folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.dataset import RiskTextDataset  # noqa: E402
from src.training.trainer import build_focal_loss, build_trainer, load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def generate_oof_logits(
    config_path: Path,
    project_root: Path,
    n_splits: int = 5,
    out_path: Path | None = None,
) -> Path:
    cfg = load_config(config_path)
    seed = cfg["training"]["seed"]
    set_seed(seed)

    model_cfg = cfg["model"]
    max_length = model_cfg["max_length"]
    num_labels = model_cfg["num_labels"]

    processed = project_root / "data" / "processed"
    train_path = processed / "train.parquet"
    train_df = pd.read_parquet(train_path).reset_index(drop=True)
    y = train_df["label"].to_numpy()

    if out_path is None:
        out_path = processed / "oof_train_logits.npy"

    oof = np.full((len(train_df), num_labels), np.nan, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print(f"=== OOF logits 생성: {len(train_df)}행, {n_splits}-fold (seed={seed}) ===")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits}: train {len(tr_idx)} / oof {len(va_idx)} ---")
        # fold별 클래스 비율 점검 (Stratified 검증)
        va_dist = pd.Series(y[va_idx]).value_counts(normalize=True).sort_index().round(3).to_dict()
        print(f"    oof fold 클래스 비율: {va_dist}")

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["name"], num_labels=num_labels
        )

        tr_ds = RiskTextDataset(train_df.iloc[tr_idx], tokenizer, max_length)
        va_ds = RiskTextDataset(train_df.iloc[va_idx], tokenizer, max_length)

        focal, _, _ = build_focal_loss(cfg["loss"])
        fold_out = project_root / "models" / "checkpoints" / "_oof_tmp" / f"fold{fold}"
        trainer = build_trainer(
            cfg,
            model,
            tokenizer,
            tr_ds,
            va_ds,
            focal,
            str(fold_out),
            report_to=[],  # fold별 run으로 MLflow 오염 방지 (집계 logits만 의미 있음)
            load_best=False,  # 예측 대상 fold로 best 선택 시 누수 → 최종 epoch 사용
        )
        trainer.train()
        preds = trainer.predict(va_ds)
        oof[va_idx] = preds.predictions

    if np.isnan(oof).any():
        missing = int(np.isnan(oof).any(axis=1).sum())
        raise RuntimeError(f"OOF logits에 빈 행 {missing}개 — fold 분할이 train을 모두 덮지 못함")

    np.save(out_path, oof)
    print(f"\n✔ OOF logits 저장: {out_path.relative_to(project_root)} shape={oof.shape}")
    print("  → 이제 scripts/train_stacking.py 가 train split에 이 OOF logits을 사용합니다.")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=str(ROOT / "configs" / "module1_kcelectra_ce.yaml")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    generate_oof_logits(
        config_path=Path(args.config),
        project_root=ROOT,
        n_splits=args.folds,
        out_path=Path(args.out) if args.out else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
