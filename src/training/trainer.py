"""Module ① KcELECTRA fine-tuning entry point.

Run via ``scripts/train_module1.py`` (locally for smoke, Colab A100 for real training).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.evaluation.metrics import compute_classification_metrics
from src.models.focal_loss import FocalLoss
from src.training.dataset import RiskTextDataset
from src.utils.seed import set_seed
from src.utils.tracking import log_metrics, mlflow_run


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class FocalLossTrainer(Trainer):
    """compute_loss를 Focal Loss로 오버라이드."""

    def __init__(self, *args: Any, focal_loss: FocalLoss, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_compute_metrics():
    def _fn(eval_pred):
        logits, labels = eval_pred
        proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = np.argmax(logits, axis=-1)
        m = compute_classification_metrics(labels, preds, proba)
        # Trainer가 받을 수 있도록 평탄화
        flat = {"macro_f1": m["macro_f1"], "emergency_recall": m["emergency_recall"]}
        if "auc_pr" in m:
            flat["auc_pr"] = m["auc_pr"]
        for cls, stats in m["per_class"].items():
            flat[f"f1_class{cls}"] = stats["f1"]
        return flat

    return _fn


def build_focal_loss(loss_cfg: dict[str, Any]) -> tuple[FocalLoss, float, list[float] | None]:
    """config의 loss 섹션에서 FocalLoss와 (gamma, alpha)를 구성.

    ``type=="ce"``면 γ=0 강제 (Focal Loss γ=0 = weighted CE, 수학적 동치).
    """
    alpha = loss_cfg.get("alpha")
    alpha_t = torch.tensor(alpha, dtype=torch.float32) if alpha else None
    gamma = 0.0 if loss_cfg.get("type", "focal") == "ce" else loss_cfg["focal_gamma"]
    return FocalLoss(gamma=gamma, alpha=alpha_t), gamma, alpha


def build_trainer(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    train_ds: Any,
    val_ds: Any,
    focal: FocalLoss,
    output_dir: str,
    report_to: list[str],
    load_best: bool = True,
) -> FocalLossTrainer:
    """TrainingArguments + FocalLossTrainer 구성 (학습 본 경로와 OOF fold가 공유).

    Args:
        load_best: True면 epoch별 best(macro_f1) 모델을 끝에 로드. OOF fold에서는
            예측 대상 fold로 best를 고르면 선택 누수가 생기므로 ``False``로 두고
            최종 epoch 모델로 예측한다.
    """
    tr_cfg = cfg["training"]
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tr_cfg["num_epochs"],
        per_device_train_batch_size=tr_cfg["batch_size"],
        per_device_eval_batch_size=tr_cfg["eval_batch_size"],
        learning_rate=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
        warmup_ratio=tr_cfg["warmup_ratio"],
        gradient_accumulation_steps=tr_cfg["grad_accum_steps"],
        fp16=tr_cfg["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch" if load_best else "no",
        load_best_model_at_end=load_best,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=tr_cfg["seed"],
        report_to=report_to,
        logging_steps=50,
    )

    # transformers 5.0+ 에서 `tokenizer` 인자가 `processing_class`로 변경됨.
    # 양 버전 모두 호환: import해서 사용 가능한 키워드 동적 결정.
    import inspect

    from transformers import Trainer as _HFTrainer

    trainer_kw = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorWithPadding(tokenizer),
        "compute_metrics": build_compute_metrics(),
        "focal_loss": focal,
    }
    sig = inspect.signature(_HFTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kw["processing_class"] = tokenizer
    else:
        trainer_kw["tokenizer"] = tokenizer
    return FocalLossTrainer(**trainer_kw)


def train_module1(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """End-to-end fine-tuning entry point."""
    cfg = load_config(config_path)
    project_root = Path(project_root)
    set_seed(cfg["training"]["seed"])

    model_cfg = cfg["model"]
    tr_cfg = cfg["training"]
    loss_cfg = cfg["loss"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=model_cfg["num_labels"],
    )

    processed = project_root / "data" / "processed"
    train_ds = RiskTextDataset(processed / "train.parquet", tokenizer, model_cfg["max_length"])
    val_ds = RiskTextDataset(processed / "val.parquet", tokenizer, model_cfg["max_length"])

    focal, gamma, alpha = build_focal_loss(loss_cfg)

    ckpt_dir = project_root / model_cfg.get("checkpoint_dir", cfg["paths"]["checkpoint_dir"])
    trainer = build_trainer(
        cfg, model, tokenizer, train_ds, val_ds, focal, str(ckpt_dir), report_to=["mlflow"]
    )

    mlflow_cfg = cfg.get("mlflow", {})
    experiment = mlflow_cfg.get("experiment_name", "thisabled-module1")
    run_name = mlflow_cfg.get("run_name", Path(config_path).stem)
    cfg_params = {
        "cfg/backbone": model_cfg["name"],
        "cfg/num_labels": model_cfg["num_labels"],
        "cfg/max_length": model_cfg["max_length"],
        "cfg/loss_type": loss_cfg.get("type", "focal"),
        "cfg/focal_gamma": gamma,
        "cfg/alpha": alpha,
        "cfg/num_epochs": tr_cfg["num_epochs"],
        "cfg/lr": tr_cfg["lr"],
        "cfg/batch_size": tr_cfg["batch_size"],
        "cfg/seed": tr_cfg["seed"],
        "cfg/train_size": len(train_ds),
        "cfg/val_size": len(val_ds),
    }

    # MLflow run으로 학습 수명을 감싼다. report_to=["mlflow"]의 HF 콜백은 active run을
    # 재사용하므로 학습 중 지표와 사후 평가 지표가 같은 run에 기록된다.
    with mlflow_run(experiment, run_name=run_name, params=cfg_params):
        train_result = trainer.train()
        # load_best_model_at_end=True 라 trainer.model이 best 모델임.
        # 부모 dir에 저장해야 from_pretrained(ckpt_dir)가 바로 동작.
        trainer.save_model(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        eval_result = trainer.evaluate()
        log_metrics(eval_result, prefix="final_")

    return {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "checkpoint_dir": str(ckpt_dir),
    }
