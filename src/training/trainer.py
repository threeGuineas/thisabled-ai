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

    alpha = loss_cfg.get("alpha")
    alpha_t = torch.tensor(alpha, dtype=torch.float32) if alpha else None
    # type=ce면 γ=0 강제 (Focal Loss with γ=0 = weighted CE, 수학적 동치)
    gamma = 0.0 if loss_cfg.get("type", "focal") == "ce" else loss_cfg["focal_gamma"]
    focal = FocalLoss(gamma=gamma, alpha=alpha_t)

    ckpt_dir = project_root / model_cfg.get("checkpoint_dir", cfg["paths"]["checkpoint_dir"])
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=tr_cfg["num_epochs"],
        per_device_train_batch_size=tr_cfg["batch_size"],
        per_device_eval_batch_size=tr_cfg["eval_batch_size"],
        learning_rate=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
        warmup_ratio=tr_cfg["warmup_ratio"],
        gradient_accumulation_steps=tr_cfg["grad_accum_steps"],
        fp16=tr_cfg["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=tr_cfg["seed"],
        report_to=[],
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
    trainer = FocalLossTrainer(**trainer_kw)

    train_result = trainer.train()
    # load_best_model_at_end=True 라 trainer.model이 best 모델임.
    # 부모 dir에 저장해야 from_pretrained(ckpt_dir)가 바로 동작.
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    eval_result = trainer.evaluate()

    return {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "checkpoint_dir": str(ckpt_dir),
    }
