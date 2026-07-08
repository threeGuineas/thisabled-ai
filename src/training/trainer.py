"""Module ① KcELECTRA fine-tuning entry point.

Run via ``scripts/train_module1.py`` (locally for smoke, Colab A100 for real training).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

if TYPE_CHECKING:
    import pandas as pd
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


def build_compute_metrics(binary: bool = False):
    if binary:
        from sklearn.metrics import f1_score, precision_score, recall_score

        def _bin(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            # 이진(0=정상, 1=주의). 주의 Recall이 주 KPI(미탐 최소화).
            return {
                "macro_f1": float(
                    f1_score(labels, preds, labels=[0, 1], average="macro", zero_division=0)
                ),
                "caution_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
                "normal_precision": float(
                    precision_score(labels, preds, pos_label=0, zero_division=0)
                ),
            }

        return _bin

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


def collapse_binary(df: pd.DataFrame) -> pd.DataFrame:
    """4-class label {1,2,3}→1(주의), 0→0(정상). docs/safe_라벨_및_판정_기준.md §2."""
    df = df.copy()
    df["label"] = (df["label"].astype(int) > 0).astype("int64")
    return df


def load_extra_caution(project_root: Path, data_cfg: dict[str, Any]) -> pd.DataFrame | None:
    """추가 '주의(1)' 소스(jsonl)를 병합용 DataFrame으로. PAN12 predator 등.

    extra_caution_filter로 특정 컬럼값만 채택(예: split_role==predator). 경로 없으면 None.
    """
    import json as _json

    import pandas as pd

    paths = data_cfg.get("extra_caution_jsonl") or []
    filt = data_cfg.get("extra_caution_filter") or {}
    rows: list[dict[str, Any]] = []
    for rel in paths:
        p = project_root / rel
        if not p.exists():
            print(f"[extra_caution] 경로 없음, 건너뜀: {rel}")
            continue
        with p.open(encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = _json.loads(ln)
                if all(r.get(k) == v for k, v in filt.items()):
                    rows.append(
                        {
                            "text": r["text"],
                            "label": 1,
                            "source": r.get("source", "extra"),
                            "source_id": f"{r.get('source', 'extra')}_{r.get('conv_id', '')}"
                            f"_{r.get('win_idx', '')}",
                        }
                    )
    if not rows:
        return None
    print(f"[extra_caution] 병합 {len(rows)}건 (label=1)")
    return pd.DataFrame(rows)


def load_extra_binary_train(project_root: Path, data_cfg: dict[str, Any]) -> pd.DataFrame | None:
    """라벨을 보존하는 이진 hard-case JSONL을 로드하고 선택적으로 반복한다."""
    import json as _json

    import pandas as pd

    paths = data_cfg.get("extra_train_jsonl") or []
    repeat = int(data_cfg.get("extra_train_repeat", 1))
    if repeat < 1:
        raise ValueError("data.extra_train_repeat must be >= 1")

    rows: list[dict[str, Any]] = []
    for rel in paths:
        path = project_root / rel
        if not path.exists():
            raise FileNotFoundError(f"[extra_train] 필수 경로 없음: {rel}")
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                row = _json.loads(line)
                text = str(row.get("text", "")).strip()
                label = int(row.get("label", -1))
                if not text or label not in (0, 1):
                    raise ValueError(f"{rel}:{line_number}: text 또는 binary label(0/1) 오류")
                rows.append(
                    {
                        "text": text,
                        "label": label,
                        "source": row.get("source", "safe_hardcase_v1"),
                        "source_id": row.get("source_id", f"hardcase_{len(rows):05d}"),
                    }
                )

    if not rows:
        return None
    unique = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    repeated = pd.concat([unique] * repeat, ignore_index=True)
    distribution = repeated["label"].value_counts().sort_index().to_dict()
    print(
        f"[extra_train] 고유 {len(unique)}건 × {repeat} = {len(repeated)}건 병합, "
        f"label={distribution}"
    )
    return repeated


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
    binary: bool = False,
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
        "compute_metrics": build_compute_metrics(binary=binary),
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
    override_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """End-to-end fine-tuning entry point."""
    cfg = load_config(config_path)

    if override_params:
        for k, v in override_params.items():
            if k in cfg["training"]:
                cfg["training"][k] = v
            elif k in cfg["loss"]:
                cfg["loss"][k] = v
            elif k == "alpha":
                cfg["loss"]["alpha"] = v

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
    import json

    import pandas as pd

    data_cfg = cfg.get("data", {})
    binary = bool(data_cfg.get("binary", False))

    train_df = pd.read_parquet(processed / "train.parquet")

    warning_augment_ratio = (
        override_params.get("warning_augment_ratio", 0) if override_params else 0
    )
    if warning_augment_ratio > 0:
        warning_synth_dir = project_root / "data" / "synthetic" / "warning"
        synth_rows = []
        for p in warning_synth_dir.rglob("*.jsonl"):
            with p.open(encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        synth_rows.append(json.loads(ln))

        if synth_rows:
            synth_df = pd.DataFrame(synth_rows)
            orig_warning_count = len(train_df[train_df["label"] == 2])
            n_augment = int(orig_warning_count * warning_augment_ratio)
            if n_augment > 0:
                augment_df = synth_df.sample(
                    n=min(n_augment, len(synth_df)),
                    replace=True,
                    random_state=cfg["training"]["seed"],
                )
                train_df = pd.concat([train_df, augment_df], ignore_index=True)
                train_df = train_df.sample(
                    frac=1.0, random_state=cfg["training"]["seed"]
                ).reset_index(drop=True)

    val_df = pd.read_parquet(processed / "val.parquet")

    if binary:
        # 1) 추가 '주의' 소스(PAN12 predator 등) 병합 — 붕괴 전 원 라벨과 무관하게 label=1
        extra = load_extra_caution(project_root, data_cfg)
        if extra is not None:
            train_df = pd.concat([train_df, extra], ignore_index=True)
        # 2) 정상·주의 라벨을 보존하는 hard-case 증강 병합
        extra_binary = load_extra_binary_train(project_root, data_cfg)
        if extra_binary is not None:
            train_df = pd.concat([train_df, extra_binary], ignore_index=True)
        # 3) train/val label {1,2,3}→1 붕괴
        train_df = collapse_binary(train_df)
        val_df = collapse_binary(val_df)
        train_df = train_df.sample(frac=1.0, random_state=cfg["training"]["seed"]).reset_index(
            drop=True
        )
        dist = train_df["label"].value_counts().to_dict()
        print(f"[binary] train label 분포 (0=정상,1=주의): {dist}")

    train_ds = RiskTextDataset(train_df, tokenizer, model_cfg["max_length"])
    val_ds = RiskTextDataset(val_df, tokenizer, model_cfg["max_length"])

    focal, gamma, alpha = build_focal_loss(loss_cfg)

    ckpt_dir = project_root / model_cfg.get("checkpoint_dir", cfg["paths"]["checkpoint_dir"])
    trainer = build_trainer(
        cfg,
        model,
        tokenizer,
        train_ds,
        val_ds,
        focal,
        str(ckpt_dir),
        report_to=["mlflow"],
        binary=binary,
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
    if override_params and "warning_augment_ratio" in override_params:
        cfg_params["cfg/warning_augment_ratio"] = override_params["warning_augment_ratio"]

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
