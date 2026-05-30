"""Classification metrics for the 4-class risk classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

EMERGENCY_LABEL = 3


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """4단계 위험도 분류 메트릭 계산.

    Args:
        y_true: shape ``(N,)`` 정답 라벨.
        y_pred: shape ``(N,)`` 예측 라벨.
        y_proba: shape ``(N, C)`` 클래스 확률 (선택).

    Returns:
        ``macro_f1``, ``per_class``, ``emergency_recall``,
        (``y_proba`` 제공 시) ``auc_pr``을 담은 dict.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    per_class = {
        int(i): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(f1))
    }

    emergency_recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=[EMERGENCY_LABEL],
            average="macro",
            zero_division=0,
        )
    )

    result: dict[str, Any] = {
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "emergency_recall": emergency_recall,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        n_classes = y_proba.shape[1]
        aps: list[float] = []
        for c in range(n_classes):
            y_true_c = (y_true == c).astype(int)
            if y_true_c.sum() == 0:
                continue
            aps.append(float(average_precision_score(y_true_c, y_proba[:, c])))
        result["auc_pr"] = float(np.mean(aps)) if aps else 0.0

    return result
