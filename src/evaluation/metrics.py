"""Classification metrics for the 4-class risk classifier."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

EMERGENCY_LABEL = 3
# 위험도 4단계는 평가셋에 일부 클래스가 0건이어도 항상 고정 라벨 집합으로 채점한다.
# (labels 인자를 생략하면 sklearn이 등장한 클래스만 평균 → 긴급(3)이 0건일 때
#  macro-F1에서 조용히 빠져 점수가 부풀려지는 누락 버그가 발생한다.)
ALL_LABELS = [0, 1, 2, 3]


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
        ``macro_f1``, ``per_class``, ``emergency_recall``, ``emergency_support``,
        (``y_proba`` 제공 시) ``auc_pr``을 담은 dict.

    Note:
        macro-F1·per_class·emergency_recall 모두 ``ALL_LABELS=[0,1,2,3]`` 기준으로
        계산한다. 긴급(3)이 평가셋에 0건이어도 macro-F1에 0으로 반영되며,
        ``emergency_support==0``이면 RuntimeWarning을 발생시킨다 — 긴급 클래스 미평가를
        지표가 좋아 보이도록 숨기지 않기 위함.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1 = f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_LABELS, zero_division=0
    )
    per_class = {
        int(ALL_LABELS[i]): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(ALL_LABELS))
    }

    emergency_support = int((y_true == EMERGENCY_LABEL).sum())
    if emergency_support == 0:
        warnings.warn(
            "평가셋에 긴급(3) 클래스가 0건입니다. emergency_recall=0.0은 '성능 0'이 아니라 "
            "'측정 불가'를 뜻합니다. 긴급 클래스가 포함된 hold-out에서 재평가하세요.",
            RuntimeWarning,
            stacklevel=2,
        )

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
        "emergency_support": emergency_support,
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
