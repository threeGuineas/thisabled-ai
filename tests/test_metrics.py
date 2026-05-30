"""Tests for src.evaluation.metrics."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_classification_metrics


def test_metrics_keys_present():
    y_true = np.array([0, 1, 2, 3, 0, 3])
    y_pred = np.array([0, 1, 1, 3, 0, 2])
    out = compute_classification_metrics(y_true, y_pred)
    assert "macro_f1" in out
    assert "per_class" in out
    assert "emergency_recall" in out
    assert 0 in out["per_class"]
    assert 3 in out["per_class"]


def test_metrics_with_proba_auc_pr():
    rng = np.random.default_rng(0)
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    y_proba = rng.random((4, 4))
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    out = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
    assert "auc_pr" in out
    assert 0.0 <= out["auc_pr"] <= 1.0


def test_emergency_recall_perfect():
    y_true = np.array([3, 3, 3, 0])
    y_pred = np.array([3, 3, 3, 0])
    out = compute_classification_metrics(y_true, y_pred)
    assert out["emergency_recall"] == 1.0
