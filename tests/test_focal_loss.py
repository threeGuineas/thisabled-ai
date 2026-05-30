"""Tests for src.models.focal_loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.focal_loss import FocalLoss


def test_focal_loss_shape_and_finite():
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_focal_loss_gamma_zero_equals_ce():
    torch.manual_seed(0)
    logits = torch.randn(16, 4)
    targets = torch.randint(0, 4, (16,))
    fl = FocalLoss(gamma=0.0)(logits, targets)
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_focal_loss_reduction_none_shape():
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    loss = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    assert loss.shape == (8,)
