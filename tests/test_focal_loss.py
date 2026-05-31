"""Tests for src.models.focal_loss."""

from __future__ import annotations

import pytest
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


def test_focal_loss_with_alpha_weight():
    """alpha 클래스 가중치가 정상 동작해야 함."""
    alpha = torch.tensor([1.0, 1.5, 1.0, 1.0])
    fl = FocalLoss(gamma=2.0, alpha=alpha)
    logits = torch.randn(16, 4)
    targets = torch.randint(0, 4, (16,))
    loss = fl(logits, targets)
    assert torch.isfinite(loss)
    # 동일 입력 → 동일 출력 (alpha를 device 이동 후에도 캐싱 일관성)
    loss2 = fl(logits, targets)
    assert torch.allclose(loss, loss2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 없음 — GPU 테스트 skip")
def test_focal_loss_alpha_auto_moves_to_logits_device():
    """alpha가 CPU에 있어도 forward 시 logits.device로 자동 이동되어야 함.

    회귀 방지: GPU 학습 시 'expected weight on CUDA but on CPU' 오류 재발 차단.
    """
    alpha_cpu = torch.tensor([1.0, 1.5, 1.0, 1.0])  # CPU
    fl = FocalLoss(gamma=2.0, alpha=alpha_cpu)
    logits_cuda = torch.randn(8, 4, device="cuda")
    targets_cuda = torch.randint(0, 4, (8,), device="cuda")
    loss = fl(logits_cuda, targets_cuda)
    assert loss.device.type == "cuda"
    assert torch.isfinite(loss)
