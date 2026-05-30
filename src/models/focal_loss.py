"""Focal Loss for multi-class classification.

Reference:
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017).
    "Focal Loss for Dense Object Detection." ICCV.
    https://arxiv.org/abs/1708.02002
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    .. math::
        FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)

    ``gamma=0``인 경우 (가중) cross-entropy와 동치.

    Args:
        gamma: focusing parameter (기본 2.0).
        alpha: 클래스별 가중치 텐서 shape ``(num_classes,)`` 또는 ``None``.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be one of mean/sum/none, got {reduction!r}")
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: shape ``(N, C)``.
            targets: shape ``(N,)`` int64.
        """
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
