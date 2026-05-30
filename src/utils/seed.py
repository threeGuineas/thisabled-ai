"""Reproducibility: 모든 라이브러리의 RNG 시드 고정."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """모든 라이브러리의 RNG를 고정.

    Args:
        seed: 시드 값.
        deterministic: True면 cuDNN deterministic 모드 (학습 속도 감소, 최종 재현 시에만).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    except ImportError:
        pass
