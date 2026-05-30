"""Tests for src.utils.seed."""

from __future__ import annotations

import numpy as np

from src.utils.seed import set_seed


def test_numpy_seed_reproducible():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)
