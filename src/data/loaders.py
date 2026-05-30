"""Dataset I/O helpers: parquet for DataFrames, HDF5 (h5py) for embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """DataFrame을 parquet으로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """parquet에서 DataFrame 로드."""
    return pd.read_parquet(path)


def save_embeddings(
    emb: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
) -> Path:
    """(임베딩, 라벨)을 HDF5(h5py)로 저장.

    Args:
        emb: shape ``(N, D)`` float ndarray.
        labels: shape ``(N,)`` ndarray.
        path: 출력 ``.h5`` 경로.

    Returns:
        저장된 파일 경로.
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("embeddings", data=np.ascontiguousarray(emb))
        f.create_dataset("labels", data=np.ascontiguousarray(labels))
    return path


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """HDF5(h5py)에서 (임베딩, 라벨)을 로드."""
    import h5py

    with h5py.File(path, "r") as f:
        emb = f["embeddings"][:]
        labels = f["labels"][:]
    return emb, labels
