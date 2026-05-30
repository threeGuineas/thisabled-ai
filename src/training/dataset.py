"""Torch Dataset wrapping the processed parquet splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class RiskTextDataset(Dataset):
    """4단계 위험도 분류용 텍스트 Dataset.

    parquet 컬럼: ``text``, ``label``, ``source``, ``source_id``.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }
