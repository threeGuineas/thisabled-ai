"""Torch Dataset wrapping the processed parquet splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class RiskTextDataset(Dataset):
    """4단계 위험도 분류용 텍스트 Dataset.

    컬럼: ``text``, ``label``, ``source``, ``source_id``.

    ``source``는 parquet 경로(``str``/``Path``) 또는 이미 로드된 ``DataFrame``을
    받는다. 후자는 OOF 교차검증에서 train fold/val fold 같은 부분집합을 임시 파일
    없이 바로 감싸기 위함이다.
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        if isinstance(source, pd.DataFrame):
            self.df = source.reset_index(drop=True)
        else:
            self.df = pd.read_parquet(source).reset_index(drop=True)
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
