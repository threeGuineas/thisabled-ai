"""Build processed train/val/test parquet from raw seed datasets.

Usage:
    python scripts/build_processed_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_dataset import build_and_save  # noqa: E402


def main() -> int:
    raw = ROOT / "data" / "raw"
    out = ROOT / "data" / "processed"
    paths = build_and_save(raw_dir=raw, out_dir=out)

    print("=== Splits saved ===")
    for split, p in paths.items():
        df = pd.read_parquet(p)
        dist = df["label"].value_counts().sort_index().to_dict()
        src = df["source"].value_counts().to_dict()
        print(f"[{split}] n={len(df):,}  labels={dist}  sources={src}  -> {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
