"""Train Module ① (KcELECTRA + Focal Loss).

Usage:
    python scripts/train_module1.py
    python scripts/train_module1.py --config configs/module1_kcelectra.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.trainer import train_module1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "module1_kcelectra.yaml"),
    )
    args = parser.parse_args()

    result = train_module1(args.config, project_root=ROOT)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
