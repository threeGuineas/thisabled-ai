"""Colab 환경 보조 유틸. 로컬에서는 동작 안 함 (import만 가능)."""

from __future__ import annotations

import shutil
from pathlib import Path


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def stage_data(drive_path: str | Path, local_dir: str | Path = "/content/data") -> Path:
    """Drive 파일을 Colab 로컬 SSD로 복사 (학습 I/O 5~10배 가속).

    Args:
        drive_path: Drive 상의 원본 파일.
        local_dir: Colab 로컬 디렉터리.

    Returns:
        로컬 복사본 경로.
    """
    drive_path = Path(drive_path)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / drive_path.name
    if not target.exists():
        print(f"📥 Copying {drive_path} → {target}")
        shutil.copy2(drive_path, target)
    else:
        print(f"♻️  Already staged: {target}")
    return target
