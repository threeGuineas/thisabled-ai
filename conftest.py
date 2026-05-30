"""pytest 루트 설정: 프로젝트 루트를 sys.path에 추가하여 src 패키지 import 가능하게 함."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
