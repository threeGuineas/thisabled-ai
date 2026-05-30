"""Logger factory: 표준 Python logging 기반."""

from __future__ import annotations

import logging

_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """이름이 동일한 로거에 핸들러를 중복 부착하지 않도록 캐시.

    Args:
        name: 로거 이름. 보통 호출 모듈의 ``__name__``을 전달.
        level: 로그 레벨 (기본 INFO).

    Returns:
        설정 완료된 ``logging.Logger`` 인스턴스.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
