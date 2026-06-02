"""MLflow 실험 추적 헬퍼.

project_facts 재현성 규약: "모든 실험은 MLflow 자동 로깅." 학습·평가 스크립트가
동일한 방식으로 run을 열고 파라미터/지표/아티팩트를 기록하도록 얇은 래퍼를 제공한다.

기본 tracking URI는 리포지토리 루트의 ``mlruns/`` (파일 백엔드)이며,
``MLFLOW_TRACKING_URI`` 환경변수가 있으면 그 값을 우선한다.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_DIR = _REPO_ROOT / "mlruns"


def _resolve_tracking_uri(tracking_uri: str | None) -> str:
    if tracking_uri:
        return tracking_uri
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    return DEFAULT_TRACKING_DIR.as_uri()


@contextmanager
def mlflow_run(
    experiment: str,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """MLflow run 컨텍스트. mlflow 미설치 시 no-op으로 흘려보낸다.

    HuggingFace Trainer의 MLflowCallback(``report_to=["mlflow"]``)은 이미 active run이
    있으면 그 run을 재사용하고 직접 종료하지 않으므로, 이 컨텍스트로 run 수명을 감싸면
    학습 중 지표(callback)와 사후 지표(수동 log)가 같은 run에 모인다.

    Args:
        experiment: 실험 이름 (``mlflow.set_experiment``).
        run_name: run 표시 이름.
        tracking_uri: 미지정 시 ``MLFLOW_TRACKING_URI`` 또는 ``<repo>/mlruns``.
        params: run 시작 직후 기록할 파라미터. 키는 HF가 자동 로깅하는 키와 겹치지
            않도록 ``cfg/`` 등 접두사 사용 권장.

    Yields:
        활성 mlflow run 객체. mlflow 미설치 시 ``None``.
    """
    try:
        import mlflow
    except ImportError:
        print("⚠ mlflow 미설치 — 실험 추적을 건너뜁니다. `pip install mlflow`.")
        yield None
        return

    mlflow.set_tracking_uri(_resolve_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(_flatten_params(params))
        yield run


def _flatten_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """list/tuple/dict 값을 문자열로 평탄화 (mlflow 파라미터는 스칼라 권장)."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        out[k] = str(v) if isinstance(v, list | tuple | dict) else v
    return out


def log_metrics(metrics: Mapping[str, Any], prefix: str = "") -> None:
    """스칼라 지표만 골라 mlflow에 기록. mlflow 미설치/활성 run 없으면 no-op."""
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None:
        return
    scalar = {
        f"{prefix}{k}": float(v)
        for k, v in metrics.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }
    if scalar:
        mlflow.log_metrics(scalar)
