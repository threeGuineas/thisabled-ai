"""SAFE 서빙의 이진/4-class 공용 응답 계약."""

from __future__ import annotations

import asyncio
import sys
import types

# 이 테스트는 모델 다운로드가 아니라 응답 매핑만 검증한다. 개발 환경의
# transformers/huggingface-hub 버전 조합과 무관하도록 로더 심볼만 격리한다.
transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModelForSequenceClassification = object
transformers_stub.AutoTokenizer = object
sys.modules["transformers"] = transformers_stub

from serving.safety_server import app as safety  # noqa: E402


def _analyze(probs: list[float], monkeypatch):
    labels = safety.LABELS_BY_N[len(probs)]
    safety._state["labels"] = labels
    monkeypatch.setattr(safety, "_infer", lambda _text: probs)
    return asyncio.run(safety.analyze(safety.AnalyzeIn(text="테스트")))


def test_binary_response_uses_two_labels(monkeypatch) -> None:
    body = _analyze([0.2, 0.8], monkeypatch)
    assert body["verdict"] == "flagged"
    assert body["level"] == "주의"
    assert body["probs"] == {"정상": 0.2, "주의": 0.8}


def test_four_class_response_remains_supported(monkeypatch) -> None:
    body = _analyze([0.7, 0.1, 0.1, 0.1], monkeypatch)
    assert body["verdict"] == "safe"
    assert body["level"] == "정상"
    assert list(body["probs"]) == ["정상", "주의", "경고", "긴급"]


def test_health_exposes_calibration() -> None:
    body = asyncio.run(safety.health())
    assert body["threshold"] == safety.THRESHOLD
    assert body["threshold_minor"] == safety.THRESHOLD_MINOR
    assert "revision" in body
