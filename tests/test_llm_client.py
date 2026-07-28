"""Gemini REST 클라이언트의 오류·재시도·키 비노출 검증."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message

import pytest

from src.data import llm_client
from src.data.llm_client import (
    GeminiAPIError,
    GeminiClient,
    GeminiResponseError,
    GeminiTransportError,
    RateLimitedError,
    RequestPacer,
)


def _response(text: str = "[]") -> io.BytesIO:
    payload = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return io.BytesIO(json.dumps(payload).encode())


def _http_error(
    code: int,
    *,
    message: str,
    status: str,
    retry_after: str | None = None,
    retry_delay: str | None = None,
) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    error: dict[str, object] = {"code": code, "message": message, "status": status}
    if retry_delay is not None:
        error["details"] = [
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        ]
    body = json.dumps({"error": error}).encode()
    return urllib.error.HTTPError(
        "https://generativelanguage.googleapis.com/v1beta/models/test:generateContent",
        code,
        "request failed",
        headers,
        io.BytesIO(body),
    )


def test_api_key_is_sent_in_header_not_url(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _response()

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    client = GeminiClient("secret-test-key", temperature=None, max_retries=1)

    assert client("prompt") == "[]"
    request = captured["request"]
    assert "secret-test-key" not in request.full_url
    assert request.get_header("X-goog-api-key") == "secret-test-key"
    assert captured["timeout"] == 90
    request_body = json.loads(request.data)
    assert "temperature" not in request_body["generationConfig"]


def test_multi_part_response_ignores_thought_and_joins_text(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "내부 추론"},
                        {"text": "["},
                        {"text": "]"},
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(
        llm_client.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )

    assert GeminiClient("secret-test-key", max_retries=1)("prompt") == "[]"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"candidates": ["not-an-object"]},
        {"candidates": [{"content": {"parts": [{"text": 123}]}}]},
    ],
)
def test_malformed_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
):
    monkeypatch.setattr(
        llm_client.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )
    client = GeminiClient("secret-test-key", max_retries=1)

    with pytest.raises(GeminiResponseError):
        client("prompt")


def test_permanent_error_preserves_detail_without_retry_or_key(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise _http_error(
            403,
            message="API key lacks permission",
            status="PERMISSION_DENIED",
        )

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    client = GeminiClient("secret-test-key", max_retries=4)

    with pytest.raises(GeminiAPIError) as caught:
        client("prompt")

    assert calls == 1
    assert caught.value.status_code == 403
    assert caught.value.status == "PERMISSION_DENIED"
    assert "API key lacks permission" in str(caught.value)
    assert "secret-test-key" not in str(caught.value)


def test_transient_503_retries_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(503, message="overloaded", status="UNAVAILABLE")
        return _response("ok")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_client.random, "uniform", lambda start, end: 0.0)
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    client = GeminiClient("secret-test-key", max_retries=2, backoff_base=1.0)

    assert client("prompt") == "ok"
    assert calls == 2
    assert sleeps == [1.0]


def test_exhausted_transport_error_is_wrapped_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.URLError("temporary DNS failure")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_client.random, "uniform", lambda start, end: 0.0)
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    client = GeminiClient("secret-test-key", max_retries=2, backoff_base=1.0)

    with pytest.raises(GeminiTransportError, match="2 attempt"):
        client("prompt")

    assert calls == 2
    assert sleeps == [1.0]


def test_mid_body_connection_reset_is_retried(monkeypatch: pytest.MonkeyPatch):
    calls = 0
    sleeps: list[float] = []

    class ResetResponse(io.BytesIO):
        def read(self, size=-1):
            raise ConnectionResetError("reset mid-body")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return ResetResponse()

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_client.random, "uniform", lambda start, end: 0.0)
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    client = GeminiClient("secret-test-key", max_retries=2, backoff_base=1.0)

    with pytest.raises(GeminiTransportError):
        client("prompt")

    assert calls == 2
    assert sleeps == [1.0]


def test_429_exposes_quota_detail_and_retry_after(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(request, timeout):
        raise _http_error(
            429,
            message="Per-minute request quota exceeded",
            status="RESOURCE_EXHAUSTED",
            retry_delay="12.5s",
        )

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    client = GeminiClient("secret-test-key", max_retries=1)

    with pytest.raises(RateLimitedError) as caught:
        client("prompt")

    assert caught.value.retryable is True
    assert caught.value.retry_after == 12.5
    assert "Per-minute request quota exceeded" in str(caught.value)


def test_retry_after_above_local_cap_fails_without_early_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise _http_error(
            429,
            message="Daily quota exhausted",
            status="RESOURCE_EXHAUSTED",
            retry_delay="300s",
        )

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    client = GeminiClient("secret-test-key", max_retries=4, max_backoff=60.0)

    with pytest.raises(RateLimitedError) as caught:
        client("prompt")

    assert calls == 1
    assert sleeps == []
    assert "retry after at least 300s" in str(caught.value)


def test_retry_delay_falls_back_to_error_message():
    error = _http_error(
        429,
        message="Quota exceeded. Please retry in 30.275133886s.",
        status="RESOURCE_EXHAUSTED",
    )

    parsed = GeminiClient._api_error(error)

    assert parsed.retry_after == pytest.approx(30.275133886)


def test_request_pacer_enforces_shared_minimum_interval(monkeypatch: pytest.MonkeyPatch):
    moments = iter([0.0, 0.0, 2.0, 10.0])
    sleeps: list[float] = []
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    pacer = RequestPacer(min_interval=10.0)

    pacer.wait()
    pacer.wait()

    assert sleeps == [8.0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_pacing_and_backoff_values_are_rejected(value: float):
    with pytest.raises(ValueError, match="유한"):
        RequestPacer(value)
    with pytest.raises(ValueError, match="유한"):
        GeminiClient("secret-test-key", backoff_base=value)
