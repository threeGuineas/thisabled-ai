"""공용 Gemini 호출 클라이언트 (합성·검수에서 주입해 쓴다).

기존 scripts/translate_pan12.py의 검증된 호출 패턴을 재사용하되, 합성/검수 로직이
LLM 구현에 묶이지 않도록 callable(TextGenerator)로 분리한다. 테스트는 fake generator를
주입해 네트워크·키 없이 파이프라인을 검증한다.
"""

from __future__ import annotations

import http.client
import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_RETRYABLE_HTTP_CODES = {408, 429}
_MAX_ERROR_MESSAGE_LENGTH = 500
_MAX_ERROR_BODY_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024

# 사기 예시 합성/검수는 유해 텍스트를 다루므로 안전필터로 응답이 막히지 않게 한다.
# (탐지 모델이 배워야 할 신호 — translate_pan12와 동일 취지.)
_SAFETY = [
    {"category": category, "threshold": "BLOCK_NONE"}
    for category in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


class GeminiClientError(RuntimeError):
    """Gemini 호출 계층에서 외부로 노출하는 오류의 공통 기반."""


class GeminiTransportError(GeminiClientError):
    """재시도 후에도 회복되지 않은 네트워크·타임아웃 오류."""


class GeminiResponseError(GeminiClientError):
    """성공 HTTP 응답의 JSON 또는 후보 구조가 계약과 다른 경우."""


class GeminiAPIError(GeminiClientError):
    """Gemini API가 반환한 구조화된 HTTP 오류."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        status: str = "",
        retry_after: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.status = status
        self.retry_after = retry_after
        self.retryable = status_code in _RETRYABLE_HTTP_CODES or 500 <= status_code < 600
        detail = " ".join(str(message).split())[:_MAX_ERROR_MESSAGE_LENGTH]
        self.message = detail
        status_detail = f" {status}" if status else ""
        retry_detail = (
            f" (retry after at least {retry_after:g}s)" if retry_after is not None else ""
        )
        super().__init__(f"Gemini API HTTP {status_code}{status_detail}: {detail}{retry_detail}")


class RateLimitedError(GeminiAPIError):
    """Gemini API HTTP 429."""


class TextGenerator(Protocol):
    """프롬프트를 받아 모델 텍스트를 반환하는 최소 인터페이스."""

    def __call__(self, prompt: str) -> str: ...


class RequestPacer:
    """여러 GeminiClient가 공유할 수 있는 최소 호출 간격 제어기."""

    def __init__(self, min_interval: float = 0.0) -> None:
        if not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("min_interval은 유한한 0 이상 값이어야 함")
        self._min_interval = min_interval
        self._last_request_at: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_request_at is not None:
            remaining = self._min_interval - (now - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()


class GeminiClient:
    """generateContent 호출 + 429 백오프 재시도. responseMimeType=application/json."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-flash-latest",
        temperature: float | None = 0.9,
        timeout: int = 90,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        max_backoff: float = 60.0,
        pacer: RequestPacer | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY 없음")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout은 유한한 양수여야 함")
        if max_retries <= 0:
            raise ValueError("max_retries는 1 이상이어야 함")
        if (
            not math.isfinite(backoff_base)
            or not math.isfinite(max_backoff)
            or backoff_base < 0
            or max_backoff < 0
        ):
            raise ValueError("backoff 값은 유한한 0 이상 값이어야 함")
        if temperature is not None and not math.isfinite(temperature):
            raise ValueError("temperature는 유한한 값이어야 함")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._max_backoff = max_backoff
        self._pacer = pacer

    def __call__(self, prompt: str) -> str:
        url = _ENDPOINT.format(model=self._model)
        generation_config: dict[str, object] = {"responseMimeType": "application/json"}
        if self._temperature is not None:
            generation_config["temperature"] = self._temperature
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "safetySettings": _SAFETY,
                "generationConfig": generation_config,
            }
        ).encode()
        for attempt in range(self._max_retries):
            try:
                if self._pacer is not None:
                    self._pacer.wait()
                return self._request(url, body)
            except GeminiAPIError as exc:
                if not exc.retryable or attempt == self._max_retries - 1:
                    raise
                if exc.retry_after is not None and exc.retry_after > self._max_backoff:
                    raise
                self._sleep_before_retry(attempt, retry_after=exc.retry_after)
            except (
                TimeoutError,
                urllib.error.URLError,
                OSError,
                http.client.IncompleteRead,
            ) as exc:
                if attempt == self._max_retries - 1:
                    raise GeminiTransportError(
                        f"Gemini transport failed after {self._max_retries} attempt(s): "
                        f"{type(exc).__name__}"
                    ) from exc
                self._sleep_before_retry(attempt)
        raise RuntimeError("Gemini 호출 재시도 상태 오류")

    def _sleep_before_retry(self, attempt: int, *, retry_after: float | None = None) -> None:
        base_delay = self._backoff_base * (2**attempt)
        jitter = random.uniform(0, base_delay * 0.25)  # noqa: S311 보안 토큰 용도 아님
        delay = min(base_delay + jitter, self._max_backoff)
        if retry_after is not None:
            delay = min(max(delay, retry_after), self._max_backoff)
        time.sleep(delay)

    def _request(self, url: str, body: bytes) -> str:
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self._api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 고정 도메인
                try:
                    raw_response = resp.read(_MAX_RESPONSE_BYTES + 1)
                    if len(raw_response) > _MAX_RESPONSE_BYTES:
                        raise GeminiResponseError("Gemini response exceeds size limit")
                    data = json.loads(raw_response)
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
                    raise GeminiResponseError("Gemini response is not valid JSON") from exc
        except urllib.error.HTTPError as exc:
            error = self._api_error(exc)
            if exc.code == 429:
                raise RateLimitedError(
                    error.status_code,
                    error.message,
                    status=error.status,
                    retry_after=error.retry_after,
                ) from exc
            raise error from exc
        if not isinstance(data, dict):
            raise GeminiResponseError("Gemini response root must be an object")
        cands = data.get("candidates")
        if not isinstance(cands, list) or not cands:
            raise GeminiResponseError(f"no candidates ({data.get('promptFeedback', {})})")
        candidate = cands[0]
        if not isinstance(candidate, dict):
            raise GeminiResponseError("malformed candidate")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list) or not parts:
            raise GeminiResponseError(f"blocked (finishReason={candidate.get('finishReason')})")
        text_parts = [
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and part["text"].strip()
            and part.get("thought") is not True
        ]
        if not text_parts:
            raise GeminiResponseError("candidate has no text")
        return "".join(text_parts)

    @staticmethod
    def _api_error(exc: urllib.error.HTTPError) -> GeminiAPIError:
        message = str(exc.reason)
        status = ""
        retry_after: float | None = None
        try:
            payload = json.loads(exc.read(_MAX_ERROR_BODY_BYTES))
        except (json.JSONDecodeError, OSError, TypeError, UnicodeDecodeError):
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]
            message = str(error.get("message") or message)
            status = str(error.get("status") or "")
            details = error.get("details")
            if isinstance(details, list):
                for detail in details:
                    if not isinstance(detail, dict):
                        continue
                    retry_after = GeminiClient._parse_retry_delay(detail.get("retryDelay"))
                    if retry_after is not None:
                        break
            if retry_after is None:
                match = re.search(
                    r"(?:please\s+)?retry\s+in\s+(\d+(?:\.\d+)?)s",
                    message[:2000],
                    flags=re.IGNORECASE,
                )
                if match:
                    retry_after = float(match.group(1))

        header_value = exc.headers.get("Retry-After") if exc.headers else None
        if header_value:
            header_retry_after = GeminiClient._parse_retry_after_header(header_value)
            if header_retry_after is not None:
                retry_after = max(retry_after or 0.0, header_retry_after)
        return GeminiAPIError(
            exc.code,
            message,
            status=status,
            retry_after=retry_after,
        )

    @staticmethod
    def _parse_retry_delay(value: object) -> float | None:
        if not isinstance(value, str):
            return None
        match = re.fullmatch(r"(\d+(?:\.\d+)?)s", value.strip())
        if not match:
            return None
        return max(0.0, float(match.group(1)))

    @staticmethod
    def _parse_retry_after_header(value: str) -> float | None:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
