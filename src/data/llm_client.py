"""공용 Gemini 호출 클라이언트 (합성·검수에서 주입해 쓴다).

기존 scripts/translate_pan12.py의 검증된 호출 패턴을 재사용하되, 합성/검수 로직이
LLM 구현에 묶이지 않도록 callable(TextGenerator)로 분리한다. 테스트는 fake generator를
주입해 네트워크·키 없이 파이프라인을 검증한다.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

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


class RateLimitedError(RuntimeError):
    """HTTP 429."""


class TextGenerator(Protocol):
    """프롬프트를 받아 모델 텍스트를 반환하는 최소 인터페이스."""

    def __call__(self, prompt: str) -> str: ...


class GeminiClient:
    """generateContent 호출 + 429 백오프 재시도. responseMimeType=application/json."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-flash-latest",
        temperature: float = 0.9,
        timeout: int = 90,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY 없음")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._max_retries = max_retries

    def __call__(self, prompt: str) -> str:
        url = f"{_ENDPOINT.format(model=self._model)}?key={self._api_key}"
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "safetySettings": _SAFETY,
                "generationConfig": {
                    "temperature": self._temperature,
                    "responseMimeType": "application/json",
                },
            }
        ).encode()
        for attempt in range(self._max_retries):
            try:
                return self._request(url, body)
            except RateLimitedError:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(15 * (attempt + 1))
        raise RateLimitedError("max retries exceeded")

    def _request(self, url: str, body: bytes) -> str:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 고정 도메인
                data = json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimitedError from exc
            raise
        cands = data.get("candidates")
        if not cands:
            raise KeyError(f"no candidates ({data.get('promptFeedback', {})})")
        parts = cands[0].get("content", {}).get("parts")
        if not parts:
            raise KeyError(f"blocked (finishReason={cands[0].get('finishReason')})")
        return parts[0]["text"]
