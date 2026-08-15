"""AI 소통 코치(모듈③) 서비스 — 캐싱·프리셋·명시 실행 가드.

COMM-05: 코치는 사용자가 버튼을 눌렀을 때만 동작한다. 이 모듈에는 자동 호출 경로가
없고, run()은 invoked_by_user=True 없이는 실행되지 않는다. 타이핑 중 자동 제안 같은
흐름을 붙이려면 이 가드를 먼저 지워야 하므로, 실수로 추가되지 않는다.

비용 컷: LLM을 끄거나(enabled=False) 호출이 실패하면 프리셋으로 내려앉는다. 기능이
사라지는 대신 품질만 낮아진다. 원문을 다시 써야 하는 COMM-01/02는 프리셋으로 대체할 수
없으므로 unavailable 로 정직하게 알린다.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.coach.prompts import (
    MAX_SUGGESTIONS,
    PROMPT_REVISION,
    CoachAction,
    PromptInputError,
    build_prompt,
    parse_suggestions,
)
from src.data.matching_input import contains_contact_info

# 캐시 키 구분자 (ASCII unit separator). 사용자 입력에는 나타나지 않는다.
_KEY_SEPARATOR = "\x1f"

# 프리셋으로 대체할 수 있는 동작. 원문 변환이 필요한 동작은 대체 불가.
_PRESETABLE = frozenset({CoachAction.SUGGEST_REPLY, CoachAction.CONVERSATION_HINT})

DEFAULT_PRESETS: dict[CoachAction, tuple[str, ...]] = {
    CoachAction.SUGGEST_REPLY: (
        "그렇군요, 더 자세히 들려주세요.",
        "좋은 생각이에요. 저도 비슷하게 느껴요.",
        "알려줘서 고마워요.",
    ),
    CoachAction.CONVERSATION_HINT: (
        "요즘 어떤 걸 즐겨 보세요?",
        "주말에는 보통 뭐 하세요?",
        "그건 어떻게 시작하게 되셨어요?",
    ),
}


class CoachError(RuntimeError):
    """코치 호출이 실패했고 대체할 수단도 없을 때."""


class CoachNotInvokedError(CoachError):
    """COMM-05 위반 — 사용자 실행 없이 호출됨."""


@dataclass(frozen=True, slots=True)
class CoachResult:
    action: CoachAction
    suggestions: tuple[str, ...]
    source: str  # "llm" | "cache" | "preset"
    latency_ms: float
    prompt_revision: str
    degraded_reason: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.suggestions)


@dataclass
class _TTLCache:
    """(동작, 프롬프트 개정, 입력) → 후보. 같은 입력의 재호출 비용을 없앤다."""

    max_entries: int = 512
    ttl_seconds: float = 600.0
    _store: OrderedDict[str, tuple[float, tuple[str, ...]]] = field(
        default_factory=OrderedDict, repr=False
    )

    def get(self, key: str, *, now: float) -> tuple[str, ...] | None:
        hit = self._store.get(key)
        if hit is None:
            return None
        stored_at, value = hit
        if now - stored_at > self.ttl_seconds:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key: str, value: tuple[str, ...], *, now: float) -> None:
        self._store[key] = (now, value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


def _cache_key(
    action: CoachAction,
    *,
    text: str | None,
    context: Sequence[tuple[str, str]] | None,
    max_suggestions: int,
) -> str:
    parts = [PROMPT_REVISION, action.value, str(max_suggestions), text or ""]
    if context:
        parts.extend(f"{speaker}:{message}" for speaker, message in context)
    # 구분자는 사용자 입력에 나타날 수 없는 제어문자여야 한다. 공백으로 이으면
    # text="a b" 와 text="a", context=["b"] 가 같은 키가 된다.
    return hashlib.sha256(_KEY_SEPARATOR.join(parts).encode("utf-8")).hexdigest()


class CoachService:
    """LLM 호출을 감싼 코치. 생성기는 주입식이라 테스트에서 대역을 쓴다."""

    def __init__(
        self,
        generate: Callable[[str], str] | None = None,
        *,
        enabled: bool = True,
        presets: dict[CoachAction, tuple[str, ...]] | None = None,
        safety_check: Callable[[str], bool] | None = None,
        cache_max_entries: int = 512,
        cache_ttl_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """safety_check: 후보 하나가 안전하면 True. SAFE 서버를 물릴 지점이다.

        코치가 문장을 다듬어 주는 기능은 그루밍·사기 메시지를 매끄럽게 만드는 데도
        쓰일 수 있다. 후보를 그대로 내보내기 전에 걸러야 한다.
        """

        self._generate = generate
        self._enabled = enabled and generate is not None
        self._presets = presets if presets is not None else DEFAULT_PRESETS
        self._safety_check = safety_check
        self._cache = _TTLCache(max_entries=cache_max_entries, ttl_seconds=cache_ttl_seconds)
        self._clock = clock

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _filter(self, suggestions: Sequence[str]) -> list[str]:
        """연락처가 섞였거나 안전 검사에 걸린 후보를 뺀다."""

        kept = []
        for suggestion in suggestions:
            if contains_contact_info(suggestion):
                continue
            if self._safety_check is not None and not self._safety_check(suggestion):
                continue
            kept.append(suggestion)
        return kept

    def _preset_result(self, action: CoachAction, *, started: float, reason: str) -> CoachResult:
        suggestions = tuple(self._presets.get(action, ())) if action in _PRESETABLE else ()
        return CoachResult(
            action=action,
            suggestions=suggestions,
            source="preset",
            latency_ms=(self._clock() - started) * 1000.0,
            prompt_revision=PROMPT_REVISION,
            degraded_reason=reason,
        )

    def run(
        self,
        action: CoachAction,
        *,
        invoked_by_user: bool,
        text: str | None = None,
        context: Sequence[tuple[str, str]] | None = None,
        max_suggestions: int = MAX_SUGGESTIONS,
    ) -> CoachResult:
        """COMM-05: invoked_by_user 가 True 일 때만 동작한다."""

        if invoked_by_user is not True:
            raise CoachNotInvokedError("코치는 사용자가 직접 실행했을 때만 동작합니다 (COMM-05).")

        started = self._clock()
        # 입력 검증은 캐시·LLM 앞단에서 한다. 잘못된 입력에 비용을 쓰지 않는다.
        prompt = build_prompt(action, text=text, context=context, max_suggestions=max_suggestions)
        key = _cache_key(action, text=text, context=context, max_suggestions=max_suggestions)

        cached = self._cache.get(key, now=self._clock())
        if cached is not None:
            return CoachResult(
                action=action,
                suggestions=cached,
                source="cache",
                latency_ms=(self._clock() - started) * 1000.0,
                prompt_revision=PROMPT_REVISION,
            )

        if not self._enabled:
            return self._preset_result(action, started=started, reason="llm_disabled")

        assert self._generate is not None
        try:
            raw = self._generate(prompt)
            suggestions = parse_suggestions(raw, max_suggestions=max_suggestions)
        except PromptInputError as exc:
            return self._preset_result(action, started=started, reason=f"unparsable: {exc}")
        except Exception as exc:  # 생성기 계층의 오류를 기능 중단으로 만들지 않는다.
            return self._preset_result(action, started=started, reason=f"llm_error: {exc}")

        kept = self._filter(suggestions)
        if not kept:
            return self._preset_result(action, started=started, reason="all_candidates_filtered")

        value = tuple(kept)
        self._cache.put(key, value, now=self._clock())
        return CoachResult(
            action=action,
            suggestions=value,
            source="llm",
            latency_ms=(self._clock() - started) * 1000.0,
            prompt_revision=PROMPT_REVISION,
        )
