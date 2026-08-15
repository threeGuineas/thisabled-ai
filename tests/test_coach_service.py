"""소통 코치 서비스 회귀 테스트 — COMM-05·캐싱·프리셋·출력 필터."""

from __future__ import annotations

import json

import pytest

from src.coach import (
    CoachAction,
    CoachNotInvokedError,
    CoachService,
)


class RecordingGenerator:
    """호출 프롬프트를 기록하고 정해진 응답을 돌려주는 대역."""

    def __init__(self, suggestions=("첫 번째 제안", "두 번째 제안")):
        self.prompts: list[str] = []
        self._suggestions = list(suggestions)

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps({"suggestions": self._suggestions}, ensure_ascii=False)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _reply_context():
    return [("partner", "주말에 뭐 하세요?")]


def test_comm05_refuses_to_run_without_explicit_user_action():
    """자동 호출 경로가 없어야 한다. 가드를 지우지 않으면 실수로 붙일 수 없다."""

    generator = RecordingGenerator()
    service = CoachService(generator)

    with pytest.raises(CoachNotInvokedError, match="COMM-05"):
        service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=False, context=_reply_context())

    # 가드에 걸리면 LLM을 부르지 않는다 — 비용도 쓰지 않는다.
    assert generator.prompts == []


@pytest.mark.parametrize("falsy", [0, "", None, []])
def test_comm05_guard_is_not_satisfied_by_truthy_lookalikes(falsy):
    service = CoachService(RecordingGenerator())

    with pytest.raises(CoachNotInvokedError):
        service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=falsy, context=_reply_context())


def test_second_identical_call_is_served_from_cache():
    generator = RecordingGenerator()
    service = CoachService(generator)

    first = service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())
    second = service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())

    assert first.source == "llm"
    assert second.source == "cache"
    assert second.suggestions == first.suggestions
    assert len(generator.prompts) == 1  # 두 번째는 호출하지 않았다


def test_cache_expires_after_ttl():
    clock = FakeClock()
    generator = RecordingGenerator()
    service = CoachService(generator, cache_ttl_seconds=60.0, clock=clock)

    service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())
    clock.advance(61.0)
    again = service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())

    assert again.source == "llm"
    assert len(generator.prompts) == 2


def test_different_input_does_not_share_cache_entry():
    generator = RecordingGenerator()
    service = CoachService(generator)

    service.run(CoachAction.EASY_SENTENCE, invoked_by_user=True, text="첫 문장")
    service.run(CoachAction.EASY_SENTENCE, invoked_by_user=True, text="다른 문장")

    assert len(generator.prompts) == 2


def test_cache_key_does_not_collide_across_field_boundaries():
    """구분자가 없으면 text='주말에 뭐 하세요?' 와 맥락 한 줄이 같은 키가 된다."""

    generator = RecordingGenerator()
    service = CoachService(generator)

    service.run(
        CoachAction.SUGGEST_REPLY,
        invoked_by_user=True,
        context=[("partner", "안녕"), ("me", "하세요")],
    )
    service.run(
        CoachAction.SUGGEST_REPLY,
        invoked_by_user=True,
        context=[("partner", "안녕 하세요")],
    )

    assert len(generator.prompts) == 2
    assert service.cache_size == 2


def test_cache_key_includes_action_and_suggestion_count():
    generator = RecordingGenerator()
    service = CoachService(generator)

    service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())
    service.run(CoachAction.CONVERSATION_HINT, invoked_by_user=True, context=_reply_context())
    service.run(
        CoachAction.SUGGEST_REPLY,
        invoked_by_user=True,
        context=_reply_context(),
        max_suggestions=1,
    )

    assert service.cache_size == 3


def test_cache_evicts_oldest_beyond_capacity():
    service = CoachService(RecordingGenerator(), cache_max_entries=2)

    for i in range(5):
        service.run(CoachAction.EASY_SENTENCE, invoked_by_user=True, text=f"문장 {i}")

    assert service.cache_size == 2


def test_disabled_llm_falls_back_to_presets():
    """비용 컷 트리거 — 기능이 사라지지 않고 품질만 낮아진다."""

    service = CoachService(RecordingGenerator(), enabled=False)

    result = service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())

    assert result.source == "preset"
    assert result.available
    assert result.degraded_reason == "llm_disabled"


def test_rewrite_actions_report_unavailable_instead_of_faking_a_preset():
    """COMM-01/02는 원문을 다시 써야 하므로 정해진 문구로 대체할 수 없다."""

    service = CoachService(RecordingGenerator(), enabled=False)

    result = service.run(CoachAction.EASY_SENTENCE, invoked_by_user=True, text="문장")

    assert result.source == "preset"
    assert result.suggestions == ()
    assert result.available is False


def test_generator_failure_degrades_instead_of_raising():
    def boom(prompt: str) -> str:
        raise RuntimeError("upstream 503")

    service = CoachService(boom)

    result = service.run(
        CoachAction.CONVERSATION_HINT, invoked_by_user=True, context=_reply_context()
    )

    assert result.source == "preset"
    assert "llm_error" in result.degraded_reason


def test_unparsable_response_degrades():
    service = CoachService(lambda prompt: "이건 JSON이 아닙니다")

    result = service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())

    assert result.source == "preset"
    assert "unparsable" in result.degraded_reason


def test_candidates_containing_contact_info_are_dropped():
    generator = RecordingGenerator(
        suggestions=["제 번호는 010-1234-5678이에요", "그렇군요, 더 들려주세요"]
    )
    service = CoachService(generator)

    result = service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())

    assert result.suggestions == ("그렇군요, 더 들려주세요",)


def test_safety_check_can_reject_candidates():
    """코치가 그루밍·사기 문장을 다듬어 주지 않도록 SAFE를 물릴 수 있어야 한다."""

    generator = RecordingGenerator(suggestions=["우리 둘만의 비밀이야", "재미있었겠어요"])
    blocked = {"우리 둘만의 비밀이야"}
    service = CoachService(generator, safety_check=lambda text: text not in blocked)

    result = service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())

    assert result.suggestions == ("재미있었겠어요",)


def test_all_candidates_filtered_degrades_to_preset():
    generator = RecordingGenerator(suggestions=["위험1", "위험2"])
    service = CoachService(generator, safety_check=lambda text: False)

    result = service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())

    assert result.source == "preset"
    assert result.degraded_reason == "all_candidates_filtered"


def test_filtered_candidates_are_not_cached():
    """걸러진 결과가 캐시에 남으면 다음 호출도 오염된다."""

    generator = RecordingGenerator(suggestions=["위험"])
    service = CoachService(generator, safety_check=lambda text: False)

    service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())

    assert service.cache_size == 0


def test_result_records_latency_and_prompt_revision():
    clock = FakeClock()

    def slow(prompt: str) -> str:
        clock.advance(1.5)
        return json.dumps({"suggestions": ["답장"]}, ensure_ascii=False)

    service = CoachService(slow, clock=clock)

    result = service.run(CoachAction.SUGGEST_REPLY, invoked_by_user=True, context=_reply_context())

    assert result.latency_ms == pytest.approx(1500.0)
    assert result.prompt_revision
