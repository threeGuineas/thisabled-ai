"""소통 코치 프롬프트 회귀 테스트 — COMM-01~04."""

from __future__ import annotations

import pytest

from src.coach.prompts import (
    MAX_INPUT_CHARS,
    MAX_SUGGESTIONS,
    CoachAction,
    PromptInputError,
    build_prompt,
    parse_suggestions,
)


def test_every_action_produces_a_prompt():
    """네 동작 모두 프롬프트가 만들어져야 COMM-01~04가 채워진다."""

    text_actions = {CoachAction.EASY_SENTENCE, CoachAction.COMPLETE_SENTENCE}
    for action in CoachAction:
        if action in text_actions:
            prompt = build_prompt(action, text="오늘 같이 밥 먹을래")
        else:
            prompt = build_prompt(action, context=[("partner", "안녕하세요")])
        assert prompt.strip()
        assert "JSON" in prompt


@pytest.mark.parametrize("action", list(CoachAction))
def test_prompt_never_mentions_disability_or_ui_mode(action: CoachAction):
    """MATCH-04와 같은 원칙 — 모델에 장애·UI 모드를 넘기지 않는다."""

    if action in {CoachAction.EASY_SENTENCE, CoachAction.COMPLETE_SENTENCE}:
        prompt = build_prompt(action, text="같이 가고 싶어요")
    else:
        prompt = build_prompt(action, context=[("partner", "뭐 하세요?")])

    for banned in ("발달장애", "시각장애", "청각장애", "ui_mode", "visual", "hearing"):
        assert banned not in prompt


def test_context_uses_role_labels_not_identifiers():
    """대화 맥락에 사용자 이름·아이디가 아니라 역할만 들어가야 한다."""

    prompt = build_prompt(
        CoachAction.SUGGEST_REPLY,
        context=[("partner", "주말에 뭐 했어요?"), ("me", "영화 봤어요")],
    )

    assert "상대: 주말에 뭐 했어요?" in prompt
    assert "나: 영화 봤어요" in prompt


def test_context_is_trimmed_to_recent_messages():
    context = [("partner" if i % 2 == 0 else "me", f"메시지{i}") for i in range(20)]

    prompt = build_prompt(CoachAction.CONVERSATION_HINT, context=context)

    assert "메시지19" in prompt
    assert "메시지0" not in prompt


def test_text_actions_reject_context_and_vice_versa():
    with pytest.raises(PromptInputError, match="대화 맥락을 쓰지 않습니다"):
        build_prompt(CoachAction.EASY_SENTENCE, text="안녕", context=[("me", "안녕")])
    with pytest.raises(PromptInputError, match="context가 필요합니다"):
        build_prompt(CoachAction.SUGGEST_REPLY)


def test_oversized_input_is_rejected_before_any_call():
    with pytest.raises(PromptInputError, match="넘습니다"):
        build_prompt(CoachAction.EASY_SENTENCE, text="가" * (MAX_INPUT_CHARS + 1))


def test_blank_input_is_rejected():
    with pytest.raises(PromptInputError, match="비어 있습니다"):
        build_prompt(CoachAction.EASY_SENTENCE, text="   ")


def test_unknown_speaker_is_rejected():
    with pytest.raises(PromptInputError, match="알 수 없는 화자"):
        build_prompt(CoachAction.SUGGEST_REPLY, context=[("admin", "공지")])


def test_parse_accepts_code_fenced_json():
    raw = '```json\n{"suggestions": ["첫 번째", "두 번째"]}\n```'

    assert parse_suggestions(raw) == ["첫 번째", "두 번째"]


def test_parse_ignores_prose_around_the_object():
    raw = '알겠습니다.\n{"suggestions": ["답장이에요"]}\n도움이 되었길 바랍니다.'

    assert parse_suggestions(raw) == ["답장이에요"]


def test_parse_deduplicates_and_caps():
    raw = '{"suggestions": ["같은 말", "같은 말", "다른 말", "또 다른 말", "네 번째"]}'

    assert parse_suggestions(raw) == ["같은 말", "다른 말", "또 다른 말"]
    assert len(parse_suggestions(raw)) <= MAX_SUGGESTIONS


def test_parse_rejects_unusable_payloads():
    for raw in ('{"suggestions": []}', '{"suggestions": "문자열"}', "설명만 있음", "{}"):
        with pytest.raises(PromptInputError):
            parse_suggestions(raw)
