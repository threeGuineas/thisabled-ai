"""Tests for src.data.synthesis_prompts — 프롬프트 빌더 구조 검증 (실 API 호출 X)."""

from __future__ import annotations

from src.data.synthesis_prompts import (
    CATEGORY_PROMPTS,
    CONTEXTS,
    LANGUAGE_STYLES,
    PERSONAS,
    SYSTEM_PROMPT,
    TARGET_COUNTS,
    build_user_prompt,
    category_personas,
)


def test_all_categories_have_required_fields():
    """5개 카테고리(3a/3b/3c/3d/boundary) 모두 필수 필드 존재."""
    expected = {"3a", "3b", "3c", "3d", "boundary"}
    assert set(CATEGORY_PROMPTS.keys()) == expected
    for cat, spec in CATEGORY_PROMPTS.items():
        for key in ("name", "definition", "signals", "examples", "task"):
            assert key in spec, f"{cat} missing {key}"
        assert isinstance(spec["signals"], list) and len(spec["signals"]) >= 3
        assert isinstance(spec["examples"], list) and len(spec["examples"]) >= 2


def test_target_counts_match_spec():
    """emergency_scenarios.md §5.1의 카테고리별 목표 건수."""
    assert TARGET_COUNTS["3a"] == {"train": 200, "val": 30, "test": 50}
    assert TARGET_COUNTS["3b"] == {"train": 150, "val": 25, "test": 40}
    assert TARGET_COUNTS["3c"] == {"train": 150, "val": 25, "test": 40}
    assert TARGET_COUNTS["3d"] == {"train": 100, "val": 20, "test": 30}
    assert TARGET_COUNTS["boundary"] == {"train": 200, "val": 50, "test": 60}
    total = sum(sum(v.values()) for v in TARGET_COUNTS.values())
    assert total == 1170  # spec과 일치


def test_build_user_prompt_includes_signals_and_format():
    prompt = build_user_prompt(
        category="3a",
        n_examples=10,
        persona_pool=PERSONAS["disability"][:3],
        contexts=CONTEXTS[:2],
        styles=LANGUAGE_STYLES[:2],
        disability_ratio=0.5,
    )
    # 필수 요소 포함
    assert "그루밍" in prompt
    assert "신호 패턴" in prompt
    assert "JSON" in prompt
    assert "총 10건" in prompt
    # 장애 도메인 비중 명시
    assert "장애 도메인" in prompt
    assert "5/10" in prompt  # n_examples * 0.5


def test_boundary_prompt_includes_label_field():
    """boundary는 label/reason 필드를 JSON 스키마에 포함."""
    prompt = build_user_prompt(
        category="boundary",
        n_examples=5,
        persona_pool=PERSONAS["general"][:2],
        contexts=CONTEXTS[:2],
        styles=LANGUAGE_STYLES[:2],
    )
    assert '"label"' in prompt
    assert '"reason"' in prompt
    assert "0|1|2" in prompt  # JSON schema에 라벨 후보 명시


def test_emergency_category_prompt_excludes_boundary_fields():
    """긴급 카테고리(3a~3d)는 label/reason 필드 없이 (기본 label=3)."""
    prompt = build_user_prompt(
        category="3c",
        n_examples=5,
        persona_pool=PERSONAS["general"][:2],
        contexts=CONTEXTS[:2],
        styles=LANGUAGE_STYLES[:2],
    )
    assert '"label"' not in prompt
    assert '"reason"' not in prompt


def test_category_personas_disability_coverage():
    """모든 카테고리에 장애 도메인 페르소나가 풀에 포함돼야 함 (장애 도메인 ≥30% 강제 위해)."""
    for cat in ("3a", "3b", "3c", "3d", "boundary"):
        pool = category_personas(cat)
        has_disability = any(p in PERSONAS["disability"] for p in pool)
        assert has_disability, f"{cat}: 장애 도메인 페르소나 없음"


def test_system_prompt_mentions_safety_principles():
    """안전 원칙 명시 — 가해 스크립트 작성 금지 등."""
    assert "안전" in SYSTEM_PROMPT
    assert "신호" in SYSTEM_PROMPT
    assert "장애" in SYSTEM_PROMPT
    assert "JSONL" in SYSTEM_PROMPT
