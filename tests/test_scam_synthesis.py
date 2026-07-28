"""사기 합성 + LLM 검수 파이프라인 오프라인 검증 (fake LLM 주입 — 네트워크·키 불필요)."""

from __future__ import annotations

import json

from src.data.scam_synthesis import (
    BENIGN_SUBTYPES,
    SCAM_SUBTYPES,
    build_verify_prompt,
    filter_forbidden,
    parse_examples,
    synthesize_scam,
    verify_labels,
)


class FakeSynthLLM:
    """합성 프롬프트를 받으면 요청 라벨대로 예시 JSON을 반환한다."""

    def __call__(self, prompt: str) -> str:
        label = 0 if '"label": 0' in prompt else 1
        rows = [
            {"text": f"예시 메시지 {i} (label {label})", "label": label, "subtype": "x"}
            for i in range(3)
        ]
        return json.dumps(rows, ensure_ascii=False)


def test_synthesize_produces_scam_and_boundary_examples():
    examples = synthesize_scam(FakeSynthLLM(), per_subtype=3, include_benign=True)
    # 유형 수 × 3개.
    assert len(examples) == (len(SCAM_SUBTYPES) + len(BENIGN_SUBTYPES)) * 3
    labels = {e["label"] for e in examples}
    assert labels == {0, 1}
    scam = [e for e in examples if e["label"] == 1]
    benign = [e for e in examples if e["label"] == 0]
    assert all(e["slice"] == "scam" for e in scam)
    assert all(e["slice"] == "scam_boundary" for e in benign)
    assert all(e["source"] == "synthetic_scam_v1" for e in examples)


def test_parse_examples_rejects_malformed():
    assert parse_examples("not json") == []
    assert parse_examples('{"text": "x"}') == []  # 배열 아님
    assert parse_examples('[{"label": 1}]') == []  # text 없음
    assert parse_examples('[{"text": " ", "label": 1}]') == []  # 빈 텍스트
    assert parse_examples('[{"text": "ok", "label": 5}]') == []  # 라벨 범위 밖
    good = parse_examples('[{"text": "계좌로 송금해", "label": 1, "subtype": "y"}]')
    assert len(good) == 1 and good[0]["label"] == 1


class FakeVerifyLLM:
    """검수 프롬프트의 입력 배열을 파싱해, '송금'/'사기' 포함 시 1 아니면 0을 준다."""

    def __call__(self, prompt: str) -> str:
        payload = prompt.split("# 입력(JSON 배열, 각 메시지)\n", 1)[1].split("\n\n# 출력", 1)[0]
        texts = json.loads(payload)
        verdicts = [{"label": 1 if ("송금" in t or "사기" in t) else 0} for t in texts]
        return json.dumps(verdicts, ensure_ascii=False)


def test_verify_keeps_agreements_drops_mismatches():
    examples = [
        {"text": "계좌로 송금해 급해", "label": 1, "slice": "scam", "subtype": "a", "source": "s"},
        {
            "text": "오늘 날씨 좋다",
            "label": 1,
            "slice": "scam",
            "subtype": "a",
            "source": "s",
        },  # 오라벨
        {
            "text": "밥값 정산하자",
            "label": 0,
            "slice": "scam_boundary",
            "subtype": "b",
            "source": "s",
        },
    ]
    kept, stats = verify_labels(FakeVerifyLLM(), examples, batch_size=10)
    kept_texts = {e["text"] for e in kept}
    # 라벨1인데 사기신호 있는 것, 라벨0인데 정상인 것만 통과. 오라벨(날씨=1)은 탈락.
    assert "계좌로 송금해 급해" in kept_texts
    assert "밥값 정산하자" in kept_texts
    assert "오늘 날씨 좋다" not in kept_texts
    assert stats["kept"] == 2 and stats["dropped"] == 1


def test_verify_prompt_roundtrips_texts():
    prompt = build_verify_prompt(["a", "b"])
    payload = prompt.split("# 입력(JSON 배열, 각 메시지)\n", 1)[1].split("\n\n# 출력", 1)[0]
    assert json.loads(payload) == ["a", "b"]


def test_filter_forbidden_removes_blind_overlaps():
    examples = [
        {"text": "인증번호 여섯 자리 불러주세요 안 그러면 계정 정지됩니다", "label": 1},
        {"text": "완전히 다른 정상 문장 산책 갈까요 날씨 좋네요", "label": 0},
    ]
    forbidden = ["인증번호 여섯 자리 불러주세요 안 그러면 계정이 정지됩니다"]  # blind와 근사 중복
    kept, removed = filter_forbidden(examples, forbidden, threshold=0.6)
    kept_texts = {e["text"] for e in kept}
    assert removed == 1
    assert "완전히 다른 정상 문장 산책 갈까요 날씨 좋네요" in kept_texts
    assert all("인증번호 여섯" not in t for t in kept_texts)


def test_filter_forbidden_noop_without_forbidden():
    examples = [{"text": "x", "label": 1}]
    kept, removed = filter_forbidden(examples, [])
    assert removed == 0 and len(kept) == 1
