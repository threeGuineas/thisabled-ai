"""사기 합성 + LLM 검수 파이프라인 오프라인 검증 (fake LLM 주입 — 네트워크·키 불필요)."""

from __future__ import annotations

import json

import pytest

from src.data.llm_client import GeminiAPIError, GeminiResponseError
from src.data.scam_synthesis import (
    BENIGN_SUBTYPES,
    SCAM_SUBTYPES,
    build_quality_prompt,
    build_synthesis_prompt,
    build_verify_prompt,
    filter_forbidden,
    parse_examples,
    review_quality,
    synthesize_scam,
    verify_labels,
)


class FakeSynthLLM:
    """합성 프롬프트를 받으면 요청 라벨대로 예시 JSON을 반환한다."""

    def __call__(self, prompt: str) -> str:
        label = 0 if '"label": 0' in prompt else 1
        subtype = prompt.split("# 과제\n'", 1)[1].split("'", 1)[0]
        rows = [
            {
                "text": f"{subtype} 예시 메시지 {i} (label {label})",
                "label": label,
                "subtype": subtype,
            }
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
    assert all(e["source"] == "synthetic_scam_v3" for e in examples)


def test_parse_examples_rejects_malformed():
    assert parse_examples("not json") == []
    assert parse_examples('{"text": "x"}') == []  # 배열 아님
    assert parse_examples('[{"label": 1}]') == []  # text 없음
    assert parse_examples('[{"text": " ", "label": 1}]') == []  # 빈 텍스트
    assert parse_examples('[{"text": "ok", "label": 5}]') == []  # 라벨 범위 밖
    good = parse_examples('[{"text": "계좌로 송금해", "label": 1, "subtype": "y"}]')
    assert len(good) == 1 and good[0]["label"] == 1


def test_parse_examples_treats_llm_fields_as_untrusted():
    raw = json.dumps(
        [
            {"text": None, "label": 1, "subtype": "wrong"},
            {"text": "bool 라벨", "label": True, "subtype": "wrong"},
            {"text": "요청과 다른 라벨", "label": 0, "subtype": "wrong"},
            {"text": "하위유형 불일치", "label": 1, "subtype": "wrong"},
            {"text": "라벨 누락", "subtype": "credential_theft"},
            {"text": "하위유형 누락", "label": 1},
            {"text": "정상 파싱", "label": 1, "subtype": "credential_theft"},
            {"text": "개수 초과", "label": 1, "subtype": "credential_theft"},
        ],
        ensure_ascii=False,
    )
    parsed = parse_examples(
        raw,
        expected_label=1,
        expected_subtype="credential_theft",
        limit=1,
    )

    assert parsed == [
        {
            "text": "정상 파싱",
            "label": 1,
            "slice": "scam",
            "subtype": "credential_theft",
            "source": "synthetic_scam_v3",
        }
    ]


def test_parse_examples_rejects_phone_and_account_like_identifiers():
    rows = [
        {"text": "연락처는 010-1234-5678이야", "label": 1},
        {"text": "연락처는 010.1234.5678이야", "label": 1},
        {"text": "연락처는 010/1234/5678이야", "label": 1},
        {"text": "연락처는 010\u200b-1234‑5678이야", "label": 1},
        {"text": "연락처는 010 . 1234 . 5678이야", "label": 1},
        {"text": "연락처는 010 / 1234 / 5678이야", "label": 1},
        {"text": "연락처는 010－1234－5678이야", "label": 1},
        {"text": "연락처는 (010) 1234-5678이야", "label": 1},
        {"text": "연락처는 010(1234)5678이야", "label": 1},
        {"text": "연락처는 82 (10) 1234 5678이야", "label": 1},
        {"text": "연락처는 010_1234_5678이야", "label": 1},
        {"text": "연락처는 010·1234·5678이야", "label": 1},
        {"text": "연락처는 （010） 1234－5678이야", "label": 1},
        {"text": "신한 110-222-333333으로 보내", "label": 1},
        {"text": "123456789012로 보내", "label": 1},
        {"text": "인증번호 123456 알려줘", "label": 1},
        {"text": "인증번호 1234 알려줘", "label": 1},
        {"text": "900101-1234567 주민번호야", "label": 1},
        {"text": "계좌 123456-12로 입금해", "label": 1},
        {"text": "상품권 핀 123456 보내", "label": 1},
        {"text": "상품권 PIN: 1234 보내", "label": 1},
        {"text": "상품권 코드 ABCD1234 알려줘", "label": 1},
        {"text": "상품권 코드 ABCD-1234 알려줘", "label": 1},
        {"text": "https://malicious.example/path 열어", "label": 1},
        {"text": "bit.ly/abc123 열어", "label": 1},
        {"text": "example.com/login 열어", "label": 1},
        {"text": "192.0.2.10/login 열어", "label": 1},
        {"text": "hxxps://example[.]com/login 열어", "label": 1},
        {"text": "real.person@example.com으로 답장해", "label": 1},
        {"text": "`<가상계좌>`로 보내고 `<가상인증번호>` 알려줘", "label": 1},
    ]
    for row in rows:
        row["subtype"] = "credential_theft"
    raw = json.dumps(rows, ensure_ascii=False)

    parsed = parse_examples(raw, expected_label=1, expected_subtype="credential_theft")

    assert [row["text"] for row in parsed] == ["<가상계좌>로 보내고 <가상인증번호> 알려줘"]


def test_parse_examples_deduplicates_spacing_and_punctuation_variants():
    raw = json.dumps(
        [
            {"text": "지금 송금해 주세요", "label": 1, "subtype": "task_scam"},
            {"text": "지금, 송금해 주세요!", "label": 1, "subtype": "task_scam"},
        ],
        ensure_ascii=False,
    )

    parsed = parse_examples(raw, expected_label=1, expected_subtype="task_scam")

    assert [row["text"] for row in parsed] == ["지금 송금해 주세요"]


def test_task_scam_prompt_requires_explicit_payment_signal():
    prompt = build_synthesis_prompt(
        5,
        "task_scam",
        SCAM_SUBTYPES["task_scam"],
        label=1,
    )

    assert "단순 홍보·모집·링크 안내만으로 끝내지 말 것" in prompt
    assert "선입금·보증금·충전·상품 선구매·수수료" in prompt


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


def test_verify_treats_abstention_as_unparsed():
    class AbstainingLLM:
        def __call__(self, prompt: str) -> str:
            return json.dumps([{"label": -1}])

    examples = [
        {
            "text": "화자와 돈의 방향이 모순된 문장",
            "label": 0,
            "slice": "scam_boundary",
            "subtype": "friend_settlement",
            "source": "synthetic_scam_v3",
        }
    ]

    kept, stats = verify_labels(AbstainingLLM(), examples)

    assert kept == []
    assert stats == {"total": 1, "kept": 0, "dropped": 0, "unparsed": 1}


def test_verify_prompt_roundtrips_texts():
    prompt = build_verify_prompt(["a", "b"])
    payload = prompt.split("# 입력(JSON 배열, 각 메시지)\n", 1)[1].split("\n\n# 출력", 1)[0]
    assert json.loads(payload) == ["a", "b"]
    assert "-1=모순·비문·애매함으로 보류" in prompt


def test_quality_review_rejects_illogical_example_with_audit():
    class FakeQualityLLM:
        def __call__(self, prompt: str) -> str:
            payload = prompt.split("# 입력(JSON 배열)\n", 1)[1].split("\n\n# 출력", 1)[0]
            rows = json.loads(payload)
            return json.dumps(
                [
                    {
                        "id": row["id"],
                        "accept": "모순" not in row["text"],
                        "reason": "논리 정상" if "모순" not in row["text"] else "돈 방향 모순",
                    }
                    for row in rows
                ],
                ensure_ascii=False,
            )

    examples = [
        {
            "text": "친구가 먼저 결제한 밥값 내 몫을 보냈어",
            "label": 0,
            "subtype": "friend_settlement",
            "source": "synthetic_scam_v3",
        },
        {
            "text": "숙소비를 택시기사 계좌로 보내라는 모순 문장",
            "label": 1,
            "subtype": "impersonation_acquaintance",
            "source": "synthetic_scam_v3",
        },
    ]

    kept, stats, audit = review_quality(FakeQualityLLM(), examples)

    assert kept == [examples[0]]
    assert stats == {"total": 2, "kept": 1, "rejected": 1, "unparsed": 0}
    assert audit[1]["accepted"] is False
    assert audit[1]["reason"] == "돈 방향 모순"


def test_quality_prompt_roundtrips_subtype_and_text():
    examples = [{"text": "정상 정산", "subtype": "friend_settlement"}]
    prompt = build_quality_prompt(examples)
    payload = prompt.split("# 입력(JSON 배열)\n", 1)[1].split("\n\n# 출력", 1)[0]

    assert json.loads(payload) == [{"id": 0, "text": "정상 정산", "subtype": "friend_settlement"}]


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


def test_filter_forbidden_removes_punctuation_only_exact_variants():
    examples = [
        {"text": "지금, 송금해 주세요!", "label": 1},
        {"text": "완전히 다른 문장", "label": 0},
    ]

    kept, removed = filter_forbidden(examples, ["지금 송금해 주세요"], threshold=0.8)

    assert removed == 1
    assert kept == [{"text": "완전히 다른 문장", "label": 0}]


class _RaisingLLM:
    """한 유형에서만 예외를 던지고 나머지는 정상 응답."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("API 404 model not found")
        label = 0 if '"label": 0' in prompt else 1
        subtype = prompt.split("# 과제\n'", 1)[1].split("'", 1)[0]
        return json.dumps([{"text": f"{subtype} t{label}", "label": label, "subtype": subtype}])


def test_synthesize_survives_llm_failure():
    # 첫 유형 호출이 실패해도 전체가 죽지 않고 나머지 유형은 생성된다.
    examples = synthesize_scam(_RaisingLLM(), per_subtype=1, include_benign=True)
    n_specs = len(SCAM_SUBTYPES) + len(BENIGN_SUBTYPES)
    assert len(examples) == n_specs - 1  # 실패한 1개 유형만 빠짐


def test_synthesize_retries_empty_parse_once():
    class RetryEmpty:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return "{}"
            return FakeSynthLLM()(prompt)

    llm = RetryEmpty()
    examples = synthesize_scam(llm, per_subtype=3, include_benign=True)

    assert len(examples) == (len(SCAM_SUBTYPES) + len(BENIGN_SUBTYPES)) * 3
    assert llm.calls == len(SCAM_SUBTYPES) + len(BENIGN_SUBTYPES) + 1


def test_synthesize_tops_up_partial_subtype_responses():
    class PartialThenRemaining:
        def __init__(self) -> None:
            self.calls_by_subtype: dict[str, int] = {}

        def __call__(self, prompt: str) -> str:
            subtype = prompt.split("# 과제\n'", 1)[1].split("'", 1)[0]
            requested = int(prompt.split(" 메시지 ", 1)[1].split("개", 1)[0])
            call = self.calls_by_subtype.get(subtype, 0)
            self.calls_by_subtype[subtype] = call + 1
            count = requested - 1 if call == 0 and requested > 1 else requested
            label = 0 if '"label": 0' in prompt else 1
            return json.dumps(
                [
                    {
                        "text": f"{subtype} 고유 예시 {call}-{index}",
                        "label": label,
                        "subtype": subtype,
                    }
                    for index in range(count)
                ],
                ensure_ascii=False,
            )

    examples = synthesize_scam(
        PartialThenRemaining(),
        per_subtype=3,
        include_benign=False,
    )

    counts = {subtype: 0 for subtype in SCAM_SUBTYPES}
    for example in examples:
        counts[example["subtype"]] += 1
    assert set(counts.values()) == {3}


def test_synthesize_checkpoints_each_completed_subtype_before_api_failure():
    snapshots: list[list[dict[str, object]]] = []

    class FailOnSecondSubtype:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 2:
                raise GeminiAPIError(429, "quota", status="RESOURCE_EXHAUSTED")
            subtype = prompt.split("# 과제\n'", 1)[1].split("'", 1)[0]
            return json.dumps(
                [{"text": f"{subtype} 고유 문장", "label": 1, "subtype": subtype}],
                ensure_ascii=False,
            )

    with pytest.raises(GeminiAPIError, match="quota"):
        synthesize_scam(
            FailOnSecondSubtype(),
            per_subtype=1,
            include_benign=False,
            on_subtype_complete=lambda rows: snapshots.append(rows),
        )

    assert len(snapshots) == 1
    assert len(snapshots[0]) == 1
    assert snapshots[0][0]["subtype"] == next(iter(SCAM_SUBTYPES))


def test_synthesize_fails_fast_on_permanent_gemini_error():
    def fail(_prompt: str) -> str:
        raise GeminiAPIError(403, "permission denied", status="PERMISSION_DENIED")

    with pytest.raises(GeminiAPIError, match="permission denied"):
        synthesize_scam(fail, per_subtype=1)


def test_synthesize_stops_after_exhausted_transient_gemini_error():
    def fail(_prompt: str) -> str:
        raise GeminiAPIError(503, "overloaded", status="UNAVAILABLE")

    with pytest.raises(GeminiAPIError, match="overloaded"):
        synthesize_scam(fail, per_subtype=1)


def test_synthesize_fails_fast_on_response_contract_error():
    def fail(_prompt: str) -> str:
        raise GeminiResponseError("candidate has no text")

    with pytest.raises(GeminiResponseError, match="candidate has no text"):
        synthesize_scam(fail, per_subtype=1)


def test_verify_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        verify_labels(FakeVerifyLLM(), [], batch_size=0)
