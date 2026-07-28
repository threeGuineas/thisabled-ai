"""SAFE 이진 v7 — 금전 사기/사회공학 학습 데이터 합성 + LLM 라벨 검수 (A3 + B2).

사기는 SAFE 학습 데이터의 최대 공백(실데이터 없음). Gemini로 한국어 메신저체 대화형
사기 예시를 합성(주의=1)하고, 경계 반례(정상 금전 대화=0)도 함께 만든 뒤, 별도 LLM
검수 패스로 각 라벨을 재확인해 오라벨·애매 예시를 걸러 학습 품질을 확보한다.

LLM 호출은 TextGenerator로 주입한다(테스트는 fake 주입 → 네트워크·키 불필요).
후보 출력 스키마는 {text, label(0|1), slice, subtype, source}이며, 품질 판정 reason은
별도 검수 보고서에만 기록한다.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

from src.data.llm_client import GeminiClientError, TextGenerator


def _safe_generate(generate: TextGenerator, prompt: str, tag: str) -> str | None:
    """generate 오류를 stderr에 남긴다. API 오류는 부분 저장 방지를 위해 전파한다."""

    try:
        return generate(prompt)
    except GeminiClientError as exc:
        print(f"[scam] {tag} LLM 호출 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    except Exception as exc:  # noqa: BLE001 — API 실패 유형을 가리지 않고 로깅 후 계속
        print(f"[scam] {tag} LLM 호출 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


SOURCE = "synthetic_scam_v3"
LABEL_REVIEW_REVISION = "scam-label-v2"
QUALITY_REVIEW_REVISION = "scam-quality-v2"
_MAX_QUALITY_REASON_LENGTH = 500
_SENSITIVE_IDENTIFIER = re.compile(
    r"(?<!\d)(?:(?:\+?82[\s./-]?)?0?1[016789][\s./-]?\d{3,4}[\s./-]?\d{4}"
    r"|\d{6}[-\s]?[1-4]\d{6}"
    r"|\d{2,6}[\s./-]\d{2,6}[\s./-]\d{2,8}|\d{8,})(?!\d)"
    r"|(?:인증번호|인증코드|OTP|일회용\s*비밀번호)[^\d\n]{0,12}\d{4,8}(?!\d)"
    r"|(?<!\d)\d{4,8}[^\d\n]{0,12}(?:인증번호|인증코드|OTP|일회용\s*비밀번호)"
    r"|(?:계좌(?:번호)?|입금|송금)[^\d\n]{0,20}\d{2,6}[\s./-]\d{2,8}(?!\d)"
    r"|(?:(?:상품권|기프트\s*카드)\s*)?(?:핀(?:번호)?|PIN)"
    r"\s*[:：은는]?\s*(?=[A-Z0-9-]{4,}(?![A-Z0-9-]))(?=[A-Z0-9-]*\d)[A-Z0-9-]{4,}"
    r"|(?:상품권|기프트\s*카드)\s*코드\s*[:：은는]?\s*"
    r"(?=[A-Z0-9-]{4,}(?![A-Z0-9-]))(?=[A-Z0-9-]*\d)[A-Z0-9-]{4,}"
    r"|(?:https?|hxxps?)\s*:\s*//[^\s]+|www\.[^\s]+"
    r"|\b(?:\d{1,3}\.){3}\d{1,3}(?:/[^\s]*)?"
    r"|\b(?:[a-z0-9-]+(?:\.|\[\.\]))+[a-z]{2,24}(?:/[^\s]*)?"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    re.IGNORECASE,
)
_BACKTICKED_PLACEHOLDER = re.compile(r"`(<가상(?:계좌|전화|인증번호|상품권핀)>)`")
_DUPLICATE_KEY_NOISE = re.compile(r"[^0-9a-z가-힣<>]+", re.IGNORECASE)
_IDENTIFIER_TRANSLATION = str.maketrans(
    {
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "－": "-",
        "／": "/",
        "．": ".",
        "_": "-",
        "＿": "-",
        "·": ".",
        "‧": ".",
        "∙": ".",
        "•": ".",
        "ㆍ": ".",
        "・": ".",
        "(": "",
        ")": "",
        "（": "",
        "）": "",
    }
)


class LLMOutputError(RuntimeError):
    """LLM 텍스트가 요청한 JSON 계약을 충족하지 못한 경우."""


def contains_sensitive_identifier(text: str) -> bool:
    """전화번호·계좌번호처럼 보이는 숫자열 포함 여부."""

    normalized = text.translate(_IDENTIFIER_TRANSLATION)
    normalized = re.sub(r"\s*([./-])\s*", r"\1", normalized)
    return bool(_SENSITIVE_IDENTIFIER.search(normalized))


def normalize_generated_text(text: str) -> str:
    """모델이 placeholder에 덧붙인 Markdown 백틱을 제거한다."""

    return _BACKTICKED_PLACEHOLDER.sub(r"\1", text).strip()


def duplicate_text_key(text: str) -> str:
    """공백·구두점·대소문자 차이를 무시하는 내부 중복 판정 키."""

    return _DUPLICATE_KEY_NOISE.sub("", normalize_generated_text(text)).lower()


# 사기 하위 유형(주의=1). 대화형 사회공학 화법 중심.
SCAM_SUBTYPES: dict[str, str] = {
    "impersonation_authority": "검찰·경찰·금감원·은행·택배 등 기관 사칭으로 계좌 이체·개인정보 요구",
    "impersonation_acquaintance": "가족·지인 사칭(폰 고장·사고·긴급)으로 송금·상품권 요구",
    "loan_investment": "저금리 대출·리딩방·코인 고수익 보장 미끼로 선입금·수수료 요구",
    "giftcard_voucher": "문화상품권·기프트카드 핀번호 요구",
    "credential_theft": "인증번호·OTP·비밀번호 탈취 유도",
    "task_scam": "부업·알바·꿀팁으로 수익 미끼 후 입금·수수료 요구",
}
# 경계 반례(정상=0): 금전 언급이 있으나 사기가 아닌 일상 대화.
BENIGN_SUBTYPES: dict[str, str] = {
    "family_allowance": "가족 사이 용돈·생활비 이체 등 정상 금전 대화",
    "friend_settlement": "친구끼리 밥값·회비 정산 등 정상 송금 대화",
    "normal_shopping": "정상 쇼핑·중고거래 결제 대화(사기 신호 없음)",
}

_SYNTH_SYSTEM = """당신은 한국어 안전 분류기 학습 데이터를 만드는 보조자입니다.
목적: 메신저에서 오가는 '금전 사기/사회공학' 메시지를 탐지하는 모델 학습.
유해 신호는 완곡화하지 말고 실제 사기 화법 그대로 살리되, 실제 개인정보(실명·실제
계좌번호·전화번호)는 쓰지 마세요. 구체적인 사람 이름도 쓰지 말고 김OO·엄마·담당자처럼
익명화하세요. 계좌·전화·인증번호·상품권 PIN이 필요하면 숫자 대신 <가상계좌>,
<가상전화>, <가상인증번호>, <가상상품권핀> 표기만 사용하세요. 백틱과 임의로 만든
조어는 쓰지 마세요. 돈·정보를 누가 누구에게 보내는지, 행동의 원인과 결과가 모순되지
않는지 확인하세요. 반드시 한국 메신저 대화체(반말/축약/자연스러운 구어)로 씁니다."""

_VERIFY_SYSTEM = """당신은 한국어 메시지가 '금전 사기/편취 위험(주의)'인지 '정상'인지
판정하는 검수자입니다. 사기 신호(기관·지인 사칭, 긴급성 압박, 계좌·상품권·인증번호 요구,
고수익 보장, 선입금·보증금·상품 구매 요구 등)가 메시지 자체에 있으면 주의(1)입니다.
단순 부업·리뷰 모집이나 링크 안내만 있고 금전·인증정보 요구가 아직 없으면 정상(0)으로
판정합니다. 가족 용돈·친구 정산·정상 거래처럼 관계와 목적이 자연스럽고 사칭·압박 신호가
없으면 계좌·송금 언급이 있어도 정상(0)입니다. 총무·계좌가 갑자기 바뀌었거나 새 계좌로
급히 보내라는 설정은 정상 정산으로 보지 마세요. 문장에 화자·돈의 방향·행동 순서 모순,
깨진 조어, 의미 불명확성이 있으면 라벨을 억지로 고르지 말고 보류(-1)하세요. 입력 문장
밖의 후속 상황은 추정하지 마세요."""

_QUALITY_SYSTEM = """당신은 한국어 안전 분류기 학습 예시의 품질 검수자입니다.
각 예시가 지정된 subtype에 정확히 맞고, 독립된 한 메시지로서 자연스럽고 논리적인지
검토하세요. 다음 중 하나라도 있으면 reject:
- 화자·수신자·돈/정보의 방향 또는 원인·결과가 모순됨
- 깨진 조어, 붙어 버린 단어, 명백한 문법 오류, 역할이 뒤바뀐 표현
- 서로 다른 사기 수법을 불필요하게 섞거나 지정 subtype을 이탈함
- 정상 예시에 계좌 변경·사칭·긴급 압박 등 사기 신호가 섞임
- 주의 예시에 편취·자격증명 탈취 요구가 명확하지 않음
특히 돈이 필요한 목적과 실제 수취인이 다르거나, 대출 상담자와 피해자의 호칭이 뒤바뀌거나,
주식·코인처럼 서로 다른 투자 대상을 한 상품처럼 섞으면 reject하세요. 단, 가족끼리
용돈·생활비·선물비를 요청하거나 나누는 것은 family_allowance 범위로 허용하고, 정상
하위유형끼리의 약한 경계 중첩만으로 reject하지 마세요.
문장을 고치거나 숨은 맥락을 추정하지 말고, 현재 문장 그대로 학습 가능한지만 판정하세요."""


def build_synthesis_prompt(
    n: int,
    subtype: str,
    description: str,
    *,
    label: int,
    exclude_texts: Sequence[str] = (),
) -> str:
    kind = "사기(주의)" if label == 1 else "정상 금전 대화(경계 반례)"
    rules: list[str] = []
    if label == 1:
        rules.extend(
            [
                "각 메시지 자체에 편취 위험 신호(송금·선입금·상품권·인증정보 요구, "
                "고수익 보장 등)를 최소 1개 명확히 포함.",
                "단순 홍보·모집·링크 안내만으로 끝내지 말 것.",
            ]
        )
    else:
        rules.extend(
            [
                "사칭·긴급 압박·고수익 보장·인증번호 요구를 넣지 말 것.",
                "관계와 정산·구매 목적이 문장 안에서 자연스럽게 드러나게 할 것.",
            ]
        )

    subtype_rules: dict[str, list[str]] = {
        "impersonation_authority": [
            "현금을 계좌로 이체한다는 식의 모순을 피하고 계좌 잔액·보유 자금처럼 정확히 표현.",
            "기관 사칭 시나리오를 이체·원격 앱·인증정보·미납금 등으로 다양화.",
        ],
        "impersonation_acquaintance": [
            "지인 관계 사칭과 연락수단 문제를 밝히고, 송금·상품권·인증정보 요구를 같은 "
            "메시지에서 논리적으로 연결.",
            "다른 subtype의 대출 상담·투자 권유를 가족이 대신 전달하는 구조는 금지.",
        ],
        "loan_investment": [
            "대출 상담자·투자 권유자가 피해자에게 직접 비용이나 입금을 요구하게 작성.",
            "가족·지인이 대신 돈을 빌려 달라는 구조와 섞지 말 것.",
        ],
        "giftcard_voucher": [
            "상품권 구매 명분과 핀 전달 요구의 인과를 자연스럽게 연결.",
            "상품권 코드는 <가상상품권핀>으로만 표기하고 <가상인증번호>와 혼동하지 말 것.",
        ],
        "credential_theft": [
            "인증번호·비밀번호를 발신자에게 말하거나 채팅으로 보내거나 가짜 링크에 "
            "입력하도록 요구.",
            "사용자가 자기 기기에서 정상적으로 입력·버튼 조작만 하는 안내는 금지.",
        ],
        "task_scam": [
            "선입금·보증금·충전·상품 선구매·수수료 중 하나를 같은 메시지 안에서 반드시 요구.",
            "초기비용만 반복하지 말고 구매 미션, 충전, 출금 잠금 해제비, 교육비 등 "
            "편취 단계를 서로 다르게 구성.",
        ],
        "family_allowance": [
            "송금 완료 통지만 반복하지 말고 용돈 요청·생활비 협의·상환을 고르게 포함.",
            "누가 누구에게 돈을 보내는지 문법적으로 명확히 표현.",
        ],
        "friend_settlement": [
            "누가 먼저 결제했고 누가 얼마를 갚는지, 송금 요청인지 완료 통지인지 명확히 표현.",
            "총무·계좌가 갑자기 바뀌거나 새 계좌로 급히 보내라는 설정은 금지.",
        ],
        "normal_shopping": [
            "정상 결제 요청·결제 완료·배송·교환 상황을 고르게 포함.",
            "구매자·판매자와 결제·배송의 주체 및 방향을 명확히 표현.",
        ],
    }
    rules.extend(subtype_rules.get(subtype, []))
    quality_rules = "\n# 필수 품질 조건\n" + "".join(f"- {rule}\n" for rule in rules)
    exclusion = ""
    if exclude_texts:
        exclusion = (
            "\n# 이미 생성된 문장\n"
            f"{json.dumps(list(exclude_texts), ensure_ascii=False)}\n"
            "위 문장과 같은 상황·표현은 반복하지 마세요.\n"
        )
    return (
        f"{_SYNTH_SYSTEM}\n\n"
        f"# 과제\n'{subtype}' 유형의 {kind} 메시지 {n}개를 서로 다른 상황·화자·말투로 생성.\n"
        f"유형 설명: {description}\n\n"
        f"{quality_rules}\n"
        f"{exclusion}\n"
        f"# 출력(JSON 배열만)\n"
        f'[{{"text": "메시지", "label": {label}, "subtype": "{subtype}"}}]'
    )


def parse_examples(
    raw: str,
    *,
    expected_label: int | None = None,
    expected_subtype: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """LLM JSON 응답을 파싱해 유효한 예시만 반환한다."""

    if expected_label is not None and expected_label not in (0, 1):
        raise ValueError("expected_label은 0 또는 1이어야 함")
    if limit is not None and limit <= 0:
        raise ValueError("limit은 양수여야 함")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in data:
        if not isinstance(item, dict) or "text" not in item:
            continue
        if not isinstance(item["text"], str):
            continue
        text = normalize_generated_text(item["text"])
        if not text:
            continue
        if contains_sensitive_identifier(text):
            continue
        duplicate_key = duplicate_text_key(text)
        if not duplicate_key or duplicate_key in seen_keys:
            continue
        label = item.get("label")
        if isinstance(label, bool) or not isinstance(label, int):
            continue
        if label not in (0, 1):
            continue
        if expected_label is not None and label != expected_label:
            continue
        subtype = item.get("subtype")
        if not isinstance(subtype, str) or not subtype:
            continue
        if expected_subtype is not None and subtype != expected_subtype:
            continue
        canonical_label = label
        out.append(
            {
                "text": text,
                "label": canonical_label,
                "slice": "scam" if canonical_label == 1 else "scam_boundary",
                "subtype": subtype,
                "source": SOURCE,
            }
        )
        seen_keys.add(duplicate_key)
        if limit is not None and len(out) >= limit:
            break
    return out


def synthesize_scam(
    generate: TextGenerator,
    *,
    per_subtype: int = 20,
    include_benign: bool = True,
    parse_attempts: int = 2,
    initial_examples: Sequence[dict[str, Any]] = (),
    on_subtype_complete: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """유형별 목표 개수까지 합성하고, 완료된 유형마다 선택적으로 체크포인트한다."""

    if per_subtype <= 0:
        raise ValueError("per_subtype은 양수여야 함")
    if parse_attempts <= 0:
        raise ValueError("parse_attempts는 양수여야 함")
    examples = [dict(example) for example in initial_examples]
    specs = list(SCAM_SUBTYPES.items())
    if include_benign:
        specs += list(BENIGN_SUBTYPES.items())
    expected_subtypes = {subtype for subtype, _ in specs}
    if any(example.get("subtype") not in expected_subtypes for example in examples):
        raise ValueError("initial_examples에 예상하지 않은 subtype이 있음")
    seen_text_keys = {
        duplicate_text_key(example["text"])
        for example in examples
        if isinstance(example.get("text"), str)
    }
    for subtype, description in specs:
        label = 0 if subtype in BENIGN_SUBTYPES else 1
        subtype_examples = [example for example in examples if example.get("subtype") == subtype]
        if len(subtype_examples) > per_subtype:
            raise ValueError(f"{subtype} initial_examples가 목표 개수를 초과함")
        if len(subtype_examples) == per_subtype:
            continue
        for attempt in range(1, parse_attempts + 1):
            remaining = per_subtype - len(subtype_examples)
            prompt = build_synthesis_prompt(
                remaining,
                subtype,
                description,
                label=label,
                exclude_texts=[example["text"] for example in subtype_examples],
            )
            raw = _safe_generate(generate, prompt, f"synth:{subtype}")
            if raw is None:
                break
            parsed = parse_examples(
                raw,
                expected_label=label,
                expected_subtype=subtype,
            )
            novel: list[dict[str, Any]] = []
            for example in parsed:
                key = duplicate_text_key(example["text"])
                if key in seen_text_keys:
                    continue
                seen_text_keys.add(key)
                novel.append(example)
                if len(novel) >= remaining:
                    break
            if novel:
                examples.extend(novel)
                subtype_examples.extend(novel)
            if len(subtype_examples) == per_subtype:
                if on_subtype_complete is not None:
                    on_subtype_complete(list(examples))
                break
            print(
                f"[scam] synth:{subtype} 유효 누적 {len(subtype_examples)}/{per_subtype}건 "
                f"({attempt}/{parse_attempts})",
                file=sys.stderr,
            )
    return examples


def filter_forbidden(
    examples: Sequence[dict[str, Any]],
    forbidden_texts: Sequence[str],
    *,
    threshold: float = 0.8,
) -> tuple[list[dict[str, Any]], int]:
    """holdout/blind과 근사 중복인 합성 예시를 제거한다(train 누수 방지). (통과, 제거수)."""

    if not forbidden_texts:
        return list(examples), 0
    from src.data.dedup import find_duplicate_indices

    forbidden_keys = {duplicate_text_key(text) for text in forbidden_texts}
    exact_duplicate_indices = {
        index
        for index, example in enumerate(examples)
        if duplicate_text_key(example["text"]) in forbidden_keys
    }
    remaining_indices = [
        index for index in range(len(examples)) if index not in exact_duplicate_indices
    ]
    dup = find_duplicate_indices(
        forbidden_texts,
        [examples[index]["text"] for index in remaining_indices],
        threshold=threshold,
    )
    near_duplicate_indices = {remaining_indices[index] for index in dup}
    removed_indices = exact_duplicate_indices | near_duplicate_indices
    kept = [example for index, example in enumerate(examples) if index not in removed_indices]
    return kept, len(removed_indices)


def build_verify_prompt(texts: Sequence[str]) -> str:
    payload = json.dumps(list(texts), ensure_ascii=False)
    return (
        f"{_VERIFY_SYSTEM}\n\n"
        f"# 입력(JSON 배열, 각 메시지)\n{payload}\n\n"
        f"# 출력\n입력 순서·개수 그대로 각 메시지의 판정을 JSON 배열로만 응답:\n"
        f'[{{"label": -1|0|1}}]  (1=주의/사기, 0=정상, -1=모순·비문·애매함으로 보류)'
    )


def build_quality_prompt(examples: Sequence[dict[str, Any]]) -> str:
    payload = [
        {
            "id": index,
            "text": example["text"],
            "subtype": example.get("subtype", ""),
        }
        for index, example in enumerate(examples)
    ]
    subtype_rubric = {**SCAM_SUBTYPES, **BENIGN_SUBTYPES}
    return (
        f"{_QUALITY_SYSTEM}\n\n"
        f"# subtype 기준\n{json.dumps(subtype_rubric, ensure_ascii=False)}\n\n"
        f"# 입력(JSON 배열)\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        f"# 출력\n입력 순서·개수·id 그대로 JSON 배열만 응답:\n"
        f'[{{"id": 0, "accept": true|false, "reason": "한 문장 사유"}}]'
    )


def _parse_verdicts(raw: str, n: int) -> list[int] | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) != n:
        return None
    verdicts: list[int] = []
    for item in data:
        value = item.get("label") if isinstance(item, dict) else item
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if value not in (-1, 0, 1):
            return None
        verdicts.append(value)
    return verdicts


def verify_labels(
    generate: TextGenerator,
    examples: Sequence[dict[str, Any]],
    *,
    batch_size: int = 20,
    initial_verdicts: Sequence[int] = (),
    on_batch_complete: Callable[[list[int]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """LLM 검수로 라벨이 일치하는 예시만 남긴다(B2). (통과 목록, 통계) 반환."""

    if batch_size <= 0:
        raise ValueError("batch_size는 양수여야 함")
    verdicts_all = list(initial_verdicts)
    if len(verdicts_all) > len(examples):
        raise ValueError("initial_verdicts가 examples보다 김")
    if len(verdicts_all) < len(examples) and len(verdicts_all) % batch_size:
        raise ValueError("initial_verdicts는 완료된 batch 경계여야 함")
    if any(
        isinstance(verdict, bool) or not isinstance(verdict, int) or verdict not in (-1, 0, 1)
        for verdict in verdicts_all
    ):
        raise ValueError("initial_verdicts 값 오류")
    for start in range(len(verdicts_all), len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        raw = _safe_generate(generate, build_verify_prompt([e["text"] for e in batch]), "verify")
        verdicts = _parse_verdicts(raw, len(batch)) if raw is not None else None
        if verdicts is None:
            raise LLMOutputError(f"label 검수 JSON 계약 불일치: batch_start={start}")
        verdicts_all.extend(verdicts)
        if on_batch_complete is not None:
            on_batch_complete(list(verdicts_all))

    kept: list[dict[str, Any]] = []
    stats = {"total": len(examples), "kept": 0, "dropped": 0, "unparsed": 0}
    for example, verdict in zip(examples, verdicts_all, strict=True):
        if verdict == -1:
            stats["unparsed"] += 1
        elif verdict == example["label"]:
            kept.append(example)
        else:
            stats["dropped"] += 1
    stats["kept"] = len(kept)
    return kept, stats


def _parse_quality_reviews(raw: str, n: int) -> list[tuple[bool, str]] | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) != n:
        return None

    reviews: list[tuple[bool, str]] = []
    for expected_id, item in enumerate(data):
        if not isinstance(item, dict) or item.get("id") != expected_id:
            return None
        accepted = item.get("accept")
        reason = item.get("reason")
        if (
            not isinstance(accepted, bool)
            or not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _MAX_QUALITY_REASON_LENGTH
            or contains_sensitive_identifier(reason)
        ):
            return None
        reviews.append((accepted, reason.strip()))
    return reviews


def review_quality(
    generate: TextGenerator,
    examples: Sequence[dict[str, Any]],
    *,
    batch_size: int = 20,
    initial_reviews: Sequence[tuple[bool, str]] = (),
    on_batch_complete: Callable[[list[tuple[bool, str]]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    """subtype 적합성·문장 논리를 별도 검수해 통과 예시와 감사 기록을 반환한다."""

    if batch_size <= 0:
        raise ValueError("batch_size는 양수여야 함")
    reviews_all = list(initial_reviews)
    if len(reviews_all) > len(examples):
        raise ValueError("initial_reviews가 examples보다 김")
    if len(reviews_all) < len(examples) and len(reviews_all) % batch_size:
        raise ValueError("initial_reviews는 완료된 batch 경계여야 함")
    for accepted, reason in reviews_all:
        if (
            not isinstance(accepted, bool)
            or not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _MAX_QUALITY_REASON_LENGTH
            or contains_sensitive_identifier(reason)
        ):
            raise ValueError("initial_reviews 값 오류")
    for start in range(len(reviews_all), len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        raw = _safe_generate(generate, build_quality_prompt(batch), "quality")
        reviews = _parse_quality_reviews(raw, len(batch)) if raw is not None else None
        if reviews is None:
            raise LLMOutputError(f"품질 검수 JSON 계약 불일치: batch_start={start}")
        reviews_all.extend(reviews)
        if on_batch_complete is not None:
            on_batch_complete(list(reviews_all))

    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    stats = {"total": len(examples), "kept": 0, "rejected": 0, "unparsed": 0}
    for index, (example, review) in enumerate(zip(examples, reviews_all, strict=True)):
        accepted, reason = review
        audit.append(
            {
                "index": index,
                "text": example["text"],
                "subtype": example.get("subtype", ""),
                "accepted": accepted,
                "reason": reason,
            }
        )
        if accepted:
            kept.append(example)
        else:
            stats["rejected"] += 1
    stats["kept"] = len(kept)
    return kept, stats, audit
