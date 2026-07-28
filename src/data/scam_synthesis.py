"""SAFE 이진 v7 — 금전 사기/사회공학 학습 데이터 합성 + LLM 라벨 검수 (A3 + B2).

사기는 SAFE 학습 데이터의 최대 공백(실데이터 없음). Gemini로 한국어 메신저체 대화형
사기 예시를 합성(주의=1)하고, 경계 반례(정상 금전 대화=0)도 함께 만든 뒤, 별도 LLM
검수 패스로 각 라벨을 재확인해 오라벨·애매 예시를 걸러 학습 품질을 확보한다.

LLM 호출은 TextGenerator로 주입한다(테스트는 fake 주입 → 네트워크·키 불필요).
출력 스키마는 기존 합성 데이터와 동일: {text, label(0|1), slice, subtype, source, reason}.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

from src.data.llm_client import TextGenerator


def _safe_generate(generate: TextGenerator, prompt: str, tag: str) -> str | None:
    """generate 호출을 감싸 한 번의 실패가 전체를 죽이지 않게 한다(원인은 stderr로)."""

    try:
        return generate(prompt)
    except Exception as exc:  # noqa: BLE001 — API 실패 유형을 가리지 않고 로깅 후 계속
        print(f"[scam] {tag} LLM 호출 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


SOURCE = "synthetic_scam_v1"

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
계좌번호·전화번호)는 쓰지 말고 가짜 예시로 만드세요. 반드시 한국 메신저 대화체(반말/축약/
자연스러운 구어)로 씁니다."""

_VERIFY_SYSTEM = """당신은 한국어 메시지가 '금전 사기/편취 위험(주의)'인지 '정상'인지
판정하는 검수자입니다. 사기 신호(기관·지인 사칭, 긴급성 압박, 계좌·상품권·인증번호 요구,
고수익 보장 등)가 있으면 주의(1), 정상적인 금전 대화면 정상(0)으로 판정합니다."""


def build_synthesis_prompt(n: int, subtype: str, description: str, *, label: int) -> str:
    kind = "사기(주의)" if label == 1 else "정상 금전 대화(경계 반례)"
    return (
        f"{_SYNTH_SYSTEM}\n\n"
        f"# 과제\n'{subtype}' 유형의 {kind} 메시지 {n}개를 서로 다른 상황·화자·말투로 생성.\n"
        f"유형 설명: {description}\n\n"
        f"# 출력(JSON 배열만)\n"
        f'[{{"text": "메시지", "label": {label}, "subtype": "{subtype}"}}]'
    )


def parse_examples(raw: str) -> list[dict[str, Any]]:
    """LLM JSON 응답을 파싱해 유효한 예시만 반환한다."""

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict) or "text" not in item:
            continue
        text = str(item["text"]).strip()
        if not text:
            continue
        try:
            label = int(item.get("label", 1))
        except (TypeError, ValueError):
            continue
        if label not in (0, 1):
            continue
        out.append(
            {
                "text": text,
                "label": label,
                "slice": "scam" if label == 1 else "scam_boundary",
                "subtype": str(item.get("subtype", "")),
                "source": SOURCE,
            }
        )
    return out


def synthesize_scam(
    generate: TextGenerator,
    *,
    per_subtype: int = 20,
    include_benign: bool = True,
) -> list[dict[str, Any]]:
    """유형별로 사기(및 경계 반례) 예시를 합성한다."""

    examples: list[dict[str, Any]] = []
    specs = list(SCAM_SUBTYPES.items())
    if include_benign:
        specs += list(BENIGN_SUBTYPES.items())
    for subtype, description in specs:
        label = 0 if subtype in BENIGN_SUBTYPES else 1
        prompt = build_synthesis_prompt(per_subtype, subtype, description, label=label)
        raw = _safe_generate(generate, prompt, f"synth:{subtype}")
        if raw is not None:
            examples.extend(parse_examples(raw))
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

    dup = find_duplicate_indices(
        forbidden_texts, [e["text"] for e in examples], threshold=threshold
    )
    kept = [example for index, example in enumerate(examples) if index not in dup]
    return kept, len(dup)


def build_verify_prompt(texts: Sequence[str]) -> str:
    payload = json.dumps(list(texts), ensure_ascii=False)
    return (
        f"{_VERIFY_SYSTEM}\n\n"
        f"# 입력(JSON 배열, 각 메시지)\n{payload}\n\n"
        f"# 출력\n입력 순서·개수 그대로 각 메시지의 판정을 JSON 배열로만 응답:\n"
        f'[{{"label": 0|1}}]  (1=주의/사기, 0=정상)'
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
        try:
            label = int(value)
        except (TypeError, ValueError):
            return None
        if label not in (0, 1):
            return None
        verdicts.append(label)
    return verdicts


def verify_labels(
    generate: TextGenerator,
    examples: Sequence[dict[str, Any]],
    *,
    batch_size: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """LLM 검수로 라벨이 일치하는 예시만 남긴다(B2). (통과 목록, 통계) 반환."""

    kept: list[dict[str, Any]] = []
    stats = {"total": len(examples), "kept": 0, "dropped": 0, "unparsed": 0}
    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        raw = _safe_generate(generate, build_verify_prompt([e["text"] for e in batch]), "verify")
        verdicts = _parse_verdicts(raw, len(batch)) if raw is not None else None
        if verdicts is None:
            stats["unparsed"] += len(batch)
            continue
        for example, verdict in zip(batch, verdicts, strict=True):
            if verdict == example["label"]:
                kept.append(example)
            else:
                stats["dropped"] += 1
    stats["kept"] = len(kept)
    return kept, stats
