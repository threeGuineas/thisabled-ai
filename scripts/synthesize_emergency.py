"""Gemini API로 긴급(3) 합성 데이터 생성.

각 카테고리(3a/3b/3c/3d) + boundary 반례를 emergency_scenarios.md §5 spec대로 생성.
환경변수: GEMINI_API_KEY 필수 (Google AI Studio에서 무료 발급).

무료 tier 제약:
- gemini-2.0-flash: 분당 15회, 일 1,500회 → 60/15 = 4초 간격 강제
- 우리 1,170건 / 20건당 호출 = 63 calls → 약 5분 소요

Usage:
    python scripts/synthesize_emergency.py                   # 전체 생성
    python scripts/synthesize_emergency.py --categories 3a   # 특정 카테고리만
    python scripts/synthesize_emergency.py --dry-run         # 호출 수만 출력
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# pyrefly: ignore [missing-import]
from src.data.synthesis_prompts import (  # noqa: E402
    CONTEXTS,
    LANGUAGE_STYLES,
    SYSTEM_PROMPT,
    TARGET_COUNTS,
    build_user_prompt,
    category_personas,
)

OUT_ROOT = ROOT / "data" / "synthetic" / "emergency"
MODEL = "gemini-2.5-flash"  # 무료 tier — 2.0-flash 쿼터 소진으로 전환
BATCH_SIZE = 20  # 한 LLM 호출당 생성 건수
RATE_LIMIT_SLEEP = 4.5  # 15 RPM 안전 마진 (60/15 = 4초)
MAX_OUTPUT_TOKENS = 8192  # flash 최대 — 20건 JSONL 잘림 방지
MAX_EMPTY_STREAK = 5  # 한 split에서 연속 빈/실패 배치 허용치 (무한 루프 가드)

# 안전 분류기 학습용 합성이므로 Gemini 기본 필터를 끈다.
# (그루밍/성적유인/협박/자해 신호 = 탐지 대상. 필터 켜면 3b/3c/3d가 차단됨)
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


def _parse_jsonl(text: str) -> list[dict]:
    """LLM 출력을 JSONL로 파싱, 실패 라인은 skip. 코드펜스도 제거."""
    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return items


def _extract_text(resp) -> str:
    """응답에서 텍스트를 안전하게 추출.

    후보가 차단(SAFETY)되거나 비면 resp.text 접근이 예외를 던지므로
    candidates.parts를 직접 순회한다. 추출 실패 시 빈 문자열 반환.
    """
    try:
        return resp.text or ""
    except Exception:
        pass
    text = ""
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            text += getattr(part, "text", None) or ""
    return text


def _call_llm(client, category: str, n: int, seed: int) -> tuple[list[dict], int, int]:
    """Gemini 호출 → (생성된 dict 리스트, prompt tokens, completion tokens)."""
    rng = random.Random(seed)
    personas = rng.sample(category_personas(category), min(4, len(category_personas(category))))
    contexts = rng.sample(CONTEXTS, min(3, len(CONTEXTS)))
    styles = rng.sample(LANGUAGE_STYLES, min(3, len(LANGUAGE_STYLES)))

    user_prompt = build_user_prompt(
        category=category,
        n_examples=n,
        persona_pool=personas,
        contexts=contexts,
        styles=styles,
    )

    resp = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.9,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "safety_settings": SAFETY_SETTINGS,
        },
    )

    text = _extract_text(resp)
    items = _parse_jsonl(text)
    in_tok = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0
    out_tok = resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0
    return items, in_tok, out_tok


def _enrich(items: list[dict], category: str, split: str) -> list[dict]:
    """라벨·source 등 메타데이터 보강."""
    enriched = []
    for item in items:
        if not isinstance(item, dict) or "text" not in item:
            continue
        text = str(item["text"]).strip()
        if not text or len(text) > 500:
            continue
        # boundary는 item['label'] 사용, 나머지는 모두 긴급(3)
        if category == "boundary":
            label = int(item.get("label", 0))
            if label not in (0, 1, 2):
                continue
        else:
            label = 3
        enriched.append(
            {
                "text": text,
                "label": label,
                "subcategory": category,
                "split": split,
                "source": "synthetic_emergency_v1",
                "persona": item.get("persona", ""),
                "context": item.get("context", ""),
                "style": item.get("style", ""),
                "disability": bool(item.get("disability", False)),
                "reason": item.get("reason", "") if category == "boundary" else "",
            }
        )
    return enriched


def synthesize(
    categories: list[str],
    dry_run: bool = False,
) -> dict:
    if not dry_run:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY 환경변수가 없습니다 (또는 GOOGLE_API_KEY).")
        from google import genai

        client = genai.Client(api_key=api_key)
    else:
        client = None

    total_calls = 0
    total_in_tok = 0
    total_out_tok = 0
    summary: dict = {}

    for cat in categories:
        if cat not in TARGET_COUNTS:
            print(f"⚠ unknown category {cat}, skip")
            continue
        cat_dir = OUT_ROOT / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        summary[cat] = {}

        for split, target in TARGET_COUNTS[cat].items():
            out_path = cat_dir / f"{split}.jsonl"
            existing: list[dict] = []
            if out_path.exists():
                with out_path.open(encoding="utf-8") as f:
                    existing = [json.loads(ln) for ln in f if ln.strip()]
                if len(existing) >= target:
                    print(f"[{cat}/{split}] {len(existing)}/{target} 이미 충족, skip")
                    summary[cat][split] = len(existing)
                    continue

            needed = target - len(existing)
            print(f"[{cat}/{split}] 추가 생성 필요: {needed}건")

            if dry_run:
                n_calls = (needed + BATCH_SIZE - 1) // BATCH_SIZE
                eta_sec = n_calls * RATE_LIMIT_SLEEP
                print(f"  → {n_calls} LLM 호출, 예상 시간 ~{eta_sec:.0f}초")
                total_calls += n_calls
                summary[cat][split] = len(existing)
                continue

            batch_seed = 42
            empty_streak = 0  # 연속 빈/실패 배치 — MAX_EMPTY_STREAK 도달 시 split 중단
            while needed > 0:
                n = min(BATCH_SIZE, needed)
                try:
                    items, in_tok, out_tok = _call_llm(client, cat, n, batch_seed)
                    enriched = _enrich(items, cat, split)
                    total_calls += 1
                    total_in_tok += in_tok
                    total_out_tok += out_tok
                    batch_seed += 1

                    if not enriched:
                        empty_streak += 1
                        print(
                            f"  call#{total_calls}: +0건 ⚠ 빈 배치 "
                            f"({empty_streak}/{MAX_EMPTY_STREAK}) — 차단/파싱 실패 가능"
                        )
                        if empty_streak >= MAX_EMPTY_STREAK:
                            print(
                                f"  ✗ [{cat}/{split}] 연속 {MAX_EMPTY_STREAK}회 빈 결과 — "
                                f"중단 ({len(existing)}/{target})"
                            )
                            break
                        time.sleep(RATE_LIMIT_SLEEP)
                        continue

                    empty_streak = 0
                    existing.extend(enriched)
                    print(
                        f"  call#{total_calls}: +{len(enriched)}건 "
                        f"(in {in_tok}/out {out_tok} tok)"
                    )
                    # 누적 저장 (중간 실패 대비)
                    with out_path.open("w", encoding="utf-8") as f:
                        for it in existing:
                            f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    needed = target - len(existing)
                    time.sleep(RATE_LIMIT_SLEEP)  # 15 RPM 준수
                except Exception as e:
                    msg = str(e)
                    empty_streak += 1
                    if empty_streak >= MAX_EMPTY_STREAK:
                        print(
                            f"  ✗ [{cat}/{split}] 연속 {MAX_EMPTY_STREAK}회 실패 — 중단: {msg[:120]}"
                        )
                        break
                    # 429 (rate limit) 시 더 길게 대기
                    wait = 30 if "429" in msg or "RESOURCE_EXHAUSTED" in msg else 5
                    print(
                        f"  ⚠ 호출 실패 ({empty_streak}/{MAX_EMPTY_STREAK}): "
                        f"{msg[:120]}, {wait}초 후 재시도"
                    )
                    time.sleep(wait)
                    batch_seed += 1

            summary[cat][split] = len(existing)
            print(f"[{cat}/{split}] 최종 {len(existing)}/{target}건 → {out_path.relative_to(ROOT)}")

    summary["total_calls"] = total_calls
    summary["total_prompt_tokens"] = total_in_tok
    summary["total_completion_tokens"] = total_out_tok
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(TARGET_COUNTS.keys()),
        choices=list(TARGET_COUNTS.keys()),
    )
    parser.add_argument("--dry-run", action="store_true", help="호출 수·예상 시간만 계산")
    args = parser.parse_args()

    print(f"카테고리: {args.categories}")
    print(f"모델: {MODEL} (무료 tier, 15 RPM)")
    print(f"batch_size: {BATCH_SIZE}, 호출 간격: {RATE_LIMIT_SLEEP}초")
    print(f"출력 경로: {OUT_ROOT.relative_to(ROOT)}/\n")

    summary = synthesize(args.categories, dry_run=args.dry_run)
    print("\n=== 요약 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
