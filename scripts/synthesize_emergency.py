"""GPT-4o-mini로 긴급(3) 합성 데이터 생성.

각 카테고리(3a/3b/3c/3d) + boundary 반례를 emergency_scenarios.md §5 spec대로 생성.
환경변수: OPENAI_API_KEY 필수.
비용 캡: COST_CAP_USD (기본 $5).

Usage:
    python scripts/synthesize_emergency.py                   # 전체 생성
    python scripts/synthesize_emergency.py --categories 3a   # 특정 카테고리만
    python scripts/synthesize_emergency.py --dry-run         # 비용·요청 수만 출력
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

from src.data.synthesis_prompts import (  # noqa: E402
    CONTEXTS,
    LANGUAGE_STYLES,
    SYSTEM_PROMPT,
    TARGET_COUNTS,
    build_user_prompt,
    category_personas,
)

OUT_ROOT = ROOT / "data" / "synthetic" / "emergency"
COST_CAP_USD = 5.0
MODEL = "gpt-4o-mini"
BATCH_SIZE = 20  # 한 LLM 호출당 생성 건수

# gpt-4o-mini 단가 (per 1M tokens, 2025년 기준)
PRICE_INPUT = 0.15 / 1e6
PRICE_OUTPUT = 0.60 / 1e6


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * PRICE_INPUT + completion_tokens * PRICE_OUTPUT


def _parse_jsonl(text: str) -> list[dict]:
    """LLM 출력을 JSONL로 파싱, 실패 라인은 skip."""
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


def _call_llm(client, category: str, n: int, seed: int) -> tuple[list[dict], int, int]:
    """LLM 호출 → (생성된 dict 리스트, prompt tokens, completion tokens)."""
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

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        max_tokens=4000,
        seed=seed,
    )

    text = resp.choices[0].message.content or ""
    items = _parse_jsonl(text)
    return items, resp.usage.prompt_tokens, resp.usage.completion_tokens


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
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY 환경변수가 없습니다.")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
    else:
        client = None

    total_cost = 0.0
    total_calls = 0
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
                est_cost = n_calls * _estimate_cost(1500, 1600)
                print(f"  → {n_calls} LLM 호출, 예상 비용 ${est_cost:.4f}")
                total_calls += n_calls
                total_cost += est_cost
                summary[cat][split] = len(existing)
                continue

            batch_seed = 42
            while needed > 0:
                if total_cost >= COST_CAP_USD:
                    print(f"💸 비용 캡 ${COST_CAP_USD} 도달 — 중단")
                    summary[cat][split] = len(existing)
                    return summary | {"total_cost": total_cost, "total_calls": total_calls}

                n = min(BATCH_SIZE, needed)
                try:
                    items, in_tok, out_tok = _call_llm(client, cat, n, batch_seed)
                    enriched = _enrich(items, cat, split)
                    existing.extend(enriched)
                    cost = _estimate_cost(in_tok, out_tok)
                    total_cost += cost
                    total_calls += 1
                    print(
                        f"  call#{total_calls}: +{len(enriched)}건 "
                        f"(in {in_tok}/out {out_tok} tok, ${cost:.4f}, 누계 ${total_cost:.4f})"
                    )
                    # 누적 저장 (중간 실패 대비)
                    with out_path.open("w", encoding="utf-8") as f:
                        for it in existing:
                            f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    needed = target - len(existing)
                    batch_seed += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  ⚠ 호출 실패: {e}, 재시도")
                    time.sleep(2)
                    batch_seed += 1

            summary[cat][split] = len(existing)
            print(f"[{cat}/{split}] 최종 {len(existing)}/{target}건 → {out_path.relative_to(ROOT)}")

    summary["total_cost"] = round(total_cost, 4)
    summary["total_calls"] = total_calls
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(TARGET_COUNTS.keys()),
        choices=list(TARGET_COUNTS.keys()),
    )
    parser.add_argument("--dry-run", action="store_true", help="비용·호출 수만 계산")
    args = parser.parse_args()

    print(f"카테고리: {args.categories}")
    print(f"모델: {MODEL}, batch_size: {BATCH_SIZE}, 비용 캡: ${COST_CAP_USD}")
    print(f"출력 경로: {OUT_ROOT.relative_to(ROOT)}/\n")

    summary = synthesize(args.categories, dry_run=args.dry_run)
    print("\n=== 요약 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
