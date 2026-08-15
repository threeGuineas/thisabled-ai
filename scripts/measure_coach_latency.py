"""소통 코치 응답 지연 측정 (W3 DoD: LLM 응답 < 3초).

캐시가 비어 있는 상태의 실제 LLM 호출만 잰다. 캐시 적중은 밀리초라 섞으면 수치가
무의미해진다. 동작별 p50/p95 와 게이트 판정을 artifacts/ 에 남긴다.

    GEMINI_API_KEY=... python3 scripts/measure_coach_latency.py --repeats 5

키가 없으면 --dry-run 으로 경로만 확인할 수 있다(지연 수치는 의미 없음).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coach import CoachAction, CoachService  # noqa: E402
from src.data.llm_client import GeminiClient, RequestPacer  # noqa: E402

SCHEMA_VERSION = 1
LATENCY_GATE_SECONDS = 3.0

# 동작별 대표 입력. 실제 사용에 가까운 길이여야 측정이 의미 있다.
SAMPLES: dict[CoachAction, list[dict[str, Any]]] = {
    CoachAction.EASY_SENTENCE: [
        {
            "text": "저번에 말씀하신 모임 건은 일정 조율이 어려울 것 같아서 부득이하게 참석이 곤란할 듯합니다."
        },
        {"text": "혹시 괜찮으시다면 다음 주 중으로 다시 한번 논의해보는 것은 어떨까 싶습니다."},
    ],
    CoachAction.COMPLETE_SENTENCE: [
        {"text": "오늘 영화를 봤는데"},
        {"text": "같이 산책하고 싶은데 혹시"},
    ],
    CoachAction.SUGGEST_REPLY: [
        {"context": [("partner", "주말에 등산 다녀왔어요. 날씨가 정말 좋더라고요.")]},
        {
            "context": [
                ("partner", "요즘 무슨 드라마 보세요?"),
                ("me", "저는 요리 예능을 자주 봐요."),
                ("partner", "오 저도 그거 좋아해요!"),
            ]
        },
    ],
    CoachAction.CONVERSATION_HINT: [
        {"context": [("partner", "안녕하세요, 반가워요."), ("me", "안녕하세요!")]},
        {"context": [("partner", "저는 사진 찍는 걸 좋아해요.")]},
    ],
}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1])


def measure(
    service: CoachService, *, repeats: int, cache_busting: bool
) -> dict[str, dict[str, Any]]:
    per_action: dict[str, dict[str, Any]] = {}
    for action, samples in SAMPLES.items():
        latencies: list[float] = []
        sources: list[str] = []
        degraded: list[str] = []
        for repeat in range(repeats):
            for index, sample in enumerate(samples):
                payload = dict(sample)
                if cache_busting:
                    # 같은 입력을 반복하면 2회차부터 캐시라 실호출이 안 잡힌다.
                    # text·context 양쪽 다 흔들어야 동작별 표본 수가 같아진다.
                    marker = f"({repeat}-{index})"
                    if "text" in payload:
                        payload["text"] = f"{payload['text']} {marker}"
                    if "context" in payload:
                        turns = list(payload["context"])
                        speaker, message = turns[-1]
                        turns[-1] = (speaker, f"{message} {marker}")
                        payload["context"] = turns
                started = time.monotonic()
                result = service.run(action, invoked_by_user=True, **payload)
                latencies.append(time.monotonic() - started)
                sources.append(result.source)
                if result.degraded_reason:
                    degraded.append(result.degraded_reason)

        llm_latencies = [
            latency for latency, source in zip(latencies, sources, strict=True) if source == "llm"
        ]
        per_action[action.value] = {
            "n_calls": len(latencies),
            "n_llm_calls": len(llm_latencies),
            "sources": {s: sources.count(s) for s in sorted(set(sources))},
            "p50_seconds": _percentile(sorted(llm_latencies), 50),
            "p95_seconds": _percentile(sorted(llm_latencies), 95),
            "max_seconds": max(llm_latencies) if llm_latencies else float("nan"),
            "degraded_reasons": sorted(set(degraded)),
        }
    return per_action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--request-interval", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "coach_latency.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="LLM 없이 경로만 확인 (지연 수치는 의미 없음)",
    )
    args = parser.parse_args()

    if args.dry_run:
        service = CoachService(
            lambda prompt: json.dumps({"suggestions": ["예시 후보"]}, ensure_ascii=False)
        )
        model_name = "dry-run"
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("GEMINI_API_KEY 가 없습니다. 셸 env로 넘기거나 --dry-run 을 쓰세요.")
            return 2
        service = CoachService(
            GeminiClient(
                api_key,
                model=args.model,
                temperature=0.7,
                pacer=RequestPacer(args.request_interval),
            )
        )
        model_name = args.model

    per_action = measure(service, repeats=args.repeats, cache_busting=True)
    all_p95 = [row["p95_seconds"] for row in per_action.values() if row["n_llm_calls"] > 0]
    worst_p95 = max(all_p95) if all_p95 else float("nan")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "repeats": args.repeats,
        "per_action": per_action,
        "gate": {
            "metric": "p95_seconds",
            "threshold": LATENCY_GATE_SECONDS,
            "worst_p95_seconds": worst_p95,
            "pass": bool(all_p95) and worst_p95 < LATENCY_GATE_SECONDS,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")

    print(f"{'동작':<22}{'호출':>6}{'LLM':>6}{'p50':>9}{'p95':>9}{'max':>9}")
    for action, row in per_action.items():
        print(
            f"{action:<22}{row['n_calls']:>6}{row['n_llm_calls']:>6}"
            f"{row['p50_seconds']:>9.2f}{row['p95_seconds']:>9.2f}{row['max_seconds']:>9.2f}"
        )
        if row["degraded_reasons"]:
            print(f"    저하: {row['degraded_reasons']}")
    gate = payload["gate"]
    print(
        f"\n게이트 p95 < {LATENCY_GATE_SECONDS}s → "
        f"{'PASS' if gate['pass'] else 'FAIL'} (최악 p95 {worst_p95:.2f}s)"
    )
    print(f"saved: {args.out}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
