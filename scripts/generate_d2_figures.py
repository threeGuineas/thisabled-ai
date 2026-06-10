"""D-2 보고서용 도표 생성: 합성↔실데이터 긴급 Recall, 실데이터 클래스별 F1.

`reports/validation_reports/module1/real_holdout_eval.json`의 수치를 읽어 그린다.
혼동행렬은 per-cell 예측이 export되지 않아(집계 지표만) 생략 — 대신 클래스별 F1로 대체.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import koreanize_matplotlib  # noqa: F401, E402
except ImportError:
    print("⚠ koreanize_matplotlib 없음 — 한글 깨질 수 있음")

ROOT = Path(__file__).resolve().parents[1]
EVAL_JSON = ROOT / "reports" / "validation_reports" / "module1" / "real_holdout_eval.json"
FIG_DIR = ROOT / "reports" / "figures"
GATE_RECALL, GATE_F1 = 0.80, 0.75


def main() -> int:
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    aihub = data["aihub"]
    synth_only = data.get("emergency_recall_synthetic_only_baseline", 0.0047)
    real_after = aihub["emergency_recall"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Fig 1 — 긴급 Recall: 순환 vs 비순환 / 합성↔실데이터
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [
        "합성 hold-out\n(순환 평가)",
        "실데이터\n(합성-only 학습)",
        "실데이터\n(+AI-Hub 실데이터 학습)",
    ]
    vals = [1.0, synth_only, real_after]
    colors = ["#bdbdbd", "#e06666", "#4a86e8"]
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(GATE_RECALL, ls="--", color="green", lw=1.3, label=f"게이트 {GATE_RECALL}")
    for b, v in zip(bars, vals, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("긴급(3) Recall")
    ax.set_title("긴급 Recall — 합성 과적합과 실데이터 보강 효과")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p1 = FIG_DIR / "module1_emergency_recall_synth_vs_real.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  ✔ {p1.relative_to(ROOT)}")

    # Fig 2 — 실데이터 hold-out 클래스별 F1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = ["정상(0)", "주의(1)", "경고(2)", "긴급(3)"]
    f1s = [aihub["per_class"][str(i)]["f1"] for i in range(4)]
    bars = ax.bar(names, f1s, color=["#4a86e8", "#f6b26b", "#e06666", "#6aa84f"])
    ax.axhline(GATE_F1, ls="--", color="green", lw=1.3, label=f"Macro-F1 게이트 {GATE_F1}")
    ax.axhline(
        aihub["macro_f1"], ls=":", color="black", lw=1.2, label=f"Macro-F1 {aihub['macro_f1']:.3f}"
    )
    for b, v in zip(bars, f1s, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("실데이터 hold-out(AI-Hub) 클래스별 F1 — 경고(2) 병목")
    ax.legend(loc="upper center")
    fig.tight_layout()
    p2 = FIG_DIR / "module1_real_holdout_per_class_f1.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  ✔ {p2.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
