"""비순환 실데이터 hold-out 평가 (긴급 Recall '측정 가능').

`scripts/build_real_holdout.py`가 만든 로컬 jsonl을 사용(라이선스상 저장소에는 커밋하지 않음):
- `data/eval/aihub_real_holdout.jsonl` — AI-Hub 실긴급 포함 4-class → **긴급 Recall + 4-class Macro-F1**
- `data/eval/beep_real_holdout.jsonl` — BEEP 0/1/2 → 비긴급 도메인 일반화

학습은 합성 긴급으로, 평가는 실데이터 긴급으로 분리되어 있으므로 이 수치가 비로소
'합성→실데이터 일반화'를 반영한다(순환평가 아님). 합성 hold-out 결과 JSON이 있으면
긴급 Recall의 합성↔실데이터 갭도 함께 보고한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_classification_metrics  # noqa: E402
from src.utils.tracking import log_metrics, mlflow_run  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"


def _predict(pipe, texts: list[str]) -> list[int]:
    preds = pipe(texts, batch_size=64, truncation=True, max_length=128)
    return [int(p["label"].split("_")[-1]) for p in preds]


def evaluate_real_holdout(model_dir: Path, out_path: Path) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== [모듈 ①] 실데이터 hold-out 평가 ({device}) ===")

    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {model_dir.resolve()}\n"
            "models/ 는 git에 포함되지 않으므로 git pull로 받아지지 않습니다(가중치는 .safetensors).\n"
            "→ 같은 Colab 세션에서 먼저 학습(scripts/train_module1.py 또는 노트북 3-2)을 돌려 "
            "models/checkpoints/module1_ce 를 생성한 뒤 이 평가를 실행하세요. "
            "세션이 재시작되면 체크포인트가 사라지니 학습→평가를 한 세션에서 이어서 하세요."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    result: dict = {"model_dir": str(model_dir)}

    # 1) AI-Hub 실긴급 hold-out — 긴급 Recall + 4-class
    aihub_path = EVAL_DIR / "aihub_real_holdout.jsonl"
    if aihub_path.exists():
        df = pd.read_json(aihub_path, lines=True)
        df["pred"] = _predict(pipe, df["text"].tolist())
        m = compute_classification_metrics(df["label"].to_numpy(), df["pred"].to_numpy())
        result["aihub"] = {
            "n": len(df),
            "macro_f1": m["macro_f1"],
            "emergency_recall": m["emergency_recall"],
            "emergency_support": m["emergency_support"],
            "per_class": m["per_class"],
        }
        print(
            f"[AI-Hub 실긴급] n={len(df)} | Macro-F1(4cls)={m['macro_f1']:.4f} | "
            f"긴급 Recall={m['emergency_recall']:.4f} (support={m['emergency_support']})"
        )
    else:
        print(f"⚠ {aihub_path} 없음 — scripts/build_real_holdout.py 먼저 실행")

    # 2) BEEP 0/1/2 hold-out — 비긴급 도메인 일반화
    beep_path = EVAL_DIR / "beep_real_holdout.jsonl"
    if beep_path.exists():
        df = pd.read_json(beep_path, lines=True)
        df["pred"] = _predict(pipe, df["text"].tolist())
        m = compute_classification_metrics(df["label"].to_numpy(), df["pred"].to_numpy())
        result["beep"] = {
            "n": len(df),
            "macro_f1": m["macro_f1"],
            "per_class": m["per_class"],
        }
        print(f"[BEEP 0/1/2] n={len(df)} | Macro-F1(4cls 채점)={m['macro_f1']:.4f}")
    else:
        print(f"⚠ {beep_path} 없음 (네트워크 환경에서 build_real_holdout.py 재실행)")

    # 3) 합성↔실데이터 긴급 Recall 갭 (순환평가 함정 정량화)
    synth_json = (
        ROOT / "reports" / "validation_reports" / "module1" / "emergency_synthetic_eval.json"
    )
    if synth_json.exists() and "aihub" in result:
        try:
            synth = json.loads(synth_json.read_text(encoding="utf-8"))
            synth_recall = float(synth.get("overall_recall", 0.0))
            real_recall = result["aihub"]["emergency_recall"]
            result["emergency_recall_gap_synth_minus_real"] = synth_recall - real_recall
            print(
                f"[갭] 긴급 Recall 합성={synth_recall:.4f} − 실데이터={real_recall:.4f} "
                f"= {synth_recall - real_recall:+.4f} (클수록 합성 과적합)"
            )
        except (json.JSONDecodeError, ValueError):
            pass

    with mlflow_run("thisabled-module1", run_name=f"real-holdout-eval/{model_dir.name}"):
        if "aihub" in result:
            log_metrics(
                {
                    "real_aihub_macro_f1": result["aihub"]["macro_f1"],
                    "real_aihub_emergency_recall": result["aihub"]["emergency_recall"],
                    "real_aihub_emergency_support": result["aihub"]["emergency_support"],
                }
            )
        if "beep" in result:
            log_metrics({"real_beep_macro_f1": result["beep"]["macro_f1"]})
        if "emergency_recall_gap_synth_minus_real" in result:
            log_metrics({"emergency_recall_gap": result["emergency_recall_gap_synth_minus_real"]})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✔ 저장: {out_path.relative_to(ROOT)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=str, default=str(ROOT / "models" / "checkpoints" / "module1_ce")
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(ROOT / "reports" / "validation_reports" / "module1" / "real_holdout_eval.json"),
    )
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    # 상대경로가 cwd 기준으로 없으면 리포지토리 루트 기준으로 재해석
    if not model_dir.is_absolute() and not model_dir.exists():
        alt = ROOT / model_dir
        if alt.exists():
            model_dir = alt
    evaluate_real_holdout(model_dir, Path(args.out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
