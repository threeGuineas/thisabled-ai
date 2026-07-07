"""Optuna 하이퍼파라미터 튜닝 — 긴급 Recall 제약 조건부 최적화.

Usage:
    python scripts/optuna_search.py --n-trials 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.trainer import train_module1


def objective(trial: optuna.Trial, config_path: Path) -> float:
    # ── 탐색 공간 ──
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    # 클래스 가중치 (0: 정상, 1: 주의, 2: 경고, 3: 긴급)
    # 긴급은 중요하므로 상한을 높게 준다.
    alpha_0 = 1.0  # 기준
    alpha_1 = trial.suggest_float("alpha_1", 1.0, 3.0)
    alpha_2 = trial.suggest_float("alpha_2", 1.0, 5.0)
    alpha_3 = trial.suggest_float("alpha_3", 1.0, 8.0)
    alpha = [alpha_0, alpha_1, alpha_2, alpha_3]

    # 경고(2) 증강 배율
    warning_augment_ratio = trial.suggest_float("warning_augment_ratio", 0.0, 5.0)

    override_params = {
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "alpha": alpha,
        "warning_augment_ratio": warning_augment_ratio,
        # epoch과 batch_size는 베이스라인 검증 결과에 따라 고정
        "num_epochs": 2,
    }

    # ── 학습 실행 ──
    # 매 trial 마다 별도의 디렉토리에 체크포인트 저장 후 덮어쓰기 방지
    # MLflow에 기록할 수 있도록 trainer.py를 바로 호출
    try:
        res = train_module1(config_path, project_root=ROOT, override_params=override_params)
        eval_metrics = res["eval_metrics"]

        macro_f1 = eval_metrics["eval_macro_f1"]
        emergency_recall = eval_metrics["eval_emergency_recall"]

        # ── 긴급 Recall 제약 조건 ──
        # 주 KPI인 긴급 Recall >= 0.80을 달성하지 못한 경우 페널티 부여
        if emergency_recall < 0.80:
            penalty = (0.80 - emergency_recall) * 10
            return macro_f1 - penalty

        return macro_f1

    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--config", type=str, default="configs/module1_kcelectra_ce.yaml")
    args = parser.parse_args()

    config_path = ROOT / args.config

    # TPE sampler와 MedianPruner 설정
    sampler = TPESampler(seed=42)
    # 첫 10회는 완주하여 분포 파악, 1 epoch 후 나쁜 trial 조기 종료
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    study = optuna.create_study(
        study_name="thisabled-module1-optuna", direction="maximize", sampler=sampler, pruner=pruner
    )

    print(f"=== Optuna Search 시작 ({args.n_trials} trials) ===")
    study.optimize(lambda t: objective(t, config_path), n_trials=args.n_trials, gc_after_trial=True)

    print("\n=== Optuna 최적화 완료 ===")
    best_trial = study.best_trial
    print(f"Best Value (Macro-F1 + Constraint): {best_trial.value:.4f}")
    print("Best Params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
