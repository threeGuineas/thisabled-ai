"""v6 학습 데이터 주입 검증 (transformers 불필요).

트레이너의 실제 로더(load_extra_binary_train, load_extra_caution, collapse_binary)를
그대로 호출해 config v6의 신규 소스가 올바르게 이진 학습셋으로 들어가는지 검증한다.
transformers/torch 의존 모듈은 sys.modules 스텁으로 우회(로더는 순수 pandas/json).

실행: python scripts/verify_v6_ingestion.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- transformers/torch 의존 모듈 스텁 (로더 함수는 이들을 실제로 쓰지 않음) ---
for name in ["transformers", "torch"]:
    if name not in sys.modules:
        m = types.ModuleType(name)
        # 어떤 속성이든 새 클래스로 반환 → Trainer 등 base class 상속 가능
        m.__getattr__ = lambda attr: type(attr, (), {})
        sys.modules[name] = m
# trainer가 import하는 내부 모듈도 스텁(로더 경로에서 미사용)
for name in [
    "src.evaluation.metrics",
    "src.models.focal_loss",
    "src.training.dataset",
    "src.utils.seed",
    "src.utils.tracking",
]:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr: (lambda *a, **k: None)
        sys.modules[name] = m

from src.training.trainer import (  # noqa: E402
    collapse_binary,
    load_extra_binary_train,
    load_extra_caution,
)

CONFIG = ROOT / "configs" / "module1_binary_hardcases_v6.yaml"


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    print(f"=== config: {CONFIG.relative_to(ROOT)} ===")
    print(
        f"binary={data_cfg.get('binary')}  extra_train_repeat={data_cfg.get('extra_train_repeat')}"
    )

    # 1) extra_caution (PAN12 predator → 전량 주의)
    print("\n[extra_caution_jsonl]")
    caution = load_extra_caution(ROOT, data_cfg)
    if caution is not None:
        print(f"  로드 {len(caution)}건, label 분포 {dict(caution['label'].value_counts())}")
        print(f"  source 분포 {dict(caution['source'].value_counts())}")

    # 2) extra_train (하드케이스 + DKTC + K-MHaS + APEACH, 라벨 보존)
    print("\n[extra_train_jsonl] (경로별)")
    total = None
    for path in data_cfg["extra_train_jsonl"]:
        single = load_extra_binary_train(
            ROOT, {"extra_train_jsonl": [path], "extra_train_repeat": 1}
        )
        n0 = int((single["label"] == 0).sum())
        n1 = int((single["label"] == 1).sum())
        print(f"  {path:48s}  총 {len(single):6d}  (정상 {n0:6d} / 주의 {n1:6d})")
        total = single if total is None else pd.concat([total, single], ignore_index=True)

    # 3) 병합 후 분포 + 이진 붕괴 idempotency
    print("\n[신규 주입 합계 (base 제외)]")
    merged = pd.concat([caution, total], ignore_index=True) if caution is not None else total
    merged = collapse_binary(merged)  # {1,2,3}→1, 0 유지 (라벨 무결성 확인)
    n0 = int((merged["label"] == 0).sum())
    n1 = int((merged["label"] == 1).sum())
    print(f"  총 {len(merged)}  정상 {n0} / 주의 {n1}  (주의 비율 {n1/len(merged):.1%})")
    print(f"  라벨 도메인 {sorted(merged['label'].unique())} (이진 {{0,1}} 이어야 함)")
    print(f"  source별\n{merged['source'].value_counts().to_string()}")

    # 4) 기존 base(README 실측)와 합산 시 규모 감
    base_normal, base_caution = 19941, 28386  # module1_binary train 분포(README)
    print("\n[base(README) + 신규 주입 = v6 학습셋 추정]")
    print(
        f"  정상 {base_normal + n0:,}  주의 {base_caution + n1:,}  총 {base_normal+base_caution+len(merged):,}"
    )


if __name__ == "__main__":
    main()
