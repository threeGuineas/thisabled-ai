#!/usr/bin/env bash
# 모듈 ① 전체 파이프라인 (D-3 누수차단·재현성 반영) — 정석 실행 순서.
#
# docs/reproducibility_pipeline.md 와 1:1 대응. fail-fast(set -euo pipefail)이며
# 각 단계는 멱등(재실행 안전)하도록 설계된 하위 스크립트를 호출한다.
# GPU 권장(2·3단계는 KcELECTRA fine-tune). Colab A100 기준.
#
# 사용:
#   bash scripts/run_full_pipeline.sh
#   CONFIG=configs/module1_kcelectra_final.yaml SYNTH_REPEAT=8 OOF_FOLDS=5 \
#     bash scripts/run_full_pipeline.sh
#
# 환경변수 (기본값):
#   PYTHON         실행 파이썬           (python)
#   CONFIG         학습/스태킹 config    (configs/module1_kcelectra_ce.yaml)
#   SYNTH_REPEAT   합성 oversample 배수  (8; 0이면 합성 미포함)
#   OOF_FOLDS      OOF Stratified fold 수 (5)
#   MODEL_DIR      base 체크포인트 dir   (config의 paths.checkpoint_dir와 일치해야 함)
#   SKIP_DATA      "1"이면 1단계 데이터 빌드 건너뜀 (이미 빌드된 경우)

set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/module1_kcelectra_ce.yaml}"
SYNTH_REPEAT="${SYNTH_REPEAT:-8}"
OOF_FOLDS="${OOF_FOLDS:-5}"
MODEL_DIR="${MODEL_DIR:-models/checkpoints/module1_ce}"
SKIP_DATA="${SKIP_DATA:-0}"

# 리포지토리 루트에서 실행되도록 고정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

step() { printf '\n\033[1;36m=== [%s/7] %s ===\033[0m\n' "$1" "$2"; }

echo "PYTHON=$PYTHON  CONFIG=$CONFIG  SYNTH_REPEAT=$SYNTH_REPEAT  OOF_FOLDS=$OOF_FOLDS  MODEL_DIR=$MODEL_DIR"
"$PYTHON" -c "import torch;print('CUDA available:', torch.cuda.is_available())"

if [ "$SKIP_DATA" != "1" ]; then
  step 1 "시드 데이터 다운로드 + processed 빌드 + 합성 병합(MinHash 중복 제거)"
  "$PYTHON" scripts/download_seed_datasets.py
  "$PYTHON" scripts/build_processed_dataset.py
  "$PYTHON" scripts/build_final_dataset.py --synth-repeat "$SYNTH_REPEAT"
else
  step 1 "데이터 빌드 건너뜀 (SKIP_DATA=1)"
fi

step 2 "base KcELECTRA fine-tune (MLflow 자동 로깅)"
"$PYTHON" scripts/train_module1.py --config "$CONFIG"

step 3 "OOF logits 생성 (Stratified ${OOF_FOLDS}-fold, seed=42 — in-fold 누수 차단)"
"$PYTHON" scripts/generate_oof_logits.py --config "$CONFIG" --folds "$OOF_FOLDS"

step 4 "Stacking 메타러너 학습 (train=OOF, val/test=full-train base)"
"$PYTHON" scripts/train_stacking.py --config "$CONFIG" --model-dir "$MODEL_DIR"

step 5 "합성 hold-out 평가 (⚠ 합성→합성 순환 — 실데이터 일반화 아님)"
"$PYTHON" scripts/evaluate_synthetic.py --model-dir "$MODEL_DIR"

# 실데이터 hold-out은 커밋된 jsonl(data/eval/*.jsonl)을 그대로 사용한다.
# build_real_holdout.py는 AI-Hub raw가 필요해 여기(Colab)서 돌리지 않는다.
step 6 "실데이터 hold-out 평가 (AI-Hub 실긴급 + BEEP — 긴급 Recall '측정 가능', 비순환)"
"$PYTHON" scripts/evaluate_real_holdout.py --model-dir "$MODEL_DIR"

step 7 "완료 — MLflow 결과 확인"
echo "mlruns/ 에 run 기록됨. 확인:  $PYTHON -m mlflow ui --backend-store-uri mlruns"
echo "또는 노트북 마지막 셀에서 mlflow.search_runs()로 지표 표 출력."
