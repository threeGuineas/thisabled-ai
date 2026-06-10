# 모듈 ① 재현·누수차단 파이프라인 실행 순서

D-3(누수·재현성 인프라) 적용 후의 올바른 실행 순서. GPU 단계는 Colab A100 권장.

## 1. 데이터 빌드 (시드 + 합성, MinHash 중복 제거 포함)

```bash
# 시드 train/val/test 생성 (StratifiedGroupKFold 80/10/10, seed=42)
python scripts/build_processed_dataset.py

# 합성 긴급 데이터를 train에 oversample 병합.
# 이때 시드 train과 근사 중복인 합성 행은 MinHash(threshold=0.8)로 자동 제거된다.
python scripts/build_final_dataset.py --synth-repeat 8
```

- 중복 제거 구현: [src/data/dedup.py](../src/data/dedup.py) (`datasketch` 필요).
- 누수 차단 의도: 합성 행이 시드와 사실상 같은 문장이면 train↔평가 누수 통로가 되므로 제거.

## 2. base 모델 fine-tune (KcELECTRA) — GPU

```bash
python scripts/train_module1.py --config configs/module1_kcelectra_ce.yaml
```

- 모든 run은 MLflow에 자동 기록(`report_to=["mlflow"]`). 기본 tracking URI는 `mlruns/`.
- 파라미터(`cfg/*`)·epoch별 지표·최종 평가(`final_*`)가 한 run에 모인다.

## 3. Stacking용 OOF logits 생성 — GPU (필수, in-fold 누수 차단)

```bash
python scripts/generate_oof_logits.py --config configs/module1_kcelectra_ce.yaml --folds 5
```

- train을 Stratified 5-fold(seed=42)로 나눠 **각 fold를 본 적 없는 base로 예측**한 OOF logits을
  `data/processed/oof_train_logits.npy`로 저장.
- 예측 대상 fold로 best epoch를 고르는 선택 누수를 피하려 fold 학습은 `load_best=False`(최종 epoch).

## 4. Stacking 메타러너 학습 — train은 OOF, val/test는 full-train base

```bash
python scripts/train_stacking.py --config configs/module1_kcelectra_ce.yaml \
    --model-dir models/checkpoints/module1_ce
```

- train split은 3단계의 OOF logits을 사용(없으면 에러로 중단 → 절대 in-fold로 폴백하지 않음).
- val/test는 full-train base 모델 logits 사용(해당 split은 base 학습에 안 쓰였으므로 정상).

## 5. 평가

```bash
python scripts/evaluate_synthetic.py --model-dir models/checkpoints/module1_ce
```

- **주의(코드/로그에 명시됨):** 합성 hold-out 긴급 Recall은 합성→합성 순환 평가다. 실데이터
  긴급 일반화를 보장하지 않는다. 긴급 클래스가 포함된 실데이터 hold-out 구성은 D-2(별도 작업).

## 지표 채점 규약 (D-1)

- [src/evaluation/metrics.py](../src/evaluation/metrics.py)의 macro-F1·per_class는 항상 `[0,1,2,3]`
  4개 라벨로 채점한다. 긴급(3)이 평가셋에 0건이면 macro-F1에 0으로 반영되고 `RuntimeWarning`을
  발생시킨다 — 긴급 미평가를 점수가 좋아 보이도록 숨기지 않기 위함.
- `emergency_support`(긴급 실제 건수)를 함께 반환한다. 0이면 `emergency_recall=0.0`은
  "성능 0"이 아니라 "측정 불가"를 뜻한다.
