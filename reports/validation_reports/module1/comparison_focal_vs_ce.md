# Focal Loss vs Weighted CE 비교 — 모듈 ①

**작성**: 2026-06-01 (D2 잔여) · **실행 환경**: Colab A100-SXM4-40GB · **연계**: [baseline.md](baseline.md)

---

## TL;DR

**Weighted CE (α=[1,1.5,1,1], 3 epoch)가 Focal Loss (γ=2, 5 epoch) 대비 모든 지표에서 이김**.

| 지표 | Focal | CE | Δ |
|---|---:|---:|---:|
| Macro-F1 (test) | 0.7464 | **0.7663** | **+0.0199** |
| 주의(1) F1 | 0.6246 | **0.6575** | **+0.0330** ← 가장 큰 개선 |
| AUC-PR (macro) | 0.8163 | **0.8499** | **+0.0336** |

**결론**: D3 본 학습은 **Weighted CE 채택**. Focal Loss는 정상/주의/경고 사이 불균형이 가벼운 본 데이터에서 가치가 작음 (긴급(3) 합성 후 재검토 필요).

---

## 1. 실험 설정 비교

| 항목 | Focal (베이스라인) | CE α=[1,1.5,1,1] |
|---|---|---|
| Loss | Focal Loss γ=2.0, α=None | Weighted CE (= Focal γ=0) |
| α 가중치 | 없음 | [정상 1.0, 주의 1.5, 경고 1.0, 긴급 1.0] |
| Epochs | 5 (best epoch 2) | 3 (best epoch 2~3) |
| 학습 시간 | 9분 4초 | ~6분 (33% 단축) |
| Config | [module1_kcelectra.yaml](../../../configs/module1_kcelectra.yaml) | [module1_kcelectra_ce.yaml](../../../configs/module1_kcelectra_ce.yaml) |
| 체크포인트 | `module1_baseline_20260531_1302` | `models/checkpoints/module1_ce/` |

CE는 베이스라인 분석에서 확인된 두 가지를 반영:
- **3 epoch**으로 단축 (epoch 3+에서 overfitting)
- **주의(1)에 α=1.5** (베이스라인에서 F1 가장 약했던 클래스)

---

## 2. Test Set 메트릭 (Hold-out, n=5,918)

### 전체
| 지표 | Focal γ=2, 5ep | CE α=[1,1.5,1,1], 3ep | Δ |
|---|---:|---:|---:|
| Macro-F1 | 0.7464 | **0.7663** | +0.0199 |
| 정상(0) F1 | 0.8274 | **0.8407** | +0.0133 |
| 주의(1) F1 | 0.6246 | **0.6575** | +0.0330 |
| 경고(2) F1 | 0.7873 | **0.8008** | +0.0135 |
| AUC-PR (macro) | 0.8163 | **0.8499** | +0.0336 |

### 데이터셋별 분리 ([label_mapping.md §6.L4](../../../docs/label_mapping.md) 정책)
| 데이터셋 | Focal | CE | Δ |
|---|---:|---:|---:|
| UnSmile (n=1,891) | 0.7748 | **0.7931** | +0.0183 |
| KOLD (n=4,027) | 0.7175 | **0.7388** | +0.0213 |
| **격차** | 0.0573 | **0.0543** | −0.0030 |

CE에서 양쪽 모두 개선 + 격차도 살짝 좁아짐. KOLD 개선폭이 약간 더 큼.

---

## 3. 분석 — 왜 CE가 이겼나

### 3.1 Focal Loss의 가치가 작은 이유
- Focal Loss는 **극단적 클래스 불균형**(positive ~1%)에서 효과가 크게 나옴.
- 본 데이터 분포 **42 : 20 : 38 : 0** — 0/1/2 사이는 가벼운 불균형. Focal의 (1-pt)^γ 항이 큰 이득 못 줌.
- 오히려 **easy examples 의 가중치를 낮추는 부작용**이 일부 작용 (epoch 3+ 빠른 overfitting).

### 3.2 α 가중치의 직접 효과
- 주의(1)이 가장 약한 클래스 (베이스라인 F1 0.6246) → α=1.5 부여.
- 결과: 주의(1) F1 **+0.033 개선** (전체 평균 개선 +0.020보다 큼).
- 정상·경고도 함께 개선된 건 균형 잡힌 학습이 다른 클래스 boundary도 깔끔하게 만든 효과로 추정.

### 3.3 3 epoch의 적절성
- 베이스라인 분석에서 "epoch 2 best, 3+ overfitting" 진단대로 3 epoch가 적절.
- 학습 시간 33% 단축 + 동등 이상 성능 → 효율성 측면도 우위.

### 3.4 긴급(3)은 변함없음
- 둘 다 `emergency_recall = 0` (시드 0건이라 당연).
- 합성 데이터 학습 후 별도 비교 필요.

---

## 4. 5일판 목표 도달도 갱신

| 목표 | 기준 | 베이스라인 (Focal) | CE | 평가 |
|---|---|---:|---:|---|
| Macro-F1 ≥ 0.60 (1차) | Test | 0.746 | **0.766** | ✅ 둘 다 통과 |
| Macro-F1 ≥ 0.68 (Stretch) | Test | 0.746 | **0.766** | ✅ 둘 다 통과, CE가 +0.02 |
| 긴급 Recall ≥ 0.75 | Test | 0.0 | 0.0 | ❌ 합성 필요 (둘 다) |
| 데이터셋별 격차 | 측정·보고 | 0.057 | 0.054 | ✅ CE 약간 좁힘 |

---

## 5. 결론 + D3 결정

### D3에 사용할 학습 설정
- **Loss**: Weighted CE, α=[1.0, 1.5, 1.0, 1.0]
- **Epochs**: 3
- **나머지**: 베이스라인과 동일 (KcELECTRA-base-v2022, lr 2e-5, batch 32, max_length 128, fp16)

### 향후 검토 사항
1. **긴급(3) 합성 후 α[3] 재튜닝** — 합성량에 따라 1.0~2.0 사이.
2. **Stacking에서 두 모델 앙상블 가능성** — Focal과 CE가 다른 부분에서 잘 맞으면 stacking 시 보완 효과 (D3에서 시도).
3. **데이터셋별 격차 0.054** — 여전히 작지 않음. D3 Stacking에서 `source` feature 추가로 추가 완화 시도.

---

## 6. 산출물

| 파일 | 내용 |
|---|---|
| 본 리포트 | `reports/validation_reports/module1/comparison_focal_vs_ce.md` |
| CE 메트릭 JSON | `reports/validation_reports/module1/ce_20260601.json` (Drive 백업 권장) |
| CE 체크포인트 | `models/checkpoints/module1_ce/checkpoint-*` (Drive 백업 권장) |
| Focal 메트릭 (비교 기준) | `reports/validation_reports/module1/baseline_20260531_1302.json` |
