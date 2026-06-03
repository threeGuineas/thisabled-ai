# [DEGRADED RUN · 긴급(3) 데이터 부재] D-3 파이프라인 인프라 검증 · Go/No-Go 판정 보고서

> ## ⚠️ 인용 금지 경고 — 이 보고서의 수치는 공식 평가치가 아니다
> 본 실행은 **긴급(3) 클래스가 학습·검증·테스트 어디에도 0건인 degraded 런**(train 47,336행,
> UnSmile+KOLD만, 합성 emergency 디렉토리 미push + AI-Hub 미연결)이다.
> **Macro-F1 0.573 / 긴급 Recall 0.0 을 Step 4·6의 공식 평가 수치로 인용하지 말 것.**
> 이 문서는 "모델 성능"의 기록이 아니라 **"왜 아직 유효 평가를 못 하는가(데이터 부재)"와
> D-3 인프라가 정상 작동함**을 남기는 진단 기록이다. 유효 수치는 긴급 데이터 확보 +
> D-2 실데이터 hold-out 이후의 재실행에서만 산출된다.

**작성일**: 2026-06-03 · **대상**: 모듈 ① 안전 감시(4단계 위험도 분류)
**실행 환경**: Colab A100-SXM4-40GB · **브랜치**: `fix/d3-leakage-reproducibility` (`25a2838`) **한정** (main·Step4/6 산출물 아님)
**근거**: `notebooks/04_run_full_pipeline.ipynb` 셀 출력(임베드된 MLflow run + stdout)
**판정 기준 출처**: `references/project_facts.md`(KPI 임계값), `references/step4_evaluate_algorithms.md`(선별·로깅), `references/step6_present_results.md`(최종 게이트)

> **저장 경로 안내**: 요청 경로 `/mnt/user-data/outputs/`는 본 실행 환경(로컬)에서 쓰기 불가(read-only)라 리포지토리 `reports/`에 저장했습니다. 파일명에 `DEGRADED-RUN_emergency-absent`를 명시해 공식 결과물과 구분합니다.

---

## ⛔ 한 줄 결론

**조건부 보류(Conditional Hold) — 현 실행 기준으로는 GO 불가.** 두 주 KPI(긴급 Recall ≥ 0.80, Macro-F1 ≥ 0.75)를 모두 미달했으나, **결정적으로 이번 학습 데이터에 긴급(3) 클래스가 0건**이어서 주 KPI(긴급 Recall)는 *실패*가 아니라 **측정 자체가 불가능**했다. 따라서 백업 축소(NO-GO)로 직행하기 전에 **긴급 데이터 통합 후 재실행이 선결조건**이다. 단, 앙상블 복잡도 축에 한해서는 별도의 NO-GO 신호가 관측됐다(아래 Decision 참조).

---

## 1. Problem

모듈 ①은 한국어 댓글/문장을 4단계 위험도(정상0·주의1·경고2·긴급3)로 분류한다. 본 과제의 **주 KPI는 긴급(3) 클래스의 미탐 최소화(Recall ≥ 0.80)**이며, 부 KPI는 4-class 균형 성능(Macro-F1 ≥ 0.75)이다(`project_facts.md` 최종 합격 게이트). 본 보고서는 D-3(누수 차단·재현성 인프라)을 반영한 파이프라인을 Colab에서 끝까지 돌린 뒤, W10~W11 Go/No-Go 기준에 대조해 과제 계속 진행 여부를 판정한다.

## 2. Method

`docs/reproducibility_pipeline.md`의 정석 6단계를 그대로 실행:

1. 시드 다운로드 → processed 빌드 → 합성 병합(MinHash 중복 제거)
2. base KcELECTRA(`beomi/KcELECTRA-base-v2022`) + Weighted CE(α=[1,1.5,1,1]), 3 epoch fine-tune — MLflow 자동 로깅
3. **OOF logits** 생성: Stratified 5-fold(seed=42), 각 fold를 본 적 없는 base로 예측(in-fold 누수 차단)
4. Stacking 메타러너(LightGBM): **train=OOF logits**, val/test=full-train base logits
5. 합성 hold-out(긴급) 평가
6. MLflow run 집계

채점은 D-1에서 수정한 대로 **긴급(3)을 항상 포함한 4-class 고정 채점**(`metrics.py`, `ALL_LABELS=[0,1,2,3]`).

### ⚠️ 실행 데이터 실태 (판정의 핵심 전제)

본 실행은 사전 가정했던 "현재 113k(AI-Hub 긴급 10,935 포함)"가 **아니었다.** 노트북 cell 10 stdout 기준:

| 항목 | 실제 값(Colab) | 비고 |
|---|---|---|
| train | **47,336행** | labels `{0: 19836, 1: 9460, 2: 18040}` — **긴급(3) 0건** |
| val / test | 5,917 / 5,918 | 각 `{0,1,2}`만, **긴급(3) 0건** |
| 시드 출처 | UnSmile + KOLD | **AI-Hub 558 미포함** (`download_seed_datasets.py`가 받지 않음) |
| 합성 긴급 | **없음** | `⚠ 합성 데이터 디렉토리가 존재하지 않습니다` → 시드만 유지 |

즉 `data/synthetic/emergency/`가 브랜치에 포함되지 않았고(미push/gitignore 추정), AI-Hub도 다운로드 스크립트 범위 밖이라, **긴급 클래스가 학습·검증·테스트 어디에도 존재하지 않은 상태로 전 과정이 돌았다.**

## 3. Result

### 3.1 Run별 지표 (MLflow, 4-class 정직 채점)

| Run | 평가 split | Macro-F1 (4-class) | (참고) 3-class macro | 긴급 Recall | AUC-PR | 긴급 support |
|---|---|---:|---:|---:|---:|---:|
| base KcELECTRA CE (3ep) | val (시드) | **0.5735** | 0.7646 | 0.0 (측정불가) | 0.8433 | 0 |
| OOF fold 평균(5-fold) | oof | ≈0.57 | — | 0.0 (측정불가) | ≈0.84 | 0 |
| **Stacking (LGBM, train=OOF)** | test (시드) | **0.5731** | 0.7641 | 0.0 (측정불가) | 0.8509 | 0 |
| 합성 hold-out 평가 | synth(80건) | — | — | **0.0 (0/80)** | — | 80 |

- 4-class Macro-F1가 0.57대인 이유: 긴급(3) F1=0이 1/4 비중으로 들어가기 때문. 이는 **버그가 아니라 D-1 가드가 의도한 정직 채점** — 긴급이 0건이면 점수가 부풀려지지 않는다. (이전 파이프라인이었다면 3-class 평균 0.76으로 게이트 통과처럼 보였을 것.)
- 모든 평가에서 `RuntimeWarning: 평가셋에 긴급(3) 클래스가 0건입니다 … 측정 불가`가 발생 — 가드가 정상 동작.

### 3.2 클래스별 상세 (Stacking, test)

| 클래스 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 정상 | 0.8184 | 0.8685 | 0.8427 | 2,480 |
| 1 주의 | 0.6869 | 0.6027 | 0.6421 | 1,183 |
| 2 경고 | 0.8087 | 0.8062 | 0.8075 | 2,255 |
| **3 긴급** | **0.0** | **0.0** | **0.0** | **0** |

- 주의(1)가 최약 클래스(F1 0.642), 정상·경고는 0.80~0.84. 이는 긴급을 제외한 3-class 문제로서는 합리적 성능.
- LightGBM 학습 로그에 `Start training from score -34.5`(클래스 3) — 긴급 사전확률이 사실상 0이라 **모델 구조상 긴급을 절대 예측하지 않음**. 합성 hold-out 80건 전부 미탐(Recall 0)도 같은 이유.

### 3.3 base → Stacking 향상폭

- 같은 split이 아니라 base=val, stacking=test라 엄밀 비교는 아니나(주의), 동일 클래스 구성에서:
  - 4-class Macro-F1: 0.5735(base/val) vs 0.5731(stacking/test) → **Δ ≈ −0.0004**
  - 3-class Macro-F1: 0.7646 vs 0.7641 → **Δ ≈ −0.0005**
- **Stacking이 base 대비 사실상 향상 없음(±0).** OOF 누수를 제거하니 메타러너의 가짜 이득이 사라졌다 — D-3가 의도한 정직한 결과.

### 3.4 운영 제약 (참고)

- CPU 단건 추론 < 100ms: **이번 실행에서 미측정(불명)**. eval throughput ~1,817 samples/s는 A100 배치 기준이라 CPU 단건 지연과 다름 → 별도 측정 필요.

### 3.5 그림

- `reports/figures/`에 `model_progression.png`, `module1_confusion_baseline.png` 등이 있으나 **모두 이전(구버전) 데이터 산출물**이라 본 실행 수치와 불일치 → 오인 방지를 위해 임베드하지 않음. 본 실행 기준 혼동행렬/진행도 그림은 재생성 필요.

## 4. Discussion

### 4.1 KPI 게이트 대조 (`project_facts.md` / step6 최종 게이트)

| KPI | 임계값 | 실측 | 판정 |
|---|---|---|---|
| 긴급 Recall (hold-out) | ≥ 0.80 | 0.0 (**측정 불가**, support=0) | ❌ 미달 — 단, 데이터 부재로 *무효* |
| Macro-F1 (hold-out, 4-class) | ≥ 0.75 | 0.573 | ❌ 미달 |
| 추론 < 100ms (CPU) | < 100ms | 미측정 | ⚠ 불명 |
| Demographic Parity / 공정성 | — | 모듈 ② 영역, 본 실행 무관 | — |

긴급이 0건이므로 두 주 KPI 모두 **현 데이터로는 충족 불가**. 그러나 이는 "모델이 긴급을 못 잡는다"가 아니라 **"긴급을 학습·측정할 데이터가 환경에 없었다"**는 인프라/데이터 가용성 실패다.

### 4.2 합성 과적합 위험 · 미구현 항목의 영향 (필수 caveat)

- **합성→합성 순환평가 한계 여전**: 합성 hold-out은 합성으로 학습→같은 분포 합성으로 평가하는 구조다. 이번엔 합성 자체가 환경에 없어 *순환평가의 신호조차 측정 못 했고*, 긴급 일반화는 전혀 검증되지 않았다.
- **D-2 실데이터 hold-out 미구현**: `project_facts.md`가 요구한 BEEP(Korean HateSpeech ~1,500건) 실데이터 hold-out이 아직 없다. 더구나 `build_dataset.py`는 현재 BEEP을 **로드조차 하지 않는다**. 이게 없으면 "합성 과적합" 진단도, 신뢰할 만한 긴급 Recall도 불가능 → **결론의 신뢰도를 직접 제약**한다.
- **SMOTE(KoBERT 임베딩) / Optuna(50 trials) 미구현**: 불균형 보정·하이퍼파라미터 최적화가 빠져 있어, 현 점수는 "최적화 이전의 하한선"으로 봐야 한다. 다만 이들을 적용하기 전에 **긴급 데이터부터 확보**하는 것이 순서다.
- **공정성**: 사전 기록된 `fairness_before.json`은 Source gap 0.0586, UnSmile max_gap 0.082, KOLD 0.065로 10%p 기준 이내였으나 **구버전 데이터 기준**이며, '장애' 집단은 `n<30`으로 측정 불가 표기. 본 실행으로 갱신되지 않았으므로 공정성 결론은 보류한다.

### 4.3 절차(Procedure) 측면은 정상

데이터 문제와 별개로, D-3 인프라는 **설계대로 작동**했다: MinHash 중복 제거 동작, OOF logits `(47336,4)` 정상 생성·정렬, train=OOF 적용 확인, 모든 run MLflow 기록, 긴급 0건 경고 발생. 즉 **재현성·누수차단 게이트(체크리스트 A·B·G)는 코드 레벨에서 통과**했고, 막힌 것은 오직 입력 데이터다.

## 5. Decision & Next Steps

### 5.1 판정

**① 과제 전체: 조건부 보류 (Conditional Hold) — GO 아님, 백업 직행도 아님.**
- 근거: 주 KPI(긴급 Recall 0.0 / Macro-F1 0.573) 모두 미달이나, 긴급 데이터가 학습·평가 어디에도 없어 **주 KPI 판정이 무효**다. 유효한 Go/No-Go를 내리려면 긴급 데이터가 포함된 재실행이 **선결조건**이다.

**② 앙상블 복잡도 축: NO-GO 신호 관측 → Soft Voting 축소 권고.**
- 근거: `project_facts.md`의 W10~W11 백업 규칙("Baseline과 PLM Top-2 차이가 유의하지 않으면 앙상블을 단순 Soft Voting으로 축소"). 본 실행에서 **Stacking이 base 대비 Δ≈0** (4-class −0.0004). OOF로 누수를 제거하자 2-layer Stacking의 이득이 사라졌다 → 복잡한 Stacking을 고집할 근거가 약하다. 단, 이 역시 긴급 포함 데이터에서 재확인 후 확정한다.

### 5.2 다음 우선순위 (선결조건부 GO 경로)

1. **[선결·최우선] 긴급 데이터 환경 확보** — ⓐ `data/synthetic/emergency/`를 브랜치에 포함(또는 Drive 스테이징), ⓑ AI-Hub 558 또는 동등 실데이터 긴급을 빌드 파이프라인에 연결. `build_final_dataset.py --synth-repeat 8`이 실제로 긴급을 주입하는지 라벨 분포로 검증.
2. **[D-2] 실데이터 hold-out 구성** — BEEP을 `build_dataset.py`에 로드 추가 → train/합성과 완전 분리한 ~1,500건 hold-out → `evaluate` 경로가 이 실데이터를 보도록 연결. **합성→합성 순환 폐기.**
3. **재실행 후 진짜 Go/No-Go** — 위 1~2 반영해 파이프라인 재실행, 긴급 Recall·Macro-F1을 *실데이터 긴급 포함* 기준으로 측정. 이때 비로소 GO/NO-GO가 유효.
4. **[조건부] 게이트 미달 시** — 긴급 Recall < 0.80이면 Focal α/γ·threshold 조정(Step 5.4); 그래도 미달이면 백업(Soft Voting + 모듈② SBERT+LogReg 축소)으로 전환.
5. **[부수] CPU 단건 추론 시간 측정**, SMOTE·Optuna는 긴급 데이터 확보 이후로 순연.

### 5.3 요약

이번 실행은 **D-3 절차 인프라가 정상 작동함을 입증**했지만, **긴급 데이터 부재로 주 KPI 판정은 무효**다. 정직 채점 가드 덕분에 "3-class 0.76으로 통과한 것처럼 보이는" 함정을 피했고, OOF 적용으로 Stacking의 가짜 이득(누수)이 사라졌음을 확인했다. **GO를 선언하려면 긴급 데이터 확보 + D-2 실데이터 hold-out이 반드시 선행**되어야 한다.
