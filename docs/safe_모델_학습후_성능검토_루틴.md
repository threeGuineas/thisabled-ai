# SAFE 모델 학습 후 성능 검토 루틴

작성: 2026-07-08
적용 대상: ThisAbled SAFE 분류 모델 및 모델+규칙 하이브리드

## 1. 목적

모델 학습이 정상 종료됐다는 사실과 서비스에 배포할 수 있다는 판단은 별개다. 모든 학습 run은
이 문서의 데이터 무결성, 오탐, 미탐, calibration, 서빙 검증을 통과해야 한다. Recall 또는
Macro-F1 하나만으로 배포를 승인하지 않는다.

## 2. 배포 판정

| 판정 | 의미 |
|---|---|
| REJECTED | 데이터 누수, 필수 지표 미달, artifact 불완전 중 하나 이상 |
| STAGING ONLY | 기본 성능은 통과했지만 blind test·운영 검증이 남음 |
| PRODUCTION APPROVED | 고정된 blind test와 서빙 계약까지 전부 통과 |

성공 기준은 학습 전에 정한다. 결과를 본 뒤 기준이나 평가셋을 바꾸면 해당 평가는 calibration 또는
개발 평가로 격하한다.

## 3. 평가 데이터 분리

다음 네 역할을 물리적으로 분리한다.

1. **Train**: 파라미터 학습에만 사용한다.
2. **Validation**: epoch/best checkpoint 선택에만 사용한다.
3. **Calibration**: 운영 threshold와 규칙을 선택한다.
4. **Blind test**: 모든 선택이 끝난 뒤 한 번만 사용한다.

Calibration 결과를 보고 threshold를 선택했다면 그 데이터의 수치는 최종 성능으로 보고하지 않는다.
회귀셋 문장을 보고 데이터·규칙·threshold를 수정했다면 해당 버전도 blind test가 아니다. 회귀셋은
버전과 생성일을 고정하고, 모델 선택에 사용한 버전과 최종 검증 버전을 구분한다.

## 4. 학습 직후 필수 검증

### 4.1 재현성 및 artifact

- [ ] config 경로와 SHA-256 기록
- [ ] seed=42 기록
- [ ] Git commit SHA 기록
- [ ] train/val/test 건수, 라벨 분포, source 분포 기록
- [ ] `config.json`, `model.safetensors`, tokenizer 파일 존재
- [ ] `num_labels`와 classifier head shape 일치
- [ ] best checkpoint/epoch와 eval loss 기록
- [ ] 학습 종료 즉시 영속 저장소 또는 HF에 후보 artifact 백업

### 4.2 누수 검사 — 하나라도 실패하면 즉시 REJECTED

- [ ] conversation/group/source ID train↔holdout 교집합 0
- [ ] 정규화 완전 일치 train↔holdout 0
- [ ] MinHash/Jaccard 등 근사 중복 검사 결과 기록
- [ ] 합성 train과 합성 holdout split 분리
- [ ] real holdout 파일이 데이터 빌더의 입력 source로 병합되지 않았는지 확인
- [ ] 제거 건수와 source별 내역 기록

정규화 완전 일치 기준은 공백·문장부호·대소문자를 제거한 문자열이다. 근사 중복 임계값은 기본
Jaccard 0.8이며 변경 시 근거를 기록한다.

## 5. 모델 단독 평가

각 데이터셋과 slice별로 다음을 보고한다.

- confusion matrix (`[[TN, FP], [FN, TP]]`)
- 위험 Recall/Precision/F1
- 정상 Recall = Specificity
- FPR = `FP / (TN + FP)`
- Macro-F1, PR-AUC
- threshold별 결과와 표본 수
- 가능하면 95% Wilson confidence interval

필수 slice:

- 일반 정상 대화
- warm-normal: 안부, 도움, 약속, 칭찬, 건강 걱정
- 그루밍/고립/비밀 유도
- 협박·폭력
- 금전·인증번호·선입금 사기
- 장애 관련 정상/위험 문장
- 성인/미성년 운영점

## 6. 모델+규칙 하이브리드 평가

같은 평가셋에서 아래 세 결과를 분리 보고한다.

1. 모델 단독
2. 규칙 단독
3. 모델 OR 규칙

규칙으로 추가된 TP와 FP를 각각 기록한다. 소규모 회귀셋에서만 좋아진 규칙은 배포하지 않고,
대규모 정상 holdout에서 FPR 증가를 확인한다.

## 7. 기본 승인 게이트

프로젝트 요구가 바뀌면 학습 전에 수치와 근거를 갱신한다.

| 항목 | 기본 기준 |
|---|---:|
| Blind real 위험 Recall | ≥ 0.80 |
| Blind warm-normal Specificity | ≥ 0.90 |
| Blind warm-normal FPR | ≤ 0.10 |
| 그루밍 Recall | ≥ 0.80 |
| train↔holdout 완전 중복 | 0 |
| 필수 artifact 누락 | 0 |

표본이 작은 회귀셋은 점 추정치뿐 아니라 confidence interval을 함께 보고한다. 운영 위험에 따라
Recall과 FPR 기준을 더 엄격하게 설정할 수 있다.

## 8. Threshold 선택 절차

1. Calibration set에서 후보 threshold를 sweep한다.
2. 사전에 정한 Recall·Specificity·slice 기준을 모두 만족하는 범위만 남긴다.
3. 그중 FPR이 가장 낮은 운영점을 선택한다.
4. 성인과 미성년 threshold를 별도로 기록한다.
5. 선택된 threshold를 고정한 뒤 blind test를 한 번 실행한다.
6. blind test 실패 시 threshold를 다시 맞추지 않고 모델 개발 단계로 되돌아간다.

## 9. 서빙 전 검증

- [ ] HF repo와 commit SHA 기록
- [ ] 추론 파일만 업로드되고 `checkpoint-*`, optimizer 등 학습 상태 제외
- [ ] `SAFE_MODEL_REVISION`으로 commit SHA 고정
- [ ] 성인/미성년 threshold 환경값 고정
- [ ] `/health`에서 revision, `num_labels`, labels, threshold 확인
- [ ] `/analyze` 정상·위험 smoke test
- [ ] 모델 로딩 실패/timeout/응답 계약 테스트
- [ ] 이전 승인 모델로 rollback 절차 확인

## 10. 완료 보고 형식

```text
모델/HF revision:
Git/config SHA:
학습 분포 및 source:
누수 검사 결과:
Calibration threshold:
Blind test confusion matrix:
위험 Recall / 정상 Specificity / FPR / Macro-F1:
slice별 결과:
모델 단독 vs 규칙 결합:
artifact 및 /health 확인:
판정: REJECTED | STAGING ONLY | PRODUCTION APPROVED
잔여 위험과 다음 액션:
```

---

## 부록 A. 2026-07-08 후보 모델 검토 기록

### 모델

- HF commit: `48048bdb6c883560fa3c3fab18e8fa3869f01f72`
- 학습: AI-Hub in-domain 포함, PAN12 제외, seed=42
- train: 68,074건 (`정상 25,459 / 주의 42,615`)
- 정규화 완전 일치 train↔holdout: 0
- 운영 후보: 성인 τ=0.66, 미성년 τ=0.50

### 실데이터 calibration 결과

| 운영점 | 혼동행렬 | 위험 Recall | Specificity | FPR | Macro-F1 |
|---|---|---:|---:|---:|---:|
| 성인 0.66 | `[[1790,320],[503,2878]]` | 0.851 | 0.848 | 0.152 | 0.844 |
| 미성년 0.50 | `[[1668,442],[387,2994]]` | 0.886 | 0.791 | 0.209 | 0.840 |

성인 Recall 95% Wilson CI는 약 `[0.839, 0.863]`, Specificity CI는 약
`[0.832, 0.863]`이다.

### 고정 회귀셋 결과

- warm-normal: 28/30 safe, Specificity 0.933, FPR 0.067
- 위험 대조: 15/18 flagged, Recall 0.833
- 3a 합성 그루밍: 78/80 flagged, Recall 0.975
- 모델+기존 규칙은 실데이터에서 TP 1건과 FP 1건을 각각 추가해 전체 결과 변화가 거의 없었다.

회귀셋 표본이 작아 warm-normal Specificity 95% CI는 약 `[0.787, 0.982]`, 위험 Recall CI는
약 `[0.608, 0.942]`로 넓다.

### 판정

**STAGING ONLY**

이유:

1. 실데이터 5,491건을 보고 τ=0.66을 선택했으므로 해당 데이터는 calibration set이다.
2. warm-normal/위험 회귀셋도 모델 선택에 사용돼 blind test가 아니다.
3. 최종 merged train에 대한 근사 중복 감사 결과가 별도 기록되지 않았다.

프로덕션 승인 전 다음이 필요하다.

- 현재 threshold와 규칙을 고정
- 학습·calibration에 사용하지 않은 새 blind real/warm-normal test 구성
- 최종 merged train↔blind test 근사 중복 0 확인
- blind test를 한 번 실행해 기본 승인 게이트 판정
