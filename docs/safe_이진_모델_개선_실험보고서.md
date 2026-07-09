# SAFE 이진 안전 분류기 개선 실험 보고서

> 대상 시스템: ThisAbled SAFE 모듈 — KcELECTRA 이진(정상/주의) 대화 안전 분류기
> 기간: 2026-07-09 / 작업 브랜치: `feature/grooming-augmentation`
> 최종 산출: v4_r1 모델 (HF revision `3e9c0b80…`), 배포 준비 완료

---

## 1. 배경

ThisAbled는 장애인 대상 서비스이며, SAFE 모듈은 사용자 간 대화에서 **사기·그루밍·강압적 통제·디지털 갈취·협박** 등 위험 발화를 탐지한다. 모델은 KcELECTRA 기반 **이진 분류기**(라벨 `["정상","주의"]`)로, 서빙은 FastAPI `/analyze`가 `verdict`(safe/flagged)를 반환한다. 미성년 수신자에는 더 민감한 임계값을 적용한다.

- 성인 임계값 `SAFE_FLAG_THRESHOLD`, 미성년 `SAFE_FLAG_THRESHOLD_MINOR`
- 규칙 보조 레이어(`SAFE_RULE_ASSIST`)는 정상 부정문을 오탐해 **비활성(OFF)** 운영

## 2. 문제 정의

두 축의 문제가 주어졌다.

- **(A) 안전한 배포**: 재학습된 이진 모델을 Git·Hugging Face·서빙 설정에 반영하되, 재현 가능하고 롤백 가능하게 한다.
- **(B) 지속적 개선 루프**: 모델의 취약 유형(어떤 위험 슬라이스를 놓치는가, 어떤 정상문을 오탐하는가)을 반복적으로 찾아 개선하되, **평가 누수 없이 정직하게** 성능을 측정한다.

핵심 제약(방법론 규칙):
1. 한 번 소비한 blind 세트는 최종 평가에 재사용하지 않는다.
2. blind 원문은 학습 데이터에 절대 병합하지 않는다.
3. 결과를 보고 임계값을 되맞춰 같은 blind로 재평가하지 않는다(라운드당 1회 소비).

## 3. 방법론

### 3.1 데이터·평가 파이프라인
```
시드(UnSmile/KOLD/AIHub/PAN12) + 합성 긴급 데이터 ──► 전처리·MinHash(0.8) 중복제거 ──► base train
hard-case base 문장 ──(prefix/suffix 결정론적 확장)──► 증강셋 ──► base train에 병합 학습
KcELECTRA fine-tune ──► 후보 체크포인트 ──► dev 게이트 ──► fresh blind 1회 소비
```

### 3.2 hard-case 증강
- 손으로 작성한 **base 문장**을 접두/접미 조합으로 결정론적 확장 → 정규화 dedup.
- 버전 누적: v5 ⊇ v4 ⊇ v3 ⊇ v2 ⊇ v1.
- 라운드마다 **소비된 blind 전부**를 `--forbidden`으로 넘겨 exact+near-dup(0.8) 누수를 빌드 시점에 차단.

### 3.3 fresh blind 방법론 (정직 평가의 핵심)
- 라운드마다 **새 blind**를 40행(정상 20/위험 20, 슬라이스 10×4)으로 작성.
- 후보 모델 고정 → **정확히 1회** 평가 → 평가 전후 **SHA-256**으로 원문 불변 검증.
- 소비된 blind는 다음 라운드부터 **dev 회귀셋**으로 격하.
- **한계(명시)**: blind도 저자 생성 합성셋이라 *패턴 일반화*를 측정하며 완전 독립 실데이터가 아니다. 독립 실데이터 회귀는 AIHub/BEEP holdout이 담당한다.

### 3.4 dev 게이트 (후보 채택 기준)
실데이터 holdout + 소비된 blind로 threshold 스윕(성인 0.40~0.85, 미성년=성인−0.16, 하한 0.35) 후, 아래를 모두 만족하는 지점만 통과:
- 실데이터 recall ≥ 0.80, specificity ≥ 0.80
- dev recall ≥ 0.85, specificity ≥ 0.90
- 타깃 위험 슬라이스 recall ≥ 0.80
- 규칙 보조 **OFF**(운영과 동일)

repeat 1→3 학습 중 게이트를 통과하면 즉시 중단.

## 4. 실험 경과

### 4.1 출발점 — 배포 모델 v2_r1의 약점 (blind v4)
직전 프로덕션 후보 `module1_binary_hardcases_v2_r1`(HF `79bbd16`, 0.85/0.69, 규칙 OFF)을 fresh blind v4로 평가한 결과가 개선 루프의 출발점이 됐다.

| 지표 | 값 |
| --- | --- |
| risk_recall / specificity | 0.90 / 0.95 |
| macro-F1 | 0.925 |
| **fraud_credentials** | **0.75** ← 약점 |
| **coercive_control** | **0.75** ← 약점 |
| routine_recon / digital_extortion / grooming | 1.0 / 1.0 / 1.0 |

대표 오판: OTP 구어체 사기 FN(p=0.19), 복장 통제 FN(p=0.11), 정당한 삭제요청 FP(p=0.98).

### 4.2 라운드별 결과

각 라운드는 직전 fresh blind에서 드러난 실패를 정조준해 hard-case를 추가하고, **새 fresh blind**로 검증했다.

| 라운드 | 타깃 | 증강 | 모델 | fresh blind | 결과 |
| --- | --- | --- | --- | --- | --- |
| **v3** | fraud·coercive 0.75, 삭제/환급 FP | hardcase v3 (1,744행) | v3_r1 @0.85/0.69 | **v5** | 통과. risk_recall 0.90, macro-F1 0.925. **fraud 0.75→1.0 해결**. coercive·routine_recon은 fresh에서 여전 0.75, 접근성 FP(p=0.93) 노출 |
| **v4** | 접근성 FP, 기기·통신 감시형 coercive, 혼자있는시간 recon | hardcase v4 (2,000행) | v4_r1 @0.85/0.69 | **v6** | **통과(최고).** risk_recall **1.0**, macro-F1 **0.975**, 전 위험슬라이스 1.0, 접근성 FP **0**, coercive·recon **0.75→1.0**. 잔여 FP 1건(정서지지문 p=0.997) |
| **v5** | 정서지지문 warm_normal FP | hardcase v5 (2,160행) | v5_r1 @**0.75/0.59** | **v7** | **실패(회귀).** risk_recall 0.90, macro-F1 0.90, **routine_recon 0.5로 붕괴**, warm FP 못 고침(여전 1건)+안전안내문 FP 추가, 운영점 하락 |

### 4.3 핵심 관찰 — dev 개선 ≠ 일반화
매 라운드 dev 회귀셋(소비된 blind 포함)에서는 타깃 슬라이스가 잘 올랐으나, 그 상승의 상당 부분은 **패턴 암기**였다. 실제 판정은 항상 **처음 보는 fresh blind**에서 갈렸다.
- v3: dev coercive 0.95 ↔ fresh(v5) 0.75
- v5: dev warm_specificity 0.955 ↔ fresh(v7)에서 warm FP 미해결 + routine 붕괴

이 격차가 "fresh blind 1회 소비" 방법론의 존재 이유를 실증한다.

## 5. 결과 분석

- **v3 → v4**로 fraud_credentials·coercive_control·routine_recon·접근성 오탐이 fresh 기준으로 순차 해결되어, v4_r1이 위험 재현율 1.0에 도달했다.
- **v5는 음의 수익**이었다. 정서지지 정상문 오탐(1건)을 잡으려 지지문·하드네거티브를 늘리자 모델이 전반적으로 둔감해져 운영점이 0.75/0.59로 내려가고 routine_recon이 0.5로 붕괴했다. 목표한 FP도 못 고쳤다.
- 따라서 **v4_r1으로 수렴**하고 반복 루프를 종료했다. FP 폴리싱이 한계 효용에 도달한 지점을 fresh blind가 명확히 신호했다.

## 6. 최종 배포

- **모델**: `module1_binary_hardcases_v4_r1`
- **HF revision**: `3e9c0b800661db9ce099782a76fbe181e8b23ab5` (main HEAD, 추론 파일 7개만, 학습상태 제외)
- **model.safetensors sha256**: `1088c558…7e76`
- **무결성 검증**: 업로드 revision을 로컬에서 blind v6로 재평가 → CM `[[19,1],[0,20]]`, risk_recall 1.0, macro-F1 0.975 **완전 재현**(= 검증된 v4_r1과 바이트 충실)

서빙 설정:
```
SAFE_MODEL_DIR=soyuncj/thisabled-safety-kcelectra
SAFE_MODEL_REVISION=3e9c0b800661db9ce099782a76fbe181e8b23ab5
SAFE_FLAG_THRESHOLD=0.85
SAFE_FLAG_THRESHOLD_MINOR=0.69
SAFE_RULE_ASSIST=0
```
`/health`가 로드된 커밋을 `revision`으로 보고하므로 배포 후 이 값이 위 SHA와 일치하는지로 확인한다. (직전 프로덕션 `79bbd16`=v2_r1 → 서빙 갱신·재시작 후 라이브 전환.)

## 7. 부수 인프라 개선

실험을 견고하게 만들기 위한 코드 개선(모두 검증·커밋).

| 개선 | 내용 | 커밋 |
| --- | --- | --- |
| 규칙 보조 OFF | 정상 부정문 오탐하던 규칙 레이어 기본 비활성 | `81bb2b2` |
| revision 고정/보고 | `SAFE_MODEL_REVISION` 소비 → `from_pretrained(revision=)` 전달, `/health`가 실제 로드 커밋 보고 | `7de6efb` |
| 시드 다운로드 재시도 | GitHub 429(Colab 공유 IP)에 지수 백오프+Retry-After+UA로 대응 | `39482c6` |

## 8. 한계 및 향후 과제

- **잔여 오탐(허용 범위)**: v4_r1은 일부 1:1 정서지지문과 특정 접근성 어법("큰 글씨 유인물")을 아직 과플래그한다. 접근성은 제품 핵심 시나리오라 향후 우선 개선 대상이나, v5 시도가 오히려 위험 재현율을 훼손해 이번엔 보류했다.
- **평가셋의 독립성**: blind v1~v7은 저자 생성 합성셋으로 패턴 일반화를 측정한다. 완전 독립 검증은 실서비스 로그 기반 holdout 확보가 필요하다.
- **재개 방법**: blind v1~v7은 모두 소비되어 dev 회귀셋이다. 다음 개선은 **fresh blind v8**부터 시작한다. 절차와 템플릿은 [safe_이진_재학습_평가_파이프라인.md](safe_이진_재학습_평가_파이프라인.md), 노트북 08~11 참고.

## 부록 A. 산출물

| 종류 | 경로 |
| --- | --- |
| 증강 빌더 | `scripts/build_safe_hardcase_dataset.py` (`--include-v3/v4/v5`) |
| 학습 config | `configs/module1_binary_hardcases_v{3,4,5}.yaml` |
| 반복 노트북 | `notebooks/09~11_iterative_hardcase_retrain_v{3,4,5}.ipynb` (v2 라운드는 08) |
| fresh blind | `tests/fixtures/safe_blind_v{5,6,7}.jsonl` |
| 파이프라인 문서 | `docs/safe_이진_재학습_평가_파이프라인.md` |

## 부록 B. fresh blind SHA-256 (무결성 앵커)

| blind | sha256 | 소비 라운드 |
| --- | --- | --- |
| v4 | `1455462c…d74226` | v2_r1 평가(출발점) |
| v5 | `904c7178…57a3e` | v3 |
| v6 | `1ac7672e…ac4b6` | v4 |
| v7 | `211117dc…19fc5` | v5 |

## 부록 C. 라운드 요약 지표 (fresh blind 기준)

| | v4(baseline) | v5(v3) | v6(v4) | v7(v5) |
| --- | --- | --- | --- | --- |
| 운영점 | 0.85/0.69 | 0.85/0.69 | 0.85/0.69 | 0.75/0.59 |
| risk_recall | 0.90 | 0.90 | **1.0** | 0.90 |
| specificity | 0.95 | 0.95 | 0.95 | 0.90 |
| macro-F1 | 0.925 | 0.925 | **0.975** | 0.90 |
| fraud_credentials | 0.75 | 1.0 | 1.0 | 1.0 |
| coercive_control | 0.75 | 0.75 | 1.0 | 1.0 |
| routine_recon | 1.0 | 0.75 | 1.0 | **0.5** |
| 접근성/정서지지 FP | — | 접근성 1 | 정서지지 1 | 정서지지+안내 2 |

→ 채택: **v6(v4_r1)**.
