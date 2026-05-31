# D1 진행 보고서 — ThisAbled AI

**작성일**: 2026-05-30 (D1 종료) · **commit**: `6000fc1` · **잔여 일정**: D2~D5 (4일)

---

## 1. 컨텍스트

본 프로젝트는 원래 7주(Week 8~14) 일정으로 설계되었으나, **실가용 기간이 5일(2026-05-30 ~ 06-03)** 로 압축되었다. 본 보고서는 D1 종료 시점의 상태와, 압축으로 인해 조정된 목표·범위를 정리한다.

전체 일정·일자별 목표는 §10에, 압축으로 격하·삭제된 항목은 §11에 명시한다.

---

## 2. 프로젝트 목표 (5일 압축판)

### 모듈 ① — 한국어 텍스트 4단계 위험도 분류
- **입력**: 한국어 자연어 텍스트 (메시지·게시물)
- **출력**: `정상(0) / 주의(1) / 경고(2) / 긴급(3)`
- **방법**: KcELECTRA fine-tuning + Focal Loss + LightGBM Stacking
- **목표**: Macro-F1 ≥ **0.60** (1차) / **0.68** (stretch). 긴급(3) Recall ≥ 0.75.

### 모듈 ② — 사용자 호환성 매칭
- **입력**: 사용자 프로필 쌍 (텍스트 + 메타데이터)
- **출력**: 호환성 점수 (랭킹)
- **방법**: `ko-sroberta-multitask` 임베딩 + LightGBM Ranker
- **목표**: NDCG@10 ≥ **0.60** (파이프라인 동작 증명 수준)

### 공통
- 재현 가능한 학습·평가 파이프라인 (config 한 줄 변경으로 백본 교체)
- SHAP 기반 XAI 리포트 (모듈 ① 오분류 Top-10 케이스)
- Fairlearn 기반 공정성 표 (타겟 집단별 F1 격차 측정)

---

## 3. D1 완료 항목

| # | 항목 | 산출물 | 상태 |
|---|---|---|---|
| 1 | UnSmile + KOLD 시드 다운로드 (재실행 안전, sha256 검증) | [scripts/download_seed_datasets.py](../scripts/download_seed_datasets.py) | ✅ |
| 2 | EDA 노트북 + figure 4종 | [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb), [reports/figures/eda/](figures/eda/) | ✅ |
| 3 | 4단계 라벨 매핑 정책 문서화 | [docs/label_mapping.md](../docs/label_mapping.md) | ✅ |
| 4 | 라벨 매핑 구현 + 단위·통합 테스트 | [src/data/label_mapping.py](../src/data/label_mapping.py), [tests/test_label_mapping.py](../tests/test_label_mapping.py) | ✅ |
| 5 | Stratified 80/10/10 split → parquet | [scripts/build_processed_dataset.py](../scripts/build_processed_dataset.py), `data/processed/{train,val,test}.parquet` | ✅ |
| 6 | KcELECTRA + Focal Loss Trainer 골격 | [src/training/trainer.py](../src/training/trainer.py) | ✅ (실행은 D2 Colab) |

**커밋**: `6000fc1` — 55 files, +3,014 lines insertions.

---

## 4. 데이터

### 4.1 시드 출처·라이선스
| 출처 | 용도 | 라이선스 | URL |
|---|---|---|---|
| Smilegate UnSmile | 시드 (한국어 혐오 댓글) | 출처 정책 (비상업·연구) | [github.com/smilegate-ai/korean_unsmile_dataset](https://github.com/smilegate-ai/korean_unsmile_dataset) |
| KOLD | 시드 (Korean Offensive Language) | 연구용 | [github.com/boychaboy/KOLD](https://github.com/boychaboy/KOLD) |

### 4.2 무결성·재현성
- **KOLD**는 Git LFS (32.79 MB). `media.githubusercontent.com/media/...` 경로 + **sha256 `c11c29b9...` 검증**.
- 다운로드 스크립트는 stdlib만 사용 → 추가 의존성 없음, Colab 호환.
- 이미 존재하면 skip (해시 일치 시) → 재실행 안전.

### 4.3 통계
| 데이터 | n | char p50 / p99 / max |
|---|---:|:---:|
| UnSmile train | 15,005 | 31 / 137 / 155 |
| UnSmile valid | 3,737  | 30 / 136 / 149 |
| KOLD comment  | 40,429 | 36 / 142 / 203 |

→ **`max_length=128` 유지** (WordPiece 변환 후 p99 이내).

### 4.4 통합 데이터셋 (라벨 매핑 후)
| split | n | 정상 | 주의 | 경고 | 긴급 |
|---|---:|---:|---:|---:|---:|
| train | 47,336 | 19,836 | 9,460 | 18,040 | 0 |
| val | 5,917 | 2,479 | 1,183 | 2,255 | 0 |
| test | 5,918 | 2,480 | 1,183 | 2,255 | 0 |
| **합계** | **59,171** | 24,795 (41.9%) | 11,826 (20.0%) | 22,550 (38.1%) | **0 (0%)** |

**긴급(3) 0건은 매핑 오류가 아닌 시드 데이터의 본질적 한계** — §5.1·§7.L1 참조.

---

## 5. 라벨 매핑 정책 — 핵심 결정 4개

전문은 [docs/label_mapping.md](../docs/label_mapping.md). 본 절은 *왜 그렇게 결정했는가* 만 요약.

### 5.1 결정 ① — 긴급(3)은 다른 축
- `0/1/2`는 **한 문장의 혐오 강도** 축, `3 (긴급)`은 **그루밍·유인·협박·자해 신호 등 행위 기반 위험** 축.
- 시드(UnSmile, KOLD)는 단발 혐오 댓글 데이터셋이라 긴급 축을 **애초에 안 담고 있다** → 0건은 당연.
- **결과**: D2 합성은 "더 심한 욕설"이 아니라 **"그루밍 시나리오 정의서"** 부터 만들어야 함.

### 5.2 결정 ② — KOLD TGT=other (n=1,402) → 주의(1) 흡수
- 40건 샘플 검증: 대부분 언론·정부·기관에 대한 분노/조롱/짜증. 보호집단 직접 차별은 0~2건 (≤5%).
- **결과**: 라벨 노이즈 수용 가능 수준으로 판단, 흡수.

### 5.3 결정 ③ — KOLD TGT=individual (n=3,899) → 주의(1) 유지 + 후처리 분리
- 학습 라벨은 "혐오 강도" 축에 충실하게 주의(1) 유지.
- 1:1 채팅 시나리오에서는 **제품 레이어 후처리 룰**로 경고(2)로 격상.
- **이유**: 모델은 단순하고 재사용 가능하게, 제품 정책은 컨텍스트 의존. 분리하면 모델 재학습 없이 정책 변경 가능.

### 5.4 결정 ④ — UnSmile·KOLD의 경고(2) 정의 이질성 인정
- UnSmile은 "혐오 라벨 존재 여부", KOLD는 "타겟이 group인지"로 경고를 정의 → 같은 라벨이지만 경계가 다름.
- **정책**: 멀티라벨 → max severity collapse. 두 데이터셋 정의 병기.
- **결과**: 평가 시 **데이터셋별 F1 분리 보고** 필요 (통합 F1만 보면 정의 이질성이 가려짐).

---

## 6. EDA 핵심 발견

| 발견 | 의사결정 |
|---|---|
| 두 데이터 모두 char p99 ≤ 142 | `max_length=128` 유지 |
| UnSmile clean 25%, KOLD OFF=False 50% → 통합 정상 비율 42% | 클래스 가중치 가벼운 조정 충분 (0/1/2 한정) |
| 멀티라벨 중 2개 이상: 994건 / 3+개: 89건 | max severity collapse로 흡수 |
| **'장애' 키워드**: UnSmile 86건, KOLD comment 141건, **KOLD GRP에 '장애-' 카테고리 0건** | D2 합성은 전 클래스의 장애 도메인 보강으로 설계 |
| 긴급(3) 매핑 결과 시드에 0건 | 100% 합성 의존 → §7.L1 |

---

## 7. 알려진 제약 (Known Limitations) — **본 보고서의 가장 중요한 절**

본 매핑 정책·시드 특성으로부터 **본 프로젝트가 측정·주장할 수 없는 것들**.

### L1. 긴급(3) 성능 측정 불가
- 시드 0건 → train·eval **둘 다 합성**으로 채워야 함.
- 합성으로 학습하고 합성으로 평가하면 모델이 실제 그루밍을 얼마나 잡는지 **숫자로 증명할 수 없다**.
- **완화**:
  - (a) 합성 다양화 (페르소나·시나리오·언어 스타일) + 외부 키워드/룰 백스톱 → **"AI 단독"이 아니라 "AI + 룰 안전망" 프레이밍**.
  - (b) 리포트에서 긴급 클래스 성능은 "합성 hold-out 기준"임을 명시, 일반화 주장 자제.
  - (c) 실제 그루밍 데이터 확보는 본 학기 범위 밖 — 후속 연구 항목.

### L2. "심한 불균형 X" 결론은 0/1/2에 한정
- 통합 분포 42:20:38:0 — 0/1/2 간에는 가벼운 가중치로 충분.
- 긴급(3)은 합성량 결정 전까지는 극소수 클래스 → Focal Loss `alpha`만으로는 부족할 수 있음.
- **완화**: D2에서 합성 규모 결정 후 oversampling 또는 `alpha[3]` 별도 튜닝.

### L3. 장애 도메인 시드 희박
- '장애' 포함 UnSmile 86건, KOLD 141건, KOLD GRP 카테고리에 0건.
- 시드만으로는 도메인 적응 불충분.
- **완화**: D2 합성은 긴급(3)만이 아니라 **전 클래스(0/1/2/3)의 장애 도메인 보강**으로 설계.

### L4. 경고(2) 정의 이질성
- §5.4 참조.
- **완화**: 평가 시 UnSmile only / KOLD only F1 분리 보고. 격차가 크면 정의 이질성이 노이즈 증거.

---

## 8. 아키텍처·구현

### 8.1 모델 선택
- **백본**: `beomi/KcELECTRA-base-v2022`
  - KcBERT 대비 한국어 혐오 분류에서 +2~3 F1 (UnSmile/KOLD 벤치마크 보고치).
  - HuggingFace `AutoTokenizer/AutoModelForSequenceClassification` API 동일 → 백본 교체 비용 0.
  - 파라미터·VRAM 비슷 (~110M), max 학습 길이 512 (우리는 128 사용).

### 8.2 파이프라인
```
data/raw/{unsmile,kold}/        (시드)
        ↓ build_unified_dataset
[text, label, source, source_id] (n=59,171)
        ↓ stratified_three_way_split (seed=42)
data/processed/{train,val,test}.parquet
        ↓ RiskTextDataset + tokenizer
HuggingFace Trainer (FocalLossTrainer subclass)
        ↓
models/checkpoints/module1/   + reports/validation_reports/module1/
```

### 8.3 Focal Loss 위치
- `src/models/focal_loss.py` (`gamma=2.0` default, optional `alpha` 가중치)
- `src/training/trainer.py::FocalLossTrainer.compute_loss` 에서 표준 Cross-Entropy 대신 사용.

### 8.4 코드 규모
- 24 Python 파일 (src + tests + scripts) — 535 LOC for 핵심 모듈만.
- pre-commit (ruff + ruff-format) 적용, 모든 hook 통과.

---

## 9. 품질 보증

### 9.1 테스트 결과
**34 passed in 5.03s** — 카테고리별:

| 카테고리 | 파일 | 건수 |
|---|---|---:|
| 라벨 매핑 규칙 | `tests/test_label_mapping.py` | 17 |
| 벡터화 / per-row 일치 | (같은 파일) | 2 |
| 실데이터 분포 회귀 방지 | (같은 파일) | 3 |
| Split (크기·disjoint·분포·결정론·invalid ratio) | `tests/test_build_dataset.py` | 5 |
| Trainer 골격 import·config·dataset | `tests/test_trainer_skeleton.py` | 4 |
| 기존 (Focal Loss, metrics, seed) | `tests/test_focal_loss.py` 외 2 | 8 |
| **합계** | | **34** |

### 9.2 재현성 보장
- 모든 RNG seed=42 (`src/utils/seed.py`)
- 데이터 다운로드 sha256 검증 (KOLD LFS)
- Split 결정론 (`test_split_is_deterministic` 검증)
- 모든 산출물 경로 명시 (config + 보고서)

---

## 10. D2~D5 계획

### D2 — 5/31 (일) | 모듈 ① 베이스라인
- Colab A100 환경 세팅 (private repo clone, requirements-colab 설치, raw·processed 재생성)
- **긴급(3) 시나리오 정의서** 먼저 작성 (`docs/emergency_scenarios.md`)
- KcELECTRA fine-tuning 베이스라인 학습 (3~5 epoch)
- Focal Loss + class weight 적용·비교
- 첫 평가 → `reports/validation_reports/module1/baseline.md`
- *여유 시*: 소규모 GPT-4o 합성 (라벨당 200~500개, 비용 캡 $5)

🎯 **D2 종료 기준**: Macro-F1 ≥ 0.55, confusion matrix 확보

### D3 — 6/1 (월) | 모듈 ① 본 학습 + 모듈 ② 시작
- LightGBM Stacking 메타 학습기
- 하이퍼파라미터: Optuna ~~20 trials~~ → 그리드 3~5 조합
- 모듈 ① 최종 학습 + 체크포인트 → Drive 백업
- 모듈 ② 데이터 스키마 설계 (사용자 프로필 페어, 룰 기반 페어 생성)

🎯 **D3 종료 기준**: 모듈 ① Macro-F1 ≥ 0.60, 모듈 ② 학습 데이터 형태 확정

### D4 — 6/2 (화) | 모듈 ② + XAI/공정성
- `ko-sroberta` 임베딩 + LightGBM Ranker
- 모듈 ② NDCG@10 측정
- SHAP 최소 분석 (오분류 Top-10)
- Fairlearn 최소 분석 (타겟 집단별 F1 격차 표)

🎯 **D4 종료 기준**: 모듈 ② 동작, XAI/공정성 결과 2종 이상

### D5 — 6/3 (수) | 최종 평가 + 리포트
- Hold-out 최종 평가 (모듈 ①·② 둘 다)
- 시각화 (`reports/figures/`)
- 최종 리포트 (`reports/final.md`)
- README 업데이트 + 최종 commit

🎯 **D5 종료 기준**: 제출 가능 리포트 + 재현 가능 코드

---

## 11. 압축으로 격하·삭제된 항목

| 원래 7주 계획 | 5일 압축판 조치 | 이유 |
|---|---|---|
| GPT-4o 대규모 합성 (수천~수만 건) | 라벨당 200~500개, 비용 캡 $5 | 검증·비용·시간 부족 |
| Claude 교차 라벨링 + Cohen's κ ≥ 0.6 | 삭제 | IAA 측정 시간 없음 |
| Optuna 20 trials | 그리드 3~5개 | 5일 안 들어감 |
| Fairlearn 개선 사이클 | 측정·리포트만 | 개선까지 시간 X |
| SHAP 전체 분석 | 오분류 Top-10만 | 동일 |
| 모듈 ② 합성 페어 학습 | 룰 기반 페어 + 파이프라인 동작 증명 | 동일 |

---

## 12. 리스크 및 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| 긴급(3) 시드 0건, 100% 합성 → 성능 측정 불가 | 보고서 클레임 불가 | "AI + 룰 안전망" 프레이밍 + 합성 다양화 |
| Colab A100 가용성 / 학습 시간 초과 | D2 학습 지연 | D2 시작 즉시 GPU 확보, fallback: T4 + batch size 축소 |
| KcELECTRA-base-v2022 토크나이저 미세 차이 | 베이스라인 재현성 영향 미미 | `from_pretrained` 픽스 → 영향 0 예상 |
| 합성 데이터 비용 초과 | D2 후반 작업 차질 | `$5` 캡 + tenacity 재시도, 실패 시 룰 기반 합성으로 대체 |
| 장애 도메인 합성 품질 저하 | 도메인 적응 실패 → 일반 혐오 분류기로 회귀 | 장애인 커뮤니티 페르소나 사전 정의 후 합성 |
| 모듈 ② 학습 시간 부족 | D4·D5 압박 | 모듈 ②는 "동작 증명" 수준 (NDCG@10 ≥ 0.60)에 한정 |

---

## 13. 산출물 인덱스 (재현·검증용)

### 코드
- [src/data/label_mapping.py](../src/data/label_mapping.py) — 4단계 매핑 함수 4종
- [src/data/build_dataset.py](../src/data/build_dataset.py) — Stratified split
- [src/data/loaders.py](../src/data/loaders.py) — parquet·HDF5 I/O
- [src/models/focal_loss.py](../src/models/focal_loss.py) — Focal Loss
- [src/training/dataset.py](../src/training/dataset.py) — Torch Dataset
- [src/training/trainer.py](../src/training/trainer.py) — FocalLossTrainer
- [src/evaluation/metrics.py](../src/evaluation/metrics.py) — Macro-F1, 긴급 Recall, AUC-PR
- [src/utils/seed.py](../src/utils/seed.py) — RNG 고정

### 설정
- [configs/module1_kcelectra.yaml](../configs/module1_kcelectra.yaml)
- [configs/module2_matching.yaml](../configs/module2_matching.yaml)

### 스크립트
- [scripts/download_seed_datasets.py](../scripts/download_seed_datasets.py)
- [scripts/build_processed_dataset.py](../scripts/build_processed_dataset.py)
- [scripts/train_module1.py](../scripts/train_module1.py)

### 문서·노트북
- [docs/label_mapping.md](../docs/label_mapping.md) — 정책 전문
- [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) — EDA 실행 결과 포함

### Figures
- [reports/figures/eda/text_length.png](figures/eda/text_length.png)
- [reports/figures/eda/unsmile_labels.png](figures/eda/unsmile_labels.png)
- [reports/figures/eda/kold_grp.png](figures/eda/kold_grp.png)
- [reports/figures/eda/label4_distribution.png](figures/eda/label4_distribution.png)

### 데이터 (gitignored, 재생성 가능)
- `data/raw/unsmile/unsmile_{train,valid}_v1.0.tsv`
- `data/raw/kold/kold_v1.json` (sha256: `c11c29b9...`)
- `data/processed/{train,val,test}.parquet`

### 테스트
- `tests/test_label_mapping.py` (17건)
- `tests/test_build_dataset.py` (5건)
- `tests/test_trainer_skeleton.py` (4건)
- `tests/test_focal_loss.py` (3건), `tests/test_metrics.py` (3건), `tests/test_seed.py` (2건)
