# ThisAbled AI

> 장애인 소셜 안전·매칭을 위한 한국어 NLP 예측 모델링 프로젝트

**진행 상태**: 🎉 **모델링 완료** → 🚀 **서빙·백엔드 연동 단계** (한이음 중간평가 7/14)

---

## 🎯 프로젝트 개요

ThisAbled AI는 장애인 커뮤니티의 소통 안전성과 사용자 간 호환성을 예측하기 위한 두 개의 머신러닝 모듈을 개발합니다.

### 모듈 ① — 한국어 텍스트 위험도 4단계 분류
- **입력**: 한국어 자연어 텍스트 (메시지, 게시물 등)
- **출력**: 4단계 라벨 `정상(0) / 주의(1) / 경고(2) / 긴급(3)`
- **모델**: KcELECTRA fine-tuning + Focal Loss + LightGBM Stacking
- **데이터**: Smilegate Unsmile, KOLD (시드) + GPT-4o 합성 (메인)
- **서빙**: KcELECTRA 단독(`module1_ce`) + 금전 사기 규칙 보조 레이어 — `serving/README.md` 참조
  (스태커는 학습 데이터 전용 `source` 피처 의존으로 서빙 제외)

### 모듈 ② — 사용자 호환성 매칭
- **입력**: 사용자 프로필 쌍 (텍스트 + 메타데이터)
- **출력**: 호환성 점수 (랭킹)
- **모델**: `ko-sroberta-multitask` 임베딩 + LightGBM Ranker
- **서빙**: 임베딩 특성(f_cosine·f_l2·f_dis_match) 실시간 계산 → LambdaMART 점수를
  태그 교집합·연령대 일치와 가중 블렌드(0.5/0.3/0.2) + 일반화 추천 사유
  (학습↔서빙 입력 스키마 드리프트 보완 — `serving/README.md` 설계 메모 참조)

---

## 🛠 환경

| 항목 | 값 |
|---|---|
| OS | macOS (개발) |
| 학습 | Google Colab Pro (A100) |
| Python | 3.11 |
| 패키지 매니저 | `uv` |
| 저장소 | GitHub private + Google Drive (대용량) |

---

## 🚀 빠른 시작

### 1) 가상환경 생성
```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 2) 의존성 설치
```bash
# 로컬 개발 (CPU)
uv pip install -r requirements.txt

# 서빙 서버 실행 시 (fastapi·uvicorn 등 추가)
uv pip install -r serving/safety_server/requirements.txt -r serving/match_server/requirements.txt

# Colab 환경에서는
# !pip install -r requirements-colab.txt
```

### 3) 환경변수 설정
```bash
cp .env.example .env
# .env 를 열어 API 키 등 실제 값을 채우세요
```

### 4) pre-commit 훅 설치 (선택)
```bash
pre-commit install
```

---

## 📁 디렉터리 구조

```
thisabled-ai/
├── .github/workflows/      # CI/CD
├── configs/                # YAML 학습/평가 설정
├── data/
│   ├── raw/                # 외부 시드 데이터 (git ignored)
│   ├── synthetic/          # GPT-4o 합성 데이터 (git ignored)
│   └── processed/          # 전처리 완료 (git ignored)
├── models/checkpoints/     # 학습 체크포인트 (git ignored — 최종본은 Drive, 아래 '모델 아티팩트' 참조)
├── notebooks/              # 탐색/리포트 노트북
├── serving/                # ⭐ SAFE·MATCH 모델 서빙 서버 (백엔드 mock 교체용) — serving/README.md
│   ├── safety_server/      # POST /analyze — KcELECTRA + 규칙 하이브리드 (:9001)
│   ├── match_server/       # POST /score — SBERT+LambdaMART (:9002)
│   └── smoke_test.py       # 계약·지연 검증
├── docs/                   # 시연 시나리오 등 프로젝트 문서
├── src/
│   ├── data/               # 데이터 로딩·전처리
│   ├── models/             # 모델 정의
│   ├── training/           # 학습 루프
│   ├── evaluation/         # 평가·메트릭
│   └── utils/              # 공용 유틸
├── tests/                  # pytest
├── scripts/                # 일회성 실행 스크립트
└── reports/
    ├── figures/            # 그래프·이미지
    └── validation_reports/ # 검증 리포트
```

---

## 📅 일정

기획된 7주(Week 8~14) 일정을 성공적으로 완수하였습니다.

| 단계 | 달성 항목 |
|---|---|
| **1** | 환경 셋업, 데이터 파이프라인 (UnSmile+KOLD) 라벨 매핑 |
| **2** | 모듈 ① KcELECTRA 베이스라인 학습, 합성 데이터 정책 수립 |
| **3** | 모듈 ① Stacking 메타 학습기, 합성 데이터 Train 병합, 모듈 ② 시작 |
| **4** | 모듈 ② LambdaMART 학습, SHAP XAI, Fairlearn 공정성 평가 스크립트 |
| **5** | 최종 리포팅 (README.md, final.md), 결과물 산출 |
| **6 (연장)** | 한이음 연동: `serving/` 서빙 서버 2종, 백엔드 compose mock 교체, 시연 시나리오 |

---

## 🏆 최종 성과 요약 (실측치)

### 모듈 ① 이진(정상/주의) — **실데이터 홀드아웃 5,491건 (최신, PAN12 반영)**

| 지표 | 기준 | 성인 τ=0.5 | 미성년 τ=0.35 | 평가 |
|:---|:---:|:---:|:---:|:---:|
| **주의 Recall** (미탐 최소화) | ≥ 0.80 | **0.859** | **0.899** | ✅ |
| **정상 Precision** (과블러 방지) | ≥ 0.60 | **0.752** | **0.780** | ✅ |
| **Macro-F1** | — | **0.775** | — | ✅ |

- 학습: 시드+합성+AI-Hub 이진 붕괴({1,2,3}→주의) + **PAN12 predator 190건 병합**
  (train 분포 {정상 19941, 주의 28386}). best=epoch1(eval_loss 0.322). seed=42.
- 혼동행렬(τ=0.5, 행=정답·열=예측): `[[1440, 670], [476, 2905]]`
- **실데이터 홀드아웃 첫 시도 통과** — 추가 튜닝 없음. HF: soyuncj/thisabled-safety-kcelectra.

### 이전 4-class·모듈② (연구 단계 이력)

| 항목 | 1차 목표 | Stretch | **실측** | 평가 |
|:---|:---:|:---:|:---:|:---:|
| 모듈 ① Macro-F1 (시드 test, 4-class) | ≥ 0.60 | ≥ 0.68 | **0.7643** | ✅ stretch |
| 모듈 ① 긴급(3) Recall (합성 hold-out) | ≥ 0.75 | — | **1.0000** | ⚠ template-circular |
| 모듈 ① UnSmile/KOLD 격차 | ≤ 0.10 | — | **0.0586** | ✅ |
| **모듈 ② NDCG@10** (cold-start, embedding) | ≥ 0.60 | ≥ 0.70 | **0.9070** | ✅ stretch |

⚠ **honest disclaimers**:
- (4-class) 긴급(3) Recall 1.0 = 합성 학습 + 합성 평가 = 동일 템플릿 풀 → 실 일반화 입증 아님.
  이진 전환 + PAN12 실그루밍 반영으로 **실데이터 주의 Recall 0.859 달성**이 이 한계의 해소 결과.
- 모듈 ② 0.907 = 룰 기반 ground truth 복원 측정 (실제 사용자 만족도 별개)
- 장애 도메인 공정성 측정 불가 (test n=28, 통계 신뢰 임계 미달)
- 금전 사기 유형(SAFE-02 ①)은 시드가 혐오표현 중심이라 커버리지 약함
  (서빙 스모크에서 사기 문장 risk_prob 0.07 실측) → 서빙 단계에서 규칙 보조 레이어로
  보완, 근본 해결은 사기 유형 합성 증강 + 재학습(후속)
- 모듈 ② 학습 입력(템플릿 자기소개: 지역·나이·장애유형 명시)과 서빙 계약 입력
  (bio·tags·age_band·ui_mode)의 스키마 드리프트 → 서빙에서 태그·연령 신호 가중 블렌드로
  보완, 근본 해결은 계약 스키마 프로필 합성 재생성 + 재학습(후속)

---

## 🔌 서빙 (한이음 백엔드 연동)

백엔드(`thisabled-backend`)의 mock 모델 컨테이너를 동일 계약으로 교체하는 실모델 서버.
실행·Docker 교체·운영 설정값은 **`serving/README.md`** 참조.

| 서버 | 계약 | 스모크 실측 (M2 CPU) |
|---|---|---|
| safety_server :9001 | `POST /analyze` → `{verdict}` | median **48ms** (백엔드 예산 2s, 판정 4/4 일치) |
| match_server :9002 | `POST /score` → `{results}` | 정상 상태 **364ms** (예산 10s) |

### 모델 아티팩트 (서빙 최종본)

`models/checkpoints/`는 git ignored — 최종본 배포 채널은 **Hugging Face private repo**
(업로드·연동 절차는 `serving/README.md`). Docker 서빙은 기동 시 HF에서 자동 다운로드하므로
빌드 머신에 파일이 없어도 된다. Drive `ThisAbled/checkpoints/`의 `module1_baseline_*`은
구버전 베이스라인 — 서빙에 사용 금지.

| 아티팩트 | 용도 | 필요 파일 |
|---|---|---|
| `module1_ce/` | SAFE 서빙 | config.json, model.safetensors, tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab.txt (optimizer.pt 등 학습 상태 불필요) |
| `module2_lambdamart_embedding.pkl` | MATCH 서빙 | 단일 파일 |

---

## 📝 라이선스 / 비고

본 저장소는 학기말 과제용 **private repo**입니다.
외부 데이터셋(Smilegate Unsmile, KOLD 등)은 각 출처 라이선스를 따릅니다 — `data/README.md` 참조.
