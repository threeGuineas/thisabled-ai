# ThisAbled AI

> 장애인 소셜 안전·매칭을 위한 한국어 NLP 예측 모델링 프로젝트

**진행 상태**: 🚧 **Week 8 진행 중** (총 7주 / Week 8 ~ Week 14)

---

## 🎯 프로젝트 개요

ThisAbled AI는 장애인 커뮤니티의 소통 안전성과 사용자 간 호환성을 예측하기 위한 두 개의 머신러닝 모듈을 개발합니다.

### 모듈 ① — 한국어 텍스트 위험도 4단계 분류
- **입력**: 한국어 자연어 텍스트 (메시지, 게시물 등)
- **출력**: 4단계 라벨 `정상(0) / 주의(1) / 경고(2) / 긴급(3)`
- **모델**: KcELECTRA fine-tuning + Focal Loss + LightGBM Stacking
- **데이터**: Smilegate Unsmile, KOLD (시드) + GPT-4o 합성 (메인)

### 모듈 ② — 사용자 호환성 매칭
- **입력**: 사용자 프로필 쌍 (텍스트 + 메타데이터)
- **출력**: 호환성 점수 (랭킹)
- **모델**: `ko-sroberta-multitask` 임베딩 + LightGBM Ranker

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
├── models/checkpoints/     # 학습 체크포인트 (git ignored)
├── notebooks/              # 탐색/리포트 노트북
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

## 📅 일정 (7주 계획)

| 주차 | 목표 |
|---|---|
| **Week 8** (현재) | 환경 셋업, 데이터 시드 수집 |
| Week 9–10 | 모듈 ① 데이터 합성 + 베이스라인 |
| Week 11–12 | 모듈 ① 본 학습, 모듈 ② 시작 |
| Week 13 | 모듈 ② 학습, XAI/공정성 분석 |
| Week 14 | 최종 평가, 리포트 작성 |

---

## 📝 라이선스 / 비고

본 저장소는 학기말 과제용 **private repo**입니다.
외부 데이터셋(Smilegate Unsmile, KOLD 등)은 각 출처 라이선스를 따릅니다 — `data/README.md` 참조.
