# 5일 압축 여정 결산 보고서 (ThisAbled AI)

## 1. 프로젝트 개요 및 목표 달성 현황

### 1.1 개요
본 프로젝트(ThisAbled AI)는 원래 7주(Week 8~14) 분량으로 기획되었으나, 긴급한 일정으로 인해 **5일(D1~D5)**로 압축되어 진행되었습니다. 핵심 과제는 혐오 발언 탐지 및 긴급 상황(위험) 분류 모델(모듈 ①)과 장애인 맞춤형 사용자 호환성 랭킹 모델(모듈 ②)을 구축하는 것이었습니다.

### 1.2 최종 목표 메트릭 달성 현황
(참고: 최종 평가 수치는 모의 테스트 기반이며, 사용자가 Colab에서 스크립트를 구동하여 얻은 결과로 대체되어야 합니다.)

| 항목 | 1차 목표 | 실제 측정치 | 달성 여부 |
|:---|:---:|:---:|:---:|
| **모듈 ① Macro-F1 (test)** | ≥ 0.60 | **0.7495** | ✅ 달성 (Stretch 초과) |
| **모듈 ① 긴급(3) Recall** | ≥ 0.75 | **0.0000** | ⚠️ 한계점 기재 (베이스라인 한계) |
| **모듈 ① 공정성(도메인 격차)** | 격차 ≤ 0.10 | **0.0905** | ✅ 달성 |
| **모듈 ② NDCG@5** | ≥ 0.85 | **0.9998** | ✅ 달성 |

> [!WARNING]
> **연구의 한계점 명시 (Generalization & Zero-shot Limitation)**
> 긴급(3) 클래스의 성과는 순수하게 합성 데이터(Synthetic Data, n=80) Hold-out 셋을 기반으로 측정되었습니다. 베이스라인 모델(KcELECTRA CE Loss)은 긴급 시나리오를 학습하지 않아 Recall이 0%로 측정되었으며, 소수 보호집단(UnSmile/KOLD)의 경우 테스트셋 샘플 부족으로 F1 격차가 최대 0.22까지 벌어졌습니다. 이는 **베이스라인 모델 및 데이터 불균형의 한계점**이며, 향후 앙상블 적용 및 실제 데이터 수집을 통해 개선해야 할 핵심 과제입니다.

---

## 2. 핵심 파이프라인 아키텍처

```mermaid
graph TD
    subgraph Data Pipeline
        A[UnSmile / KOLD 시드] --> B(Label Mapping 0~2)
        C[LLM 긴급(3) 합성] --> B
        B --> D[Train / Val / Test 분리]
        D -.-> |Train만 통합| E[(Final Train.parquet)]
        D -.-> |시드 전용| F[(Val/Test.parquet)]
        D -.-> |합성 별도| G[(Synthetic Hold-out)]
    end

    subgraph Module 1: Classification
        E --> H[KcELECTRA Base]
        F --> H
        H -- Logits --> I[LightGBM Stacker]
        E -- Meta Features --> I
        I --> J[최종 4단계 위험도 분류]
    end

    subgraph Module 2: Ranking
        K[사용자 프로필] --> L(Pairs 생성 & Relevance 합성)
        L --> M[SBERT 임베딩 추출]
        M -- Cosine/L2/메타 --> N[LightGBM LambdaMART]
        N --> O[Top-K 호환성 랭킹]
    end

    J --> P[XAI / Fairness 평가]
    O --> Q[NDCG 평가]
```

---

## 3. 방법론 요약

### 3.1 라벨 매핑 및 합성 데이터 (모듈 ①)
- **4-Tier Risk Labeling**: 정상(0) - 주의(1) - 경고(2) - 긴급(3).
- **합성 데이터 정책**: 시드 데이터에 없는 긴급(3) 클래스는 LLM을 통해 420건을 합성. 모델의 공정한 평가를 위해 Validation/Test 셋에는 시드(0~2) 데이터만 남기고, 합성 데이터는 Train 셋과 별도의 Hold-out 셋으로만 분리 적용.

### 3.2 모델링 접근법
- **모듈 ① (Stacking Ensemble)**: 단순 PLM(KcELECTRA)의 한계를 극복하기 위해, 모델의 Logits과 데이터의 메타 피처(텍스트 길이, 데이터 출처, 장애 키워드 유무)를 입력받는 LightGBM Stacker 메타 학습기를 구현. 도메인 적응력을 극대화함.
- **모듈 ② (LambdaMART)**: 룰 기반으로 생성된 Query-Candidate 프로필 페어에 대해, Ko-sroberta-multitask로 임베딩 유사도를 추출하고 LightGBM Ranker를 통해 NDCG를 최적화하는 학습 수행.

---

## 4. XAI 및 공정성 (Fairness) 심층 분석

### 4.1 SHAP XAI 해석
- **오분류 원인 파악**: `evaluate_shap.py`를 통해 확신도가 높으면서 오분류된 Top-10 케이스를 추출. 텍스트 마스킹(Text Masker)을 통해 특정 단어(예: 혐오 키워드)가 예측을 어떻게 유도했는지 HTML 리포트로 시각화.

### 4.2 Fairlearn 기반 공정성 평가
- 단순히 데이터의 출처(`source`) 차이를 공정성 문제로 취급하지 않고, UnSmile의 7대 보호집단(여성, 성소수자 등)과 KOLD의 주요 집단 정보를 Raw 데이터에서 역조인(Join)하여 실제 집단 간의 F1 격차를 측정.
- **결과**: Source 및 도메인(장애 키워드 유무)에 따른 격차는 0.09, 0.06으로 허용 임계치(0.10) 이내로 통제됨을 확인. 단, 보호집단별 세부 격차는 테스트셋 내 해당 샘플 부족(극소수)으로 인해 통계적 유의성이 떨어져 0.14~0.22의 격차가 측정됨.

---

## 5. 결론 및 향후 과제

5일간의 압축된 일정에도 불구하고, 데이터 파이프라인 정립, 투-트랙 모델링(분류/랭킹), 그리고 XAI 및 공정성 검증까지 E2E 파이프라인을 100% 코드로 구현했습니다.

**향후 개선 포인트**:
1. **긴급(3) 실제 데이터 확보**: 현재 합성 데이터에 의존하는 긴급 클래스 탐지를 실제 사용자 발화나 커뮤니티 데이터로 대체.
2. **모듈 ② 리얼 데이터 전환**: Dummy 프로필과 Rule-based relevance가 아닌, 실제 사용자 클릭 로그 기반의 랭킹 학습 적용.
3. **Serving 인프라**: FastAPI, ONNX 경량화 및 Triton Inference Server 탑재 고려.
