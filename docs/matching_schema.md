# 모듈 ② 사용자 호환성 매칭 데이터 스키마 및 피처 설계 정의서

본 문서는 **모듈 ② (사용자 호환성 매칭)** 의 학습 데이터 형태를 정의하고, SBERT 임베딩 및 메타데이터를 활용한 피처 엔지니어링 스펙을 확정합니다.

---

## 1. 데이터 아키텍처 개요

사용자 호환성 매칭은 특정 사용자($User_A$)에게 가장 적합한 다른 사용자($User_B$)를 추천 및 정렬하는 **Learning-to-Rank (LTR)** 문제로 정의합니다.
- **백본**: `jhgan/ko-sroberta-multitask` (SBERT 텍스트 임베딩 추출)
- **랭킹 모델**: `LightGBM LambdaMART` (`objective="lambdarank"`, `metric="ndcg"`)
- **평가 메트릭**: `NDCG@5`, `NDCG@10` (목표: NDCG@10 ≥ 0.60)

---

## 2. 사용자 프로필 스키마 (User Profile Schema)

각 사용자는 개인 성향, 관심사, 도메인 맥락(장애 도메인)을 포함하는 정형/비정형 정보를 가집니다.

| 필드명 | 데이터 타입 | 설명 | 예시 |
|---|---|---|---|
| `user_id` | `str` | 고유 식별자 (UUID 또는 인덱스) | `"usr_10028"` |
| `introduction` | `str` | 자기소개 자연어 텍스트 (SBERT 임베딩 대상) | `"안녕하세요! 운동과 음악을 좋아하는 발달장애 청년입니다. 같이 소통할 친구 찾아요."` |
| `age` | `int` | 나이 | `24` |
| `gender` | `str` | 성별 (`"남성"`, `"여성"`, `"기타"`) | `"남성"` |
| `region` | `str` | 주요 활동 지역 (시/도 단위) | `"서울"`, `"경기"`, `"부산"`, `"대구"`, `"인천"` |
| `disability_type` | `str` | 장애 유형 | `"발달장애"`, `"시각장애"`, `"청각장애"`, `"지체장애"`, `"비장애"` |
| `interests` | `list[str]` | 관심사 태그 목록 | `["운동", "음악", "독서", "게임", "맛집"]` |
| `mbti` | `str` | MBTI 성향 (16종) | `"ENFP"`, `"ISTJ"` |

---

## 3. 랭킹 쿼리 및 relevance 라벨 정책

LambdaMART 학습을 위해 **질의(Query, 주 사용자 $User_A$)** 와 **후보군(Candidates, 대상 사용자 $User_B$)** 의 페어를 구성하고, 호환성 강도를 나타내는 **Relevance Label ($y \in \{0, 1, 2, 3\}$)** 을 다음과 같은 규칙 기반 규칙으로 설계 및 합성합니다.

> [!IMPORTANT]
> 랭킹 데이터셋은 하나의 `query_group` (동일한 $User_A$)으로 묶여 있어야 LambdaMART가 NDCG 손실 함수를 통해 정렬을 올바르게 학습할 수 있습니다.

### Relevance Label 4단계 정의 (호환성 강도)

- **`3` (최상 호환 - Strongly Relevant)**:
  - 지역이 일치하고, 나이 차이가 5세 이하이며,
  - 겹치는 관심사가 2개 이상이고,
  - SBERT 자기소개 임베딩 코사인 유사도가 임계치(예: 0.70) 이상.
- **`2` (우수 호환 - Relevant)**:
  - 지역이 일치하고, 나이 차이가 10세 이하이며,
  - 겹치는 관심사가 1개 이상이거나 SBERT 유사도가 0.60 이상.
- **`1` (보통 호환 - Marginally Relevant)**:
  - 지역 또는 연령대 중 최소 하나가 매칭되고, 공통점이 최소한으로 존재.
- **`0` (비호환 - Irrelevant)**:
  - 공통 관심사가 전혀 없거나, 지역/연령 등 주요 매칭 조건이 크게 불일치.

---

## 4. 피처 엔지니어링 명세 (Feature Engineering Specification)

모델의 입력 피처 벡터 $\mathbf{x}_{AB}$ 는 SBERT 임베딩 간의 유사도 특징량과 사용자 간 메타데이터 연산 특징량의 결합으로 정의됩니다.

### 4.1 SBERT 임베딩 피처 (비정형 텍스트 매칭)
$User_A$와 $User_B$의 자기소개 임베딩 벡터를 각각 $\mathbf{u}_A, \mathbf{u}_B \in \mathbb{R}^{d}$ ($d=768$) 라고 할 때:

1. **Cosine Similarity**: 두 벡터 간 각도 유사도
   $$f_{\text{cosine}} = \frac{\mathbf{u}_A \cdot \mathbf{u}_B}{\|\mathbf{u}_A\| \|\mathbf{u}_B\|}$$
2. **L1 Distance**: 원소별 절대 차이 합
   $$\mathbf{f}_{\text{L1}} = |\mathbf{u}_A - \mathbf{u}_B| \in \mathbb{R}^{d}$$
3. **L2 Distance**: Euclidean 거리
   $$f_{\text{L2}} = \|\mathbf{u}_A - \mathbf{u}_B\|_2$$
4. **Hadamard Product**: 원소별 곱 (공통 활성화 영역 인지)
   $$\mathbf{f}_{\text{hadamard}} = \mathbf{u}_A \odot \mathbf{u}_B \in \mathbb{R}^{d}$$

### 4.2 메타데이터 피처 (정형 데이터 매칭)

1. **`age_diff`**: 연령 차이의 절대값
   $$f_{\text{age\_diff}} = |age_A - age_B|$$
2. **`region_match`**: 활동 지역 동일 여부 (바이너리)
   $$f_{\text{region\_match}} = \mathbb{I}(region_A == region_B)$$
3. **`gender_compatibility`**: 성별 선호도 매칭 여부 (바이너리)
4. **`disability_type_match`**: 장애 유형 일치/유사 여부 (바이너리)
   $$f_{\text{disability\_match}} = \mathbb{I}(disability\_type_A == disability\_type_B)$$
5. **`interest_overlap_count`**: 공통 관심사 개수
   $$f_{\text{interest\_overlap}} = |interests_A \cap interests_B|$$
6. **`mbti_compatibility`**: MBTI 궁합 점수 (전통적인 MBTI 궁합 척도에 따라 1~5점 부여)

---

## 5. LambdaMART 학습 데이터셋 파일 명세

최종적으로 학습에 입력되는 데이터프레임 스키마는 다음과 같습니다.

```
[query_id, user_a_id, user_b_id, f_cosine, f_l2, f_age_diff, f_region_match, f_disability_match, f_interest_overlap, ..., relevance_label]
```

- **학습용 그룹 포맷 (group)**: LightGBM lambdarank는 각 쿼리 그룹별 데이터 크기 배열(`group` 파일 또는 API 파라미터)이 필요합니다.
  - 예: `group = [10, 10, 10, ...]` (각 쿼리 사용자당 10명의 후보자 페어가 정렬되어 구성됨을 뜻함)

이 스키마 설계를 기반으로 **D4에서 `ko-sroberta` 임베딩 캐싱 기술 및 LightGBM Ranker 학습 파이프라인**을 성공적으로 구축할 것입니다.
