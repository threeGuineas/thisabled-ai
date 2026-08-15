# 모듈 ② MATCH 입력·특성 스키마

이 문서는 커뮤니티 친구 추천용 MATCH 입력 파이프라인의 현재 계약을 정의한다.
연애·데이트 매칭, 장애 유형 추정 및 장애 인증 정보 사용은 범위에서 제외한다.

## 1. 책임 경계

- 백엔드: 인증, DB 조회, 생년월일→만 나이/연령대 계산, TAG-01 조회, 후보 관계 선필터링
- MATCH 입력 파이프라인: 허용 필드 검증, 개인정보 차단, 콘텐츠 활성 상태 선별, 일괄 임베딩,
  사용자·페어 특성 생성
- LambdaMART: 검증된 수치 특성으로 후보 순위 계산
- 백엔드 응답 계층: 공개 프로필 결합과 최종 추천 사유 allowlist 검증

정확한 생년월일은 MATCH 모델 서비스에 전달하지 않는다.

## 2. `/score` 호환 계약

현재 백엔드가 보내는 기존 5필드 계약은 그대로 유효하다.

```json
{
  "me": {
    "user_id": "uuid",
    "bio": "영화와 야구를 좋아해요.",
    "tags": ["movie", "walking"],
    "age_band": "25-34",
    "ui_mode": "visual"
  },
  "candidates": []
}
```

입력 파이프라인 v2는 다음 선택 필드를 추가로 받는다.

```json
{
  "me": {
    "user_id": "uuid",
    "bio": "영화와 야구를 좋아해요.",
    "tags": ["movie", "walking"],
    "age_years": 27,
    "age_band": "25-34",
    "ui_mode": "visual",
    "authored_items": [],
    "liked_items": []
  },
  "candidates": [
    {
      "user_id": "candidate-uuid",
      "bio": "영화 이야기를 나눌 친구를 찾아요.",
      "tags": ["movie"],
      "age_years": 29,
      "age_band": "25-34",
      "ui_mode": "hearing",
      "authored_items": [],
      "liked_items": [],
      "relationship": {
        "blocked_either_direction": false,
        "already_friends": false,
        "last_rejected_at": null,
        "common_friend_count": 2
      }
    }
  ]
}
```

Wire 이름 `tags`는 기존 백엔드 호환을 위해 유지하고 내부에서는 `tag_ids`로 변환한다.
`relationship`이 없으면 현재 백엔드가 이미 후보를 선필터링한 legacy 요청으로 처리한다.
관계 정보가 전달된 경우에만 MATCH 서비스가 방어적 2차 검사를 수행할 수 있다.

응답은 다음 공개 필드만 포함한다.

```json
{
  "results": [
    {
      "user_id": "candidate-uuid",
      "score": 0.81,
      "reasons": ["관심사가 비슷해요", "공통 친구가 있어요"]
    }
  ]
}
```

`model_score`, UI 모드, 관계 상태, 원문 콘텐츠, 내부 특성은 응답하지 않는다.

## 3. 입력 규칙

### 사용자

| 필드 | 규칙 |
|---|---|
| `user_id` | 비어 있지 않은 최대 128자 식별자 |
| `bio` | 선택 입력, 정규화 후 최대 300자, 연락처 발견 시 요청 거부 |
| `tags` | TAG-01 코드, 중복 제거 후 최대 10개 |
| `age_years` | 선택적 만 나이 14~120, 생년월일 원문 금지 |
| `age_band` | 선택적 `14-18`, `19-24`, `25-34`, `35-44`, `45-54`, `55+` |
| `ui_mode` | `visual`, `hearing`, `developmental`; 추천 사유·응답 비노출 |

연령대의 한글 명세 표기(`25~34세` 등)도 입력 alias로 받고 내부에서 한 값으로 정규화한다.
`age_years`와 `age_band` 중 하나 이상은 반드시 있어야 하며, 함께 있으면 서로 일치해야 한다.
TAG-01 허용 코드는 `configs/module2_matching.yaml`의 43개 레지스트리에서 읽는다.

### 콘텐츠

| 필드 | 규칙 |
|---|---|
| `content_id` | 콘텐츠 중복 제거 키 |
| `source_type` | `post` 또는 `comment`만 허용 |
| `text` | 빈 문자열과 연락처 포함 항목 제외, 임베딩 길이 상한 적용 |
| `created_at` | 시간대가 포함된 시각, 미래·분석 기간 밖 항목 제외 |
| `is_deleted` | true이면 즉시 제외 |
| `is_accessible` | false이면 즉시 제외 |
| `is_blocked_author` | true이면 즉시 제외 |
| `is_like_active` | 좋아요 취소 시 false, 다음 갱신부터 제외 |

### 명시적으로 허용하지 않는 데이터

- 닉네임, 프로필 이미지
- 1:1 채팅 메시지
- 전화번호, 이메일, 외부 메신저 ID
- 정확한 위치
- 장애 유형 및 인증 정보
- AI 위험 메시지 판정 기록
- 생년월일 원문

알 수 없는 필드는 HTTP 경계에서 거부한다. 오류 응답에는 입력값을 포함하지 않는다.

## 4. 처리 순서

1. 요청 필드와 크기 상한을 검증한다.
2. 자기소개에서 연락처를 발견하면 `CONTACT_INFO_DETECTED`로 거부한다.
3. 후보 관계 제외를 콘텐츠 검증과 encoder 호출보다 먼저 수행한다.
4. 삭제·접근 불가·차단 작성자·연락처·기간 밖 콘텐츠를 제거한다.
5. 허용된 모든 사용자 텍스트를 한 번의 Sentence-BERT 배치로 임베딩한다.
6. profile/authored/liked 벡터를 분리 집계하고 페어 특성을 만든다.
7. 현재 모델 스키마에 맞는 열만 명시적으로 투영해 예측한다.
8. 정확한 allowlist 문장으로 추천 사유를 만든다.

## 5. 후보 제외

다음 후보는 임베딩 및 점수 계산 전에 제외한다.

- 자기 자신
- 어느 방향이든 차단된 사용자
- 이미 친구인 사용자
- 친구 요청 거절 후 30일이 지나지 않은 사용자
- `14~18세`와 성인 연령대 사이의 양방향 조합

현재 백엔드는 이 규칙을 DB 후보 조회 단계에서 강제한다. 확장 `relationship`이 제공되면 MATCH
서비스도 같은 규칙을 재검사한다. 정확히 30일이 지난 거절은 다시 후보가 될 수 있다.

## 6. 임베딩과 집계

- `profile_vector`: 자기소개와 태그를 합친 텍스트의 SBERT 벡터
- `authored_vector`: 허용된 게시물·댓글 벡터의 단위 정규화 평균
- `liked_vector`: 활성 좋아요 콘텐츠 벡터의 단위 정규화 평균
- `effective_vector`: 존재하는 세 성분 벡터의 단위 정규화 평균

성분이 없으면 0 벡터로 가장하지 않고 `*_available` 특성을 0으로 둔다. 자기소개가 없어도 태그,
작성물, 좋아요, 공통 친구, UI 모드를 사용할 수 있다. 이 신호도 없으면 `insufficient_signal`로
처리하며 `/score`는 빈 결과와 `추천 정보가 부족합니다` 메시지를 반환한다.

## 7. 페어 특성

입력 파이프라인 v2는 다음 계열을 생성한다.

- 프로필·작성물·좋아요·유효 관심 벡터 cosine
- 프로필 raw SBERT L2
- 질의 사용자의 좋아요와 후보 작성물 간 cosine
- 태그 교집합 수와 Jaccard
- 공통 친구 수
- 만 나이 차이와 연령대 일치
- UI 모드 일치(내부 전용)
- 각 벡터/연령 특성의 availability

## 8. 랭커 스키마 (legacy-v1 ↔ match-input-v2)

서빙은 `MATCH_FEATURE_SCHEMA`(config `features.feature_schema`)로 두 스키마를 전환한다.

**legacy-v1** (기본): 구형 3열 pickle.

```text
f_cosine, f_l2, f_dis_match
```

- `f_cosine`, `f_l2`는 프로필 raw SBERT 벡터에서 계산한다.
- pickle의 구형 열 이름 `f_dis_match`에는 서빙 시 `f_ui_mode_match`만 투영한다(shim).
- 점수식: `W_MODEL·sigmoid(ranker) + W_TAG·norm_tag_overlap + W_AGE·age_band_match`.

**match-input-v2**: 재학습된 15열 LambdaMART(`configs/module2_matching.yaml`의
`features.v2_pair_features`). pickle은 `{model, columns, params, metrics}` 번들이며 열
순서를 함께 저장한다.

- 15열 전체를 순서대로 랭커에 전달한다(투영 shim 없음).
- 모델이 태그·나이·콘텐츠·공통친구를 이미 학습했으므로 서빙에서 재가산하지 않는다.
  점수식: `score = sigmoid(ranker)` (순수 모델 점수).
- 배포는 HF revision을 고정한다: repo `soyuncj/module2`, 파일 `module2_lambdamart_v2.pkl`,
  revision `ecb31a428e74dfc393617a6a4a95ecc4cb7e6d67`. 상세 env는 `serving/README.md`.
- 학습·평가·공정성: `scripts/train_match_v2.py`(잠재라벨 합성 + AI허브 실문장 코퍼스),
  `scripts/evaluate_match_v2_fairness.py`(ui_mode DP ≤ 0.10). 특성 생성은 학습·서빙 모두
  `matching_input.build_pair_features`를 그대로 호출한다.

**롤백**: `MATCH_FEATURE_SCHEMA=legacy-v1`로 되돌리고 재기동. legacy pickle·shim은 유지한다.

## 9. 추천 사유 allowlist

- `관심사가 비슷해요`
- `관심 있는 콘텐츠가 비슷해요`
- `공통 친구가 있어요`
- `비슷한 연령대예요`

UI 모드와 장애 관련 문구, 특정 게시물·댓글 원문은 추천 사유에 사용할 수 없다.

## 10. 운영 설정

기본값은 `configs/module2_matching.yaml`과 MATCH 서버 환경변수로 관리한다.

- 작성물/좋아요 분석 기간: 각각 90일
- 작성물/좋아요 최대 수: 각각 100개
- 콘텐츠 임베딩 입력: 항목당 최대 2,000자
- 요청당 후보: 최대 200명
- 거절 제외 기간: 30일

삭제·차단·접근 변경의 즉시 반영은 백엔드가 최신 상태를 전달하거나 캐시 무효화 이벤트를
제공해야 완전히 보장된다. 현재 입력 파이프라인은 요청에 전달된 상태를 항상 재검사한다.

## 11. 검증

```bash
.venv/bin/python -m pytest -q tests/test_matching_input.py tests/test_match_server_input.py
.venv/bin/ruff check src/data/matching_input.py serving/match_server/app.py \
  tests/test_matching_input.py tests/test_match_server_input.py
.venv/bin/python -m pytest -q tests/test_module2_pair.py
```
