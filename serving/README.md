# serving — SAFE·MATCH 모델 서버 (백엔드 mock 교체)

백엔드(`thisabled-backend`)는 `safety-model:9001`, `match-model:9002`를 HTTP로 호출한다.
이 디렉터리의 두 서버는 그 mock과 **동일 계약**을 구현한 실모델 서버다.
계약만 유지되므로 **백엔드 코드 수정은 없다.**

| 서버 | 계약 | 모델 |
| --- | --- | --- |
| safety_server (9001) | `POST /analyze {text, receiver_is_minor}` → `{verdict: safe\|flagged}` | KcELECTRA 4-class (`module1_ce`) |
| match_server (9002) | `POST /score {me, candidates}` → `{results:[{user_id, score, reasons}]}` | ko-sroberta 임베딩 + LambdaMART |

MATCH는 기존 `user_id,bio,tags,age_band,ui_mode` 5필드 요청을 그대로 지원한다. 입력
파이프라인 v2에서는 `age_years`, `authored_items`, `liked_items`와 후보별 선택
`relationship`을 추가할 수 있다. 알 수 없는 필드는 거부하며 검증 오류에는 입력값을 되돌리지
않는다. 상세 계약은 `docs/matching_schema.md`를 따른다.

## 로컬 실행 (Docker 없이)

```bash
cd thisabled-ai
source .venv/bin/activate    # 반드시 저장소 venv 사용
pip install -r serving/safety_server/requirements.txt -r serving/match_server/requirements.txt
# `uvicorn` 단독 실행 금지 — PATH의 다른 파이썬을 잡을 수 있음. 항상 `python -m`:
python -m uvicorn serving.safety_server.app:app --port 9001 &
python -m uvicorn serving.match_server.app:app --port 9002 &
python serving/smoke_test.py    # /health 폴링 후 계약·지연 검증
```

## 모델 배포 (Hugging Face Hub — private)

모델 최종본은 HF private repo에서 로드한다. 이미지에 모델을 굽지 않으므로
**빌드하는 머신에 체크포인트 파일이 없어도 된다.**

```bash
# 최초 1회 업로드 (모델 소유자 = AI 파트)
hf auth login   # write 토큰
hf upload soyuncj/thisabled-safety-kcelectra models/checkpoints/module1_ce . \
  --repo-type model --private \
  --exclude "optimizer.pt" --exclude "rng_state.pth" --exclude "scheduler.pt" --exclude "training_args.bin"
hf upload soyuncj/thisabled-match-lambdamart models/checkpoints/module2_lambdamart_embedding.pkl \
  --repo-type model --private
```

- BE에게는 두 repo만 읽을 수 있는 **fine-grained read 토큰**을 발급해 전달 (write 토큰 공유 금지).
- 로컬 개발 시에는 `models/checkpoints/` 경로가 있으면 그대로 사용 (HF 불필요).

## 백엔드 compose 교체

`thisabled-backend/docker-compose.yml`의 두 서비스만 수정 (AI 저장소를 백엔드 옆에 clone해둔 기준 —
체크포인트는 clone에 없어도 됨):

```yaml
  safety-model:
    build:
      context: ../thisabled-ai
      dockerfile: serving/safety_server/Dockerfile
    environment:
      SAFE_MODEL_DIR: soyuncj/thisabled-safety-kcelectra
      HF_TOKEN: ${HF_TOKEN}
    volumes:
      - hf_cache_safety:/srv/hf-cache
    restart: unless-stopped

  match-model:
    build:
      context: ../thisabled-ai
      dockerfile: serving/match_server/Dockerfile
    environment:
      MATCH_HF_REPO: soyuncj/thisabled-match-lambdamart
      HF_TOKEN: ${HF_TOKEN}
    volumes:
      - hf_cache_match:/srv/hf-cache
    restart: unless-stopped

# volumes: 에 hf_cache_safety, hf_cache_match 추가
```

백엔드 `.env`에 `HF_TOKEN=<read 토큰>` 추가 후:

```bash
docker compose up -d --build safety-model match-model
docker compose logs -f safety-model      # 첫 기동 시 모델 다운로드(~500MB) 후 startup complete
docker compose exec -T app pytest -q     # 백엔드 테스트 그린 확인
```

포트·서비스명·계약이 mock과 동일하므로 이것으로 끝. 되돌리려면 원래 mock 설정으로 복구
(§18.3 장애 시연·영상 촬영 폴백용으로 mock 설정을 지우지 말 것).

## MATCH 랭커 스키마 (legacy-v1 ↔ match-input-v2)

MATCH는 두 랭커 스키마를 `MATCH_FEATURE_SCHEMA`로 전환한다. 기본은 `legacy-v1`(구형 3열
pickle + UI모드→f_dis_match 투영 shim). `match-input-v2`는 재학습된 15열 LambdaMART를
그대로 쓰며, 모델이 이미 태그·나이·콘텐츠·공통친구를 학습했으므로 **서빙에서 tag/age를
재가산하지 않고 순수 모델 점수로 정렬**한다(`MATCH_W_*` 미사용).

v2 배포는 HF revision을 고정한다(train/serve 일관성):

```yaml
  match-model:
    environment:
      MATCH_FEATURE_SCHEMA: match-input-v2
      MATCH_HF_REPO: soyuncj/module2
      MATCH_HF_FILE: module2_lambdamart_v2.pkl
      MATCH_HF_REVISION: ecb31a428e74dfc393617a6a4a95ecc4cb7e6d67
      HF_TOKEN: ${HF_TOKEN}   # 해당 repo read 토큰
```

**롤백**: `MATCH_FEATURE_SCHEMA: legacy-v1`로 바꾸고(또는 env 제거) 재기동하면 구형
pickle+shim 경로로 즉시 복귀한다. legacy 자산은 유지한다.

## 운영 설정값 (env)

| 변수 | 기본 | 의미 |
| --- | --- | --- |
| `SAFE_FLAG_THRESHOLD` | 0.5 | P(주의+경고+긴급) ≥ τ → flagged |
| `SAFE_FLAG_THRESHOLD_MINOR` | 0.35 | 미성년 수신자 민감 임계값 (§4.5) |
| `SAFE_RULE_ASSIST` | 0 | 금전 사기 규칙 보조 레이어 on/off. 최신 이진 모델은 기본 비활성 |
| `SAFE_MAX_LENGTH` | 128 | 토크나이저 max_length |
| `TORCH_NUM_THREADS` | 2 | CPU 스레드 |
| `MATCH_COSINE_REASON_MIN` | 0.5 | "소개 내용이 비슷해요" 사유 최소 코사인 |
| `MATCH_CONFIG_PATH` | `configs/module2_matching.yaml` | 입력 정책·TAG-01·특성 스키마 설정 파일 |
| `MATCH_ALLOWED_TAG_IDS` | 설정 파일의 43개 코드 | 쉼표로 구분한 TAG-01 허용 코드 전체 재정의 |
| `MATCH_FEATURE_SCHEMA` | `legacy-v1` | 랭커 스키마 `legacy-v1`\|`match-input-v2`. 전환·롤백 스위치 |
| `MATCH_HF_FILE` | `module2_lambdamart_embedding.pkl` | HF에서 받을 모델 파일명 (v2는 `module2_lambdamart_v2.pkl`) |
| `MATCH_HF_REVISION` | (최신) | 고정할 HF commit hash. v2 배포는 반드시 지정 |
| `MATCH_W_MODEL` / `MATCH_W_TAG` / `MATCH_W_AGE` | 0.5 / 0.3 / 0.2 | legacy-v1 점수 블렌드 가중치 (v2는 미사용 — 순수 모델 점수) |
| `MATCH_AUTHORED_LOOKBACK_DAYS` / `MATCH_LIKED_LOOKBACK_DAYS` | 90 / 90 | 작성물·좋아요 분석 기간 |
| `MATCH_MAX_AUTHORED_ITEMS` / `MATCH_MAX_LIKED_ITEMS` | 100 / 100 | 사용자별 최대 콘텐츠 수 |
| `MATCH_MAX_CONTENT_CHARS` | 2000 | 콘텐츠 한 건의 임베딩 입력 문자 상한 |
| `MATCH_MAX_CANDIDATES` | 200 | `/score` 요청당 후보 상한 |
| `MATCH_EMBEDDING_BATCH_SIZE` | 64 | Sentence-BERT 인코딩 배치 크기 |
| `MATCH_CONTENT_REASON_MIN` | 0.65 | 콘텐츠 유사 추천 사유 cosine 하한 |
| `SAFE_MODEL_DIR` | 로컬 경로 | 로컬 체크포인트 경로 또는 HF repo id |
| `SAFE_MODEL_REVISION` | (없음=main) | HF repo id 사용 시 로드할 커밋 SHA 고정. `/health.revision`으로 실제 로드값 확인 |
| `MATCH_HF_REPO` | (없음) | 로컬 pkl 부재 시 다운로드할 HF repo id |
| `HF_TOKEN` | (없음) | HF private repo read 토큰 |

## 설계 메모 (보고서 반영)

- **스태커 미사용**: LightGBM 스태커의 meta feature에 학습 데이터 전용 `source` 컬럼이
  필요해 서빙에선 KcELECTRA softmax를 직접 사용. 확률의 위험 합산 + 임계값으로
  binary verdict 산출 — 임계값은 운영 설정값(명세 §4.5).
- **이진·4-class 자동 지원**: 서버가 기동 시 `model.config.num_labels`(2 또는 4)를 읽어
  라벨을 결정. `risk_prob = sum(probs[1:])`는 이진에선 P(주의), 4-class에선 P(주의+경고+긴급)
  으로 동일하게 작동 → 재학습 전환기에 코드 수정 없이 두 모델 모두 서빙. `/health`의
  `num_labels`로 로드된 모델 확인 가능. (권장: docs/재학습_프롬프트_이진_pan12.md의 이진 모델)
- **하이브리드(모델+규칙)**: 학습 시드가 혐오표현 중심이라 금전 사기 유형(SAFE-02 ①)
  커버리지가 약함 — 스모크에서 사기 문장 risk_prob 0.07 실측. 규칙 보조 레이어를 OR로
  결합해 재현율 확보(플래그 추가만, 해제 없음 → 오탐 소폭 증가 트레이드오프).
  근본 해결은 사기 유형 합성 증강 + 재학습 — 후속 계획으로 보고서에 기술.
- **f_dis_match 대응**: 학습의 disability_type 일치 → 서빙에선 ui_mode 일치.
  서버 내부 특성 전용, 추천 사유로 노출 금지 (MATCH-04).
- **bio 폴백**: bio가 비면 관심사 태그 문자열로 임베딩 (MATCH-02-8).
- **입력 파이프라인 v2**: 작성물·좋아요는 별도 집계 벡터로 보존하고, 삭제·접근 불가·차단
  작성자의 콘텐츠는 encoder 호출 전에 제외한다. 후보 관계가 요청에 포함되면 차단·친구·최근
  거절·미성년/성인을 모델 서버에서도 재검사한다.
- **레거시 모델 투영**: 현재 pickle은 3열만 학습했으므로 프로필 raw SBERT의
  `f_cosine,f_l2`와 내부 `ui_mode` 일치값만 구형 `f_dis_match` 열에 전달한다. v2 특성을
  LambdaMART 점수에 직접 쓰려면 새 스키마로 재학습해야 한다.
- **MATCH 점수 블렌드**: 학습 입력(지역·나이·장애유형이 명시된 템플릿 자기소개)과
  서빙 계약 입력(bio+tags+age_band+ui_mode) 사이에 스키마 드리프트가 있음 — 학습 라벨의
  주 결정 변수(지역)가 서빙 입력에 없고, 나이도 텍스트에서 사라짐(스모크에서 상이한 두
  후보가 동일 점수로 나온 원인). 보완으로 모델 점수에 태그 교집합·연령대 일치를 가중
  결합(0.5/0.3/0.2) — 두 신호는 학습 라벨 규칙(overlap, age_diff)과 동일 계열이라 정합적.
  근본 해결은 계약 스키마로 프로필 합성 재생성 + 재학습 — 후속 계획.
- 응답의 `risk_prob`·`level`·`probs`는 계약 외 부가 필드 — 백엔드는 `verdict`만 사용,
  데모·리포트 시연(실시간 위험도 스코어러)에 활용 가능.
