# serving — SAFE·MATCH 모델 서버 (백엔드 mock 교체)

백엔드(`thisabled-backend`)는 `safety-model:9001`, `match-model:9002`를 HTTP로 호출한다.
이 디렉터리의 두 서버는 그 mock과 **동일 계약**을 구현한 실모델 서버다.
계약만 유지되므로 **백엔드 코드 수정은 없다.**

| 서버 | 계약 | 모델 |
| --- | --- | --- |
| safety_server (9001) | `POST /analyze {text, receiver_is_minor}` → `{verdict: safe\|flagged}` | KcELECTRA 4-class (`module1_ce`) |
| match_server (9002) | `POST /score {me, candidates}` → `{results:[{user_id, score, reasons}]}` | ko-sroberta 임베딩 + LambdaMART |

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

## 백엔드 compose 교체

`thisabled-backend/docker-compose.yml`의 두 서비스만 수정 (AI 저장소를 백엔드 옆에 clone해둔 기준):

```yaml
  safety-model:
    build:
      context: ../thisabled-ai
      dockerfile: serving/safety_server/Dockerfile
    restart: unless-stopped

  match-model:
    build:
      context: ../thisabled-ai
      dockerfile: serving/match_server/Dockerfile
    restart: unless-stopped
```

```bash
docker compose up -d --build safety-model match-model
docker compose exec -T app pytest -q     # 백엔드 테스트 그린 확인
```

포트·서비스명·계약이 mock과 동일하므로 이것으로 끝. 되돌리려면 원래 mock 설정으로 복구
(§18.3 장애 시연·영상 촬영 폴백용으로 mock 설정을 지우지 말 것).

## 운영 설정값 (env)

| 변수 | 기본 | 의미 |
| --- | --- | --- |
| `SAFE_FLAG_THRESHOLD` | 0.5 | P(주의+경고+긴급) ≥ τ → flagged |
| `SAFE_FLAG_THRESHOLD_MINOR` | 0.35 | 미성년 수신자 민감 임계값 (§4.5) |
| `SAFE_RULE_ASSIST` | 1 | 금전 사기 규칙 보조 레이어 on/off (모델 판정에 OR 결합) |
| `SAFE_MAX_LENGTH` | 128 | 토크나이저 max_length |
| `TORCH_NUM_THREADS` | 2 | CPU 스레드 |
| `MATCH_COSINE_REASON_MIN` | 0.5 | "소개 내용이 비슷해요" 사유 최소 코사인 |

## 설계 메모 (보고서 반영)

- **스태커 미사용**: LightGBM 스태커의 meta feature에 학습 데이터 전용 `source` 컬럼이
  필요해 서빙에선 KcELECTRA softmax를 직접 사용. 4-class 확률의 위험 합산 + 임계값으로
  binary verdict 산출 — 임계값은 운영 설정값(명세 §4.5).
- **하이브리드(모델+규칙)**: 학습 시드가 혐오표현 중심이라 금전 사기 유형(SAFE-02 ①)
  커버리지가 약함 — 스모크에서 사기 문장 risk_prob 0.07 실측. 규칙 보조 레이어를 OR로
  결합해 재현율 확보(플래그 추가만, 해제 없음 → 오탐 소폭 증가 트레이드오프).
  근본 해결은 사기 유형 합성 증강 + 재학습 — 후속 계획으로 보고서에 기술.
- **f_dis_match 대응**: 학습의 disability_type 일치 → 서빙에선 ui_mode 일치.
  서버 내부 특성 전용, 추천 사유로 노출 금지 (MATCH-04).
- **bio 폴백**: bio가 비면 관심사 태그 문자열로 임베딩 (MATCH-02-8).
- 응답의 `risk_prob`·`level`·`probs`는 계약 외 부가 필드 — 백엔드는 `verdict`만 사용,
  데모·리포트 시연(실시간 위험도 스코어러)에 활용 가능.
