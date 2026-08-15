# SAFE 이진 v6 백엔드 배포 인계

작성일: 2026-07-11
대상: 백엔드·인프라 담당자
상태: 모델 학습, dev 게이트, fresh blind v8, Hugging Face 업로드, Google Drive 백업 완료. 실제 서빙 반영 전.

## 1. 배포 대상

| 항목 | 값 |
| --- | --- |
| HF 저장소 | `soyuncj/thisabled-safety-kcelectra` |
| 고정 revision | `31b334152010912ea979a7116b219f3e01c0bf94` |
| 모델 형식 | KcELECTRA 이진 분류 (`num_labels=2`) |
| 성인 임계값 | `0.73` |
| 미성년 임계값 | `0.57` |
| 규칙 보조 | OFF (`SAFE_RULE_ASSIST=0`) |
| 이전 revision(롤백) | `3e9c0b800661db9ce099782a76fbe181e8b23ab5` |
| 이전 임계값(롤백) | 성인 `0.85`, 미성년 `0.69` |

HF revision에서 다음 추론 파일 6개와 `num_labels=2`를 확인했다.

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`

## 2. 승인 근거

### Dev 게이트

| 지표 | 결과 | 기준 | 판정 |
| --- | ---: | ---: | --- |
| 실데이터 recall | 0.8039 | ≥ 0.80 | PASS |
| 실데이터 specificity | 0.8834 | ≥ 0.80 | PASS |
| dev recall | 0.9938 | ≥ 0.85 | PASS |
| dev specificity | 0.9250 | ≥ 0.90 | PASS |
| fraud recall | 0.9688 | ≥ 0.80 | PASS |
| coercive recall | 1.0000 | ≥ 0.80 | PASS |
| grooming recall | 1.0000 | ≥ 0.80 | PASS |
| digital extortion recall | 1.0000 | ≥ 0.80 | PASS |

### Fresh blind v8

| 항목 | 결과 |
| --- | ---: |
| 표본 수 | 40 |
| 혼동행렬 | `[[18, 2], [0, 20]]` |
| 위험 recall | 1.0000 |
| specificity | 0.9000 |
| macro F1 | 0.9499 |
| 위험 슬라이스 5종 recall | 전부 1.0000 |
| fixture SHA-256 | `e516d27779025f4a3a6b48d2c5f8eed930550bb8bf959e5324f722141c6f2fce` |

fresh blind v8은 이미 1회 소비됐다. 배포 과정에서 재평가하지 않는다. 반복 가능한 별도 smoke/regression 셋으로만 서빙을 확인한다.

## 3. 필수 환경변수

```dotenv
SAFE_MODEL_DIR=soyuncj/thisabled-safety-kcelectra
SAFE_MODEL_REVISION=31b334152010912ea979a7116b219f3e01c0bf94
SAFE_FLAG_THRESHOLD=0.73
SAFE_FLAG_THRESHOLD_MINOR=0.57
SAFE_RULE_ASSIST=0
HF_TOKEN=<private HF repo fine-grained read token>
```

`HF_TOKEN`은 저장소에 커밋하지 않는다. 배포 플랫폼 secret 또는 서버의 비공개 환경파일에 저장하며, 해당 모델 저장소를 읽을 수 있는 최소 권한 토큰을 사용한다.

## 4. Docker Compose 예시

실제 compose의 서비스명이 다르면 `safety-model`을 해당 이름으로 치환한다.

```yaml
services:
  safety-model:
    build:
      context: ../thisabled-ai
      dockerfile: serving/safety_server/Dockerfile
    environment:
      SAFE_MODEL_DIR: soyuncj/thisabled-safety-kcelectra
      SAFE_MODEL_REVISION: 31b334152010912ea979a7116b219f3e01c0bf94
      SAFE_FLAG_THRESHOLD: "0.73"
      SAFE_FLAG_THRESHOLD_MINOR: "0.57"
      SAFE_RULE_ASSIST: "0"
      HF_TOKEN: ${HF_TOKEN}
    volumes:
      - hf_cache_safety:/srv/hf-cache
    restart: unless-stopped
```

환경변수 변경 후 단순 재시작만 하지 말고 컨테이너를 재생성한다.

```bash
docker compose up -d --build --force-recreate safety-model
docker compose logs -f safety-model
```

## 5. 배포 직후 검증

### 5.1 컨테이너 환경 확인

```bash
docker compose exec safety-model printenv SAFE_MODEL_DIR
docker compose exec safety-model printenv SAFE_MODEL_REVISION
docker compose exec safety-model printenv SAFE_FLAG_THRESHOLD
docker compose exec safety-model printenv SAFE_FLAG_THRESHOLD_MINOR
docker compose exec safety-model printenv SAFE_RULE_ASSIST
```

기대값은 순서대로 다음과 같다.

```text
soyuncj/thisabled-safety-kcelectra
31b334152010912ea979a7116b219f3e01c0bf94
0.73
0.57
0
```

### 5.2 Health 확인

로컬 기본 포트가 9001인 경우:

```bash
curl -fsS http://localhost:9001/health
```

필수 확인값:

```json
{
  "status": "ok",
  "revision": "31b334152010912ea979a7116b219f3e01c0bf94",
  "loaded": true,
  "num_labels": 2
}
```

`revision`이 이전 값 `3e9c0b...`이면 환경변수가 적용되지 않은 구 컨테이너다. 배포 성공으로 간주하지 않는다.

### 5.3 `/analyze` smoke/regression

기존 38건 재평가 셋을 새 revision 로드 확인 후 다시 실행한다. HTTP 200, 응답 계약, 정상 반례 specificity, 위험·그루밍 recall을 기록한다.

주의: 직전 38건 결과는 이전 revision `3e9c0b...`에서 측정됐다. FP 4/12, specificity 66.7% 결과는 v6 성능으로 인용하면 안 된다.

검증 시 함께 기록할 항목:

- `/health.revision`
- `/health.num_labels`
- HTTP 성공 건수
- 위험 recall 및 FN 원문·`risk_prob`
- 정상 specificity 및 FP 원문·`risk_prob`
- 성인/미성년 각각의 임계값 적용 여부
- `rule_assist=false` 여부

## 6. 실패 및 롤백

다음 중 하나면 신규 revision을 승인하지 않고 롤백한다.

- 모델 다운로드·토크나이저 로딩 실패
- `/health.loaded != true`
- `/health.num_labels != 2`
- `/health.revision`이 지정 SHA와 불일치
- `/analyze` 응답 계약 또는 HTTP 안정성 회귀
- 신규 smoke/regression에서 허용하기 어려운 정상 오탐 회귀

롤백 환경값:

```dotenv
SAFE_MODEL_REVISION=3e9c0b800661db9ce099782a76fbe181e8b23ab5
SAFE_FLAG_THRESHOLD=0.85
SAFE_FLAG_THRESHOLD_MINOR=0.69
SAFE_RULE_ASSIST=0
```

롤백 후에도 컨테이너를 재생성하고 `/health`에서 이전 revision이 실제 로드됐는지 확인한다.

## 7. 백업·추적 정보

Google Drive 백업:

```text
/content/drive/MyDrive/thisabled-safe/module1_binary_hardcases_v6_r1_31b33415
```

백업에는 추론 checkpoint, 배포 메타데이터, SHA-256 manifest가 있다. 원본 `safe_blind_v8_results.json`은 런타임에 남아 있지 않아 실제 노트북 출력으로 복구한 아래 요약본을 함께 보관했다.

```text
blind_v8_evaluation_summary.recovered.json
```

이 파일은 원본 evaluator artifact가 아니라 복구 요약본임을 명시한다.

## 8. 완료 보고 양식

```text
배포 일시:
배포 환경:
서비스/컨테이너:
SAFE_MODEL_REVISION:
SAFE_FLAG_THRESHOLD / MINOR:
/health 응답:
38건 HTTP 성공:
위험 recall:
정상 specificity:
FN:
FP:
판정: PRODUCTION APPROVED | STAGING ONLY | ROLLED BACK
비고:
```
