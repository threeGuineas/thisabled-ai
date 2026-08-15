# BE 전달 — 모델 적용 안내 (8/15 기준)

> **백엔드 코드 수정 0줄.** 계약·포트·서비스명 그대로다. compose의 `environment:` 블록만 바꾸면 된다.
> 7/8자 `backend_연동_전달사항.md`를 대체한다. 그 문서에서 **바뀐 점 3가지**를 먼저 확인할 것.

## 0. 7/8 안내에서 바뀐 것

| | 7/8 안내 | 지금 |
| --- | --- | --- |
| HF 토큰 | `HF_TOKEN` 필요 (private repo) | **불필요.** 두 모델 저장소 모두 공개로 전환됨 |
| MATCH 저장소 | `soyuncj/thisabled-match-lambdamart` | **`soyuncj/module2`** (v2 재학습본) |
| revision | 미고정 (latest) | **커밋 SHA 고정 필수** |

`.env`의 `HF_TOKEN`은 지워도 되고 남겨둬도 무해하다. 남길 거면 유효한 토큰이어야 한다 —
**만료된 토큰이 있으면 공개 저장소 다운로드까지 401로 실패한다.** 애매하면 지우는 쪽이 안전하다.

## 1. SAFE (안심 채팅) — `safety-model :9001`

> **⚠️ revision 결정 대기 중. 이 절은 확정 전까지 적용하지 말 것.**
>
> v6(`31b33415`)이 보호집단 공정성 게이트를 넘지 못했다. KOLD 기준 위험 재현율 격차
> **0.1263** (게이트 0.10). 여성·성소수자 대상 혐오 표현을 다른 집단보다 12.6%p 덜 잡는다.
> 롤백 후보인 v4_r1(`3e9c0b80`)은 같은 축에서 **0.0906**으로 통과한다.
>
> | 측정 축 | v4_r1 | v6 | 게이트 |
> | --- | --- | --- | --- |
> | UnSmile 7집단 | 0.0207 | 0.0414 | 0.10 |
> | KOLD 7집단 | 0.0906 | **0.1263** | 0.10 ❌ |
> | 장애 도메인 | 0.0474 | 0.0384 | 0.10 |
>
> 근거: `artifacts/safe_binary_fairness_v4_3e9c0b80.json`, `..._v6_31b33415.json`
>
> v6은 blind v8(위험 탐지 성능)은 통과했다. 서로 다른 것을 재는 지표이므로 모순은 아니지만,
> W2 완료 조건이 "공정성 격차 < 10%p"인 이상 v6 배포는 그 조건을 충족하지 못한다.
> **어느 revision을 쓸지 정해진 뒤에 아래 값을 적용할 것.**

계약 그대로: `POST /analyze {text, receiver_is_minor}` → `{"verdict": "safe"|"flagged"}`

```yaml
  safety-model:
    environment:
      SAFE_MODEL_DIR: soyuncj/thisabled-safety-kcelectra
      SAFE_MODEL_REVISION: 31b334152010912ea979a7116b219f3e01c0bf94
      SAFE_FLAG_THRESHOLD: "0.73"
      SAFE_FLAG_THRESHOLD_MINOR: "0.57"
      SAFE_RULE_ASSIST: "0"
```

v4_r1로 간다면 세 값을 함께 바꾼다 — revision과 임계값은 짝이므로 따로 바꾸면 안 된다.

| | revision | 성인 | 미성년 |
| --- | --- | --- | --- |
| v6 | `31b334152010912ea979a7116b219f3e01c0bf94` | `0.73` | `0.57` |
| v4_r1 | `3e9c0b800661db9ce099782a76fbe181e8b23ab5` | `0.85` | `0.69` |

v6 상세 승인 근거와 검증 절차는 `docs/safe_v6_백엔드_배포_인계.md`에 있다.

- 정상/주의 2분류 모델이다. 서빙이 4분류와 동일하게 `verdict`로 흡수하므로 백엔드는 신경 쓸 것 없다.
- `SAFE_RULE_ASSIST: "0"` 은 반드시 유지. 규칙 보조가 켜지면 정상 문장을 과플래그한다.
- 임계값 0.73/0.57은 평가로 정한 운영점이다. revision과 짝이므로 따로 바꾸지 말 것.

## 2. MATCH (친구 추천) — `match-model :9002`

계약 그대로: `POST /score {me, candidates}` → `{results:[{user_id, score, reasons}]}`

```yaml
  match-model:
    environment:
      MATCH_FEATURE_SCHEMA: match-input-v2
      MATCH_HF_REPO: soyuncj/module2
      MATCH_HF_FILE: module2_lambdamart_v2.pkl
      MATCH_HF_REVISION: ecb31a428e74dfc393617a6a4a95ecc4cb7e6d67
```

- `MATCH_FEATURE_SCHEMA`가 이번 전환의 핵심 스위치다. 이 값이 없으면 구형 3열 모델로 뜬다.
- v2는 순수 모델 점수를 쓴다. `MATCH_W_MODEL`/`MATCH_W_TAG`/`MATCH_W_AGE`는 무시되므로 지워도 된다.
- `MATCH_COSINE_REASON_MIN`은 **제거됐다.** 설정돼 있어도 아무 동작도 하지 않는다(지우는 걸 권장).

### 추천 사유가 바뀌었다 — 프론트와 공유 필요

`reasons` 배열의 형식은 그대로이나 문구 구성이 달라졌다.

- `소개 내용이 비슷해요` **폐기** — 이제 내려가지 않는다.
- `관심사가 비슷해요` — 태그 3개 이상 겹칠 때만 (기존 1개)
- `비슷한 연령대예요` — 나이차 5세 이내일 때 (기존 연령대 일치)

카드당 사유 수가 평균 2.0 → 1.5개로 줄지만, 남는 문구는 모델 근거와 98% 일치한다.
되돌리려면 env 두 줄이면 된다: `MATCH_TAG_REASON_MIN_OVERLAP: "1"`, `MATCH_AGE_REASON_MAX_DIFF: "99"`.

## 3. 소통 코치 — 아직 배포 대상 아님

프롬프트만 작성된 상태다. 서버도 엔드포인트도 없다. compose에 추가할 것 없음.

## 4. 기동·검증

```bash
docker compose up -d --build safety-model match-model
docker compose logs -f safety-model     # 첫 기동은 모델 다운로드(~500MB)

docker compose exec app curl -s http://safety-model:9001/health
docker compose exec app curl -s http://match-model:9002/health
```

확인할 값:

| 서버 | 필드 | 기대값 |
| --- | --- | --- |
| safety | `revision` | `31b33415...`로 시작 |
| safety | `num_labels` | `2` |
| safety | `threshold` / `threshold_minor` | `0.73` / `0.57` |
| match | `feature_schema` | `match-input-v2` |
| match | `model` | 경로에 `ecb31a42...` 포함 |

`revision`이 `3e9c0b...`로 보이면 이전 모델이다. 환경변수가 컨테이너에 안 들어간 것이니 배포 성공으로 보지 말 것.

`feature_schema`가 `legacy-v1`로 보이면 env가 컨테이너에 안 들어간 것이다.
v2 스키마인데 v2 번들이 아니면 서버가 기동 자체를 거부하도록 돼 있으니, 뜬 채로 잘못된 모델을 쓰는 일은 없다.

E2E: 채팅에서 `계좌번호 알려주면 돈 보내줄게. 급한 거니까 빨리` 전송 → 수신자 화면 블러 확인.

## 5. 롤백

| 대상 | 방법 |
| --- | --- |
| MATCH v2 → 구형 | `MATCH_FEATURE_SCHEMA: legacy-v1` 후 재기동 |
| SAFE v6 → v4 | `SAFE_MODEL_REVISION: 3e9c0b800661db9ce099782a76fbe181e8b23ab5` + 임계값 `0.85`/`0.69` 로 함께 되돌린다 |
| 전체 → mock | 주석 처리해 둔 mock 설정 복구 (mock 삭제 금지 — 장애 시연 폴백) |

## 6. 참고

- 응답의 부가 필드(`risk_prob`, `level`, `probs`, `model_score`)는 디버그용이다. 계약은 `verdict` / `score`·`reasons`뿐.
- env 전체 목록은 `serving/README.md`.
- 성능 근거는 `artifacts/`의 JSON 4종에 모델 해시·revision과 함께 기록돼 있다.
