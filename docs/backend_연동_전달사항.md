# BE 전달 — SAFE·MATCH 실모델 연동 안내 (7/8)

> 결론: **mock 컨테이너 2개를 실모델 서버로 교체.** 계약·포트·서비스명 동일 → **백엔드 코드 수정 0줄.**
> compose 6줄 수정 + `.env` 1줄이 전부라 BE가 직접 하는 걸 제안 (원하면 PR로 보내줄 수도 있음 — 아래 진행 방식 참조).

## 1. 무엇이 준비됐나

- AI 레포(`thisabled-ai`) `serving/`에 모델 서버 2개 — **push 완료**
  - `safety-model :9001` — KcELECTRA 파인튜닝 + 금전사기 규칙 하이브리드. `POST /analyze {text, receiver_is_minor}` → `{verdict}`
  - `match-model :9002` — SBERT 임베딩 + LambdaMART + 점수 블렌드. `POST /score {me, candidates}` → `{results:[{user_id, score, reasons}]}`
- 모델 파일은 **HF private repo**에서 기동 시 자동 다운로드 → **체크포인트 파일을 받을 필요 없음** (Drive도 불필요)
- 스모크 실측: SAFE median **48ms** (타임아웃 예산 2s), MATCH **364ms** (예산 10s), 계약 테스트 통과

## 2. BE가 할 일 (10분)

### 2-1. AI 레포 clone (백엔드 레포 옆에)

```bash
cd <백엔드 레포 상위 디렉터리>
git clone https://github.com/threeGuineas/thisabled-ai.git   # 체크포인트 없어도 됨
```

### 2-2. `.env`에 토큰 추가

```
HF_TOKEN=hf_...   # 별도 채널로 전달함 (fine-grained read, 모델 repo 2개만 접근 가능)
```

### 2-3. `docker-compose.yml` 두 서비스 교체 (기존 mock 설정은 주석으로 보존!)

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
```

`volumes:` 섹션에 `hf_cache_safety:`, `hf_cache_match:` 추가.

### 2-4. 기동·검증

```bash
docker compose up -d --build safety-model match-model
docker compose logs -f safety-model    # 첫 기동: 모델 다운로드(~500MB) → "Application startup complete"
docker compose exec app curl -s http://safety-model:9001/health
docker compose exec app curl -s http://match-model:9002/health
docker compose exec -T app pytest -q   # 그린 확인
```

E2E: 채팅에서 `"계좌번호 알려주면 돈 보내줄게. 급한 거니까 빨리"` 전송 → 수신자 화면 블러 확인.

## 3. 참고

- **응답의 부가 필드** (`risk_prob`, `level`, `probs`, `rule_assist`, `model_score`)는 계약 외 디버그·시연용 — 백엔드는 기존대로 `verdict`/`score`/`reasons`만 쓰면 됨
- **롤백**: 주석 처리한 mock 설정 복구 후 재기동 (§18.3 장애 시연·영상 촬영 폴백용으로도 필요하니 mock 삭제 금지)
- **운영 설정값**(임계값·블렌드 가중치 등)은 모델 서버 env로 조정 — 목록은 `thisabled-ai/serving/README.md`
- 토큰은 중간평가 후 revoke 예정. `.env`는 커밋 금지 (기존 규칙대로)

## 4. 진행 방식 (선택해줘)

- **A. BE가 직접** (추천): 위 절차 그대로. compose는 백엔드 레포 파일이라 이 편이 자연스러움
- **B. AI가 PR**: 원하면 compose 수정 PR 보낼게 — 단, `context: ../thisabled-ai` 상대 경로가 각자 clone 위치에 의존하니 머지 전에 경로 컨벤션(레포 나란히 clone)만 합의 필요

문제 생기면 `docker compose logs safety-model match-model` 로그 캡처해서 보내줘.
