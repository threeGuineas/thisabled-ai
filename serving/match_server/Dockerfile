# MATCH 모델 서버 — v2 LambdaMART pkl은 이미지에 굽지 않고 기동 시 HF Hub에서 로드.
# 빌드: docker build -f serving/match_server/Dockerfile .   (체크포인트 파일 불필요)
# 실행: -e HF_TOKEN=<soyuncj/module2 read 토큰>   (v2 좌표는 아래 ENV 기본값)
FROM python:3.11-slim

WORKDIR /srv
# HF 캐시 위치를 먼저 고정해야 빌드 시 받은 SBERT를 런타임에서 재사용한다.
ENV HF_HOME=/srv/hf-cache

COPY serving/match_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SBERT(공개 모델)는 빌드 시 미리 내려받아 이미지에 포함 → 기동 시 재다운로드 방지
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sroberta-multitask')"

COPY src ./src
COPY configs/module2_matching.yaml ./configs/module2_matching.yaml
COPY serving/match_server/app.py .

# v2 랭커를 HF에서 받아 서빙(이미지 기본값). HF_TOKEN만 런타임에 주입한다.
# 롤백: 실행 시 -e MATCH_FEATURE_SCHEMA=legacy-v1 등으로 덮거나 이전 이미지로 되돌린다.
ENV MATCH_FEATURE_SCHEMA=match-input-v2 \
    MATCH_HF_REPO=soyuncj/module2 \
    MATCH_HF_FILE=module2_lambdamart_v2.pkl \
    MATCH_HF_REVISION=ecb31a428e74dfc393617a6a4a95ecc4cb7e6d67

EXPOSE 9002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9002"]
