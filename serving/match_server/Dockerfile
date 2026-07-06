# MATCH 모델 서버 — pkl은 이미지에 포함하지 않고 기동 시 HF Hub에서 로드.
# 빌드: docker build -f serving/match_server/Dockerfile .   (체크포인트 파일 불필요)
# 실행 필수 env: MATCH_HF_REPO=<hf-repo-id>, HF_TOKEN=<read token>
FROM python:3.11-slim

WORKDIR /srv
COPY serving/match_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SBERT(공개 모델)는 빌드 시 미리 내려받아 기동 시 네트워크 의존 최소화
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sroberta-multitask')"

COPY serving/match_server/app.py .

ENV HF_HOME=/srv/hf-cache
EXPOSE 9002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9002"]
