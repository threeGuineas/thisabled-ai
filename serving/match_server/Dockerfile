# MATCH 모델 서버 — 빌드 컨텍스트는 저장소 루트: docker build -f serving/match_server/Dockerfile .
FROM python:3.11-slim

WORKDIR /srv
COPY serving/match_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SBERT를 이미지에 미리 내려받아 기동 시 네트워크 의존 제거
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sroberta-multitask')"

COPY models/checkpoints/module2_lambdamart_embedding.pkl models/checkpoints/
COPY serving/match_server/app.py .

ENV MATCH_MODEL_PATH=models/checkpoints/module2_lambdamart_embedding.pkl
EXPOSE 9002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9002"]
