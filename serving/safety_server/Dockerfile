# SAFE 모델 서버 — 빌드 컨텍스트는 저장소 루트: docker build -f serving/safety_server/Dockerfile .
FROM python:3.11-slim

WORKDIR /srv
COPY serving/safety_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 추론에 필요한 파일만 복사 (optimizer.pt 1GB 등 학습 상태 제외)
COPY models/checkpoints/module1_ce/config.json \
     models/checkpoints/module1_ce/model.safetensors \
     models/checkpoints/module1_ce/tokenizer.json \
     models/checkpoints/module1_ce/tokenizer_config.json \
     models/checkpoints/module1_ce/special_tokens_map.json \
     models/checkpoints/module1_ce/vocab.txt \
     models/checkpoints/module1_ce/

COPY serving/safety_server/app.py .

ENV SAFE_MODEL_DIR=models/checkpoints/module1_ce
EXPOSE 9001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9001"]
