# SAFE 모델 서버 — 모델은 이미지에 포함하지 않고 기동 시 HF Hub에서 로드.
# 빌드: docker build -f serving/safety_server/Dockerfile .   (체크포인트 파일 불필요)
# 실행 필수 env: SAFE_MODEL_DIR=<hf-repo-id>, HF_TOKEN=<read token>
#   (로컬 체크포인트로 쓰려면 SAFE_MODEL_DIR에 경로 지정 + 볼륨 마운트)
FROM python:3.11-slim

WORKDIR /srv
COPY serving/safety_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/safety_server/app.py .

# HF 캐시 — compose에서 볼륨 마운트하면 재시작 시 재다운로드 없음
ENV HF_HOME=/srv/hf-cache
EXPOSE 9001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9001"]
