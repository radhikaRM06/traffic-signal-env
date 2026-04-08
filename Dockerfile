FROM python:3.11-slim

LABEL description="Traffic Signal Control — OpenEnv Environment"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/ ./src/
COPY server/ ./server/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir openenv-core || true

COPY app.py .
COPY openenv.yaml .
COPY inference.py .
COPY pyproject.toml .
COPY README.md .

ENV TRAFFIC_TASK=single_intersection_easy
ENV TRAFFIC_SEED=42
ENV PYTHONPATH=/app/src
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
