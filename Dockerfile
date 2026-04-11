FROM ghcr.io/meta-pytorch/openenv-base:latest

LABEL description="Traffic Signal Control — OpenEnv Environment"
LABEL version="1.0.0"

WORKDIR /app

COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

COPY models.py .
COPY server/ ./server/
COPY openenv.yaml .
COPY inference.py .
COPY pyproject.toml .
COPY README.md .

ENV TRAFFIC_TASK=single_intersection_easy
ENV TRAFFIC_SEED=42
ENV PORT=7860
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
