# SentinelML Multi-Service Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, configs, and pre-trained artifacts
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY README.md ./
COPY artifacts/ ./artifacts/

# Ensure runtime directories exist
RUN mkdir -p data/raw data/processed artifacts/models reports/figures

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Failsafe: if model artifacts are not present in build context, auto-generate dataset and train
RUN if [ ! -f artifacts/models/model_metadata.json ] || [ ! -f artifacts/models/fraud_pipeline.joblib ]; then \
        echo "Artifacts missing in build context. Generating baseline demo dataset and training model..." && \
        python3 scripts/generate_demo_data.py --samples 5000 --out data/raw/creditcard.csv && \
        PYTHONPATH=/app/src python3 -m sentinelml.training.train; \
    fi

EXPOSE 8000 8501 5000

# Default entrypoint starts the FastAPI inference server
CMD ["uvicorn", "sentinelml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
