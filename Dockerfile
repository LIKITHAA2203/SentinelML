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

# Copy application source code and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY README.md ./

# Create runtime directories
RUN mkdir -p data/raw data/processed artifacts/models reports/figures

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501 5000

# Default entrypoint starts the FastAPI inference server
CMD ["uvicorn", "sentinelml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
