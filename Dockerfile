# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

# Install build deps for scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy just what we need
COPY config.yaml ./config.yaml
COPY artifacts ./artifacts
COPY src ./src

# Cloud Run (and other platforms) usually set PORT
ENV PORT=8000

EXPOSE 8000

# Use $PORT if Cloud Run overrides it
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
