FROM python:3.11-slim

WORKDIR /app

# Install system deps (FAISS needs libomp, torch needs libopenblas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    libopenblas-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (sentence-transformers dependency)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements (from build context app/ to /app/requirements.txt)
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# Copy application code into /app/app/ (package structure for relative imports)
COPY app/ /app/app/

# HF Spaces sets $PORT, default 7860
EXPOSE 7860

# Run uvicorn targeting app.main:app (the app package at /app/app/)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
