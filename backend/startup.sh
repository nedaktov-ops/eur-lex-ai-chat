#!/bin/bash
# Render entry point — downloads index and starts uvicorn
set -e

cd "$(dirname "$0")"

echo "=== EUR-Lex AI Chat Backend Startup ==="
echo "Python: $(python3 --version)"
echo "Installing CPU-only PyTorch (smaller for Render)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet --no-cache-dir
pip install sentence-transformers --quiet --no-cache-dir
pip install faiss-cpu --quiet --no-cache-dir
pip install transformers --quiet --no-cache-dir

echo "Starting uvicorn..."
echo "INDEX_SUFFIX=${INDEX_SUFFIX:-<unset>}"

exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
