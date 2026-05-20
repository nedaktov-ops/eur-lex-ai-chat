#!/bin/bash
# Render entry point — downloads index and starts uvicorn
set -e

cd "$(dirname "$0")"

echo "=== EUR-Lex AI Chat Backend Startup ==="
echo "Python: $(python3 --version)"

exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
