#!/bin/bash
set -e

# Evaluation shortcut script for EUR-Lex AI Chat
# Usage: ./scripts/run_evaluation.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== EUR-Lex AI Chat Evaluation ==="

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r app/requirements.txt
pip install -r requirements-dev.txt
pip install ragas

# Check for required environment variables
if [ -z "$GROQ_API_KEY" ] || [ -z "$HF_TOKEN" ]; then
    if [ -f ".env" ]; then
        echo "Loading environment from .env"
        export $(cat .env | grep -v '^#' | xargs)
    fi
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "ERROR: GROQ_API_KEY not set. Set it in environment or .env file."
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Set it in environment or .env file."
    exit 1
fi

# Run evaluation
echo "Running evaluation..."
python scripts/evaluate.py
