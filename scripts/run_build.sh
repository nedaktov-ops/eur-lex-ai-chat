#!/bin/bash
# Script to run the full build process
# Usage: ./scripts/run_build.sh

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
elif [ -f "../../.venv/bin/activate" ]; then
    source ../../.venv/bin/activate
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    exit 1
fi

# Run the build script
python3 scripts/build_index.py