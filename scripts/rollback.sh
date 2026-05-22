#!/usr/bin/env bash
# Emergency rollback to pre-PNP-refactor checkpoint
# Usage: bash scripts/rollback.sh

set -euo pipefail

echo "=== ROLLBACK to pre-pnp-refactor checkpoint ==="
cd "$(dirname "$0")/.."

if git tag -l pre-pnp-refactor | grep -q .; then
    echo "Restoring git state..."
    git reset --hard pre-pnp-refactor
    echo "Git state restored to: $(git log --oneline -1 pre-pnp-refactor)"
else
    echo "ERROR: tag 'pre-pnp-refactor' not found"
    echo "Last known good state: $(git log --oneline -1)"
    exit 1
fi

# If build was running, stop it
if systemctl --user status build-index &>/dev/null 2>&1; then
    echo "Stopping build-index service..."
    systemctl --user stop build-index 2>/dev/null || true
fi

# Clean up build artifacts
echo "Cleaning build artifacts..."
rm -f data/index.faiss data/chunks.db data/last_updated.txt

echo "=== ROLLBACK COMPLETE ==="
echo "To verify: git log --oneline -3"
