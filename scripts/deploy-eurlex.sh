#!/usr/bin/env bash
# EURLEX-BERT Deployment: Switch Space to EURLEX-BERT
# Usage: bash scripts/deploy-eurlex.sh <HF_TOKEN>
# Run this AFTER the build-index workflow completes successfully.

set -euo pipefail

HF_TOKEN="${1:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "Usage: $0 <HF_TOKEN>"
    echo "Run me after build-index workflow completes."
    echo ""
    echo "Workflow: https://github.com/nedaktov-ops/eur-lex-ai-chat/actions"
    exit 1
fi

echo "================================================"
echo "Phase D – Verify Artifacts on HF Dataset"
echo "================================================"

echo "Checking dataset contents..."
curl -s "https://huggingface.co/api/datasets/NedAktovOps/eurlex-chat-data" \
  -H "Authorization: Bearer $HF_TOKEN" | python3 -c "
import sys, json
d = json.load(sys.stdin)
files = [f['rfilename'] for f in d.get('siblings', [])]
required = ['index_eurlex.faiss', 'chunks_eurlex.db', 'build_meta_eurlex.json']
for f in required:
    if f in files:
        print(f'  ✓ {f}')
    else:
        print(f'  ✗ {f}  -- MISSING!')
        sys.exit(1)
print()
print('All required artifacts present.')
"

echo ""
echo "================================================"
echo "Phase E – Switch Space to EURLEX-BERT"
echo "================================================"

# Set INDEX_SUFFIX and restart Space
python3 -c "
from huggingface_hub import HfApi
import time

api = HfApi(token='$HF_TOKEN')
space_id = 'nedaktovops/eurlex-chat-api'

print('Setting INDEX_SUFFIX=_eurlex...')
api.add_space_variable(space_id, 'INDEX_SUFFIX', '_eurlex')
print('Restarting Space...')
api.restart_space(space_id)
print('Space restart initiated.')
print('Polling /health endpoint...')

import requests
base_url = 'https://nedaktovops-eurlex-chat-api.hf.space'

for i in range(30):
    time.sleep(10)
    try:
        r = requests.get(f'{base_url}/health', timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get('index_loaded'):
                print(f'  ✓ Health OK: ntotal={data[\"ntotal\"]}, loaded_at={data[\"loaded_at\"]}')
                break
        print(f'  Attempt {i+1}: status={r.status_code}, index_loaded={data.get(\"index_loaded\", False)}')
    except Exception as e:
        print(f'  Attempt {i+1}: {e}')
else:
    print('  ✗ Space did not become healthy within 5 minutes')
    print('  Run rollback: api.remove_space_variable(\"$space_id\", \"INDEX_SUFFIX\"); api.restart_space(\"$space_id\")')
    exit(1)
"

echo ""
echo "================================================"
echo "Phase F – End-to-End Test"
echo "================================================"

curl -s -X POST "https://nedaktovops-eurlex-chat-api.hf.space/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the transparency obligations for employers under the Pay Transparency Directive 2023/970?"}' \
  | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f'Answer length: {len(r.get(\"answer\", \"\"))} chars')
celex_count = r.get('answer', '').count('CELEX')
print(f'CELEX citations in answer: {celex_count}')
print(f'Confidence: {r.get(\"_confidence\", \"N/A\")}')
print(f'Citations: {r.get(\"citations\", [])}')
if len(r.get('answer', '')) >= 100 and celex_count >= 2:
    print('✓ End-to-end test PASSED')
else:
    print('✗ End-to-end test FAILED')
    exit(1)
"

echo ""
echo "================================================"
echo "Deployment complete!"
echo "================================================"
echo ""
echo "If issues occur, rollback with:"
echo "  python3 -c \"from huggingface_hub import HfApi; api = HfApi(token='$HF_TOKEN'); api.remove_space_variable('nedaktovops/eurlex-chat-api', 'INDEX_SUFFIX'); api.restart_space('nedaktovops/eurlex-chat-api')\""
