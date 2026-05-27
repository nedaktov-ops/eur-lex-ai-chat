# EUR-Lex AI Chat — Next Steps After Build

> **Build state:** Incremental SQLite download in progress (~172K docs, 10 types from 1952)
> **Status:** Post-build validation + deployment checklist

---

## Phase 1: Verify Upload to HuggingFace Hub

After `build_index.py` completes, confirm the HF Hub dataset has the right files:

```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('NedAktovOps/eurlex-chat-data', repo_type='dataset')
print('Files:', files)
"
```

**Expected:** `['index.faiss', 'chunks.db', 'last_updated.txt', 'build_meta.json']`

**Check sizes:**
```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
meta = api.get_repo_revision('NedAktovOps/eurlex-chat-data', repo_type='dataset')
for f in ['index.faiss', 'chunks.db']:
    info = api.get_paths_info('NedAktovOps/eurlex-chat-data', paths=[f], repo_type='dataset')
    size_mb = info[0].size / 1e6 if info else 0
    print(f'{f}: {size_mb:.1f} MB')
"
```

**Verify FAISS loads:**
```bash
python3 -c "
import faiss
from huggingface_hub import hf_hub_download
path = hf_hub_download('NedAktovOps/eurlex-chat-data', 'index.faiss', repo_type='dataset')
idx = faiss.read_index(path)
print(f'Vectors: {idx.ntotal}, Dim: {idx.d}')
"
```

**Verify SQLite loads:**
```bash
python3 -c "
import sqlite3
from huggingface_hub import hf_hub_download
path = hf_hub_download('NedAktovOps/eurlex-chat-data', 'chunks.db', repo_type='dataset')
conn = sqlite3.connect(path)
count = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
celexes = conn.execute('SELECT COUNT(DISTINCT celex) FROM chunks').fetchone()[0]
print(f'Chunks: {count}, Unique CELEXes: {celexes}')
conn.close()
"
```

---

## Phase 2: Run RAGAS Evaluation

Evaluate the new index against the ground-truth QA dataset (121 Q/A pairs):

```bash
HF_TOKEN="hf_xxx" OPENAI_API_KEY="sk-xxx" venv/bin/python3 scripts/evaluate.py
```

**Thresholds** (from `.github/workflows/evaluate.yml`):
- `faithfulness > 0.7`
- `context_recall > 0.6`

If thresholds are met, the GitHub Actions CI gate will pass.

---

## Phase 3: Update BM25 Store

The BM25 store needs to be rebuilt with the new chunks for hybrid search:

```bash
venv/bin/python3 -c "
from app.bm25_store import BM25Store
store = BM25Store()
store.build_from_db('data/chunks.db')
store.save('data/bm25_store.pkl')
"
```

---

## Phase 4: Run Coverage Benchmark

Compare the new index coverage against EUR-Lex availability:

```bash
venv/bin/python3 -c "
from scripts.build_index import query_all_documents
import sqlite3
docs = query_all_documents()
conn = sqlite3.connect('data/chunks.db')
indexed = conn.execute('SELECT COUNT(DISTINCT celex) FROM chunks').fetchone()[0]
conn.close()
print(f'Indexed: {indexed} / {len(docs)} available ({100*indexed/len(docs):.1f}%)')
"
```

Update the coverage badge in `README.md`.

---

## Phase 5: Update README

- [ ] Update coverage percentage (current: 86.77%)
- [ ] Update chunk count (target: 3M+)
- [ ] Update document count
- [ ] Verify live demo URLs still work
- [ ] Update architecture diagram if flow changed

---

## Phase 6: Commit & Push

```bash
git add -A
git status  # verify only intended files
git commit -m "feat: full index rebuild with 10 doc types from 1952

- 172K documents, 3M+ chunks across 10 EU doc types
- CDN primary fetch + SPARQL/PDF fallback chain
- Incremental SQLite flushing for OOM-safe builds
- Streaming embedding from SQLite (low memory)"
git push origin main
```

This triggers auto-deploy to:
- **HF Spaces** — Docker build, 512MB RAM
- **Render** — uvicorn backend, 512MB RAM
- **Vercel** — Astro frontend

---

## Phase 7: Verify Deployments

### 7.1—Check HF Spaces
```bash
curl -s https://nedaktovops-eurlex-chat-api.hf.space/health | python3 -m json.tool
```
Expected: `"index_loaded": true, "ntotal": > 0, "size": > 0`

### 7.2—Check Render
```bash
curl -s https://eurlex-chat-api.onrender.com/health | python3 -m json.tool
```
Expected: same

### 7.3—Check Vercel Frontend
Visit: https://frontend-ruddy-zeta-40.vercel.app
- Page renders with chat interface
- Model selector (MiniLM / EURLEX-BERT) works
- Query returns answer with CELEX citations

### 7.4—Verify Model Selection
```bash
curl -s https://nedaktovops-eurlex-chat-api.hf.space/models
```
Expected: List of available embedding models

---

## Phase 8: End-to-End Tests

| # | Test | Expected |
|---|------|----------|
| 1 | `POST /chat {"query": "What is the DMA?"}` | Answer with CELEX citations |
| 2 | `POST /chat {"query": "A" * 2001}` | 400 error |
| 3 | `POST /chat {"query": ""}` | 400 error |
| 4 | 21 requests in 1s | 429 on 21st |
| 5 | `GET /refresh` | `"status": "current"` |
| 6 | `POST /feedback {"query": "...", "response": "...", "rating": 1}` | 200 OK |

---

## Phase 9: Memory Monitoring

After deployment, verify under 512MB limit:

- [ ] HF Space dashboard: RSS < 400MB
- [ ] Render dashboard: RSS < 400MB
- [ ] Run 50 parallel queries, check no OOM

```bash
for i in $(seq 1 50); do
  curl -X POST https://nedaktovops-eurlex-chat-api.hf.space/chat \
    -H "Content-Type: application/json" \
    -d '{"query":"What are the rules for digital markets?"}' &
done
wait
```

---

## Phase 10: CI/CD Pipeline Activation

After code is pushed, verify GitHub Actions:

- [ ] `ci.yml` — Unit tests pass (15 tests)
- [ ] `evaluate.yml` — RAGAS metrics pass thresholds
- [ ] `build-index.yml` — Triggered on schedule
- [ ] `backup.yml` — Backup created
- [ ] `refresh-index.yml` — Index refresh works

Check:
```bash
gh run list --workflow=ci.yml --limit=3
```

---

## Rollback Plan

If anything fails after upload:

```bash
# 1. Delete new files
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
for f in ['index.faiss', 'chunks.db', 'last_updated.txt', 'build_meta.json']:
    try:
        api.delete_file('NedAktovOps/eurlex-chat-data', f, repo_type='dataset')
    except: pass
"

# 2. Restore from backup
python3 scripts/build_index.py --from-backup
```

For code rollback:
```bash
git revert HEAD
git push origin main
```
