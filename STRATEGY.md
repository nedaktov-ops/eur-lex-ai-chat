# EUR-Lex AI Chat — Implementation Strategy

> **Date:** 2026-05-22
> **Project:** eur-lex-ai-chat — RAG chatbot over 38K+ EU regulations & directives
> **Constraint:** 512MB RAM on HuggingFace Spaces, 512MB on Render free tier
> **Current State:** All 13 source files modified; uncommitted; old index (739 docs) on HF Hub

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 0: Pre-Flight Verification](#2-phase-0-pre-flight-verification)
3. [Phase 1: Build Pipeline Execution](#3-phase-1-build-pipeline-execution)
4. [Phase 2: Upload & HF Hub Verification](#4-phase-2-upload--hf-hub-verification)
5. [Phase 3: Code Commit & Deployment](#5-phase-3-code-commit--deployment)
6. [Phase 4: Post-Deployment Verification](#6-phase-4-post-deployment-verification)
7. [Rollback Plan](#7-rollback-plan)
8. [Success Criteria](#8-success-criteria)
9. [Checkpoint Timeline](#9-checkpoint-timeline)

---

## 1. Architecture Overview

```
                     ┌──────────────────────────────────────┐
                     │        build_index.py (laptop)         │
                     │  SPARQL → XHTML → Parse → Embed → FAISS│
                     │         → SQLite → Upload to HF Hub    │
                     └──────────────┬───────────────────────┘
                                    │ index.faiss + chunks.db
                                    ▼
               ┌────────────────────────────────────────┐
               │     HuggingFace Hub Dataset             │
               │  NedAktovOps/eurlex-chat-data            │
               │  ├── index.faiss    (FAISS IVFPQ, ~32MB) │
               │  ├── chunks.db      (SQLite, ~35MB)     │
               │  └── last_updated.txt                   │
               └────────────┬───────────────────────────┘
                            │ hf_hub_download()
              ┌─────────────┴─────────────┐
              │                           │
     ┌────────▼────────┐       ┌─────────▼────────┐
     │  HF Space API   │       │  Render API       │
     │  (Docker)       │       │  (startup.sh)     │
     │  uvicorn main   │       │  uvicorn main     │
     │  512MB RAM      │       │  512MB RAM        │
     └────────┬────────┘       └─────────┬────────┘
              │                           │
              └──────────┬───────────────┘
                         │
              ┌──────────▼──────────┐
              │   Astro Frontend    │
              │   (Vercel)          │
              └─────────────────────┘
```

### Memory Budget (HF Space / Render)

| Component | Memory | Notes |
|-----------|--------|-------|
| SentenceTransformer model | ~250MB | all-MiniLM-L6-v2 |
| FAISS IVFPQ index | ~32MB | compressed from 768MB |
| SQLite connection | ~0MB | on-disk, zero RAM |
| Query vector + scratch | ~8MB | 384-dim float32 |
| Python/uvicorn overhead | ~50MB | interpreter + libs |
| **Total** | **~340MB** | well under 512MB limit |

---

## 2. Phase 0: Pre-Flight Verification

**Goal:** Validate every assumption before touching production data.

### Checklist

- [ ] 0.1 — `faiss-cpu==1.13.2` import works in venv
- [ ] 0.2 — SPARQL endpoint responds with expected count (~36K docs from 2004+)
- [ ] 0.3 — Cellar XHTML endpoint returns valid content with new headers
- [ ] 0.4 — Parsing produces chunks (sample 3 documents)
- [ ] 0.5 — Embedding works and produces normalized 384-dim vectors
- [ ] 0.6 — FAISS IVFPQ trains and saves/loads correctly
- [ ] 0.7 — SQLite round-trip: write chunks, read them back
- [ ] 0.8 — HF Hub token can read old files and write new files
- [ ] 0.9 — Git diff is clean (only intended files modified)
- [ ] 0.10 — Build script syntax-check passes

### Rollback Gate

If ANY of 0.1, 0.2, 0.3, or 0.8 fail → **STOP**. Fix before proceeding.

---

## 3. Phase 1: Build Pipeline Execution

**Goal:** Run `build_index.py` to produce FAISS index + SQLite DB from 36K+ EUR-Lex documents.

### Estimated Duration

| Step | Est. Time | Notes |
|------|-----------|-------|
| SPARQL query | ~30s | 26,871 docs from server-side xsd:dateTime FILTER |
| Download & parse (20 workers) | ~19 min | 0.84s avg × 26,871 ÷ 20 workers (~88% success) |
| Embedding (batch 128) | ~6 min | ~456K chunks × 384-dim, benchmarked at ~0.1s/batch |
| FAISS training | ~15 min | IVFPQ on ~456K vectors, 2,683 centroids |
| SQLite build | ~3 min | executemany batch insert, ~35MB DB |
| Upload to HF Hub | ~5 min | ~70MB total (index + db + timestamp) |
| **Total** | **~50 min** | Fast enough for daytime execution |

### Execution Steps

#### 1.1 — SPARQL Query Test (dry-run)

```
python3 -c "
from scripts.build_index import query_all_documents
docs = query_all_documents()
print(f'{len(docs)} docs found')
if docs:
    print(f'  First: {docs[0][\"celex\"]} ({docs[0][\"date\"]})')
    print(f'  Last:  {docs[-1][\"celex\"]} ({docs[-1][\"date\"]})')
"
```

Expected: ~36,000 docs, dates from 2004-01-01 to present.

#### 1.2 — Sample Fetch Test

```
python3 -c "
from scripts.build_index import fetch_document_xhtml
doc = {'celex': '32024R1234', 'title': '', 'date': '2024-01-01', 
       'type': 'REG', 'cellar_url': 'https://...'}
html = fetch_document_xhtml(doc)
print(f'Got {len(html)} chars' if html else 'FAILED')
"
```

Expected: XHTML content, 500+ chars.

#### 1.3 — Full Build

```
HF_TOKEN=hf_xxx python3 scripts/build_index.py 2>&1 | tee data/build-$(date +%Y%m%d-%H%M).log
```

#### 1.4 — Monitor Checkpoints

During the build, watch for:
- **Download phase:** Success rate should be ~80-85% (XHTML coverage). Log warns on empty content.
- **Parse phase:** Expect ~500K+ chunks from ~30K successful documents.
- **Embed phase:** Progress logged every 10 batches (1280 chunks).
- **FAISS train:** Logs centroid count and training status.
- **SQLite build:** Logs chunk count and DB size.
- **Upload:** Uploads index.faiss + chunks.db, deletes old vectors.npy + chunks.json.

#### 1.5 — Failure Recovery

If build fails mid-way:
1. Check logs for error type
2. Fix the issue in code
3. Delete partial output files (`rm -f data/index.faiss data/chunks.db`)
4. Re-run with same HF_TOKEN (upload will overwrite)

---

## 4. Phase 2: Upload & HF Hub Verification

**Goal:** Confirm HF Hub has the correct files in the new format.

### Verification Commands

```python
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('NedAktovOps/eurlex-chat-data', repo_type='dataset')
print(files)
# Expected: ['index.faiss', 'chunks.db', 'last_updated.txt']
# NOT expected: ['vectors.npy', 'chunks.json']  (should be deleted)
```

### Integrity Checks

- [ ] `index.faiss` exists and is >10MB (expected ~32MB for 500K vectors)
- [ ] `chunks.db` exists and is >10MB (expected ~35MB)
- [ ] `last_updated.txt` contains valid ISO timestamp
- [ ] Old `vectors.npy` and `chunks.json` are **gone**
- [ ] FAISS index loads without error and `ntotal` matches expected vector count
- [ ] SQLite DB opens and `SELECT COUNT(*)` matches expected chunk count

### Rollback Gate

If index.faiss or chunks.db are missing or corrupt → **STOP.** Restore old files:
```python
# Manual rollback — re-upload the old backup
api.upload_file(repo_id=..., path_in_repo="vectors.npy", path_or_fileobj="data/vectors.npy", ...)
api.upload_file(repo_id=..., path_in_repo="chunks.json", path_or_fileobj="data/chunks.json", ...)
```

---

## 5. Phase 3: Code Commit & Deployment

**Goal:** Push all source changes to GitHub → auto-deploy to HF Spaces, Render, Vercel.

### Step 3.1 — Commit

```bash
git add -A
git commit -m "feat: FAISS IVFPQ + SQLite for memory-optimized vector search

- Replace numpy brute-force search with FAISS IVFPQ index (~32MB vs 768MB)
- Replace chunks.json RAM load with on-disk SQLite (zero query-time RAM)
- Streaming build pipeline (no 7.6GB html_results dict)
- SPARQL date filter via xsd:dateTime (server-side, not in Python)
- Cellar XHTML fetch with proper User-Agent + Accept headers
- Title extraction from XHTML metadata
- Coverage disclaimer on frontend (~62%, 2004-2023 only)
- Both hfspace/ (Docker) and backend/ (Render) deployments updated"
git push origin main
```

### Step 3.2 — Verify Deployments

| Platform | Trigger | URL to Check | Expected |
|----------|---------|-------------|----------|
| HuggingFace Spaces | Auto-build from Dockerfile on push | `https://...hf.space/health` | `index_loaded: false` (no new index yet) |
| Render | Auto-deploy from repo | `https://...render.com/health` | `index_loaded: false` |
| Vercel | Auto-deploy from repo | `https://...vercel.app/` | Shows new disclaimer |

> **Note:** `index_loaded: false` is expected because the old index format (vectors.npy + chunks.json) was deleted from HF Hub. The new index (index.faiss + chunks.db) was uploaded in Phase 2. Once the services restart, they'll download the new files automatically.

### Step 3.3 — Wait for Deployments

- HF Spaces: ~5 min build + ~1 min start
- Render: ~3 min build + ~1 min start
- Vercel: ~2 min build

### Rollback Gate

If any deployment fails → Check logs. If it's a code issue:
```bash
git revert HEAD
git push origin main
```
Then fix the issue in a new branch.

---

## 6. Phase 4: Post-Deployment Verification

**Goal:** Confirm the full stack works end-to-end under the 512MB memory constraint.

### Test Matrix

| # | Test | Command/URL | Expected |
|---|------|-------------|----------|
| 4.1 | Health endpoint | `GET /health` | `status: ok, index_loaded: true, ntotal > 0, size > 0` |
| 4.2 | Simple query | `POST /chat {"query": "What is the DMA?"}` | Returns answer with citations |
| 4.3 | Long query | `POST /chat {"query": "A"*2001}` | 400 error |
| 4.4 | Rate limiting | 21 requests in 1s | 429 on 21st |
| 4.5 | Empty query | `POST /chat {"query": ""}` | 400 error |
| 4.6 | Memory check | HF Space dashboard | < 400MB RSS |
| 4.7 | Refresh endpoint | `GET /refresh` | `status: "current"` |
| 4.8 | Frontend loads | Browser to Vercel URL | Page renders, disclaimer visible |

### Memory Stress Test

```bash
# Fire 50 parallel requests to simulate load
for i in $(seq 1 50); do
  curl -X POST https://$API_URL/chat \
    -H "Content-Type: application/json" \
    -d '{"query":"What are the rules for digital markets?"}' &
done
wait
# Check memory didn't spike above 480MB
```

### Monitoring

After deployment, monitor for 24 hours:
- HF Space logs for OOM kills
- Render logs for crash loops
- Vercel analytics for error rates
- HF Hub download counts (index.faiss and chunks.db should download once per restart)

---

## 7. Rollback Plan

### Scenario A: Build Fails During Execution

**Symptom:** Script crashes or produces corrupt output.

**Action:**
1. Read the log to identify the failure point
2. Fix the code issue
3. Delete partial `data/index.faiss` and `data/chunks.db`
4. Re-run from scratch

### Scenario B: HF Hub Upload Contains Wrong Files

**Symptom:** Wrong file sizes, missing files, or old files still present.

**Action:**
```bash
# Delete bad files
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
for f in ['index.faiss', 'chunks.db']:
    api.delete_file('NedAktovOps/eurlex-chat-data', f, repo_type='dataset')
# Re-upload old backup files
api.upload_file('NedAktovOps/eurlex-chat-data', 'vectors.npy', 'data/vectors.npy', repo_type='dataset')
api.upload_file('NedAktovOps/eurlex-chat-data', 'chunks.json', 'data/chunks.json', repo_type='dataset')
"
```

### Scenario C: Deployment Fails

**Symptom:** HF Space or Render returns 503 or crashes on startup.

**Action:**
```bash
git revert HEAD
git push origin main
# Wait for re-deploy
# Debug the issue locally in a branch
```

### Scenario D: Out of Memory in Production

**Symptom:** 512MB OOM killer kills the process.

**Action:**
1. Check memory metrics in HF Space dashboard
2. If near limit: reduce `nprobe` from 8 to 4 (faster, slightly less accurate)
3. If still OOM: switch to IVF without PQ, or reduce centroid count
4. Worst case: fall back to brute-force with memory-mapped numpy

---

## 8. Success Criteria

The implementation is complete when ALL of these are true:

- [ ] `build_index.py` produces `index.faiss` (with `ntotal == expected vector count`)
- [ ] `build_index.py` produces `chunks.db` (with matching row count)
- [ ] Both uploaded to HF Hub, old files deleted
- [ ] Git committed and pushed to GitHub
- [ ] HF Space `/health` returns `index_loaded: true`
- [ ] Render `/health` returns `index_loaded: true`
- [ ] Frontend shows coverage disclaimer
- [ ] A test query returns a relevant answer with CELEX citations
- [ ] Memory stays under 400MB during query
- [ ] All 13 source file changes are live in production

---

## 9. Checkpoint Timeline

```
T+0:00 — Phase 0 Verification (15 min)
T+0:15 — Phase 1 Build starts (~50 min)
  ├── T+0:15 — SPARQL query returns (26,871 docs)
  ├── T+0:16 — Download phase begins (20 workers)
  ├── T+0:25 — MID-DOWNLOAD CHECK: ~8,000 processed
  ├── T+0:35 — Download complete (~22,800 successful), parse done
  ├── T+0:36 — Embedding begins (~456K chunks @ batch 128)
  ├── T+0:42 — Embedding complete
  ├── T+0:43 — FAISS training begins
  ├── T+0:58 — FAISS + SQLite done
  └── T+1:00 — Upload complete
T+1:02 — Phase 2 Verification (10 min)
T+1:12 — Phase 3 Commit & Deploy (10 min)
T+1:22 — Phase 4 End-to-End Verification (20 min)
T+1:42 — DONE

---

## Appendix: Key File Mapping

| File | Purpose | Deployed To |
|------|---------|-------------|
| `scripts/build_index.py` | Full build pipeline | Dev laptop only |
| `hfspace/data_loader.py` | FAISS + SQLite download/load | HF Space |
| `hfspace/search.py` | FAISS search + SQLite lookup | HF Space |
| `hfspace/main.py` | FastAPI app | HF Space |
| `hfspace/Dockerfile` | Docker build for HF Space | HF Space |
| `hfspace/requirements.txt` | Python deps | HF Space |
| `backend/data_loader.py` | FAISS + SQLite download/load | Render |
| `backend/search.py` | FAISS search + SQLite lookup | Render |
| `backend/main.py` | FastAPI app | Render |
| `backend/requirements.txt` | Python deps | Render |
| `backend/startup.sh` | Render entry point | Render |
| `frontend/src/pages/index.astro` | Frontend (Astro) | Vercel |
