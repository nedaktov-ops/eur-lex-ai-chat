# The Phoenix Plan

> **EUR-Lex AI Chat — Complete Architecture Overhaul**
> Consolidation, Recovery, and EURLEX-BERT Upgrade
> Started: 2026-05-24

---

## Table of Contents

1. [Pre-Flight Baseline](#1-pre-flight-baseline)
2. [Phase 0: Codebase Consolidation](#2-phase-0-codebase-consolidation)
3. [Phase 1: CI & Automation](#3-phase-1-ci--automation)
4. [Phase 2: EURLEX-BERT Rebuild](#4-phase-2-eurlex-bert-rebuild)
5. [Phase 3: Verification & Polish](#5-phase-3-verification--polish)
6. [Rollback Procedures](#6-rollback-procedures)
7. [File Change Manifest](#7-file-change-manifest)
8. [Appendix: Research Findings](#8-appendix-research-findings)

---

## 1. Pre-Flight Baseline

**Every action in this plan starts from a known state.**

### 1.1 — Check Git Status

```bash
git status --short     # Must show only expected modifications
git log --oneline -1   # Record current commit hash
git stash list         # Ensure no hidden changes
```

### 1.2 — Run Test Suite

```bash
source venv/bin/activate
python3 -m pytest tests/ -v  # Record pass/fail counts
```

**Acceptable baseline at start (2026-05-24):**
- tests/test_answer_validation.py: 6/6 pass
- tests/test_build_index.py: 5/12 pass, 4 fail (XHTML parsing), 3 timeout (FAISS)
- tests/test_discourse_scoring.py: unknown
- tests/test_query_expansion.py: unknown
- tests/test_question_classifier.py: unknown
- tests/test_regression.py: unknown
- tests/test_relation_extraction.py: unknown

**Total: 57 collected (higher than documented 32 due to expanded build_index tests)**

### 1.3 — Verify Platform States

```bash
# HF Space: should return 503 (broken) — this is our starting point
curl https://nedaktovops-eurlex-chat-api.hf.space/health

# Vercel: should return 200 (frontend up, API down)
curl -o /dev/null -w "%{http_code}" https://frontend-ruddy-zeta-40.vercel.app

# GitHub: repo is public, unlimited Actions minutes
gh api repos/nedaktov-ops/eur-lex-ai-chat
```

### 1.4 — Tool Verification

```bash
# Docker (via sg wrapper for group access)
sg docker -c "docker run --rm hello-world"

# Node.js version check (minimum required: >=22.12.0 for Astro v6)
node --version

# Python packages
python3 -c "import torch; print('torch OK')"
python3 -c "import sentence_transformers; print(f'ST {sentence_transformers.__version__}')"
```

---

## 2. Phase 0: Codebase Consolidation

**Goal:** Single `app/` directory with the full 9-stage pipeline, deployed to HF Space via root Dockerfile.

### Step 0.1 — Investigate Test Failures

**Action:** Before any code changes, understand why build_index tests fail.

```bash
# Run failing test in isolation with full output
python3 -m pytest tests/test_build_index.py::test_parse_eli_structured_html -v --tb=long
```

**Checkpoint:** `git commit -m "checkpoint: baseline test investigation"` (if code changes needed)

**Rollback:** `git reset --hard HEAD~1`

### Step 0.2 — Create app/ Directory

**Action:** Copy all `backend/*.py` files into `app/`.

```bash
mkdir -p app
cp backend/main.py app/main.py
cp backend/data_loader.py app/data_loader.py
cp backend/search.py app/search.py
cp backend/rag.py app/rag.py
cp backend/question_classifier.py app/question_classifier.py
cp backend/query_expander.py app/query_expander.py
cp backend/relation_extractor.py app/relation_extractor.py
cp backend/answer_validator.py app/answer_validator.py
cp backend/logging_middleware.py app/logging_middleware.py
cp backend/rate_limit.py app/rate_limit.py
```

**Verification:**
- Each file exists: `ls app/*.py | wc -l` should be 10
- Import tests: `python3 -c "from app.main import app; print('OK')"`
- Full test: `python3 -m pytest tests/ -v` (same results as baseline)

**Checkpoint:**
```bash
git add app/
git commit -m "checkpoint: create app/ directory with consolidated backend"
git tag checkpoint-phase0-2
```

**Rollback:** `git reset --hard HEAD~1`

### Step 0.3 — Create Merged requirements.txt

**Action:** Combine `backend/requirements.txt` + `hfspace/requirements.txt` into `app/requirements.txt`.

```txt
torch (installed separately via startup script)
fastapi==0.136.1
uvicorn==0.47.0
numpy==2.4.6
httpx==0.28.1
huggingface_hub>=0.27.0,<1.0
sentence-transformers>=3.4.0
faiss-cpu==1.13.2
```

**Verification:**
```bash
pip install -r app/requirements.txt
python3 -c "from sentence_transformers import SentenceTransformer; print('ST OK')"
python3 -c "import faiss; print(f'faiss {faiss.__version__}')"
```

**Checkpoint:** `git add app/requirements.txt && git commit -m "checkpoint: merged requirements.txt"`. Tag `checkpoint-phase0-3`.

### Step 0.4 — Create Root Dockerfile

**Action:** Create `Dockerfile` at project root for HF Space deployment.

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch first (smaller image)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install dependencies
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# Copy project files (excluding .dockerignore'd paths)
COPY . .

# HF Space requires port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Create `.dockerignore`:**

```
.git
__pycache__
*.pyc
.venv
venv
data/
.checkpoints/
memory/
docs/
node_modules/
frontend/node_modules/
frontend/dist/
scripts/run_build.sh
*.md
```

**Verification:**
```bash
sg docker -c "docker build -t eurlex-chat ."
```

**Checkpoint:** `git add Dockerfile .dockerignore && git commit -m "checkpoint: root Dockerfile + .dockerignore"`. Tag `checkpoint-phase0-4`.

**Rollback:** `git reset --hard HEAD~1`, `docker rmi eurlex-chat`

### Step 0.5 — Docker Local Test

**Action:** Run the container locally and test all endpoints.

```bash
sg docker -c "docker run -d --name eurlex-test \
  -p 7860:7860 \
  -e GROQ_API_KEY='YOUR_GROQ_KEY' \
  -e HF_TOKEN='YOUR_HF_TOKEN' \
  eurlex-chat"

sleep 15  # Wait for model download + startup

# Health check
curl http://localhost:7860/health
# Expected: {"status": "ok", "ntotal": 305957, ...}

# Stats check
curl http://localhost:7860/stats

# Chat check (requires GROQ_API_KEY to be set)
curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the Pay Transparency Directive?"}'
# Expected: answer with citations + confidence

# Cleanup
sg docker -c "docker stop eurlex-test && docker rm eurlex-test"
```

**If health check fails:**
1. `sg docker -c "docker logs eurlex-test"` — read error
2. Fix the issue in code
3. `sg docker -c "docker build -t eurlex-chat ."` — rebuild
4. Re-run test

**Do NOT proceed until Docker test passes.**

### Step 0.6 — Push to HF Space

**Action:** Deploy the consolidated codebase.

```bash
git push hf main
```

**Wait for HF Space build (~3-5 min). Verify:**

```bash
curl https://nedaktovops-eurlex-chat-api.hf.space/health
# Expected: 200, not 503
```

**If HF Space fails:**
1. Check HF Space build logs (dashboard)
2. Fix locally
3. `git push hf main` — retry

**Checkpoint:** After successful deployment, `git tag checkpoint-phase0-6`.

**Rollback:** HF Space dashboard → Settings → "Revert to previous build"

### Step 0.7 — Delete Dead Code

**Action:** Remove `hfspace/`, `backend/`, `render.yaml`.

```bash
git rm -r hfspace/ backend/ render.yaml
git commit -m "feat: remove hfspace/ backend/ render.yaml (consolidated into app/)"
```

**Verification:**
- `ls hfspace/` → "No such file"
- `ls backend/` → "No such file"
- `python3 -m pytest tests/ -v` — same results as baseline
- `git ls-files hfspace/ backend/ render.yaml` → empty

**Checkpoint:** `git tag checkpoint-phase0-7`.

**Rollback:** `git revert HEAD`

---

## 3. Phase 1: CI & Automation

**Goal:** Test on every push, alert on failure, deploy flexibly.

### Step 1.1 — Add CI Workflow

**Action:** Create `.github/workflows/ci.yml`.

```yaml
name: CI
on:
  push:
    branches: [main]
    paths-ignore: ['docs/**', '*.md']
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install PyTorch
        run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - name: Install dependencies
        run: pip install -r app/requirements.txt
      - name: Run tests
        run: python3 -m pytest tests/ -v

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - name: Install ruff
        run: pip install ruff
      - name: Lint check
        run: ruff check .
      - name: Format check
        run: ruff format --check .

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t eurlex-chat .
```

**Verification:** Push to GitHub → all 3 jobs pass.

**Checkpoint:** `git add .github/workflows/ci.yml && commit`. Tag `checkpoint-phase1-1`.

**Rollback:** `git revert HEAD`

### Step 1.2 — Fix keepalive.yml

**Action:** Add failure notification.

```yaml
name: Keepalive
on:
  schedule:
    - cron: "*/5 * * * *"
  workflow_dispatch:

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Check API health
        run: |
          STATUS=$(curl -sS -o /dev/null -w "%{http_code}" \
            "https://nedaktovops-eurlex-chat-api.hf.space/health" || echo "000")
          if [ "$STATUS" != "200" ]; then
            echo "FAILED with status $STATUS"
            # Log to GitHub Actions annotations
            echo "::error::API health check failed with status $STATUS"
            exit 1
          fi
          echo "Health check OK (status $STATUS)"
```

**Verification:** After HF Space is healthy, this workflow runs successfully.

**Checkpoint:** `git add .github/workflows/keepalive.yml && commit`. Tag `checkpoint-phase1-2`.

### Step 1.3 — Wire VITE_API_URL

**Action:** One-line change in `frontend/src/components/ChatWidget.jsx`.

```diff
- const API_URL = "https://nedaktovops-eurlex-chat-api.hf.space";
+ const API_URL = import.meta.env.VITE_API_URL || "https://nedaktovops-eurlex-chat-api.hf.space";
```

**Verification:**
```bash
cd frontend
grep "import.meta.env.VITE_API_URL" src/components/ChatWidget.jsx
cd ..
```

**Checkpoint:** `git commit -m "feat: wire VITE_API_URL env var in ChatWidget"`. Tag `checkpoint-phase1-3`.

**Rollback:** `git revert HEAD`

### Step 1.4 — Pre-commit Hooks + Ruff Config

**Action:** Create `.pre-commit-config.yaml` and `pyproject.toml`.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python3 -m pytest tests/ -v
        language: system
        pass_filenames: false
        always_run: true
      - id: hardcoded-url
        name: hardcoded-api-url
        entry: bash -c '! grep -r "nedaktovops-eurlex-chat-api.hf.space" frontend/src/ | grep -v "default\|fallback\|VITE_API_URL"'
        language: system
        types: [javascript, jsx]
```

```toml
# pyproject.toml
[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
```

**Verification:**
```bash
pip install pre-commit
pre-commit run --all-files
```

**Checkpoint:** `git commit -m "feat: pre-commit hooks + ruff config"`. Tag `checkpoint-phase1-4`.

**Rollback:** `git revert HEAD`

---

## 4. Phase 2: EURLEX-BERT Rebuild

**Goal:** 768-dim legal embeddings from `nlpaueb/bert-base-uncased-eurlex`, built via ONNX O3 + matrix sharding on GitHub Actions (public repo = unlimited minutes).

### Pre-condition Check

```bash
# Verify HF Space is healthy with MiniLM index
curl https://nedaktovops-eurlex-chat-api.hf.space/health
# Verify app/data_loader.py has INDEX_SUFFIX support
grep -n "INDEX_SUFFIX" app/data_loader.py
# Verify build_index.py can be imported
python3 -c "from scripts.build_index import *"
```

### Step 2.1 — Pre-export ONNX Model (Laptop, One-Time)

**Action:** Export EURLEX-BERT to ONNX O3, push to HF Hub.

```bash
pip install sentence-transformers[onnx] optimum onnx onnxruntime
```

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nlpaueb/bert-base-uncased-eurlex", backend="onnx",
                            model_kwargs={"export": True})
model.save_pretrained("data/eurlex-bert-onnx")
```

**Verify export:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("data/eurlex-bert-onnx", backend="onnx")
emb = model.encode(["test"], normalize_embeddings=True)
assert emb.shape == (1, 768), f"Expected (1, 768), got {emb.shape}"
print(f"ONNX export OK: {emb.shape}")
```

**Upload to HF Hub:**
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="data/eurlex-bert-onnx",
    repo_id="NedAktovOps/eurlex-chat-data",
    repo_type="dataset",
    path_in_repo="onnx_models/eurlex-bert/",
)
```

**Verification:**
```python
files = api.list_repo_files("NedAktovOps/eurlex-chat-data", repo_type="dataset")
onnx_files = [f for f in files if "onnx" in f.lower()]
print(f"ONNX files on Hub: {len(onnx_files)}")
```

### Step 2.2 — Modify build_index.py

**Action:** Add `--backend onnx`, `--shard-id`, `--total-shards`, `--save-embeddings`, `--mode merge`.

**The ONNX embedding path (3 lines replacing ~35 lines of manual pooling):**
```python
if backend == "onnx":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, backend="onnx",
                                 model_kwargs={"providers": ["CPUExecutionProvider"]})
    embeddings = model.encode(chunks, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=True)
```

**The merge mode:**
```python
if mode == "merge":
    import glob
    shard_files = sorted(glob.glob(embeddings_glob))
    all_embeddings = np.concatenate([np.load(f) for f in shard_files])
    # Build FAISS index and upload
    build_faiss_index(all_embeddings, chunks)
    upload_to_hub()
```

**Verification:**
```bash
# Test ONNX path with small input
python3 scripts/build_index.py --backend onnx \
  --model-path data/eurlex-bert-onnx \
  --shard-id 0 --total-shards 2 \
  --save-embeddings /tmp/test_embeddings.npy \
  --dry-run
```

**Checkpoint:** `git commit -m "feat: add ONNX backend + sharding + merge mode to build_index.py"`. Tag `checkpoint-phase2-2`.

**Rollback:** `git revert HEAD`

### Step 2.3 — Restructure backup.yml

**Action:** Matrix-sharded build with unlimited parallelism (public repo).

**Download job** — SPARQL crawl → chunk → split into 10 shards → upload 10 `shard-{0..9}` artifacts.

**Build matrix** — 10 parallel jobs, each:
1. Download pre-exported ONNX model from HF Hub
2. Download their shard artifact
3. Embed with ONNX backend
4. Upload `embeddings-shard-{0..9}.npy`

**Merge job** — after ALL build jobs complete:
1. Download all 10 `.npy` files
2. Concatenate
3. Train FAISS PQ index
4. Upload `index_eurlex.faiss` + `chunks_eurlex.db` to HF Hub

**Verification:** Trigger workflow manually → verify 11 jobs (10 build + 1 merge) succeed.

**Checkpoint:** `git commit -m "feat: matrix-sharded EURLEX-BERT rebuild in backup.yml"`. Tag `checkpoint-phase2-3`.

**Rollback:** `git revert HEAD`

### Step 2.4 — Verify INDEX_SUFFIX in app/data_loader.py

**Action:** Confirm `app/data_loader.py` loads `index{_eurlex}.faiss` and `chunks{_eurlex}.db` when `INDEX_SUFFIX` is set.

```bash
grep -n "INDEX_SUFFIX\|_eurlex\|suffix" app/data_loader.py
```

If missing, add the same logic from `backend/data_loader.py`:
```python
INDEX_SUFFIX = os.environ.get("INDEX_SUFFIX", "")
# ...
self.INDEX_SUFFIX = INDEX_SUFFIX
```

**Verification:**
```bash
INDEX_SUFFIX=_eurlex python3 -c "
from app.data_loader import get_index
idx = get_index()
print(f'Index loaded: ntotal={idx[\"ntotal\"]}')
"
```

### Step 2.5 — Toggle to EURLEX-BERT

**Action:** Set `INDEX_SUFFIX=_eurlex` in HF Space environment variables.

```bash
# Via HuggingFace Hub API
curl -X POST https://huggingface.co/api/spaces/nedaktovops/eurlex-chat-api/secrets \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"key": "INDEX_SUFFIX", "value": "_eurlex"}'
```

Or set via HF Space dashboard → Settings → Environment Variables.

**Verification:**
```bash
curl https://nedaktovops-eurlex-chat-api.hf.space/health
# Expected: ntotal matches EURLEX-BERT index (probably ~same count, different dim)

curl -X POST https://nedaktovops-eurlex-chat-api.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What are employer obligations under GDPR?"}'
# Expected: better relevance with 768-dim legal embeddings
```

**Rollback:** Remove `INDEX_SUFFIX` env var → Space restarts with MiniLM index.

---

## 5. Phase 3: Verification & Polish

### Step 3.1 — Fix feedback-analysis.yml

**Replace the echo-only workflow with actual analysis:**

```yaml
name: Weekly Feedback Analysis
on:
  schedule:
    - cron: "0 8 * * 1"
  workflow_dispatch:

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Fetch HF Space logs
        run: |
          curl -sS "https://nedaktovops-eurlex-chat-api.hf.space/logs" \
            -o pipeline_logs.txt || echo "NO_LOGS_AVAILABLE" > pipeline_logs.txt
      - name: Run feedback analysis
        run: |
          cat pipeline_logs.txt | python3 scripts/feedback_analyzer.py || true
```

**Checkpoint:** `git commit -m "fix: make feedback-analysis.yml actually analyze"`. Tag `checkpoint-phase3-1`.

### Step 3.2 — Run Full Test Suite

```bash
source venv/bin/activate
python3 -m pytest tests/ -v 2>&1 | tee docs/test-results.txt
```

### Step 3.3 — Clean Up

```bash
# Remove temporary files
rm -f scripts/run_build.sh
rm -rf .checkpoints/
```

---

## 6. Rollback Procedures

| Scenario | Action |
|----------|--------|
| **HF Space goes down after push** | HF Space dashboard → Settings → "Revert to previous build" |
| **Code consolidation breaks something** | `git revert HEAD~N` + `git push origin main` + `git push hf main --force` |
| **EURLEX-BERT index is worse than MiniLM** | Remove `INDEX_SUFFIX` env var from HF Space → Space restarts with MiniLM |
| **Matrix build fails** | Trigger workflow manually with `rebuild_index: true` — only MiniLM path builds |
| **Docker build fails** | Fix locally, rebuild, push — no production impact (not deployed yet) |
| **CI fails after merge** | `git revert HEAD` + `git push origin main` — removes broken workflow |
| **Pre-commit hooks break workflow** | `SKIP=pre-commit` on next commit, then fix and re-enable |

---

## 7. File Change Manifest

| File | Action | Purpose |
|------|--------|---------|
| `app/main.py` | Create | Full 9-stage pipeline (was `backend/main.py`) |
| `app/data_loader.py` | Create | With INDEX_SUFFIX support (was `backend/data_loader.py`) |
| `app/search.py` | Create | Discourse-aware search (was `backend/search.py`) |
| `app/rag.py` | Create | Enhanced prompts + retry (was `backend/rag.py`) |
| `app/question_classifier.py` | Create | Legal intent detection (was `backend/question_classifier.py`) |
| `app/query_expander.py` | Create | Legal synonym expansion (was `backend/query_expander.py`) |
| `app/relation_extractor.py` | Create | Legal relation extraction (was `backend/relation_extractor.py`) |
| `app/answer_validator.py` | Create | Answer validation + confidence (was `backend/answer_validator.py`) |
| `app/logging_middleware.py` | Create | Structured pipeline logging (was `backend/logging_middleware.py`) |
| `app/rate_limit.py` | Create | Per-IP rate limiting (was `backend/rate_limit.py`) |
| `app/requirements.txt` | Create | Merged dependencies |
| `Dockerfile` | Create | Root Dockerfile for HF Space (was `hfspace/Dockerfile`) |
| `.dockerignore` | Create | Exclude build artifacts from Docker context |
| `README.md` | Modify | Add HF Space YAML frontmatter |
| `frontend/src/components/ChatWidget.jsx` | Modify | Wire VITE_API_URL env var |
| `scripts/build_index.py` | Modify | Add ONNX backend + sharding + merge mode |
| `.github/workflows/backup.yml` | Modify | Matrix-sharded EURLEX-BERT rebuild |
| `.github/workflows/ci.yml` | Create | pytest + ruff + docker-build on every push |
| `.github/workflows/keepalive.yml` | Modify | Add failure alerting |
| `.github/workflows/feedback-analysis.yml` | Modify | Actually call feedback_analyzer.py |
| `.pre-commit-config.yaml` | Create | ruff + pytest + URL check hooks |
| `pyproject.toml` | Create | Ruff configuration |
| `hfspace/` | Delete | Replaced by `app/` |
| `backend/` | Delete | Replaced by `app/` |
| `render.yaml` | Delete | Render no longer in architecture |

**Total: 12 created, 4 modified, 3 deleted = 19 file changes**

---

## 8. Appendix: Research Findings

### Platform Limits (Verified)

| Platform | Free Tier Limit | Impact on Plan |
|----------|----------------|----------------|
| GitHub Actions (public repo) | **Unlimited minutes** | Matrix sharding with 10 parallel jobs is free |
| GitHub Actions (standard runner) | 2 vCPU, 8GB RAM, 6h timeout | 10 shards × 2h each = 2h wall clock (fits in 6h) |
| HF Space (CPU Basic) | 2 vCPU, 16GB RAM, 48h sleep timeout | Full 9-stage pipeline fits comfortably |
| HF Hub Dataset | 100GB storage | Index + chunks.db = ~40MB total |
| Vercel (Hobby) | 100GB bandwidth/mo | Static Astro site, negligible bandwidth |
| Groq (free tier) | 1,000 req/day, 30 req/min | Sufficient for development + testing |

### Dollar Cost

**$0.00.** Everything stays on free tiers.

### Architecture Diagram (After Phoenix Plan)

```
                        GitHub Actions
                     (unlimited minutes)
                     ┌─────────────────┐
                     │  backup.yml     │
                     │  10 shards × 2h │──→ HF Hub (eurlex-chat-data)
                     │  → merge index  │    ├── index.faiss / index_eurlex.faiss
                     └─────────────────┘    ├── chunks.db / chunks_eurlex.db
                                            └── onnx_models/eurlex-bert/
                          │                      ▲
                          │ hf_hub_download()     │
                          ▼                      │
┌──────────────┐    ┌───────────┴───────────┐    │
│ Vercel       │───→│ HF Space (app/)       │────┘
│ Astro+React  │    │ Full 9-stage pipeline │
│ ChatWidget   │    │ Classification        │
│              │    │ Expansion             │
│              │    │ Discourse Search      │
│              │    │ Relation Extraction   │
│              │    │ Enhanced RAG          │
│              │    │ Groq LLM (70B)        │
│              │    │ Validation            │
│              │    │ LLM Retry             │
│              │    │ Confidence Estimation │
└──────────────┘    └───────────────────────┘
```
