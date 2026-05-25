# EUR-Lex AI Chat: Complete Implementation Strategy

**Date:** 2025-05-24  
**Status:** Planning (0% complete)  
**Goal:** Deploy a fully functional EUR-Lex AI Chat on Hugging Face Space with Groq LLM, robust CELEX citation validation, EURLEX-BERT index support, full CI/CD, and comprehensive tests.

---

## Table of Contents
1. [Baseline & Secret Audit](#phase-0-baseline--secret-audit)
2. [Core Prompt & Validation Fixes](#phase-1-core-prompt--validation-fixes)
3. [Validation & Prompt Iteration on HF Space](#phase-2-validation--prompt-iteration-on-hf-space)
4. [Build & Deploy EURLEX-BERT Index](#phase-3-build--deploy-eurlex-bert-index)
5. [Frontend & CI Integration](#phase-4-frontend--ci-integration)
6. [Documentation & Knowledge Capture](#phase-5-documentation--knowledge-capture)
7. [Checkpoints & Rollback Strategy](#checkpoints--rollback-strategy)
8. [Testing Strategy](#testing-strategy)
9. [Risks & Mitigations](#risks--mitigations)

---

## Phase 0: Baseline & Secret Audit (Read‑Only) ✅ COMPLETED

**Objective:** Establish ground truth without changing anything.

### Steps Completed
- ✅ `git status` → uncommitted change on `app/answer_validator.py` (MIN_CITATIONS=1)
- ✅ `git log -1 --oneline --all` → commit `586da65` on main
- ✅ `gh secret list` → `GROQ_API_KEY` and `HF_TOKEN` exist in GitHub repo
- ✅ Verified HF Space secrets via API: `GROQ_API_KEY`, `HF_TOKEN`, `GROQ_MODEL` present (set earlier)
- ✅ `curl -s https://nedaktovops-eurlex-chat-api.hf.space/health` → ntotal=305957, last_updated=2026-05-22, loaded_at=2026-05-24T19:33:14
- ✅ `curl -s -X POST .../chat` with test query → baseline fallback response captured (answer: "I found documents... couldn't generate complete answer")
- ✅ `ruff check app/ scripts/` → **All checks passed!** (0 errors)
- ✅ `pytest` → **Result:** 46 passed, 6 skipped, 5 failed (see details). Failures:
  - `test_insufficient_citation_fails` – expected to fail with current MIN_CITATIONS=1 (validator expects 2)
  - `test_build_chunks_db_basic` – UNIQUE constraint (leftover DB from previous run)
  - `test_build_chunks_db_data_integrity` – database locked
  - `test_embed_chunks_shape` – UnboundLocalError (model undefined)
  - `test_embed_chunks_normalized` – UnboundLocalError

**Baseline Evidence:**
- Health: `{"status": "ok", "index_loaded": true, "ntotal": 305957, "size": 305957, "last_updated": "2026-05-22T22:07:04.827872+00:00", "loaded_at": "2026-05-24T19:33:14.910686+00:00"}`
- Chat fallback includes citations list `["32023L0970","32018L1972"]` but answer lacks inline CELEX numbers.
- Lint: 0 errors.
- Test suite: not fully green; will fix in Phase 1.

**Notes:** Missing `beautifulsoup4` caused 5 parse test failures; installed via `requirements-dev.txt`. Remaining failures are real bugs in `scripts/build_index.py` and the validator threshold mismatch. These will be addressed in Phase 1.

---

## Phase 1: Core Prompt & Validation Fixes ✅ COMPLETED

**Objective:** Align validator with prompt, fix prompt structure, ensure robust citation extraction, fix AutoExpander.

### Completed Steps

#### 1.1 Align validator threshold ✅
- `app/answer_validator.py`: `MIN_CITATIONS = 2`
- Existing test `test_insufficient_citation_fails` now passes (it expects 2 citations).

#### 1.2 Remove SYSTEM_PROMPT duplication ✅
- `app/rag.py` → `build_prompt()`: No longer prefixes user message with "System: ..."
- Verified: `test_build_prompt_no_duplicate_system` passes.

#### 1.3 ENSURE_CITATION_PROMPT only on retry ✅
- `answer_question()`: `extra_notes = None` unless `retry_with_citation_emphasis=True`
- Verified: `test_ensure_citation_prompt_only_on_retry` passes.

#### 1.4 Strengthen SYSTEM_PROMPT with few‑shot ✅
- Added concrete example showing CELEX citation format.
- Kept requirement: "Every factual claim must include an inline CELEX citation (format: CELEX 32023L0970)."

#### 1.5 Align extract_citations ✅
- Changed regex to `r"3\d{4}[A-Z]\d{4}"` (matches validator's pattern).
- Added `tests/test_extract_citations.py` (5 tests, all pass).

#### 1.6 Fix AutoExpander path ✅
- `app/query_expander.py`: `EXPANSION_FILE = "/tmp/auto_expansions.json"`
- Works in read‑only container; no FS errors on validation failures.

#### 1.7 Fix `scripts/build_index.py` bugs ✅
- `embed_chunks`: Added else branch to load `SentenceTransformer` for MiniLM.
- `build_chunks_db`: Delete existing DB at start to avoid UNIQUE constraint.
- These fixes unblock the build‑index workflow and make its tests pass.

#### 1.8 Lint and full test suite ✅
- `ruff check app/ scripts/` → **All checks passed!**
- `pytest` → **59 passed, 6 skipped, 0 failures** (green suite).

#### 1.9 Commit & push ✅
- Branch: `main`
- Commit: `659c97c` "feat: improve CELEX citation compliance and fix build_index bugs"
- Pushed to `origin/main` and `hf/main`.
- HF Space rebuild triggered automatically.

**Checkpoint:** Commit `659c97c` is the new baseline.

---

## Phase 2: Validation & Prompt Iteration on HF Space ✅ COMPLETED

**Objective:** Confirm LLM produces ≥2 CELEX citations.

### Step 2.1: Wait for Space rebuild ✅
- Poll `/health` until `status: ok` and `index_loaded: true`.
- Observed: `loaded_at` updated from `2026-05-24T19:33:14` to `2026-05-24T20:59:09`.
- ntotal unchanged: 305,957.

### Step 2.2: Automated test query ✅
- Query: *"What are the transparency obligations for employers under the Pay Transparency Directive 2023/970? Give me specific provisions."*
- Response: Real answer with multiple inline CELEX citations.
- Answer length: >100 chars.
- CELEX numbers in answer text: `32023L0970`, `32018L1972`, `32019L2034` (≥2).
- Citations array includes those numbers.
- Validation passed; no fallback.

### Step 2.3: No iteration needed ✅
- First attempt succeeded; no further prompt tuning required.

**Checkpoint:** Commit `659c97c` deployed and verified. Chat endpoint works as expected.

---

## Phase 3: Build & Deploy EURLEX‑BERT Index (Next)

**Prerequisites:** `HF_TOKEN` secret present in GitHub repo.

### Step 3.1: Verify GitHub secret
```bash
gh secret list | grep HF_TOKEN
```
If missing: `gh secret set HF_TOKEN --repo nedaktov-ops/eur-lex-ai-chat < hf_token.txt`

### Step 3.2: Trigger build-index workflow
```bash
gh workflow run build-index.yml --repo nedaktov-ops/eur-lex-ai-chat
```
- Monitor via `gh run watch` or GitHub UI.
- Duration: ~3.5 h.

### Step 3.3: Validate artifacts
```python
from huggingface_hub import list_repo_files
files = list_repo_files('NedAktovOps/eurlex-chat-data', repo_type='dataset')
assert 'index_eurlex.faiss' in files
assert 'chunks_eurlex.db' in files
assert 'onnx_models/eurlex-bert/model.quant.onnx' in files
```

### Step 3.4: Backup current index (optional)
- Call `/backup` endpoint if operational, or rely on dataset backups.

### Step 3.5: Switch Space to EURLEX‑BERT
```python
from huggingface_hub import HfApi
api = HfApi(token='<HF_TOKEN>')  # Use your HF write token
api.add_space_variable('nedaktovops/eurlex-chat-api', 'INDEX_SUFFIX', '_eurlex')
api.restart_space('nedaktovops/eurlex-chat-api')
```
- Wait for `/health` → 200.
- Check logs for “Loading EURLEX-BERT embedding model (768-dim)”.

### Step 3.6: Re‑test chat
- Same query; expect improved retrieval; still need ≥2 CELEX citations.
- If citation compliance drops, adjust prompt/k and repeat Phase 1–2.

**Rollback:** Remove `INDEX_SUFFIX` and restart → back to MiniLM.

**Checkpoint:** Record commit and index artifacts.

---

## Phase 4: Frontend & CI Integration

### Step 4.1: Vercel environment
- Verify `frontend` project has `VITE_API_URL` set to `https://nedaktovops-eurlex-chat-api.hf.space` in Production.
- If missing, set via Vercel API (token provided).
- Trigger redeploy.

### Step 4.2: CI assurance
- Ensure `ci.yml` runs on `push` to `main`.
- All PRs must pass lint and pytest.
- Keep `ruff` config stable.

**Checkpoint:** Frontend reachable; API calls succeed.

---

## Phase 5: Documentation & Knowledge Capture

### Step 5.1: Update `README.md`
- Environment Variables section.
- Deployment Steps (index rebuild, switch).
- Troubleshooting.

### Step 5.2: Add `docs/DECISIONS.md`
- Rationale for thresholds, formats, paths.

### Step 5.3: Update `.env.example`
- Include all variables with placeholders.

### Step 5.4: Record in Knowledge Graph
- Use `pnp-checkpoint` to save session context with decisions.

---

## Checkpoints & Rollback Strategy

- **Git tags:** After each successful phase, tag commit: `phase1-complete`, `phase2-complete`, etc.
- **HF Space rollback:** Use “Replicate” or “Restart from previous commit” in UI.
- **Index rollback:** Keep `index.faiss` and `chunks.db` unchanged; simply remove `INDEX_SUFFIX` to revert.
- **Backups:** Trigger `/backup` before swapping index; uploads to `eurlex-chat-backups`.

---

## Testing Strategy

- **Unit tests:** `tests/test_validator.py`, `test_extract_citations.py`, `test_rag.py`, `test_auto_expander.py`.
- **Integration test:** fixture index + mocked Groq response; validate full pipeline.
- **Manual test script:** `scripts/manual_test.py` for quick sanity checks.
- **CI:** `pytest` on all pushes; `ruff check` as separate job.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM still won’t produce 2 citations | Poor UX | Iterate prompt with more examples; test locally with Groq playground. |
| HF Space rebuild fails | Downtime | Push previous commit; pin dependencies. |
| EURLEX‑BERT artifacts missing | Runtime error | Verify artifacts before toggling; ensure HF_TOKEN works. |
| AutoExpander FS errors | 500 on validation failure | Fixed by `/tmp`; verify in logs. |
| GH secret missing → build fails | Index not published | Audit and set before triggering. |
| Vercel env missing → frontend broken | UX break | Verify via API; test from browser. |

---

## Final Success Criteria

- `/chat` returns substantive answer (≥100 chars) with ≥2 inline CELEX numbers.
- No fallback for typical queries.
- Frontend displays answers correctly.
- All GitHub Actions workflows pass.
- EURLEX‑BERT index built and optionally deployed.
- Full test coverage for core logic.
- Comprehensive documentation.

---

**Execution will begin immediately after this document is saved. Each step will be marked completed and verified with evidence before moving to the next.**
