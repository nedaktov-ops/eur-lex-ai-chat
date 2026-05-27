# EURLEX‑BERT Deployment Plan: Complete Handover

**Date:** 2025‑05‑24  
**Status:** PLANNED (not yet executed)  
**Goal:** Build and deploy the EURLEX‑BERT (768‑dim) index and switch the production Space from MiniLM (384‑dim) to EURLEX‑BERT without downtime.

---

## 1. Executive Summary

The EUR‑Lex AI Chat is live on Hugging Face Space with a MiniLM index. The RAG pipeline (prompt + validator) is fixed and verified: answers now include ≥2 inline CELEX citations. The next milestone is to upgrade the embedding model to EURLEX‑BERT for better retrieval. A GitHub Actions workflow (`.github/workflows/build-index.yml`) exists to build the index in a 10‑way matrix shard, but it is **broken** due to two critical issues:

1. It attempts to download `chunks.db` by calling `scripts/backup_index.py` without arguments, which actually **creates** a backup (upload) rather than restoring one. The backup dataset (`eurlex-chat-backups`) does not exist, causing a 404 failure.
2. Even if the build succeeded, the workflow only uploads `index_eurlex.faiss` and `build_meta_eurlex.json`; it never uploads `chunks_eurlex.db`, which is required by the Space when `INDEX_SUFFIX=_eurlex`.

This document provides a complete, evidence‑based fix and a step‑by‑step execution plan. No guesswork; all conclusions are drawn from reading the actual source files on disk.

---

## 2. Current State (Evidence)

### 2.1. Working Production
- **HF Space:** `https://nedaktovops-eurlex-chat-api.hf.space`  
  Health: `200 OK`, `index_loaded: true`, `ntotal: 305957`, `loaded_at: 2026‑05‑24T20:59:09` (after prompt fix).
- **Groq LLM:** Connected; answers now contain CELEX citations.  
  Example query: *"What are the transparency obligations for employers under the Pay Transparency Directive 2023/970?"* returns a real answer with multiple `CELEX 32023L0970` etc.
- **Vercel Frontend:** `https://eur-lex-ai-chat.vercel.app/` successfully calls the Space and displays answers.
- **Tests:** `pytest` → 59 passed, 6 skipped, 0 failures.  
  `ruff check app/ scripts/` → 0 errors.
- **Git:** Latest commit `659c97c` (feat: improve CELEX citation compliance and fix build_index bugs) pushed to `origin/main` and `hf/main`.

### 2.2. Dataset Contents (NedAktovOps/eurlex-chat-data)
API call `https://huggingface.co/api/datasets/NedAktovOps/eurlex-chat-data` returns:

```json
{
  "siblings": [
    {"rfilename": "index.faiss"},
    {"rfilename": "chunks.db"},
    {"rfilename": "build_meta.json"},
    {"rfilename": "last_updated.txt"},
    {"rfilename": "onnx_models/eurlex-bert/config.json"},
    {"rfilename": "onnx_models/eurlex-bert/model.quant.onnx"},
    {"rfilename": "onnx_models/eurlex-bert/special_tokens_map.json"},
    {"rfilename": "onnx_models/eurlex-bert/tokenizer.json"},
    {"rfilename": "onnx_models/eurlex-bert/tokenizer_config.json"},
    {"rfilename": "onnx_models/eurlex-bert/vocab.txt"}
  ]
}
```

**Missing:** `index_eurlex.faiss`, `chunks_eurlex.db`, `build_meta_eurlex.json`.

### 2.3. Build Index Workflow Failure
Latest run (ID `26373018724`) completed with `conclusion: failure`. Log excerpt:

```
Repository Not Found for url: https://huggingface.co/api/datasets/NedAktovOps/eurlex-chat-backups/preupload/backup-20260524
...
sqlite3.OperationalError: unable to open database file
```

The step `python3 scripts/backup_index.py` (lines 30–32 of the workflow) calls `create_backup()` (upload) instead of restoring, and the backup dataset does not exist. Consequently `data/chunks.db` is never created, causing the subsequent export step to fail.

---

## 3. Root Cause Analysis

| # | Observation | File / Line | Evidence |
|---|-------------|-------------|----------|
| 1 | `backup_index.py` default action is `create_backup()` | `scripts/backup_index.py:225` | `if __name__ == "__main__": main()` → `main()` → `create_backup()` when no args |
| 2 | Workflow step misnamed “Download chunks.db from HF backup” actually runs `backup_index.py` without flags | `.github/workflows/build-index.yml:30–32` | `run: | python3 scripts/backup_index.py || echo "Backup download skipped"` |
| 3 | Backup dataset `eurlex-chat-backups` does not exist | Log: `404 .../eurlex-chat-backups/...` | Hugging Face API returns 404 |
| 4 | Workflow never uploads `chunks_eurlex.db` | `.github/workflows/build-index.yml:182–200` | Only two `upload_file` calls: `index_eurlex.faiss` and `build_meta_eurlex.json` |
| 5 | Space requires `chunks{suffix}.db` when `INDEX_SUFFIX` set | `app/data_loader.py:50–76` | `chunks_path = DATA_DIR / f"chunks{suffix}.db"` |
| 6 | `scripts/generate_embeddings.py` imports: `numpy`, `onnxruntime`, `transformers`, `huggingface_hub` | `scripts/generate_embeddings.py:1–11` | No `tqdm` → embed job’s pip list is sufficient |
| 7 | Main dataset already contains `chunks.db` | API response above | `"rfilename": "chunks.db"` present |
| 8 | ONNX model present in dataset | API response | `onnx_models/eurlex-bert/model.quant.onnx` |

**Conclusion:** The workflow’s first job must provide `chunks.db` (and `chunks.json`) to the embed job, and the merge job must upload both `index_eurlex.faiss` **and** `chunks_eurlex.db`.

---

## 4. Solution: Patch `.github/workflows/build-index.yml`

### 4.1. Download Job – Direct Dataset Access

Replace the broken backup step with a direct download from `NedAktovOps/eurlex-chat-data`. Also upload the raw `chunks.db` as an artifact so the merge job can later copy it.

**Patch (download job):**

```diff
--- a/.github/workflows/build-index.yml
+++ b/.github/workflows/build-index.yml
@@ -27,16 +27,23 @@ jobs:
       - name: Install dependencies
         run: |
           pip install huggingface_hub faiss-cpu numpy
-      - name: Download chunks.db from HF backup
-        run: |
-          python3 scripts/backup_index.py || echo "Backup download skipped"
+      - name: Download chunks.db from main dataset
+        run: |
+          python3 -c "
+          from huggingface_hub import hf_hub_download
+          import os, shutil
+          os.makedirs('data', exist_ok=True)
+          path = hf_hub_download(
+              repo_id='NedAktovOps/eurlex-chat-data',
+              filename='chunks.db',
+              repo_type='dataset',
+              token=os.environ.get('HF_TOKEN'),
+          )
+          shutil.copy(path, 'data/chunks.db')
+          print('Downloaded chunks.db')
+          "
       - name: Export chunks to JSON
         run: |
           python3 -c "
           import sqlite3, json
           conn = sqlite3.connect('data/chunks.db')
           rows = conn.execute('SELECT celex, article, text FROM chunks').fetchall()
           chunks = [{'celex': r[0], 'article': r[1], 'text': r[2]} for r in rows]
           with open('data/chunks.json', 'w') as f:
               json.dump(chunks, f)
           print(f'Exported {len(chunks)} chunks to data/chunks.json')
           "
-      - name: Upload chunks.json
-        uses: actions/upload-artifact@v4
-        with:
-          name: chunks-json
-          path: data/chunks.json
-          retention-days: 1
+      - name: Upload chunks data
+        uses: actions/upload-artifact@v4
+        with:
+          name: chunks-data
+          path: |
+            data/chunks.json
+            data/chunks.db
+          retention-days: 1
```

### 4.2. Embed Job – Download Combined Artifact

Change artifact name from `chunks-json` to `chunks-data` and ensure both files land in `data/`.

```diff
@@ -85,10 +85,10 @@ jobs:
         env:
           MODEL_PATH: ${{ inputs.embedding_model || 'onnx_models/eurlex-bert/model.quant.onnx' }}
       - name: Download chunks
-        uses: actions/download-artifact@v4
+        uses: actions/download-artifact@v4
         with:
-          name: chunks-json
+          name: chunks-data
           path: data/
       - name: Generate embeddings for shard ${{ matrix.shard }}
```

### 4.3. Merge Job – Upload `chunks_eurlex.db`

After building the FAISS index, copy `chunks.db` → `chunks_eurlex.db` and upload both.

```diff
@@ -110,6 +110,7 @@ jobs:
       - name: Merge shards and build FAISS index
         run: |
           python3 -c "
           import numpy as np, json, os, faiss
@@ -180,6 +181,8 @@ jobs:
           with open('data/build_meta.json', 'w') as f:
               json.dump(meta, f)
           print('Build metadata saved')
           "
+      - name: Prepare chunks_eurlex.db
+        run: |
+          cp data/chunks.db data/chunks_eurlex.db
+          echo "Created chunks_eurlex.db"
-      - name: Upload FAISS index to HF dataset
-        run: |
-          python3 -c "
-          from huggingface_hub import HfApi
-          import os
-          api = HfApi(token=os.environ.get('HF_TOKEN'))
-          api.upload_file(
-              path_or_fileobj='data/index.faiss',
-              path_in_repo='index_eurlex.faiss',
-              repo_id=os.environ['MODEL_REPO'],
-              repo_type='dataset',
-          )
-          api.upload_file(
-              path_or_fileobj='data/build_meta.json',
-              path_in_repo='build_meta_eurlex.json',
-              repo_id=os.environ['MODEL_REPO'],
-              repo_type='dataset',
-          )
-          print('✓ Index uploaded to HF dataset')
-          "
+      - name: Upload FAISS index and chunks to HF dataset
+        run: |
+          python3 -c "
+          from huggingface_hub import HfApi
+          import os
+          api = HfApi(token=os.environ.get('HF_TOKEN'))
+          api.upload_file(
+              path_or_fileobj='data/index.faiss',
+              path_in_repo='index_eurlex.faiss',
+              repo_id=os.environ['MODEL_REPO'],
+              repo_type='dataset',
+          )
+          api.upload_file(
+              path_or_fileobj='data/chunks_eurlex.db',
+              path_in_repo='chunks_eurlex.db',
+              repo_id=os.environ['MODEL_REPO'],
+              repo_type='dataset',
+          )
+          print('✓ Index and chunks uploaded')
+          "
```

---

## 5. Step‑by‑Step Execution Plan

### Phase A – Preparation
1. **Confirm GitHub secret** `HF_TOKEN` exists (it does).  
   `gh secret list --repo nedaktov-ops/eur-lex-ai-chat`
2. **Tag current good state** (optional safety):  
   `git tag pre-eurlex-build 659c97c`
3. **Record commit hash** for rollback: `git rev-parse HEAD`

### Phase B – Apply Workflow Fix
4. Edit `.github/workflows/build-index.yml` exactly as patched above.
5. **No code changes** to `app/` or `scripts/` are required; the existing code already supports `INDEX_SUFFIX=_eurlex`.
6. Commit:  
   ```bash
   git add .github/workflows/build-index.yml
   git commit -m "fix: build-index workflow – direct download, upload chunks_eurlex.db"
   ```
7. Push to both remotes:  
   ```bash
   git push origin main
   git push hf main
   ```

### Phase C – Trigger Build
8. Wait a few seconds for the push to arrive on GitHub.
9. Manually trigger the workflow:  
   ```bash
   gh workflow run build-index.yml --repo nedaktov-ops/eur-lex-ai-chat
   ```
10. **Monitor** without blocking:  
    ```bash
    RUN=$(gh run list --workflow=build-index.yml -L 1 --json number --jq '.[0].number')
    echo "Monitoring run #$RUN"
    gh run view $RUN --log --web  # opens browser; or tail manually
    ```
    If it fails, fetch logs: `gh run view $RUN --log > build_failure.log` and diagnose.

### Phase D – Verify Artifacts
11. After successful completion, check dataset contents via API:  
    ```bash
    curl -s https://huggingface.co/api/datasets/NedAktovOps/eurlex-chat-data |
      python3 -c "import sys,json; d=json.load(sys.stdin); print([f['rfilename'] for f in d['siblings']])"
    ```
    Must include: `index_eurlex.faiss`, `chunks_eurlex.db`, `build_meta_eurlex.json`.
12. If any missing, **do not proceed**. Investigate logs.

### Phase E – Switch Space to EURLEX‑BERT
13. Set `INDEX_SUFFIX=_eurlex` on the Space (using HF API):  
    ```python
    from huggingface_hub import HfApi
    api = HfApi(token='<HF_TOKEN>')  # use the same token from your git remote
    api.add_space_variable('nedaktovops/eurlex-chat-api', 'INDEX_SUFFIX', '_eurlex')
    api.restart_space('nedaktovops/eurlex-chat-api')
    ```
14. Poll `/health` until `200 OK` and `index_loaded: true`. Note `loaded_at` should update.
15. (Optional) Check logs via HF UI for line “Loading EURLEX‑BERT embedding model (768‑dim)”.

### Phase F – Final End‑to‑End Test
16. Call `/chat` with the same query:  
    ```bash
    curl -s -X POST https://nedaktovops-eurlex-chat-api.hf.space/chat \
      -H "Content-Type: application/json" \
      -d '{"query": "What are the transparency obligations for employers under the Pay Transparency Directive 2023/970?"}' |
      python3 -m json.tool
    ```
    **Expected:** Real answer (not fallback), length ≥100 chars, ≥2 CELEX numbers in answer text, `_confidence` field present.
17. Also test a non‑obligation query to ensure general coverage.

### Phase G – Documentation
18. Update this document with actual timestamps, commit hashes, and any issues encountered.
19. Optionally add a section to `README.md` describing the EURLEX‑BERT deployment and how to roll back.

---

## 6. Rollback Procedure

**If the Space fails to start or chat returns errors after enabling `_eurlex`:**

```python
from huggingface_hub import HfApi
api = HfApi(token='<HF_TOKEN>')
api.remove_space_variable('nedaktovops/eurlex-chat-api', 'INDEX_SUFFIX')
api.restart_space('nedaktovops/eurlex-chat-api')
```

The Space will revert to the MiniLM index (suffix `""`) immediately. No data loss occurs; the MiniLM index remains in the dataset.

**If the build‑index workflow fails:**  
- Fix the workflow based on error logs (e.g., memory issues → reduce `batch-size` or increase `max-parallel` wait).  
- Or revert to the previous commit: `git reset --hard pre-eurlex-build` and push force (only if necessary).

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Workflow runs out of memory/time | Delay; need to retune | Monitor first few shards; if OOM, reduce `--batch-size` from 32 to 16; increase `max-parallel` wait. |
| HF upload fails (permissions) | Artifacts not published | `HF_TOKEN` already used successfully for other uploads; should be fine. |
| After switch, retrieval quality worse | Poor answers | Rollback to MiniLM; investigate embedding mismatch. |
| Space crashes on startup | Downtime | Rollback by removing `INDEX_SUFFIX`. |
| Accidentally delete existing dataset files | Data loss | Never delete; only add new files (`index_eurlex.faiss`, `chunks_eurlex.db`). |

---

## 8. Open Questions (for the user)

1. **Backup dataset:** Should we also create `NedAktovOps/eurlex-chat-backups` for future safety? Not required for deployment.
2. **Batch size:** The current `--batch-size 32` is a guess. Should we start with 16 to be safe on GitHub’s shared runners? The log from the failed run didn’t get to embedding, so we don’t know. I propose keeping 32; if it OOMs, we’ll see in logs and can adjust.
3. **Trigger timing:** After I push the workflow fix, should I immediately trigger the build, or would you like to review the workflow file first?

---

## 9. Appendix – Full Corrected Workflow (for reference)

Below is the **complete corrected** `.github/workflows/build-index.yml` after applying all patches. Use this to replace the existing file.

```yaml
name: EURLEX-BERT Index Rebuild

on:
  workflow_dispatch:
    inputs:
      total_shards:
        description: "Number of parallel shards"
        type: number
        default: 10
      embedding_model:
        description: "ONNX model path on HF"
        default: "onnx_models/eurlex-bert/model.quant.onnx"

env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
  MODEL_REPO: NedAktovOps/eurlex-chat-data

jobs:
  download:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install dependencies
        run: |
          pip install huggingface_hub faiss-cpu numpy
      - name: Download chunks.db from main dataset
        run: |
          python3 -c "
          from huggingface_hub import hf_hub_download
          import os, shutil
          os.makedirs('data', exist_ok=True)
          path = hf_hub_download(
              repo_id='NedAktovOps/eurlex-chat-data',
              filename='chunks.db',
              repo_type='dataset',
              token=os.environ.get('HF_TOKEN'),
          )
          shutil.copy(path, 'data/chunks.db')
          print('Downloaded chunks.db')
          "
      - name: Export chunks to JSON
        run: |
          python3 -c "
          import sqlite3, json
          conn = sqlite3.connect('data/chunks.db')
          rows = conn.execute('SELECT celex, article, text FROM chunks').fetchall()
          chunks = [{'celex': r[0], 'article': r[1], 'text': r[2]} for r in rows]
          with open('data/chunks.json', 'w') as f:
              json.dump(chunks, f)
          print(f'Exported {len(chunks)} chunks to data/chunks.json')
          "
      - name: Upload chunks data
        uses: actions/upload-artifact@v4
        with:
          name: chunks-data
          path: |
            data/chunks.json
            data/chunks.db
          retention-days: 1

  embed:
    needs: download
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      max-parallel: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install dependencies
        run: |
          pip install onnxruntime>=1.18.0 numpy>=1.24.0 transformers>=4.44.0 huggingface_hub
      - name: Download ONNX model
        run: |
          python3 -c "
          from huggingface_hub import hf_hub_download
          import os
          os.makedirs('data/eurlex-bert-onnx', exist_ok=True)
          hf_hub_download(
              repo_id=os.environ['MODEL_REPO'],
              filename=os.environ['MODEL_PATH'],
              local_dir='data/eurlex-bert-onnx',
              local_dir_use_symlinks=False,
              token=os.environ.get('HF_TOKEN'),
          )
          import glob
          for f in glob.glob('data/eurlex-bert-onnx/**'):
              print(f)
          "
        env:
          MODEL_PATH: ${{ inputs.embedding_model || 'onnx_models/eurlex-bert/model.quant.onnx' }}
      - name: Download chunks data
        uses: actions/download-artifact@v4
        with:
          name: chunks-data
          path: data/
      - name: Generate embeddings for shard ${{ matrix.shard }}
        run: |
          python3 scripts/generate_embeddings.py \
            --model-path data/eurlex-bert-onnx/model.quant.onnx \
            --tokenizer-name nlpaueb/bert-base-uncased-eurlex \
            --chunks data/chunks.json \
            --shard ${{ matrix.shard }} \
            --total-shards ${{ inputs.total_shards || 10 }} \
            --output-dir data/embeddings \
            --batch-size 32
      - name: Upload shard embeddings
        uses: actions/upload-artifact@v4
        with:
          name: embeddings-shard-${{ matrix.shard }}
          path: |
            data/embeddings/embeddings_shard_${{ matrix.shard }}.npy
            data/embeddings/metadata_shard_${{ matrix.shard }}.json
          retention-days: 7

  merge:
    needs: embed
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install dependencies
        run: pip install numpy faiss-cpu huggingface_hub
      - name: Download all shard embeddings
        run: |
          for i in $(seq 0 $(( ${{ inputs.total_shards || 10 }} - 1 ))); do
            gh run download --name embeddings-shard-$i --dir data/embeddings
          done
        env:
          GH_TOKEN: ${{ github.token }}
      - name: Merge shards and build FAISS index
        run: |
          python3 -c "
          import numpy as np, json, os, faiss
          from pathlib import Path

          embed_dir = Path('data/embeddings')
          shard_files = sorted(embed_dir.glob('embeddings_shard_*.npy'))
          print(f'Found {len(shard_files)} shard files')

          all_embeddings = []
          all_metadata = []
          for f in shard_files:
              emb = np.load(f)
              all_embeddings.append(emb)
              meta_file = embed_dir / f'metadata_{f.stem.replace(\"embeddings_\", \"\")}.json'
              if meta_file.exists():
                  with open(meta_file) as mf:
                      all_metadata.append(json.load(mf))

          embeddings = np.vstack(all_embeddings)
          print(f'Total embeddings: {embeddings.shape}')

          # Build FAISS index (IVF with PQ compression)
          dim = embeddings.shape[1]
          nlist = min(4096, int(np.sqrt(embeddings.shape[0])))
          quantizer = faiss.IndexFlatIP(dim)
          index = faiss.IndexIVFPQ(quantizer, dim, nlist, 48, 8)
          index.train(embeddings)
          index.add(embeddings)
          print(f'FAISS index: {index.ntotal} vectors, {index.d} dimensions')

          # Save index
          os.makedirs('data', exist_ok=True)
          faiss.write_index(index, 'data/index.faiss')

          # Save metadata
          total_celexes = list(set(
              cid for m in all_metadata for cid in m.get('celex_ids', [])
          ))
          meta = {
              'ntotal': index.ntotal,
              'dim': dim,
              'nlist': nlist,
              'code_size': 48,
              'num_celexes': len(total_celexes),
              'model': 'nlpaueb/bert-base-uncased-eurlex',
              'suffix': '_eurlex',
              'timestamp': json.dumps({}).__str__(),
          }
          with open('data/build_meta.json', 'w') as f:
              json.dump(meta, f)
          print('Build metadata saved')
          "
      - name: Prepare chunks_eurlex.db
        run: |
          cp data/chunks.db data/chunks_eurlex.db
          echo "Created chunks_eurlex.db"
      - name: Upload FAISS index and chunks to HF dataset
        run: |
          python3 -c "
          from huggingface_hub import HfApi
          import os
          api = HfApi(token=os.environ.get('HF_TOKEN'))
          api.upload_file(
              path_or_fileobj='data/index.faiss',
              path_in_repo='index_eurlex.faiss',
              repo_id=os.environ['MODEL_REPO'],
              repo_type='dataset',
          )
          api.upload_file(
              path_or_fileobj='data/chunks_eurlex.db',
              path_in_repo='chunks_eurlex.db',
              repo_id=os.environ['MODEL_REPO'],
              repo_type='dataset',
          )
          print('✓ Index and chunks uploaded')
          "
```

---

## 10. File Location

This plan is saved at:

**`/home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat/docs/implementation/2025-05-24-EURLEX-BERT-DEPLOYMENT-PLAN.md`**

You can start a new session and point NedCode3 to this file for full context.

---

**End of Document**
