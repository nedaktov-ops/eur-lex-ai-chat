# Post-Mortem & Revised Strategy

> **Date:** 2026-05-22
> **Author:** Build analysis after 2 failed attempts
> **Project:** eur-lex-ai-chat — RAG chatbot over 26,871 EU regulations & directives

---

## Table of Contents

1. [Failure Analysis](#1-failure-analysis)
2. [Research Findings — 50+ GitHub Links with Summaries](#2-research-findings)
3. [Revised Architecture](#3-revised-architecture)
4. [Revised Strategy & Implementation Plan](#4-revised-strategy--implementation-plan)
5. [Risk Register](#5-risk-register)

---

## 1. Failure Analysis

### Failure #1: `nohup` + bash tool → process killed on timeout

**Symptom:** Build process dies silently after 120 seconds (the bash tool's default timeout).

**Root Cause:** The bash tool runs each command in a process group. When the tool's timeout expires, it sends SIGTERM to the entire process group, including any `nohup`'d processes. `nohup` only redirects SIGHUP — it doesn't protect against SIGTERM to the process group.

**Evidence:** No log file was created. `ps aux` showed no build process.

**See links:**
- [1] https://dev.to/ajitkumar/why-your-deep-learning-job-dies-after-ssh-logout-a-practical-guide-to-persistent-linux-sessions-52ml
- [2] https://codegive.com/blog/run_python_script_continuously_on_server.php
- [3] https://sergiobelkin.com/posts/systemd-run-vs-nohup-vs-tmuxscreen/
- [4] https://makandracards.com/operations/521736-systemd-run-alternative-screen
- [5] https://acsrujan.net/why-i-say-no-to-tmux/
- [6] https://www.freedesktop.org/software/systemd/man/249/systemd.service.html

### Failure #2: 4.3GB RSS memory leak from `future_map`

**Symptom:** RSS grew from 1.45GB to 4.3GB in 10 minutes for only 1,262 documents processed.

**Root Cause:** The pattern `{executor.submit(fetch, doc): doc for doc in docs}` submits ALL 26,871 tasks at once. Each `Future` object stores its result (HTML ~350KB each). `as_completed()` iterates but the futures remain in `future_map`, holding their results. After 1,262 completed futures: 1,262 × 350KB ≈ 440MB just for completed HTML results, plus 25,609 pending futures with metadata.

**Evidence:** RSS at T+4min = 1.45GB, at T+10min = 4.3GB. Linear extrapolation: would hit 11GB system limit at ~26 min.

**The fix (already applied):** Batch submission — process 60 futures at a time, `pop()` from the dict when done, freeing the future's result immediately.

**See links:**
- [7] https://bugs.python.org/issue37909 — Thread pool return ref hold memory
- [8] https://bugs.python.org/issue16284 — ThreadPoolExecutor keeps unnecessary references
- [9] https://bugs.python.org/issue38430 — Memory leak in run_in_executor
- [10] https://stackoverflow.com/questions/74895168/python-threadpoolexecutor-concurrent-futures-memory-leak
- [11] https://stackoverflow.com/questions/37445540/memory-usage-with-concurrent-futures-threadpoolexecutor-in-python3
- [12] https://stackoverflow.com/questions/71901539/python-threadpoolexecutor-memory-leak-issues
- [13] https://stackoverflow.com/questions/34770169/using-concurrent-futures-without-running-out-of-ram
- [14] https://stackoverflow.com/questions/63487755/concurrent-futures-threadpoolexecutor-unreleased-memory
- [15] https://superfastpython.com/threadpoolexecutor-wait-first-result/
- [16] https://github.com/python/cpython/blob/main/Lib/concurrent/futures/_base.py
- [17] https://pypi.org/project/future-map/
- [18] https://github.com/Ahmedie-m/batchit
- [19] https://pypi.org/project/fastpipe/
- [20] https://medium.com/@priteshjha27/multiprocessing-in-production-pool-tuning-pipelines-and-backpressure-a09116ed31fb
- [21] https://oneuptime.com/blog/post/2026-01-30-python-asyncio-queues/view
- [22] https://medium.com/@rakotobesalimo/accelerating-synchronous-process-pipelines-with-multiprocessing-batching-762426a76ff5

### Failure #3: Highly variable Cellar fetch throughput

**Symptom:** Throughput varies wildly from 1-2 it/s to 20+ it/s. Average ~5 it/s instead of expected 20 it/s.

**Root Cause:** The ThreadPoolExecutor uses `max_workers=20`, but some Cellar requests take 10-30 seconds (corrigenda, rate-limited). When even 2-3 workers are blocked, the `as_completed` loop only yields results from the 17 remaining workers, reducing throughput proportionally.

**Evidence:** Benchmark of 50 sequential non-corrigenda docs showed 0.84s average, but the real run had bursts of 1-3 it/s followed by 14-20 it/s (pattern of slow requests clogging the pool).

**See links:**
- [23] https://github.com/maastrichtlawtech/cellar-extractor
- [24] https://github.com/Kymylyy/cellar-wrapper
- [25] https://github.com/do-me/eur-lex
- [26] https://github.com/kevin91nl/eurlex
- [27] https://pypi.org/project/pyeurlex/
- [28] https://github.com/fvanlitsenburg/BatchLegal
- [29] https://github.com/maastrichtlawtech/EU_EurLex_Cellar_reference_querier
- [30] https://github.com/nature-of-eu-rules/data-extraction/blob/main/eu_rules_metadata_extractor.py

### Failure #4: No progress monitoring via log file

**Symptom:** Log file contains only tqdm overwrites (carriage return `\r`), making progress unreadable.

**Root Cause:** tqdm uses `\r` to overwrite the same terminal line. When captured to a file, each `\r` overwrites the previous bytes rather than appending a new line.

**See links:**
- [31] https://github.com/tqdm/tqdm/issues/306 — tqdm and file logging
- [32] https://stackoverflow.com/questions/37529964/tqdm-to-log-file-not-terminal
- [33] https://github.com/tqdm/tqdm/blob/master/examples/redirect_print.py

---

## 2. Research Findings

### 2.1 Process Persistence (surviving shell exit)

| # | Link | Summary | Why useful |
|---|------|---------|------------|
| 1 | [Why Your Deep Learning Job Dies After SSH Logout](https://dev.to/ajitkumar/why-your-deep-learning-job-dies-after-ssh-logout-a-practical-guide-to-persistent-linux-sessions-52ml) | Comprehensive comparison of tmux, nohup, disown, systemd for persistent processes | Explains why nohup fails and how systemd-run is production-grade |
| 2 | [Run python script continuously on server](https://codegive.com/blog/run_python_script_continuously_on_server.php) | 5 methods ranked: nohup < screen/tmux < supervisor < systemd < Docker | Decision tree for choosing the right method |
| 3 | [systemd-run vs nohup vs tmux/screen](https://sergiobelkin.com/posts/systemd-run-vs-nohup-vs-tmuxscreen/) | Feature comparison table: systemd-run has status monitoring, lifecycle control, journald logs, cgroups | systemd-run wins on every dimension |
| 4 | [Use systemd-run as alternative for screen](https://makandracards.com/operations/521736-systemd-run-alternative-screen) | Simple `systemd-run --unit=build openssl speed` example, status via `systemctl status`, logs via `journalctl` | Cleanest approach for one-shot tasks |
| 5 | [Why I say no to tmux](https://acsrujan.net/why-i-say-no-to-tmux/) | Argues for systemd services over tmux: proper signal handling, auto-restart, cgroups, logging | Convinced me to use systemd-run |
| 6 | [systemd.service documentation](https://www.freedesktop.org/software/systemd/man/249/systemd.service.html) | Official systemd service docs: Type=simple vs fork, Restart= policies, RuntimeMaxSec= | Essential reference for unit file creation |

### 2.2 Memory-Efficient Concurrent Futures

| # | Link | Summary | Why useful |
|---|------|---------|------------|
| 7 | [Thread pool return ref hold memory (CPython bug 37909)](https://bugs.python.org/issue37909) | Python 3.7+ ThreadPoolExecutor threads keep >600MB per-thread memory after task completion | Confirms the issue is real and Python 3.8+ partially mitigates it |
| 8 | [ThreadPoolExecutor keeps references to worker functions (CPython bug 16284)](https://bugs.python.org/issue16284) | Fixed in Python 3.4 — `_WorkItem` references held until next blocking wait | Why `del future` matters |
| 9 | [Memory leak in run_in_executor (CPython bug 38430)](https://bugs.python.org/issue38430) | Forgetting `await` causes memory leak in asyncio + run_in_executor | Not directly applicable (sync code) but same root cause |
| 10 | [Python ThreadPoolExecutor memory leak (SO)](https://stackoverflow.com/questions/74895168/python-threadpoolexecutor-concurrent-futures-memory-leak) | Answer recommends chunked submission: submit N, process, repeat | Direct solution to our problem |
| 11 | [Memory usage with ThreadPoolExecutor (SO)](https://stackoverflow.com/questions/37445540/memory-usage-with-concurrent-futures-threadpoolexecutor-in-python3) | Answer recommends `wait(FIRST_COMPLETED)` loop to keep future count bounded | The key pattern we need |
| 12 | [ThreadPoolExecutor memory leak (SO)](https://stackoverflow.com/questions/71901539/python-threadpoolexecutor-memory-leak-issues) | Demonstrates that `del result, f` + `gc.collect()` still shows memory growth because Python's allocator doesn't return to OS | Memory may stay high even after `del` — use batch to cap |
| 13 | [Using Concurrent Futures without running out of RAM (SO)](https://stackoverflow.com/questions/34770169/using-concurrent-futures-without-running-out-of-ram) | Answer: use generator expression for futures — `(executor.submit(w, x) for x in items)` — no reference retained after iteration | Direct solution: swap dict comprehension for generator |
| 14 | [ThreadPoolExecutor unreleased memory (SO)](https://stackoverflow.com/questions/63487755/concurrent-futures-threadpoolexecutor-unreleased-memory) | Answer: calling `executor.shutdown()` frees worker memory | Must use `with ThreadPoolExecutor() as executor:` |
| 15 | [How to Wait For The First Task To Finish](https://superfastpython.com/threadpoolexecutor-wait-first-result/) | Clean example of `wait(futures, return_when=FIRST_COMPLETED)` + `done.pop()` | Template for our continuous streaming pattern |
| 16 | [CPython _base.py source](https://github.com/python/cpython/blob/main/Lib/concurrent/futures/_base.py) | `_yield_finished_futures()` explicitly removes future from all ref sets before yielding | `as_completed()` is safe — no hidden references after yield |
| 17 | [future-map library](https://pypi.org/project/future-map/) | `FutureMap(fn, iterable, buffersize=5)` — bounded buffer, yields unordered results | Drop-in replacement for our pattern |
| 18 | [batchit library](https://github.com/Ahmedie-m/batchit) | Weighted batching with backpressure for async/sync pipelines | Over-engineered for our case but demonstrates the pattern |
| 19 | [fastpipe library](https://pypi.org/project/fastpipe/) | Zero-dep pipeline library with `.iter()` (memory-efficient) mode | Good design but we need less abstraction |
| 20 | [Multiprocessing in Production: Pool Tuning, Pipelines, Backpressure](https://medium.com/@priteshjha27/multiprocessing-in-production-pool-tuning-pipelines-and-backpressure-a09116ed31fb) | Queue-based pipeline: workers → bounded queue → single batched writer | The `qmax = n_workers * batch_size * 2` formula is gold |
| 21 | [How to Build Asyncio Queues in Python](https://oneuptime.com/blog/post/2026-01-30-python-asyncio-queues/view) | Producer-consumer with bounded queue for backpressure | Backpressure naturally limits memory |
| 22 | [Accelerating Sync Process Pipelines with Multiprocessing & Batching](https://medium.com/@rakotobesalimo/accelerating-synchronous-process-pipelines-with-multiprocessing-batching-762426a76ff5) | Pipeline parallelism: each stage in own process, connected by bounded queues | Not needed (single-stage pipeline) but good design reference |

### 2.3 EUR-Lex / Cellar API Tooling

| # | Link | Summary | Why useful |
|---|------|---------|------------|
| 23 | [cellar-extractor (maastrichtlawtech)](https://github.com/maastrichtlawtech/cellar-extractor) | Python library for Cellar case law extraction with SPARQL → metadata → full text pipeline | Shows working `threads=N` pattern for Cellar |
| 24 | [cellar-wrapper (Kymylyy)](https://github.com/Kymylyy/cellar-wrapper) | Python wrapper + CLI + MCP server for Cellar CELEX resolution and act metadata | Can use `cellar lookup resolve-celex --celex X` for validation |
| 25 | [EUR-LEX Miner (do-me)](https://github.com/do-me/eur-lex) | High-performance mining tool with parallel parsing, joblib caching, HF Hub upload | Directly comparable project — uploads to HF too! |
| 26 | [EUR-Lex Parser (kevin91nl)](https://github.com/kevin91nl/eurlex) | `get_html_by_celex_id()` + `parse_html()` — clean API for single-doc fetching | Reference implementation for HTML parsing |
| 27 | [pyeurlex](https://pypi.org/project/pyeurlex/) | SPARQL query builder + Cellar download for all resource types | SPARQL generation pattern |
| 28 | [BatchLegal (fvanlitsenburg)](https://github.com/fvanlitsenburg/BatchLegal) | Two-step: EUR-Lex metadata → Cellar references → text content | Same two-step pattern we use |
| 29 | [EU EurLex Cellar Reference Querier (maastrichtlawtech)](https://github.com/maastrichtlawtech/EU_EurLex_Cellar_reference_querier) | Build citation networks from Cellar SPARQL | SPARQL query optimization patterns |
| 30 | [EU Rules Metadata Extractor](https://github.com/nature-of-eu-rules/data-extraction/blob/main/eu_rules_metadata_extractor.py) | CONSTRUCT SPARQL for full metadata extraction per CELEX | Alternative to SELECT for metadata |

### 2.4 FAISS IVFPQ Best Practices

| # | Link | Summary | Why useful |
|---|------|---------|------------|
| 31 | [Indexing 1G vectors — FAISS Wiki](https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors) | Detailed benchmarks: code sizes 8-64 bytes, IVF/HNSW quantizers, memory breakdowns | Reference for memory estimation formulas |
| 32 | [Indexing 1M vectors — FAISS Wiki](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors/795bb432f2371389797b66eb83fd849c3a7954aa) | Comparison: Flat-CPU (512MB, 9s) vs IVF16384,Flat (520MB, 0.5s) vs HNSW (1.3GB, 0.08s) | Confirms IVF over HNSW for memory-constrained |
| 33 | [Compress 1B sentence embeddings d=384 (FAISS issue #2624)](https://github.com/facebookresearch/faiss/issues/2624) | PQ32 gives 32 bytes/vector. 1B × 32 = 32GB. For 4.5GB target, need ~4.5 bytes/vector. | Our 48 bytes/vector × 456K = ~22MB — well within budget |
| 34 | [Production FAISS: Sharding, Quantization, GPU Memory (Markaicode)](https://markaicode.com/architecture/scalable-faiss-architecture-production/) | PQ4x8 = 8 bytes/vector, 10-15× compression, <2% recall loss. `nlist=4*sqrt(N)` formula | Confirms our nlist formula is correct |
| 35 | [FAISS IVF vs HNSW vs Flat: 10M Vector Benchmarked (Markaicode)](https://markaicode.com/benchmarks/faiss-production-benchmark-latency/) | IVF: 4.2GB RAM for 10M vectors (d=768). HNSW: 12.8GB. Flat: 2.86GB but 44ms. | IVF is the right choice for memory-constrained |
| 36 | [Bench all IVF logs deep1M — FAISS Wiki](https://github.com/facebookresearch/faiss/wiki/bench_all_ivf_logs-deep1M) | Detailed per-parameter logs: OPQ16_64,IVF1024,PQ16x4fs → index size 16MB for 1M | Real index sizes for reference |
| 37 | [Estimate memory usage of IVFPQ (FAISS issue #1520)](https://github.com/facebookresearch/faiss/issues/1520) | Formula: `size = n_vectors * 1 * code_size`. 114M × 64 = 6.8GB. Confirmed by FAISS team. | Confirms memory formula |
| 38 | [Precomputed table memory (FAISS issue #1570)](https://github.com/facebookresearch/faiss/issues/1570) | Precomputed table = `nlist × pq.M × ksub × 4 bytes`. For nlist=4096, M=64, ksub=256 = 256MB | Must watch for precomputed table bloat |
| 39 | [FAISS production benchmark latency](https://markaicode.com/benchmarks/faiss-production-benchmark-latency/) | Recommended settings: `nlist = 4*sqrt(N)`, `nprobe=64` for 0.95 recall, `OMP_NUM_THREADS=8` | Production tuning guide |
| 40 | [FAISS GPU memory optimization docs](https://markaicode.com/architecture/scalable-faiss-architecture-production/) | Use `float16` coarse quantizer, IVFPQ halves memory vs Flat on GPU | Not directly applicable (CPU only) but good reference |

### 2.5 General Python Best Practices

| # | Link | Summary | Why useful |
|---|------|---------|------------|
| 41 | [Python `concurrent.futures` docs](https://docs.python.org/3/library/concurrent.futures.html) | Official docs: `wait()`, `as_completed()`, `FIRST_COMPLETED`, `ALL_COMPLETED` | Reference for all patterns |
| 42 | [Python `concurrent.futures` cheat sheet](https://www.pythonsheets.com/notes/concurrency/python-futures.html) | Quick examples: `wait(ALL_COMPLETED)`, `wait(FIRST_COMPLETED)`, `as_completed()` | Quick reference |
| 43 | [CPython `wait()` dedup commit](https://github.com/python/cpython/commit/9a9061d1ca7e28dc2b7e326153e933872c7cd452) | `wait()` now deduplicates futures (Python 3.11+) | Can safely pass duplicate futures |
| 44 | [Tornado's concurrent.futures._base copy](https://www.tornadoweb.org/en/branch6.2/_modules/concurrent/futures/_base.html) | Same as CPython source — `_yield_finished_futures` removes references before yielding | Confirms `as_completed` is safe |
| 45 | [Wait for fastest thread in Python](https://www.pythontutorials.net/blog/how-to-wait-until-only-the-first-thread-is-finished-in-python/) | Practical tutorial on `FIRST_COMPLETED` pattern with weather API example | Code template for our use case |

---

## 3. Revised Architecture

### 3.1 Key Changes from Previous Approach

| Aspect | Old Approach | New Approach | Why |
|--------|-------------|--------------|-----|
| **Process launch** | `nohup` in bash tool | `systemd-run --user --scope` | Survives tool timeout |
| **Task submission** | All 26,871 at once (OOM) | Generator: `(executor.submit(f, d) for d in docs)` then `as_completed(gen)` | O(1) memory for pending futures |
| **Result cleanup** | `future_map[future]` keeps result | Generator naturally discards futures after yield | No `del` needed — GC handles it |
| **Progress monitoring** | tqdm with `\r` (unreadable log) | tqdm with `file=sys.stdout` redirected + periodic structured log lines | Readable log file + terminal progress |
| **Cellar timeout** | Fixed 15s per request | Adaptive: retry with 5s timeout, skip after 3 failures | Faster failure recovery |
| **Corrigenda** | Fetched and discarded | Filtered at SPARQL level: `FILTER(!CONTAINS(?celex, "R("))` | Save 15-20% download time |
| **Throughput** | 20 workers, no backpressure | 20 workers + generative `as_completed` with bounded pending set | Self-regulating throughput |
| **Checkpoint** | None — restart from scratch after failure | Periodic save: chunk count + last CELEX to `data/checkpoint.json` | Resume after crash |

### 3.2 Memory Budget (Revised)

```
Embedding phase:
  all_chunks:  456K × 1KB  = 456MB
  vectors:     456K × 1.5KB = 700MB  (384-dim float32)
  model:                 = 250MB
  Python overhead:       =  50MB
  Total:                 = 1.46GB  ← freed immediately after FAISS write

Query phase (HF Space / Render):
  FAISS index:     22MB  (PQ codes + IVF lists)
  SQLite:           0MB  (on-disk)
  Model:          250MB
  Python overhead:  50MB
  Total:          322MB  ← under 512MB limit
```

---

## 4. Revised Strategy & Implementation Plan

### Phase 0: Prepare the Environment

#### Step 0.1 — Install `systemd-run` (user scope)
```bash
# Verify user instance is running
systemctl --user status
# Set lingering so our service survives logout
sudo loginctl enable-linger $USER
```

#### Step 0.2 — Verify systemd-run works
```bash
systemd-run --user --scope --unit=test-build sleep 60
systemctl --user status test-build.scope
# Wait, then check logs
journalctl --user -u test-build.scope --no-pager
```

### Phase 1: Fix the Build Script

#### Step 1.1 — Apply the generator pattern for futures

Replace the dict comprehension / batch approach with a generator:

```python
with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
    # Generator: futures are created lazily, no materialized collection
    futures = (executor.submit(fetch_document_xhtml, doc) for doc in docs)
    
    # as_completed also doesn't hold refs to completed futures
    for future in tqdm(as_completed(futures), total=len(docs), desc="Fetching"):
        try:
            html = future.result()
            ...
        except Exception:
            ...
```

**Why this works:** The generator expression `(... for doc in docs)` doesn't create a list. Iteration over the generator lazily submits tasks. `as_completed()` internally has its own set of futures, but `_yield_finished_futures()` explicitly removes the yield-ed future from all internal ref sets before yielding it. After `future.result()`, the `future` variable goes out of scope at the top of the next loop iteration, and the generator has already discarded it.

**Key constraint:** We lose the `doc` metadata mapping. Solution: return metadata from `fetch_document_xhtml` as a tuple `(html, celex)` or include it in the fetch function's return value.

#### Step 1.2 — Add checkpoint/resume capability

Save progress every 500 documents:

```python
def save_checkpoint(celex, chunk_count, success_count):
    with open(os.path.join(DATA_DIR, "checkpoint.json"), "w") as f:
        json.dump({"last_celex": celex, "chunk_count": chunk_count, 
                    "success_count": success_count}, f)

def load_checkpoint():
    try:
        with open(os.path.join(DATA_DIR, "checkpoint.json")) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
```

#### Step 1.3 — Add adaptive HTTP timeout with retry

```python
def fetch_document_xhtml(doc, max_retries=3, timeout=10):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)
            r.raise_for_status()
            if len(r.content) >= 500:
                return r.text
            return None  # empty content
        except requests.Timeout:
            if attempt == max_retries - 1:
                return None
            timeout *= 1.5  # exponential backoff
        except requests.RequestException:
            return None  # 4xx/5xx, don't retry
    return None
```

#### Step 1.4 — Add structured progress logging

```python
logger.info(f"PROGRESS: {i}/{total} documents | {success_count} OK "
            f"| {len(all_chunks)} chunks | {elapsed:.1f}s elapsed"
            f"| {(i/elapsed):.1f} it/s")
```

### Phase 2: Launch the Build

#### Step 2.1 — Pre-filter corrigenda at SPARQL level

Add to the SPARQL query:
```sparql
FILTER(!CONTAINS(?celex, "R("))
```

#### Step 2.2 — Launch with systemd-run

```bash
systemd-run --user --scope --unit=eurlex-build \
    -E HF_TOKEN="hf_xxx" \
    -E PATH="$HOME/Desktop/EUProjects/.venv/bin:$PATH" \
    bash -c "source $HOME/Desktop/EUProjects/.venv/bin/activate && \
             python3 scripts/build_index.py 2>&1 | tee -a data/build-$(date +%Y%m%d-%H%M).log"
```

#### Step 2.3 — Monitor via systemd

```bash
# Check status
systemctl --user status eurlex-build.scope
# Tail logs
journalctl --user -u eurlex-build.scope -f --no-pager
# Check resource usage
systemd-cgtop --user
```

### Phase 3: Deployment

Same as before — commit to git, push to GitHub, auto-deploy to HF Spaces + Render + Vercel.

### Phase 4: Verification

Same as before — health check, query test, memory monitoring.

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| systemd-run not available | Low (Ubuntu 22.04+) | High | Fallback: `setsid` with explicit `--fork` |
| KillUserProcesses kills build | Medium (systemd 230+) | High | Enable lingering: `loginctl enable-linger $USER` |
| Generator pattern doesn't free memory fast enough | Low | Medium | Add explicit `del future` at end of loop |
| Cellar rate-limits aggressively | Low | High | Add `time.sleep(0.1)` between batch submissions |
| SPARQL endpoint times out | Low | Medium | Add timeout to SPARQL query, retry |
| Embedding model OOM at 456K chunks | Low | Medium | Batch embed in chunks of 10K |
| HF Hub upload fails mid-way | Low | Medium | Retry with exponential backoff |
| Build takes longer than expected | Medium | Low | Checkpoints save every 500 docs — resume if killed |
