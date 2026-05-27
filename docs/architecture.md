# System Architecture

## Overview

EUR-Lex AI Chat is a RAG-based system that answers questions about EU law using a sophisticated 9-stage pipeline. The system processes queries through query understanding, hybrid retrieval, answer generation, validation, and confidence estimation.

## Architecture Diagram

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 1: Query Received             │
│  - Request logging                  │
│  - Client IP tracking               │
│  - Request ID generation            │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  STAGE 2: Query Classification      │
│  - EUQuestionClassifier             │
│  - Legal intent detection           │
│    (obligation, definition, entity, │
│     procedural, right, prohibition) │
│  - Actor extraction                 │
│  - Clarification gating             │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  STAGE 3: Query Expansion           │
│  - Legal synonym injection (55+    │
│    term dictionary)                │
│  - Obligation pattern transform    │
│    (e.g., "employer responsibilities"│
│    → "obligations of undertakings")│
│  - AutoExpander (learns from failures)│
│  - Up to 5 query variations         │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  STAGE 4: Hybrid Search + Rerank    │
│  1. BM25 sparse retrieval (top 20) │
│  2. FAISS dense retrieval (top 20) │
│  3. RRF fusion (Reciprocal Rank    │
│     Fusion, k=60) across all       │
│     query variations               │
│  4. Cross-encoder reranking        │
│     (ms-marco-MiniLM-L-6-v2)       │
│  - Returns top 10 chunks           │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  STAGE 5: Confidence Gating         │
│  - Assess if retrieved chunks      │
│    sufficiently answer query       │
│  - Check relevance thresholds      │
│  - May return partial citations    │
│    without full answer             │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  STAGE 6: Answer Generation         │
│  - RAG prompt with:                │
│    • Query + classification        │
│    • Retrieved chunks              │
│    • Per-chunk relation summaries │
│    • CELEX citation guidance      │
│  - LLM: llama-3.3-70b-versatile   │
│    via Groq API                    │
│  - Exponential backoff retry       │
│  - 413 auto-chunk-reduction        │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  STAGE 7: Answer Validation        │
│  AnswerValidator checks:           │
│  - Minimum length (≥100 chars)     │
│  - CELEX citations (≥2 for         │
│    obligation queries)            │
│  - Deontic language presence      │
│    (shall/must for obligations)   │
│  - Keyword overlap                │
└────────────────┬────────────────────┘
                 │
          ┌──────┴───────┐
          ▼              ▼
     ✅ Passes        ❌ Fails
          │              │
          ▼              ▼
┌──────────────┐  ┌──────────────────┐
│ STAGE 8:     │  │ Retry with       │
│ Confidence   │  │ citation emphasis│
│ Estimation   │  └────────┬─────────┘
│ + Response   │           │
└──────────────┘           │
          │          ┌──────┴──────┐
          │          ▼            ▼
          │      ✅ Passes    ❌ Fails
          │          │            │
          │          ▼            ▼
          │   ┌──────────┐  ┌─────────────┐
          │   │ Return   │  │ Informative │
          │   │ answer   │  │ fallback    │
          │   │ + conf   │  │ + CELEX refs│
          │   └──────────┘  └─────────────┘
          ▼
┌──────────────┐
│ STAGE 9:     │
│ Response     │
│ Returned     │
└──────────────┘
```

## Pipeline Stages in Detail

### Stage 1: Query Received
- Request ID generation (UUID)
- Client IP extraction
- Timestamp logging

### Stage 2: Query Classification
Uses `EUQuestionClassifier` (spaCy patterns):
- Intent detection: obligation, definition, entity, procedural, right, prohibition
- Actor identification (employer, member state, commission, etc.)
- Confidence gating — may trigger clarification prompt

### Stage 3: Query Expansion
- `expand_query()`: Injects legal synonyms (55+ curated terms)
- `expand_obligation_query()`: Transforms plain language to legal patterns
- `AutoExpander`: Records failed query terms for future improvements

### Stage 4: Hybrid Search + Reranking
**Retrievers:**
- **BM25**: Keyword-based sparse retrieval from `rank-bm25` index
- **FAISS**: Dense vector similarity (384-dim MiniLM or 768-dim EURLEX-BERT)

**Fusion:**
- Reciprocal Rank Fusion (RRF): score = 1 / (k + rank), k=60
- Combines top 20 from each retriever (across all query variations)
- RRF scores aggregated across retrievers

**Reranking:**
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Re-ranks top 20 candidates to produce final top 10
- Adds `rerank_score` field to results

### Stage 5: Confidence Gating
- `classifier.should_answer()` decides if retrieved chunks are adequate
- May short-circuit with "I don't have enough information" or partial citations
- Considers: number of results, presence of operative articles, intent type

### Stage 6: Answer Generation
- `answer_question()` from `app/rag.py`
- Constructs RAG prompt with:
  - System instruction (intent-specific)
  - Retrieved chunks (with CELEX, article, type, text)
  - Per-chunk relation summaries (obligations, rights, actors)
  - CELEX citation guidance
- Calls Groq API with Llama 3.3 70B
- Handles 413 payload too large by reducing chunks

### Stage 7: Answer Validation
`AnswerValidator` performs 4 checks:
1. **Length**: ≥ 100 characters
2. **Citations**: ≥ 2 CELEX numbers for obligation queries
3. **Deontic language**: "shall"/"must" present for obligation queries
4. **Keyword overlap**: Query keywords appear in answer

If validation fails → Stage 8 retry.

### Stage 8: Retry or Fallback
- Retry once with `ENSURE_CITATION_PROMPT` emphasizing CELEX citations
- If retry fails:
  - Record failure in `AutoExpander`
  - Generate informative fallback with CELEX references
  - Returns "I found documents..." with citations

### Stage 9: Response Return
- Final response JSON with:
  ```json
  {
    "answer": "...",
    "citations": ["32023L0970", "32016R0679"],
    "sources": [...],
    "_confidence": "high|medium|low"
  }
  ```
- Pipeline logging middleware writes structured JSON log entry

## Data Flow

```
┌────────────┐
│   Query    │
└─────┬──────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│           Retrieval Layer                   │
│  ┌─────────────┐      ┌─────────────┐    │
│  │    BM25     │      │    FAISS    │    │
│  │   (Sparse)  │      │   (Dense)   │    │
│  └──────┬──────┘      └──────┬──────┘    │
│         │                   │            │
│         └──────────┬────────┘            │
│                    ▼                      │
│           RRF Fusion (top 60)            │
│                    ▼                      │
│           Cross-Encoder Reranker (top 10)│
│                    ▼                      │
│           Retrieved Chunks (with         │
│           metadata: celex, title,        │
│           article, type, text, score)   │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│         LLM Generation Layer               │
│  • RAG prompt construction                │
│  • Groq API call (llama-3.3-70b-versatile)│
│  • Streaming not supported (full answer)  │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│         Validation Layer                   │
│  • AnswerValidator (4 checks)             │
│  • Confidence estimator                   │
│  • Fallback generator                     │
└─────────────────────────────────────────────┘
      │
      ▼
┌────────────┐
│  Response  │
└────────────┘
```

## Storage & Indexes

### HuggingFace Dataset

Index files stored at `NedAktovOps/eurlex-chat-data`:

```
data/
├── index.faiss          # MiniLM embeddings (384-dim, IVFPQ ~28MB)
├── index_eurlex.faiss   # EURLEX-BERT embeddings (768-dim, IVFPQ ~56MB)
├── chunks.db            # SQLite with chunk text, metadata (377MB)
├── chunks_eurlex.db     # Copy of chunks.db (for EURLEX index)
└── build_meta.json      # Build timestamp, CELEX count, etc.
```

### SQLite Schema

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    celex TEXT NOT NULL,
    title TEXT,
    article TEXT,
    type TEXT DEFAULT 'section',
    text TEXT NOT NULL,
    embedding BLOB NOT NULL  -- FAISS index stored separately
);
CREATE INDEX idx_chunks_celex ON chunks(celex);
```

### FAISS Index

- Type: `IndexIVFPQ` (Product Quantization for memory efficiency)
- Dimensions: 384 (MiniLM) or 768 (EURLEX-BERT)
- Trained on 305K+ embeddings
- Stored in `.faiss` file (binary)

### BM25 Store

- Serialized `rank_bm25.BM25Okapi` index
- Stored as `bm25_store.pkl` (approx 50MB)
- Optional: system works without BM25 (FAISS-only fallback)

## Component Reference

| Component | File | Purpose |
|-----------|------|---------|
| FastAPI app | `app/main.py` | HTTP server, pipeline orchestration |
| Search | `app/search.py` | FAISS search, discourse boost, RRF fusion |
| Hybrid Search | `app/hybrid_search.py` | RRF fusion implementation |
| Reranker | `app/reranker.py` | Cross-encoder reranking |
| Classifier | `app/question_classifier.py` | Intent/actor detection |
| Query Expander | `app/query_expander.py` | Legal synonym expansion |
| Relation Extractor | `app/relation_extractor.py` | Per-chunk legal relation extraction |
| Answer Validation | `app/answer_validator.py` | Quality checks, fallback |
| RAG | `app/rag.py` | Prompt construction, Groq API |
| Data Loader | `app/data_loader.py` | Index download, embedding model factory |
| Embedders | `app/data_loader.py` (EURLEXEmbedder, SentenceTransformer) | Text → vector |
| Logging | `app/logging_middleware.py` | Structured JSON pipeline logs |

## Configuration

### Environment Variables

See [`docs/DEPLOYMENT.md`](./DEPLOYMENT.md#environment-variables-reference) for full list.

Key ones:

- `GROQ_API_KEY` — Required, Groq API authentication
- `HF_TOKEN` — Required, HuggingFace dataset access
- `INDEX_SUFFIX` — Optional, `""` (MiniLM) or `"_eurlex"` (EURLEX-BERT)
- `PORT` — Optional, server port (default 8000)

### Model Selection

Switch embedding models at startup:

```bash
export INDEX_SUFFIX=""          # MiniLM (default, 384-dim)
export INDEX_SUFFIX="_eurlex"   # EURLEX-BERT (768-dim, better legal)
```

Both use the same `chunks.db`; only the embeddings differ.

## Performance Characteristics

| Stage | Typical Latency (p95) | Notes |
|-------|----------------------|-------|
| Classification | 50ms | Fast spaCy pattern matching |
| Query Expansion | 30ms | Dictionary lookup |
| BM25 Search | 100ms | Depends on index size |
| FAISS Search | 150ms | IVFPQ ~1ms per 10k vectors |
| RRF Fusion | 10ms | Merge + score aggregation |
| Reranking | 200ms | Cross-encoder (slowest) |
| Answer Generation | 1500ms | Groq API, model-dependent |
| Validation | 50ms | Rule-based checks |
| **Total** | **~2100ms** | Median on warm cache |

Memory footprint: ~450MB (MiniLM) or ~550MB (EURLEX-BERT) on HF Spaces `cpu-basic`.

## Coverage

Current index covers **86.77%** of eligible EU documents:

- Indexed: 15,112 distinct CELEXes
- Total available (REG+DIR from 2004-01-01): 17,417
- Missing: 2,305 (13.23%)

See [`docs/coverage_benchmark_20260526.md`](./coverage_benchmark_20260526.md) for details.

## Future Improvements

- RAGAS evaluation integration (present in `scripts/evaluate.py`)
- Chunkweaver LEGAL_EU for structure-aware splitting (already integrated in build)
- Dynamic model routing based on query complexity
- Multi-turn conversation memory
- Citation linking to specific articles
