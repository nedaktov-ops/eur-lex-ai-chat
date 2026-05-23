# EUR-Lex AI Chat

> An intelligent legal reasoning assistant for EU law — powered by RAG, discourse-aware search, and answer validation.

Ask questions about EU directives, regulations, and legislation in plain English. The system understands legal intent, expands queries with legal synonyms, performs discourse-aware search across 305K+ chunks of EUR-Lex documents, generates answers with CELEX citations, validates answer quality, and provides confidence estimates.

**Live demo:** [frontend-ruddy-zeta-40.vercel.app](https://frontend-ruddy-zeta-40.vercel.app)  
**API backend:** [nedaktovops-eurlex-chat-api.hf.space](https://nedaktovops-eurlex-chat-api.hf.space)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  1. Query Classification           │
│     - Legal intent detection       │
│     - Actor extraction             │
│     - Clarification gating         │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  2. Query Expansion                │
│     - Legal synonym injection      │
│     - Obligation pattern transform │
│     - Auto-expansion from failures │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  3. Discourse-Aware FAISS Search   │
│     - 384-dim MiniLM embeddings    │
│     - IVFPQ index (305K vectors)   │
│     - Operative article boosting   │
│     - Deontic language scoring     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  4. Legal Relation Extraction      │
│     - Obligation/prohibition/right │
│     - Actor identification         │
│     - Per-chunk relation summaries │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  5. Enhanced Prompt Construction   │
│     - Intent-specific instructions │
│     - Relation metadata injection  │
│     - CELEX citation guidance      │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  6. Groq LLM Call (Llama 3.3 70B) │
│     - Exponential backoff retry    │
│     - 413 auto-chunk-reduction     │
│     - 4096 max tokens              │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  7. Answer Validation              │
│     - Length & specificity check   │
│     - CELEX citation verification  │
│     - Deontic language check       │
│     - Keyword overlap analysis     │
└────────────────┬────────────────────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
    ✅ Passes      ❌ Fails
         │               │
         ▼               ▼
┌──────────────┐  ┌──────────────────┐
│ Confidence   │  │ Retry with       │
│ Estimation   │  │ citation emphasis│
│ + Response   │  └────────┬─────────┘
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
         ▼   └──────────┘  └─────────────┘
    Return response
```

## Features

### Phase 0 — Foundation & Safety
- Structured JSON pipeline logging (7 stages)
- HuggingFace Hub index backup/restore
- Checkpoint save/restore for rollback safety
- Automated GitHub backup workflow (every 6 hours)

### Phase 1 — Query Understanding
- **`EUQuestionClassifier`** — detects legal intent (obligation, definition, entity, procedural), extracts legal actors, confidence gating for clarification
- **`expand_obligation_query()`** — transforms plain language into legal text patterns (e.g., "employer responsibilities" → "obligations of undertakings")
- **`expand_query()`** — injects legal synonyms from a curated 55+ term dictionary
- Pay Transparency obligation recall improved 37% (0.496 → 0.698)

### Phase 2 — Legal Reasoning
- **`discourse_boost()`** — re-ranks search results: operative articles > recitals > annexes; boosts chunks with deontic language (shall/must) for obligation queries
- **`search_discourse_aware()`** — combines FAISS similarity with discourse scoring
- **`extract_legal_relations()`** — identifies obligations, prohibitions, rights, conditions, and actors per chunk
- **`DualIndexManager`** — supports MiniLM (384-dim) and EURLEX-BERT (768-dim) indexes with seamless rollback
- **Enhanced prompt** — per-chunk relation summaries injected into context, intent-specific system instructions

### Phase 3 — Answer Validation
- **`AnswerValidator`** — 4 checks: min length (100 chars), CELEX citation count (≥2), deontic language for obligation queries, keyword overlap
- **`estimate_confidence()`** — scores by relevance, operative article ratio, deontic presence → high/medium/low
- **`make_fallback_answer()`** — informative fallback with CELEX numbers and document titles
- Pipeline logs include `validation_passed`, `confidence_level`, `confidence_score`

### Phase 4 — Continuous Improvement
- **LLM retry** — on validation failure, retries once with `ENSURE_CITATION_PROMPT` emphasizing CELEX citation
- **`AutoExpander`** — records failed query terms for future synonym expansion
- **`feedback_analyzer.py`** — analyzes pipeline logs for pass rates, latency, intent distribution, confidence breakdown
- **EURLEX-BERT support** — `INDEX_SUFFIX=_eurlex` env var enables 768-dim embeddings from `nlpaueb/bert-base-uncased-eurlex`

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.12, FastAPI |
| **Vector Search** | FAISS IVFPQ (305K vectors, 384-dim) |
| **Chunk Storage** | SQLite (~35MB, on-disk) |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) or nlpaueb/bert-base-uncased-eurlex (768-dim) |
| **LLM** | Groq API — llama-3.3-70b-versatile |
| **NLP** | sentence-transformers, spaCy, transformers |
| **Logging** | Structured JSON (stdout) |
| **Frontend** | Astro + React (Tailwind CSS) |
| **Hosting** | HuggingFace Spaces (backend), Vercel (frontend) |
| **CI/CD** | GitHub Actions (backup, feedback analysis, index rebuild) |

## Quick Start

### Prerequisites
- Python 3.12+
- `pip` and `venv`

### Setup

```bash
git clone https://github.com/nedaktov-ops/eur-lex-ai-chat.git
cd eur-lex-ai-chat

python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
```

### Run

```bash
# Set API keys
export GROQ_API_KEY="your_groq_key"
export HF_TOKEN="your_hf_token"

# Start the server
uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
```

### Test

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the responsibilities of employers under the Pay Transparency Directive?"}'
```

### Run Tests

```bash
python3 -m pytest tests/ -v
```

## API Reference

### `POST /chat`

Ask a question about EU law.

**Request:**
```json
{
  "query": "What are employer obligations under GDPR?"
}
```

**Response (validated):**
```json
{
  "answer": "Based on the provided legal texts, data controllers have specific obligations...",
  "citations": ["32016R0679", "32023L0970"],
  "sources": [
    {
      "celex": "32016R0679",
      "title": "General Data Protection Regulation",
      "article": "art_5",
      "score": 0.68
    }
  ],
  "_confidence": "high"
}
```

**Response (fallback):**
```json
{
  "answer": "I found documents related to your question...",
  "citations": ["32023L0970"],
  "sources": [...]
}
```

### `GET /health`

Health check. Returns `{"status": "ok", "vectors": 305957}`.

### `GET /stats`

Index statistics: vector count, file sizes, last updated timestamp.

### `POST /backup`

Trigger manual index backup to HuggingFace Hub.

## Project Structure

```
eur-lex-ai-chat/
├── backend/
│   ├── main.py                    # FastAPI app — all pipeline stages
│   ├── rag.py                     # RAG prompt builder + Groq caller
│   ├── search.py                  # FAISS+SQLite search, discourse scoring
│   ├── data_loader.py             # Index loader, DualIndexManager, EURLEXEmbedder
│   ├── logging_middleware.py      # Structured JSON pipeline logging
│   ├── answer_validator.py        # Answer quality validation + confidence
│   ├── question_classifier.py     # Legal intent/actor detection
│   ├── query_expander.py          # Legal synonym expansion + AutoExpander
│   ├── relation_extractor.py      # Legal relation extraction
│   ├── rate_limit.py              # Groq rate limiting
│   ├── startup.sh                 # Render/HF Space entry point
│   └── requirements.txt
├── frontend/
│   ├── src/components/ChatWidget.jsx  # React chat widget
│   ├── package.json                   # Astro + React
│   ├── vercel.json                    # Vercel deployment config
│   └── dist/                          # Built output
├── scripts/
│   ├── build_index.py             # Full index builder (SPARQL→XHTML→chunks→FAISS→HF)
│   ├── backup_index.py            # HF Hub backup/restore tool
│   ├── checkpoint_save.py         # Checkpoint creation
│   ├── checkpoint_restore.py      # Checkpoint restoration
│   ├── feedback_analyzer.py       # Pipeline log analysis
│   └── rollback.sh                # Emergency rollback
├── tests/
│   ├── test_question_classifier.py    # 5 tests
│   ├── test_query_expansion.py        # 5 tests
│   ├── test_discourse_scoring.py      # 5 tests
│   ├── test_relation_extraction.py    # 5 tests
│   ├── test_answer_validation.py      # 6 tests
│   └── test_regression.py             # 6 tests
├── docs/
│   ├── IMPROVEMENT_STRATEGY.md    # Full 4-phase improvement plan
│   └── phase4-plan.md             # Phase 4 implementation plan
├── data/                          # FAISS index + SQLite (gitignored, on HF Hub)
├── .github/workflows/
│   ├── backup.yml                 # Automated backup + EURLEX-BERT rebuild trigger
│   └── feedback-analysis.yml      # Weekly log analysis
├── .checkpoints/                  # Checkpoint snapshots (gitignored)
└── .gitignore
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM inference |
| `HF_TOKEN` | Yes | HuggingFace token for index download/backup |
| `INDEX_SUFFIX` | No | Set to `_eurlex` for EURLEX-BERT 768-dim index |
| `GROQ_MODEL` | No | Model name (default: `llama-3.3-70b-versatile`) |
| `PORT` | No | Server port (default: `8000`) |

## Deployment

### Backend (HuggingFace Spaces)

The backend auto-deploys via git push to the HF Space:

```bash
git remote add hf https://huggingface.co/spaces/nedaktovops/eurlex-chat-api
git push hf main --force
```

Set environment variables in the HF Space dashboard:
- `GROQ_API_KEY`
- `HF_TOKEN`
- `INDEX_SUFFIX` (optional, for EURLEX-BERT)

### Frontend (Vercel)

```bash
cd frontend
npm install
npm run build
npx vercel --prod
```

The frontend expects the API at `https://nedaktovops-eurlex-chat-api.hf.space`. Override with `VITE_API_URL` env var if needed.

## EURLEX-BERT Index Rebuild

To rebuild the FAISS index with 768-dim legal embeddings:

```bash
EMBEDDING_MODEL=nlpaueb/bert-base-uncased-eurlex \
INDEX_SUFFIX=_eurlex \
HF_TOKEN=hf_xxx \
python3 scripts/build_index.py
```

Or trigger via GitHub Actions: go to Actions → Daily Backup → "Run workflow" → check "Rebuild index".

## Checkpoints

Save and restore session state for rollback safety:

```bash
# Save checkpoint
python3 scripts/checkpoint_save.py --phase 3 --message "my changes"

# List checkpoints
python3 scripts/checkpoint_restore.py --list

# Restore checkpoint
python3 scripts/checkpoint_restore.py --id ckpt-20260523-204006
```

## Feedback Analysis

Analyze pipeline logs for quality metrics:

```bash
cat /path/to/server.log | python3 scripts/feedback_analyzer.py
```

Outputs: validation pass rate, latency (avg/p50/p95), confidence distribution, intent breakdown, answer length, citation counts.

## License

MIT
