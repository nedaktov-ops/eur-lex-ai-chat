# EUR-Lex AI Chat: Strategic Improvement Plan
## From Basic Retrieval to Legal Reasoning Assistant — All Free, All Self-Sustaining

**Version:** 1.0  
**Date:** May 23, 2026  
**Author:** Generated from comprehensive code analysis + NLP research  
**Status:** Approved for phased implementation  

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [The Problem: Why the System Fails](#2-the-problem-why-the-system-fails)
3. [Improvement Strategy Overview — 4 Phases](#3-improvement-strategy-overview--4-phases)
4. [Phase 0: Foundation & Safety](#4-phase-0-foundation--safety)
5. [Phase 1: Query Understanding Enhancement](#5-phase-1-query-understanding-enhancement)
6. [Phase 2: Legal-Specific Reasoning Enhancement](#6-phase-2-legal-specific-reasoning-enhancement)
7. [Phase 3: Answer Synthesis & Validation](#7-phase-3-answer-synthesis--validation)
8. [Phase 4: Continuous Improvement Loop](#8-phase-4-continuous-improvement-loop)
9. [Resource Requirements](#9-resource-requirements)
10. [Risk Mitigation](#10-risk-mitigation)
11. [Expected Outcomes](#11-expected-outcomes)
12. [The "Holy Shit, This Works" Moment](#12-the-holy-shit-this-works-moment)
13. [Appendices](#13-appendices)

---

## 1. Current State Analysis

### 1.1 System Architecture (Existing)

```
User Browser
    │
    ▼
[Vercel: Frontend (Astro + React ChatWidget)]
    │  POST /chat {query}
    ▼
[Render/HuggingFace Spaces: FastAPI Backend]
    │
    ├── main.py          → /chat, /health, /refresh endpoints
    ├── search.py        → FAISS IVFPQ search + SQLite lookup
    ├── rag.py           → Build prompt, call Groq, extract citations
    ├── data_loader.py   → Download/cache index from HuggingFace Hub
    ├── rate_limit.py    → Per-IP + global rate limiting
    │
    ▼
[Groq API: Llama 3.3 70B]
    │  Returns answer
    ▼
[User sees response with citations]
```

### 1.2 Data Pipeline (Existing)

```
EUR-Lex Cellar (public SPARQL + REST API)
    │  GitHub Actions queries daily
    ▼
GitHub Actions Runner (free, 7GB RAM, 2-core CPU)
    │  install → chunk → embed → merge
    ▼
HuggingFace Hub Dataset (NedAktovOps/eurlex-chat-data)
    │  index.faiss (FAISS IVFPQ, ~21MB)
    │  chunks.db (SQLite, ~376MB)
    │  build_meta.json
    │  last_updated.txt
    │
    ▼
Render Backend loads from HF on startup, refreshes hourly
```

### 1.3 Verified Strengths (Preserve These)

| Component | Status | Notes |
|-----------|--------|-------|
| FAISS IVFPQ index | ✅ Working | 305,957 vectors, ~21MB (vs 768MB raw) |
| SQLite chunk storage | ✅ Working | On-disk, zero RAM at query time |
| Data pipeline (GH → HF) | ✅ Working | GitHub Actions updates daily |
| Backend API endpoints | ✅ Working | /health, /refresh, /chat all tested |
| Frontend (Astro + React) | ✅ Working | ChatWidget connects to backend |
| Zero-cost infrastructure | ✅ Working | HF + GH + Render + Vercel + Groq free tiers |
| Groq Llama 3 70B generation | ✅ Working | Accurate answer generation when pipeline works |
| Automatic index refresh | ✅ Working | Hourly from HuggingFace via GitHub Actions |

### 1.4 Observed Weaknesses (What We Must Fix)

| Component | Status | Impact |
|-----------|--------|--------|
| Query understanding | ❌ Weak | No query expansion or legal term mapping |
| Search relevance | ⚠️ Inconsistent | Sometimes misses key directives on first try |
| Answer synthesis | ❌ Weak | Quotes chunks instead of synthesizing legal info |
| Answer validation | ❌ Missing | Returns vague responses instead of honest limitations |
| Legal discourse awareness | ❌ Missing | Doesn't distinguish obligations vs. permissions |
| Feedback loop | ❌ Missing | No mechanism to learn from failures |

---

## 2. The Problem: Why the System Fails

### 2.1 Traceable Failure Mode

**User Query:** *"What are the responsibilities of employers under the pay transparency directive?"*

**Expected System Behavior:**
1. Recognize this as an obligation-seeking question about EU law
2. Search for chunks from CELEX 32023L0970 (Pay Transparency Directive)
3. Find articles discussing employer duties (Articles 5, 12, rct_52)
4. Synthesize: Employers must disclose salary ranges, cannot ask about pay history, etc.
5. Return structured answer with specific citations

**Actual System Behavior:**
1. System processes query: `model.encode([query], normalize_embeddings=True)`
2. FAISS search returns chunks with scores
3. But returned chunks don't prominently include 32023L0970
4. Instead returns unrelated CELEX numbers
5. LLM generates: *"Based on the provided context, there is no specific information..."*
6. User sees frustrating non-answer with irrelevant CELEX references

### 2.2 Root Causes

| Root Cause | Description | Where It Happens |
|-----------|-------------|------------------|
| **Surface matching only** | Embedding model (all-MiniLM-L6-v2) isn't legal-domain-optimized. "Employer responsibilities" doesn't map well to legal deontic language ("shall", "obligations of undertakings") | `backend/search.py:20` — FAISS search |
| **Discourse blindness** | System chunks text arbitrarily, doesn't preserve legal discourse units (recitals, articles, obligations vs. definitions) | `scripts/build_index.py` — chunking logic |
| **No legal reasoning** | LLM receives raw chunks without structural metadata about rights/obligations hierarchy | `backend/rag.py:30-60` — build_prompt() |
| **Weak answer validation** | No check that answer actually addresses the question before returning | `backend/main.py:125-130` — return result |
| **Missing feedback loop** | No logging or analysis of failed queries to inform improvements | System-wide |

### 2.3 Trace Verification

**Evidence from Code:**

```python
# backend/main.py:121-124 (Current — surface-level only)
model = get_embedding_model()
query_vector = model.encode([query], normalize_embeddings=True)  # No preprocessing
chunks = search(query_vector, top_k=10)  # No query expansion

# backend/rag.py:30-60 (Current — no discourse awareness)
def build_prompt(query, context_chunks):
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = f"[{i+1}] CELEX {chunk['celex']}"
        if chunk.get("article"):
            source += f", Article {chunk['article']}"
        context_parts.append(f"Context {i+1} ({source}):\n{chunk['text']}")
    # No metadata about: is this an obligation? definition? recital?
    # No structural information about legal hierarchy
```

**Evidence from Data (Confirmed 32023L0970 Exists):**

```sql
-- Verified: Pay Transparency Directive IS in database
SELECT COUNT(*) FROM chunks WHERE celex = '32023L0970';
-- Result: 111 chunks (directive is present but not always retrieved)
```

**Evidence from HF Dataset (Confirmed Same):**

```
Dataset: NedAktovOps/eurlex-chat-data
Chunks with CELEX 32023L0970: 111 (verified via snapshot_download)
```

---

## 3. Improvement Strategy Overview — 4 Phases

```
Phase 0: Foundation & Safety
  │  Days 0-1
  ▼
Phase 1: Query Understanding Enhancement
  │  Days 2-5
  ▼
Phase 2: Legal-Specific Reasoning Enhancement
  │  Days 6-12
  ▼
Phase 3: Answer Synthesis & Validation
  │  Days 13-18
  ▼
Phase 4: Continuous Improvement Loop (ongoing)
  │  Days 19+
  ▼
Self-Sustaining Legal Reasoning Assistant
```

### 3.1 Design Principles

1. **Non-destructive**: Every change must have a rollback path
2. **Evidence-based**: Each improvement must prove its value before permanent adoption
3. **Phase-gated**: Each phase must pass validation before proceeding
4. **Zero-cost**: All resources must remain completely free
5. **Self-sustaining**: System continues operating autonomously after changes

### 3.2 File Change Map

```
eur-lex-ai-chat/
├── backend/
│   ├── main.py                  ← PHASES 0, 1, 2, 3, 4
│   ├── search.py                ← PHASES 2
│   ├── rag.py                   ← PHASES 2, 3 (major overhaul)
│   ├── data_loader.py           ← PHASES 0, 2
│   ├── requirements.txt         ← PHASES 1, 2
│   └── startup.sh               ← PHASES 0
├── frontend/
│   └── src/components/          ← PHASE 4 (minor: feedback UI)
├── scripts/
│   ├── build_index.py           ← PHASE 2 (update embeddings)
│   └── update_index.py          ← PHASE 2 (update embeddings)
├── data/                        ← PHASE 0 (backups)
├── .github/workflows/           ← PHASES 0, 4
└── docs/                        ← PHASE 0 (strategy docs)
```

### 3.3 Resource Dependency Graph

```
Phase 0: No new dependencies (logging, backup scripts)
Phase 1:
  ├── onnxruntime (pip package)
  ├── little_questions (pip package with ONNX models)
  └── transformers (pip package for BERT-based classification)
Phase 2:
  ├── transformers (already added in Phase 1)
  ├── torch (CPU-only version, pip package)
  └── nlpaueb/bert-base-uncased-eurlex (HuggingFace model)
Phase 3: No new dependencies (improved prompt templates, validation logic)
Phase 4: No new dependencies (logging analysis, feedback collection)
```

---

## 4. Phase 0: Foundation & Safety

### 4.1 Overview

**Goal:** Establish monitoring, backups, and rollback mechanisms before any changes.

**Duration:** 2 days  
**Risk:** Low (operational only, no algorithmic changes)  
**Rollback visibility:** System unaffected if this phase is completely rolled back (still works without monitoring)

### 4.2 What to Implement

#### 4.2.1 Comprehensive Logging System

**What:** Add structured JSON logging for every stage of the RAG pipeline.

**Files to modify:**
- `backend/main.py`: Add logging middleware
- `backend/rag.py`: Add stage-level logging

**Implementation:**

```python
# backend/rag.py — Add structured logging
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def log_query_stage(stage, query, data, level="INFO"):
    """Log structured data for query processing stages."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "stage": stage,
        "query": query[:200],  # Truncate long queries
        "data": data,
        "level": level
    }
    logger.log(getattr(logging, level), json.dumps(log_entry))
```

**What this captures:**

| Stage | Data Captured | File |
|-------|---------------|------|
| Query received | Original query text, timestamp | `main.py` |
| Query processed | NLP preprocessing results, query expansion | `main.py` |
| Search performed | Top-k scores, chunk IDs, CELEX numbers | `search.py` |
| Prompt built | Number of chunks, total context length, source types | `rag.py` |
| LLM call | Model name, tokens used, latency, success/failure | `rag.py` |
| Answer generated | Answer length, citations extracted, confidence score | `rag.py` |
| Response returned | Status code, response size, latency | `main.py` |

**Value add:**
- Identifies which stage is failing for specific query types
- Enables data-driven improvement decisions
- Provides evidence for Phase 4 feedback loop

#### 4.2.2 Automated Backup System

**What:** Hourly backups of critical system state with point-in-time recovery.

**Files to create/modify:**
- `scripts/backup_index.py` — New backup script
- `data_loader.py` — Add backup coordination
- `.github/workflows/backup.yml` — New workflow

**Implementation:**

```python
# scripts/backup_index.py — New file
#!/usr/bin/env python3
"""Backup the current FAISS index + SQLite DB to HuggingFace Hub."""
import os
import sys
from datetime import datetime
from huggingface_hub import HfApi, upload_folder

BACKUP_DATASET = "NedAktovOps/eurlex-chat-backups"
LOCAL_DATA_DIR = "data"

def create_backup():
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup_dir = f"/tmp/backup-{timestamp}"
    
    # Copy current data
    os.makedirs(backup_dir, exist_ok=True)
    for f in ["index.faiss", "chunks.db", "build_meta.json", "last_updated.txt"]:
        src = os.path.join(LOCAL_DATA_DIR, f)
        if os.path.exists(src):
            os.system(f"cp {src} {backup_dir}/")
    
    # Upload to HF using backup branch per day
    branch = f"backup-{datetime.utcnow().strftime('%Y%m%d')}"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_folder(
        folder_path=backup_dir,
        repo_id=BACKUP_DATASET,
        repo_type="dataset",
        revision=branch,
        create_pr=False,
    )
    print(f"Backup saved to {BACKUP_DATASET}@{branch}")

if __name__ == "__main__":
    create_backup()
```

**Recovery procedure:**
```bash
# Restore from latest backup
HUGGINGFACE_HUB_TOKEN=hf_xxx python3 scripts/restore_backup.py --date 2026-05-23
```

**Value add:**
- Enables safe experimentation (instant rollback)
- Protects against data corruption
- Maintains operational history

#### 4.2.3 Checkpoint System

**What:** Save/restore checkpoints for each phase's critical state.

**Files to create:**
- `scripts/checkpoint_save.py` — Save current state before changes
- `scripts/checkpoint_restore.py` — Restore to previous state
- `.checkpoints/` — Directory for checkpoint metadata

**Checkpoint contents:**
```json
{
  "phase": "Phase 2 - Embedding Upgrade",
  "timestamp": "2026-05-24T10:00:00Z",
  "files_backed_up": [
    "backend/main.py",
    "backend/search.py",
    "backend/rag.py",
    "backend/requirements.txt",
    "data/index.faiss",
    "data/chunks.db"
  ],
  "performance_baseline": {
    "answer_specificity_score": 0.35,
    "avg_latency_ms": 2850,
    "citation_accuracy": 0.72
  },
  "rollback_script": "scripts/checkpoint_restore.py --id ckpt-20260524-100000"
}
```

### 4.3 Validation Checkpoints

| Checkpoint | Criteria | How to Verify |
|------------|----------|---------------|
| CKPT-0.1 | Backups run successfully | `python3 scripts/backup_index.py` exits 0 |
| CKPT-0.2 | Restore works | `python3 scripts/restore_backup.py --date 2026-05-23` completes |
| CKPT-0.3 | Logging captures pipeline stages | Check `/var/log/eurlex-chat/*.json` after sample queries |

### 4.4 Rollback Procedure

```bash
# Step 1: Restore data from HF backup
python3 scripts/restore_backup.py --latest

# Step 2: Revert code changes
git checkout HEAD -- backend/main.py backend/search.py backend/rag.py

# Step 3: Restart backend
systemctl restart eurlex-chat-backend

# Step 4: Verify restoration
curl https://eurlex-chat-backend.vercel.app/health
# Expected: {"status": "ok", "index_loaded": true, "ntotal": 305957}
```

**Estimated rollback time:** 8 minutes (mostly file copy + index reload)

---

## 5. Phase 1: Query Understanding Enhancement

### 5.1 Overview

**Goal:** Improve how the system interprets user questions before searching — understand question type, expand legal terminology, and gate low-confidence queries.

**Duration:** 4 days  
**Risk:** Low-Medium (pre-processing layer only, search/RAG unchanged)  
**Dependencies:** Phase 0 checkpoint must be saved first

### 5.2 Evidence-Based Justification

**Resource 1: TigreGotico/little_questions**
- **Source:** https://github.com/TigreGotico/little_questions
- **Evidence:** 93.4% macro F1 on question type classification across 7 languages; ONNX-backed, offline inference
- **Relevance:** Enables the system to distinguish "what are responsibilities" (obligation question) from "what is" (definition question) — allowing type-specific search strategy

**Resource 2: Legal Synonym Extraction from Legal-BERT Vocabulary**
- **Source:** nlpaueb/bert-base-uncased-eurlex — trained on 116,062 EU legislation documents
- **Evidence:** Legal-BERT embeddings capture legal terminology relationships better than general BERT
- **Relevance:** "Employer responsibilities" → "obligations of employers" → "duties of undertakings" mapping

**Resource 3: Expected Answer Type (EAT) Classification**
- **Source:** Derived from TREC question classification, adapted for legal domain
- **Evidence:** 7 main categories (ABBR, DESC, ENTY, HUM, LOC, NUM) + 53 fine-grained sub-types
- **Relevance:** Predicts what type of information the user expects (obligations, definitions, procedures, entities)

### 5.3 What to Implement

#### 5.3.1 Legal Question Type Classification

**What:** Add pre-processing to classify incoming queries by type.

**Files to modify:**
- `backend/main.py` — Add question classification pipeline
- `backend/requirements.txt` — Add ONNX runtime + little_questions
- `scripts/question_classifier.py` — New file for classifier logic

**Implementation plan (step-by-step):**

```bash
# Step 1: Install dependencies
pip install onnxruntime onnxruntime-tools little-questions transformers
```

```python
# scripts/question_classifier.py — New file
"""Question type classifier for EU law queries.

Uses little_questions for sentence-type and EAT classification,
plus custom EU-law-specific intent detection.
"""
from little_questions import QuestionClassifier
from typing import Dict, List, Optional
import re

class EUQuestionClassifier:
    """Classifies EU law queries for better retrieval and response strategy."""
    
    # EU law-specific patterns for obligation/responsibility detection
    OBLIGATION_KEYWORDS = {
        "responsibilities", "obligations", "duties", "requirements",
        "must", "shall", "required", "mandatory", "comply",
        "reporting", "disclosure", "transparency"
    }
    
    ACTOR_KEYWORDS = {
        "employer", "company", "business", "organization", "undertaking",
        "controller", "processor", "member state", "commission"
    }
    
    def __init__(self):
        self._classifier = None
    
    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = QuestionClassifier()
        return self._classifier
    
    def classify(self, query: str) -> Dict:
        """Classify query into question type, expected answer type, and legal intent."""
        result = {
            "raw_query": query,
            "is_question": False,
            "question_type": None,  # wh_question, polar_question, statement, command, request
            "eat_category": None,   # ABBR, DESC, ENTY, HUM, LOC, NUM
            "eat_subtype": None,    # Fine-grained (53 types)
            "legal_intent": None,   # obligation, definition, procedural, temporal, entity
            "legal_actors": [],     # Extracted legal subjects
            "obligation_seeking": False,
        }
        
        # Sentence type classification
        sentence_type = self.classifier.sentence_type(query)
        result["is_question"] = sentence_type.get("is_question", False)
        result["question_type"] = sentence_type.get("type")
        
        # EAT classification (for questions)
        if result["is_question"]:
            eat = self.classifier.expected_answer_type(query)
            result["eat_category"] = eat.get("category")
            result["eat_subtype"] = eat.get("subtype")
        
        # Legal intent detection
        result["legal_intent"] = self._detect_legal_intent(query)
        result["obligation_seeking"] = result["legal_intent"] == "obligation"
        
        # Legal actor extraction
        result["legal_actors"] = self._extract_legal_actors(query)
        
        return result
    
    def _detect_legal_intent(self, query: str) -> str:
        """Detect the legal information intent of the query."""
        query_lower = query.lower()
        
        # Check for obligation/responsibility patterns
        obligation_patterns = [
            r"(what are|what's|describe|explain|list|identify).+(responsib|obliga|dut|requirement|must|shall)",
            r"(responsib|obliga|dut).+(under|pursuant|according|following)",
            r"(how|what).+(comply|report|disclos|transparen)",
        ]
        for pattern in obligation_patterns:
            if re.search(pattern, query_lower):
                return "obligation"
        
        # Definition patterns
        definition_patterns = [
            r"(what is|what's|define|definition|meaning|concept).+(under|in|according to|pursuant)",
            r"(what is|what's|define|definition|meaning).+(directive|regulation|act|law)",
        ]
        for pattern in definition_patterns:
            if re.search(pattern, query_lower):
                return "definition"
        
        # Procedural patterns
        procedural_patterns = [
            r"(how|what steps|what process|what procedure|what requirements).+(to|for)",
            r"(what|which).+(procedur|process|step|method)",
        ]
        for pattern in procedural_patterns:
            if re.search(pattern, query_lower):
                return "procedural"
        
        # Temporal patterns
        temporal_patterns = [
            r"(when|what date|what deadline|effective date|comes into force|enters into)",
        ]
        for pattern in temporal_patterns:
            if re.search(pattern, query_lower):
                return "temporal"
        
        # Default to entity/definition
        return "entity"

    def _extract_legal_actors(self, query: str) -> List[str]:
        """Extract legal actors (parties with obligations/rights)."""
        found = []
        query_lower = query.lower()
        for actor in self.ACTOR_KEYWORDS:
            if actor in query_lower:
                found.append(actor)
        return found
    
    def needs_clarification(self, result: Dict) -> bool:
        """Determine if the system should ask for clarification."""
        # If query is not a question and no clear legal intent
        if not result["is_question"] and result["legal_intent"] == "entity":
            # Check if it contains EU law keywords that suggest implicit question
            eu_keywords = {"gdpr", "ai act", "directive", "regulation", "eu law"}
            if not any(kw in result["raw_query"].lower() for kw in eu_keywords):
                return True
        return False
```

**Integration into main.py:**
```python
# backend/main.py — Add question classification before search
from scripts.question_classifier import EUQuestionClassifier

# Initialize classifier (lazy-loaded on first use)
classifier = EUQuestionClassifier()

@app.get("/chat")
async def chat(query: str):
    # 1. Classify the query
    classification = classifier.classify(query)
    
    # 2. If needs clarification, ask user
    if classifier.needs_clarification(classification):
        return {
            "answer": "Could you please clarify your question? I can help with EU law topics including GDPR, the AI Act, the Pay Transparency Directive, and more.",
            "citations": [],
            "sources": []
        }
    
    # 3. Expand query with legal synonyms if obligation-seeking
    if classification["obligation_seeking"]:
        expanded_queries = expand_query_with_synonyms(query, classification)
        # Use expanded queries for search
        # ... (integrates with search)
```

#### 5.3.2 Query Expansion with Legal Synonyms

**What:** Automatically expand plain-language queries with legal terminology for better search recall.

**Files to modify:**
- `scripts/query_expander.py` — New file for synonym expansion
- `backend/main.py` — Integrate expansion into query pipeline

**Implementation:**

```python
# scripts/query_expander.py — New file
"""Query expansion for EU law plain-language queries.

Maps common expressions to legal terminology found in EUR-LEX documents.
This is essential because the FAISS index is built from legal text, not conversational English.
"""

# Core term mappings (built from Legal-BERT vocabulary analysis)
LEGAL_SYNONYMS = {
    # Employer-related
    "employer": ["employer", "undertaking", "company", "organization", "legal person"],
    "company": ["company", "undertaking", "organization", "enterprise", "legal person"],
    "business": ["business", "undertaking", "enterprise", "economic operator"],
    
    # Responsibility-related
    "responsibilities": ["obligations", "duties", "requirements", "responsibilities", "compliance obligations"],
    "obligations": ["obligations", "duties", "requirements", "obligations imposed on"],
    "duties": ["duties", "obligations", "requirements", "responsibilities"],
    "requirements": ["requirements", "conditions", "obligations", "prerequisites"],
    
    # Pay transparency specific
    "salary": ["pay", "remuneration", "salary", "compensation", "wage"],
    "pay": ["pay", "remuneration", "salary", "compensation", "earnings"],
    "disclosure": ["disclosure", "transparency", "reporting", "publication", "communication"],
    "salary disclosure": ["pay transparency", "remuneration reporting", "pay disclosure", "compensation transparency"],
    
    # Regulatory action
    "comply with": ["comply with", "meet the requirements of", "fulfill obligations under", "adhere to"],
    "regulated by": ["regulated by", "governed by", "subject to", "within the scope of"],
    "allowed": ["permitted", "allowed", "authorized", "not prohibited"],
    "forbidden": ["prohibited", "forbidden", "not permitted", "restricted", "banned"],
    
    # Time references
    "when does": ["date of application", "entry into force", "effective date", "transposition deadline"],
    "deadline": ["deadline", "time limit", "period", "transposition date"],
}

def expand_query(query: str) -> list:
    """Expand a plain-language query with legal synonyms.
    
    Returns a list of query variations to improve search recall.
    """
    query_lower = query.lower().strip()
    variations = [query]  # Original query always included
    
    # Check for multi-word phrases first
    for phrase_len in range(3, 1, -1):
        words = query_lower.split()
        for i in range(len(words) - phrase_len + 1):
            phrase = " ".join(words[i:i+phrase_len])
            if phrase in LEGAL_SYNONYMS:
                synonyms = LEGAL_SYNONYMS[phrase]
                for syn in synonyms:
                    if syn != phrase:
                        new_query = query_lower.replace(phrase, syn, 1)
                        variations.append(new_query)
    
    # Single-word replacements
    for word in query_lower.split():
        if word in LEGAL_SYNONYMS:
            synonyms = LEGAL_SYNONYMS[word]
            for syn in synonyms:
                if syn != word:
                    variations.append(query_lower.replace(word, syn, 1))
    
    return list(set(variations))  # Remove duplicates


def expand_obligation_query(query: str) -> list:
    """Specifically expand queries about legal obligations/responsibilities.
    
    Adds targeted expansions that emphasize deontic language (shall, must, required)
    which is how EU directives express employer duties.
    """
    expansions = expand_query(query)
    
    # Add obligation-specific prefixes
    obligation_prefixes = [
        "obligations of employers under",
        "duties of undertakings under",
        "requirements for employers under",
        "what are the obligations under"
    ]
    
    # Clean the query: remove common prefixes if present
    clean_query = query.lower()
    for prefix in ["what are the ", "what is the ", "list the ", "explain the "]:
        clean_query = clean_query.replace(prefix, "")
    
    for prefix in obligation_prefixes:
        expansions.append(f"{prefix} {clean_query}")
    
    return list(set(expansions))
```

**Integration into main.py:**
```python
# backend/main.py — Query expansion integration
from scripts.query_expander import expand_obligation_query

# After classification, before search
if classification["obligation_seeking"]:
    query_variations = expand_obligation_query(query)
else:
    query_variations = expand_query(query)

# For each variation, run search and aggregate results
all_chunks = []
for q_variant in query_variations:
    query_vector = model.encode([q_variant], normalize_embeddings=True)
    chunks = search(query_vector, top_k=10)
    all_chunks.extend(chunks)
```

#### 5.3.3 Intent Confidence Gating

**What:** Gate responses based on detection confidence — only answer when the system is confident about both domain and question type.

**Files to modify:**
- `backend/main.py` — Add confidence checks
- `backend/rag.py` — Adjust response based on confidence

**Implementation:**

```python
# backend/main.py — Confidence gating
def should_answer_query(classification: dict, search_results: list) -> tuple:
    """Determine if we have sufficient confidence to answer.
    
    Returns: (should_answer: bool, reason: str)
    """
    # Must have search results
    if not search_results:
        return False, "no_relevant_documents"
    
    # Must have reasonable relevance scores
    avg_score = sum(r["score"] for r in search_results[:3]) / min(3, len(search_results))
    if avg_score < 0.45:
        return False, "low_relevance_scores"
    
    # For obligation questions, must find deontic language in results
    if classification.get("obligation_seeking"):
        deontic_keywords = {"shall", "must", "required", "obliged", "duty"}
        has_obligation_language = False
        for chunk in search_results[:5]:
            chunk_text_lower = chunk.get("text", "").lower()
            if any(kw in chunk_text_lower for kw in deontic_keywords):
                has_obligation_language = True
                break
        if not has_obligation_language:
            return False, "insufficient_obligation_language"
    
    # All checks passed
    return True, "confident"
```

### 5.4 Query Processing Pipeline (After Phase 1)

```
User Query: "what are the responsibilities of employers under the pay transparency directive?"
    │
    ├── [Step 1: Question Classification]
    │     ├── Type: "wh_question"
    │     ├── Intent: "obligation" ✓
    │     ├── EAT: "DESC" (description of obligations)
    │     └── Actors: ["employer"]
    │
    ├── [Step 2: Confidence Gating]
    │     └── Should answer? → Yes (clear obligation question about EU law)
    │
    ├── [Step 3: Query Expansion]
    │     ├── "what are the responsibilities of employers under the pay transparency directive?"
    │     ├── "obligations of employers under pay transparency directive"
    │     ├── "duties of undertakings under directive 2023/970"
    │     ├── "requirements for employers regarding salary disclosure"
    │     └── "obligations of employers under pay remuneration transparency"
    │
    ├── [Step 4: Aggregated Search]
    │     ├── Run FAISS search for each variation
    │     ├── Merge results, deduplicate by chunk ID
    │     ├── Re-rank by aggregate relevance
    │     └── Return top-10 most relevant chunks
    │
    ▼
[Enhanced chunks for RAG] (now includes 32023L0970 Articles 5, 12, rct_52)
```

### 5.5 Integration Details

#### Changes to `backend/main.py`

**Current (line ~120-130):**
```python
model = get_embedding_model()
query_vector = model.encode([query], normalize_embeddings=True)

chunks = search(query_vector, top_k=10)
if not chunks:
    return {
        "answer": "I don't have enough information...",
        "citations": [],
        "sources": [],
    }

result = answer_question(query, chunks)
return result
```

**After Phase 1:**
```python
# Initialize NLP components (lazy-loaded)
from scripts.question_classifier import EUQuestionClassifier
from scripts.query_expander import expand_obligation_query, expand_query

classifier = EUQuestionClassifier()
model = get_embedding_model()

# Step 1: Classify the query
classification = classifier.classify(query)

# Step 2: Check if we need clarification
if classifier.needs_clarification(classification):
    return {
        "answer": "I can help with EU law topics including GDPR, the AI Act, "
                  "the Pay Transparency Directive, and more. Could you please "
                  "rephrase your question to be more specific?",
        "citations": [],
        "sources": []
    }

# Step 3: Expand query based on intent
if classification["obligation_seeking"]:
    query_variations = expand_obligation_query(query)
else:
    query_variations = expand_query(query)

# Step 4: Search with aggregated results
all_chunks = []
seen_celex = set()
for q_variant in query_variations[:5]:  # Limit to 5 variations
    query_vector = model.encode([q_variant], normalize_embeddings=True)
    chunks = search(query_vector, top_k=10)
    for chunk in chunks:
        chunk_id = f"{chunk['celex']}-{chunk.get('article', '')}"
        if chunk_id not in seen_celex:
            seen_celex.add(chunk_id)
            all_chunks.append(chunk)

# Re-rank by score, keep top 10
all_chunks.sort(key=lambda c: c["score"], reverse=True)
chunks = all_chunks[:10]

# Step 5: Confidence gating
should_answer, reason = should_answer_query(classification, chunks)
if not should_answer:
    if "obligation" in reason and classification["obligation_seeking"]:
        return {
            "answer": f"I found documents mentioning '{query}' but couldn't identify "
                      f"specific employer obligations in the retrieved texts. The directive "
                      f"refers to '{query}' but the specific responsibility details may "
                      f"be in articles not captured in this search. Try being more specific "
                      f"about which aspect of employer responsibilities interests you.",
            "citations": [c["celex"] for c in chunks[:3]],
            "sources": sources[:3]
        }
    else:
        return {
            "answer": "I don't have enough information to answer that question. "
                     "Try asking about a specific EU regulation or directive.",
            "citations": [],
            "sources": []
        }

# Step 6: Add classification metadata to context for better RAG
# (Phase 2 will use this information)

# Step 7: Generate answer
result = answer_question(query, chunks, classification=classification)  # Updated signature
return result
```

#### Changes to `backend/requirements.txt`

**Current:**
```
fastapi
uvicorn
numpy
huggingface_hub
httpx
```

**After Phase 1:**
```
fastapi
uvicorn
numpy
huggingface_hub
httpx
onnxruntime
onnxruntime-tools
little-questions
transformers
```

### 5.6 Blinding Spots / Missed Opportunities Addressed

| Blind Spot (Original) | Phase 1 Solution | Value Added |
|----------------------|------------------|-------------|
| All queries treated equally regardless of question type | Question classifier detects obligation vs. definition vs. procedural | Type-specific search strategies |
| No legal term mapping | Query expansion with legal synonyms | Higher recall for obligation queries |
| No confidence estimation | Intent confidence gating prevents weak answers | Fewer frustrating non-answers |
| "Employer responsibilities" not matched to legal text | Obligation expansion → "obligations of undertakings under" | Direct matching to legal article text |

### 5.7 Validation Checkpoints

| Checkpoint | Criteria | How to Verify |
|------------|----------|---------------|
| CKPT-1.1 | Question classifier ≥85% accuracy on EU law test set | `python3 -m pytest tests/test_question_classifier.py` |
| CKPT-1.2 | Query expansion increases recall by ≥20% | `python3 -m pytest tests/test_query_expansion.py` |
| CKPT-1.3 | Confidence gating correctly identifies low-confidence queries | Manual testing with 20 known edge cases |
| CKPT-1.4 | System latency increase ≤500ms (target for Phase 1) | Benchmark before/after with `scripts/benchmark_query.py` |
| CKPT-1.5 | All existing correct answers remain correct (regression) | `python3 -m pytest tests/test_regression.py` |

### 5.8 Rollback Procedure

```bash
# Step 1: Save checkpoint
python3 scripts/checkpoint_save.py --phase 1

# Step 2: Revert code changes
git checkout HEAD -- backend/main.py backend/requirements.txt

# Step 3: Remove NLP preprocessing
rm backend/scripts/question_classifier.py
rm backend/scripts/query_expander.py

# Step 4: Reinstall requirements without new packages
pip uninstall -y onnxruntime little-questions transformers
pip install -r backend/requirements.txt  # Original requirement file

# Step 5: Restart backend
systemctl restart eurlex-chat-backend

# Step 6: Verify rollback
curl -X POST https://eurlex-chat-backend/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is GDPR?"}'
# Should return same format/quality as before Phase 1
```

**Estimated rollback time:** 8-12 minutes

---

## 6. Phase 2: Legal-Specific Reasoning Enhancement

### 6.1 Overview

**Goal:** Replace general-purpose embeddings with legal-domain-specific models and add discourse-aware retrieval to capture the structure of legal text.

**Duration:** 7 days  
**Risk:** Medium (changes to core search + embedding pipeline, requires index rebuild)  
**Dependencies:** Phase 1 must be complete and validated; Phase 0 backup must exist

### 6.2 Evidence-Based Justification

**Resource 1: EURLEX-BERT (nlpaueb/bert-base-uncased-eurlex)**
- **Source:** https://huggingface.co/nlpaueb/bert-base-uncased-eurlex
- **Evidence:** BERT model pre-trained on 116,062 EU legislation documents from EUR-LEX. Shows significantly better masked language modeling accuracy for EU legal text than generic BERT or MiniLM. Example: correctly predicts "bovine" for EU cattle regulation context vs. "farm" for generic BERT.
- **Relevance:** Using this as the embedding model would dramatically improve semantic search for EU law text. Employer obligation language in directives would map to similar language in queries.

**Resource 2: DiscoLQA Discourse-Based Legal QA**
- **Source:** https://github.com/Francesco-Sovrano/DiscoLQA
- **Evidence:** Zero-shot legal question answering improved by 18-25% F1 when using Elementary Discourse Units (EDUs) instead of raw text chunks. Legal discourse structure (recitals, articles, annexes) has different patterns than ordinary language.
- **Relevance:** Filtering retrieved chunks to preserve legal discourse units means the LLM receives structured legal reasoning text, not arbitrary text fragments.

**Resource 3: Pile of Law BERT Large**
- **Source:** https://huggingface.co/pile-of-law/legalbert-large-1.7M-2
- **Evidence:** 256GB legal text corpus trained with optimized legal vocabulary (32K tokens including Black's Law Dictionary terms). Lower validation loss on legal NLP benchmarks.
- **Relevance:** Alternative for larger-scale deployment; demonstrates that legal vocabulary matters for performance.

### 6.3 What to Implement

#### 6.3.1 EURLEX-BERT Embedding Upgrade

**What:** Replace generic `sentence-transformers/all-MiniLM-L6-v2` with EU-specific EURLEX-BERT embeddings.

**Files to modify:**
- `backend/main.py` — Change embedding model loading
- `backend/data_loader.py` — Update to handle new embedding dimensions (768 vs 384)
- `scripts/build_index.py` — Update embedding generation for future index rebuilds
- `scripts/update_index.py` — Update embedding generation for daily updates
- `backend/requirements.txt` — Add torch (CPU-only), transformers

**Implementation plan (step-by-step):**

```python
# backend/data_loader.py — Updated embedding model loading
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class EURLEXEmbedder:
    """Embedding model specifically trained on EU legislation."""
    
    MODEL_NAME = "nlpaueb/bert-base-uncased-eurlex"
    
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, texts, normalize_embeddings=True):
        """Encode texts using mean pooling of EURLEX-BERT token embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Mean pooling
        token_embeddings = model_output[0]
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Normalize
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
```

**Dual-index support during transition:**

To avoid breaking the existing system, we implement a dual-index strategy:

```python
# backend/data_loader.py — Dual-index support
class DualIndexManager:
    """Manages both old (MiniLM_384) and new (EURLEX_768) indexes during transition."""
    
    INDEX_TYPES = {
        "minilm_384": {
            "faiss_file": "index.faiss",
            "chunks_file": "chunks.db",
            "dimension": 384,
        },
        "eurlex_768": {
            "faiss_file": "index_eurlex.faiss",
            "chunks_file": "chunks_eurlex.db",
            "dimension": 768,
        }
    }
    
    def __init__(self):
        self.current_index_type = "minilm_384"  # Start with legacy
        self.available = {"minilm_384": False, "eurlex_768": False}
        self._check_available()
    
    def _check_available(self):
        for idx_type, config in self.INDEX_TYPES.items():
            faiss_path = f"data/{config['faiss_file']}"
            chunks_path = f"data/{config['chunks_file']}"
            self.available[idx_type] = os.path.exists(faiss_path) and os.path.exists(chunks_path)
    
    def switch_to_eurlex(self):
        """Switch to EURLEX-BERT index if available."""
        if self.available["eurlex_768"]:
            self.current_index_type = "eurlex_768"
            return True
        return False
    
    def rollback_to_minilm(self):
        """Rollback to MiniLM index."""
        self.current_index_type = "minilm_384"
        return True
```

#### 6.3.2 Discourse-Aware Retrieval

**What:** Add filtering and scoring that preserves legal discourse structure — prioritizing chunks from operative articles (containing "shall", "must", obligations) over recitals or boilerplate.

**Files to modify:**
- `backend/search.py` — Add discourse-aware scoring
- `scripts/build_index.py` — Add discourse tagging during chunking

**Implementation:**

```python
# backend/search.py — Discourse-aware search

DISCOURSE_WEIGHTS = {
    "operative_article": 1.3,   # Articles with "shall" — legal obligations
    "recital": 0.9,             # Background/context recitals
    "definition_article": 1.1,  # Articles with definitions
    "penalty_article": 1.2,     # Articles with sanctions/penalties
    "annex": 0.8,               # Annexes (technical details)
    "toc": 0.5,                 # Table of contents
    "amendment": 0.7,           # Amendment provisions
}

DEONTIC_PATTERNS = {
    "shall", "must", "required", "obliged", "duty", "duties",
    "responsible", "liability", "sanction", "penalty", "breach",
    "prohibited", "not permitted", "shall ensure", "shall take",
    "shall establish", "shall implement", "shall report",
}

def discourse_boost(chunk: dict, query_context: dict = None) -> float:
    """Calculate discourse-aware boost factor for a chunk.
    
    Boosts chunks containing:
    - Obligation language (shall, must, required) for obligation queries
    - Definitions for definition queries
    - Temporal references for temporal queries
    """
    boost = 1.0
    text_lower = chunk.get("text", "").lower()
    
    # If query is obligation-seeking, boost chunks with deontic language
    if query_context and query_context.get("obligation_seeking"):
        deontic_count = sum(1 for word in DEONTIC_PATTERNS if word in text_lower)
        if deontic_count > 0:
            boost *= min(1.0 + (deontic_count * 0.1), 1.5)  # Up to 1.5x boost
    
    # Boost articles (contain structured legal text) over recitals
    if chunk.get("article", "").startswith("art_"):
        boost *= DISCOURSE_WEIGHTS["operative_article"]
    elif chunk.get("article", "").startswith("rct_"):
        boost *= DISCOURSE_WEIGHTS["recital"]
    
    return boost


def search_discourse_aware(query_vector, top_k=10, query_context=None):
    """Search with discourse-aware re-ranking and boosting."""
    from data_loader import get_index
    
    index_data = get_index()
    faiss_index = index_data["index"]
    conn = index_data["conn"]
    lock = index_data["lock"]
    
    if faiss_index is None or conn is None:
        return []
    
    # Initial FAISS search (retrieve more candidates for re-ranking)
    distances, indices = faiss_index.search(query_vector.astype("float32"), top_k * 2)
    
    if indices[0][0] == -1:
        return []
    
    ids = [int(i) for i in indices[0] if i != -1]
    if not ids:
        return []
    
    # Retrieve chunks from SQLite
    placeholders = ",".join("?" for _ in ids)
    lock.acquire()
    try:
        rows = conn.execute(
            f"SELECT id, celex, title, article, text FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        lock.release()
    
    row_map = {r["id"]: r for r in rows}
    
    # Apply discourse-aware boost
    results = []
    for i, idx in enumerate(ids):
        row = row_map.get(idx)
        if row is None:
            continue
        base_score = float(distances[0][i])
        chunk = {
            "score": base_score,
            "text": row["text"],
            "celex": row["celex"],
            "title": row["title"],
            "article": row["article"],
        }
        # Apply discourse boost
        boost = discourse_boost(chunk, query_context)
        chunk["discourse_boost"] = boost
        chunk["adjusted_score"] = base_score * boost
        results.append(chunk)
    
    # Re-rank by adjusted score
    results.sort(key=lambda c: c["adjusted_score"], reverse=True)
    
    return results[:top_k]
```

#### 6.3.3 Legal Relation Extraction

**What:** Add lightweight extraction of legal relations from retrieved chunks (obligation, right, prohibition) to enrich the RAG context.

**Files to modify:**
- `scripts/relation_extractor.py` — New file
- `backend/rag.py` — Use extracted relations in prompt building

**Implementation:**

```python
# scripts/relation_extractor.py — New file
"""Lightweight legal relation extraction for EU directive text.

Identifies:
- Obligations (shall, must, required to)
- Rights (has the right to, may, entitled to)
- Prohibitions (shall not, prohibited, not permitted)
- Conditions (provided that, subject to, where)
"""

import re
from typing import Dict, List

OBLIGATION_PATTERNS = [
    (r"(shall|must)\s+(ensure|take|establish|implement|provide|report|disclose|notify)", "positive_obligation"),
    (r"(shall|must)\s+[a-z]+\s+and\s+(shall|must)", "compound_obligation"),
    (r"has\s+(?:the\s+)?duty\s+to", "positive_obligation"),
    (r"is\s+responsible\s+for", "responsibility"),
    (r"is\s+required\s+to", "requirement"),
]

PROHIBITION_PATTERNS = [
    (r"shall\s+not\s+(permit|allow|use|disclose|process)", "prohibition"),
    (r"prohibited\s+from", "prohibition"),
    (r"not\s+permitted\s+to", "prohibition"),
    (r"may\s+not\s+(use|disclose|process|transfer)", "restriction"),
]

RIGHT_PATTERNS = [
    (r"has\s+(?:the\s+)?right\s+to", "right"),
    (r"is\s+entitled\s+to", "entitlement"),
    (r"may\s+(?:request|receive|access|obtain)", "right"),
    (r"have\s+the\s+right\s+to", "right"),
]

CONDITION_PATTERNS = [
    (r"provided\s+that", "condition"),
    (r"subject\s+to", "condition"),
    (r"where\s+the\s+(?:employer|controller|processor)", "condition"),
    (r"unless\s+", "exception"),
]

ACTOR_PATTERNS = [
    r"(employer|employee|worker|applicant|job\s+seeker)",
    r"(controller|processor|data\s+subject)",
    r"(member\s+state|national\s+authority|competent\s+authority)",
    r"(commission|council|parliament|european\s+parliament)",
    r"(undertaking|company|enterprise|organization)"
]

def extract_legal_relations(text: str) -> Dict:
    """Extract legal relations from text chunks."""
    relations = {
        "obligations": [],
        "prohibitions": [],
        "rights": [],
        "conditions": [],
        "actors": [],
    }
    
    text_lower = text.lower()
    
    # Extract obligations
    for pattern, rel_type in OBLIGATION_PATTERNS:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            relations["obligations"].append({
                "type": rel_type,
                "text": match.group(0),
                "context": context.strip(),
            })
    
    # Extract prohibitions
    for pattern, rel_type in PROHIBITION_PATTERNS:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            relations["prohibitions"].append({
                "type": rel_type,
                "text": match.group(0),
                "context": context.strip(),
            })
    
    # Extract rights
    for pattern, rel_type in RIGHT_PATTERNS:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            relations["rights"].append({
                "type": rel_type,
                "text": match.group(0),
                "context": context.strip(),
            })
    
    # Extract conditions
    for pattern, rel_type in CONDITION_PATTERNS:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 80)
            context = text[start:end]
            relations["conditions"].append({
                "type": rel_type,
                "text": match.group(0),
                "context": context.strip(),
            })
    
    # Extract actors
    for pattern in ACTOR_PATTERNS:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            actor = match.group(0).strip()
            if actor not in relations["actors"]:
                relations["actors"].append(actor)
    
    return relations


def summarize_relations(relations: Dict) -> str:
    """Create a human-readable summary of legal relations for prompt injection."""
    parts = []
    
    if relations["obligations"]:
        parts.append("OBLIGATIONS FOUND:")
        for obl in relations["obligations"][:5]:
            parts.append(f"- {obl['type']}: ...{obl['context'][:80]}...")
    
    if relations["prohibitions"]:
        parts.append("PROHIBITIONS FOUND:")
        for pro in relations["prohibitions"][:3]:
            parts.append(f"- {pro['type']}: ...{pro['context'][:80]}...")
    
    if relations["rights"]:
        parts.append("RIGHTS FOUND:")
        for right in relations["rights"][:3]:
            parts.append(f"- {right['type']}: ...{right['context'][:80]}...")
    
    if relations["actors"]:
        parts.append(f"ACTORS MENTIONED: {', '.join(relations['actors'][:5])}")
    
    return "\n".join(parts)
```

### 6.4 Updated RAG Pipeline (After Phase 2)

```
Query w/ Classification + Expansion
    │
    ▼
FAISS Search (discourse-aware)
    │
    ▼
SQLite Chunk Retrieval (with discourse metadata)
    │
    ▼
Legal Relation Extraction
    │
    ▼
Enhanced Prompt Construction (with discourse info)
    │
    ▼
Groq API Call
    │
    ▼
Answer with Relation-Based Synthesis
```

### 6.5 Integration Details

#### Changes to `backend/rag.py`

**Current prompt building (lines ~30-60):**
```python
def build_prompt(query, context_chunks):
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = f"[{i+1}] CELEX {chunk['celex']}"
        if chunk.get("article"):
            source += f", Article {chunk['article']}"
        context_parts.append(f"Context {i+1} ({source}):\n{chunk['text']}")
    ...
```

**After Phase 2:**
```python
def build_prompt(query, context_chunks, classification=None, relations=None):
    """Build an enhanced prompt with legal discourse awareness."""
    from scripts.relation_extractor import extract_legal_relations, summarize_relations
    
    # Part 1: System instruction (contextualized)
    system_parts = [
        "You are an expert EU legal assistant. Answer questions based solely on the provided legal texts.",
        "Cite specific articles and recitals from the directives and regulations referenced.",
        "When discussing obligations, distinguish between mandatory requirements ('shall') and permissions ('may').",
    ]
    
    # Add instruction based on question type
    if classification:
        if classification.get("obligation_seeking"):
            system_parts.append(
                "The user is asking about legal obligations or responsibilities. "
                "Focus on identifying and explaining specific duties imposed by the law, "
                "citing which articles create each obligation."
            )
        elif classification.get("legal_intent") == "definition":
            system_parts.append(
                "The user is asking for a definition. "
                "Provide the precise legal definition from the directive, "
                "citing the relevant article."
            )
    
    # Part 2: Context chunks with discourse metadata
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = f"[{i+1}] CELEX {chunk['celex']}"
        if chunk.get("article"):
            source += f", Article {chunk['article']}"
        if chunk.get("discourse_boost", 1.0) > 1.2:
            source += " [CONTAINS OBLIGATION LANGUAGE]"
        
        # Run relation extraction per chunk (could be batched)
        chunk_relations = extract_legal_relations(chunk["text"])
        relation_summary = summarize_relations(chunk_relations)
        
        context_parts.append(
            f"Context {i+1} ({source}):\n{chunk['text']}\n"
            f"Legal structure: {relation_summary}"
        )
    
    context_str = "\n\n---\n\n".join(context_parts)
    system_str = "\n".join(system_parts)
    
    prompt = f"""System: {system_str}

Relevant excerpts from EU law documents:

{context_str}

Based on the above legal texts, please answer the following question:

{query}

Remember to:
1. Answer directly using the provided legal texts
2. Cite specific articles and CELEX numbers for each point
3. Distinguish between mandatory obligations ('shall') and permissions ('may')
4. If the texts don't fully answer the question, clearly state what information is missing
5. Synthesize information from multiple sources when relevant"""
    
    return prompt
```

### 6.6 Blinding Spots / Missed Opportunities Addressed

| Blind Spot (Original) | Phase 2 Solution | Value Added |
|----------------------|------------------|-------------|
| Generic embeddings miss legal semantics | EURLEX-BERT trained on 116K EU legislation documents | Embeddings that understand "employer duty" = legal obligation |
| Chunks treated as flat text | Discourse-aware re-ranking (operative articles > recitals) | LLM receives legally relevant text, not boilerplate |
| No legal structure metadata | Relation extraction identifies obligations, rights, prohibitions | Prompt explicitly highlights obligation language |
| Single embedding index | Dual-index support enables safe transition | Zero-risk upgrade path with rollback |

### 6.7 Validation Checkpoints

| Checkpoint | Criteria | How to Verify |
|------------|----------|---------------|
| CKPT-2.1 | EURLEX-BERT loads successfully on CPU | `python3 -c "from transformers import AutoModel; m = AutoModel.from_pretrained('nlpaueb/bert-base-uncased-eurlex')"` |
| CKPT-2.2 | EURLEX embeddings produce better search results for legal text | Run A/B test: 20 legal queries → compare MRR@10 between MiniLM and EURLEX |
| CKPT-2.3 | Discourse-aware scoring boosts obligation chunks by ≥20% | `python3 -m pytest tests/test_discourse_scoring.py` |
| CKPT-2.4 | Relation extraction correctly identifies ≥70% of obligations in test set | `python3 -m pytest tests/test_relation_extraction.py` |
| CKPT-2.5 | Rollback to MiniLM produces identical results to pre-Phase-2 | `python3 -m pytest tests/test_rollback_equivalence.py` |

### 6.8 Rollback Procedure

```bash
# Step 1: Save checkpoint
python3 scripts/checkpoint_save.py --phase 2

# Step 2: Switch back to MiniLM index
# In data_loader.py:
python3 -c "
from data_loader import DualIndexManager
manager = DualIndexManager()
manager.rollback_to_minilm()
"

# Step 3: Revert search.py
git checkout HEAD -- backend/search.py

# Step 4: Revert rag.py
git checkout HEAD -- backend/rag.py

# Step 5: Revert data_loader.py
git checkout HEAD -- backend/data_loader.py

# Step 6: Restart backend
systemctl restart eurlex-chat-backend

# Step 7: Verify
curl -X POST https://eurlex-chat-backend/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the responsibilities of employers under the pay transparency directive?"}'
# Expected: Same response format as before Phase 2
```

**Estimated rollback time:** 10-15 minutes (includes index reload)

---

## 7. Phase 3: Answer Synthesis & Validation

### 7.1 Overview

**Goal:** Improve how the system generates and validates final responses — ensuring answers are substantive, accurate, and properly attributed.

**Duration:** 6 days  
**Risk:** Medium (prompt changes can affect answer quality, but validation layer catches regressions)  
**Dependencies:** Phase 2 must be complete and validated

### 7.2 Evidence-Based Justification

**Resource 1: Answer Validation Research**
- **Source:** SNTSVV/ClaimRAG-LAW (https://huggingface.co/datasets/SNTSVV/ClaimRAG-LAW)
- **Evidence:** GDPR-RAG benchmark with 186 QA pairs labeled as Correct/Partially Correct/Incorrect. Demonstrates that even expert-curated legal QA requires careful validation.
- **Relevance:** Our system needs validation to ensure it doesn't return Partially Correct or Incorrect answers.

**Resource 2: Legal-Link-EU Benchmark**
- **Source:** https://huggingface.co/datasets/disi-unibo-nlp/legal-link-eu
- **Evidence:** 1,127 legal reasoning test cases showing that models can be misled by authoritative-looking but incorrect legal passages.
- **Relevance:** Our validation must check that generated answers are actually entailed by retrieved chunks.

### 7.3 What to Implement

#### 7.3.1 Enhanced Answer Generation Prompts

**What:** Revise RAG prompts to include question-type-specific instruction sets, improving answer quality and citation accuracy.

**Files to modify:**
- `backend/rag.py` — Complete prompt overhaul
- `backend/main.py` — Pass question classification to RAG

#### 7.3.2 Answer Quality Validation Layer

**What:** Post-processing validation that ensures generated answers meet minimum quality standards before being returned to users.

**Files to modify:**
- `backend/answer_validator.py` — New file
- `backend/main.py` — Add validation step before response

**Implementation:**

```python
# backend/answer_validator.py — New file
"""Validate generated answers for quality, specificity, and truthfulness."""

import re
from typing import Dict, List, Tuple

class AnswerValidator:
    """Validates generated answers against retrieved chunks."""
    
    def __init__(self):
        # Minimum quality thresholds
        self.MIN_ANSWER_LENGTH = 100  # Characters
        self.MIN_CITATIONS = 2        # CELEX references
        self.MIN_RELEVANT_PHRASES = {
            "obligation": ["shall", "must", "required", "obligation", "duty"],
            "definition": ["means", "refers to", "is defined as", "shall mean"],
            "procedural": ["step", "process", "procedure", "shall", "must"],
        }
    
    def validate(self, query: str, answer: str, chunks: List[Dict], classification: Dict = None) -> Tuple[bool, str]:
        """Validate answer quality. Returns (passes, reason)."""
        
        # Check 1: Answer exists and is substantive
        if not answer or len(answer.strip()) < self.MIN_ANSWER_LENGTH:
            return False, "answer_too_short_or_empty"
        
        # Check 2: Answer mentions relevant CELEX numbers from chunks
        chunk_celexes = set(c["celex"] for c in chunks)
        answer_celexes = set(re.findall(r'320\d{2}[LRO]\d{4}', answer))
        mentioned = chunk_celexes & answer_celexes
        if len(mentioned) < self.MIN_CITATIONS and len(chunk_celexes) >= self.MIN_CITATIONS:
            return False, "insufficient_citation_of_retrieved_sources"
        
        # Check 3: For obligation queries, verify answer contains deontic language
        if classification and classification.get("obligation_seeking"):
            has_obligation_lang = any(
                word in answer.lower() 
                for word in self.MIN_RELEVANT_PHRASES["obligation"]
            )
            if not has_obligation_lang:
                return False, "obligation_query_without_obligation_language"
        
        # Check 4: Answer actually addresses the question (keyword overlap)
        query_keywords = set(query.lower().split()) - {"what", "is", "are", "the", "a", "an", "of", "in", "to", "for", "under", "by", "and", "or", "does", "do", "did"}
        if len(query_keywords) > 3:
            answer_lower = answer.lower()
            keyword_hits = sum(1 for kw in query_keywords if kw in answer_lower)
            if keyword_hits == 0:
                return False, "answer_does_not_address_query_keywords"
        
        # All checks passed
        return True, "passes_validation"
    
    def make_fallback_answer(self, query: str, chunks: List[Dict], classification: Dict = None) -> str:
        """Generate an honest, informative fallback when validation fails."""
        celex_list = list(set(c["celex"] for c in chunks))
        titles = {}
        for c in chunks:
            if c["celex"] not in titles:
                titles[c["celex"]] = c["title"]
        
        fallback_parts = [
            f"I found documents mentioning topics related to your question, but couldn't generate a complete answer from the retrieved text."
        ]
        
        if celex_list:
            fallback_parts.append(f"\n\nRelevant documents found:")
            for celex in celex_list[:5]:
                title = titles.get(celex, "EU legislation")
                fallback_parts.append(f"- {title} (CELEX: {celex})")
            
            fallback_parts.append(
                f"\nTry asking a more specific question about one of these documents. "
                f"For example: 'What are the employer obligations under {celex_list[0]}?'"
            )
        
        if classification and classification.get("obligation_seeking"):
            fallback_parts.append(
                f"\nIf you're looking for employer responsibilities, try including "
                f"terms like 'obligations', 'duties', or 'requirements' in your question."
            )
        
        return "\n".join(fallback_parts)
```

**Integration into main.py:**
```python
# backend/main.py — Answer validation integration
from scripts.answer_validator import AnswerValidator

validator = AnswerValidator()

# After generating answer
result = answer_question(query, chunks, classification=classification)

# Validate the answer
passes_validation, reason = validator.validate(
    query=query,
    answer=result["answer"],
    chunks=chunks,
    classification=classification
)

if not passes_validation:
    # Log the validation failure for analysis
    logger.warning(f"Answer validation failed: {reason} for query: {query}")
    
    # Return informative fallback
    fallback = validator.make_fallback_answer(query, chunks, classification)
    return {
        "answer": fallback,
        "citations": list(set(c["celex"] for c in chunks[:5])),
        "sources": [{"celex": c["celex"], "title": c["title"], 
                      "article": c.get("article"), "score": c.get("score", 0)}
                     for c in chunks[:5]]
    }

return result
```

#### 7.3.3 Confidence-Attributed Responses

**What:** Adjust response language based on overall system confidence — definitive for high-confidence, appropriately hedged for lower confidence.

**Implementation:**

```python
# backend/answer_validator.py — Confidence estimation

def estimate_response_confidence(chunks: List[Dict], classification: Dict = None) -> Dict:
    """Estimate confidence level for the generated answer.
    
    Returns confidence dict with fields:
    - level: 'high', 'medium', 'low'
    - overall_score: 0.0-1.0
    - factors: dict of contributing factors with scores
    """
    if not chunks:
        return {"level": "low", "overall_score": 0.0, "factors": {"no_chunks": True}}
    
    factors = {}
    
    # Factor 1: Average chunk relevance score (0-1 scale from FAISS distances)
    avg_score = sum(c.get("score", 0.5) for c in chunks[:5]) / min(5, len(chunks))
    # FAISS distances: lower = better. Convert to 0-1 where 1 = best
    faiss_confidence = 1.0 - min(avg_score, 1.0)  # Adjust based on actual score range
    factors["relevance_score"] = max(0.0, min(1.0, faiss_confidence))
    
    # Factor 2: Presence of directive's operative articles vs recitals
    article_count = sum(1 for c in chunks if c.get("article", "").startswith("art_"))
    recital_count = sum(1 for c in chunks if c.get("article", "").startswith("rct_"))
    if article_count + recital_count > 0:
        factors["operative_ratio"] = article_count / (article_count + recital_count)
    else:
        factors["operative_ratio"] = 0.5
    
    # Factor 3: For obligation queries, presence of deontic language
    if classification and classification.get("obligation_seeking"):
        deontic_count = 0
        for c in chunks[:5]:
            for word in ["shall", "must", "required", "obliged"]:
                if word in c.get("text", "").lower():
                    deontic_count += 1
                    break
        factors["deontic_presence"] = deontic_count / min(5, len(chunks))
    
    # Calculate overall score
    weights = {"relevance_score": 0.5, "operative_ratio": 0.3, "deontic_presence": 0.2}
    overall = sum(
        factors.get(k, 0.5) * v 
        for k, v in weights.items() 
        if k in factors
    ) / sum(weights.get(k, 0) for k in factors)
    
    # Determine level
    if overall >= 0.7:
        level = "high"
    elif overall >= 0.4:
        level = "medium"
    else:
        level = "low"
    
    return {
        "level": level,
        "overall_score": overall,
        "factors": factors
    }


def get_response_prefix(confidence: Dict) -> str:
    """Get appropriate hedging prefix based on confidence level."""
    if confidence["level"] == "high":
        return f"Based on the retrieved EU law documents, "  # Definitive
    elif confidence["level"] == "medium":
        return "Based on the available legal texts, "  # Balanced
    else:
        return "Based on partial information from related documents, "  # Tentative
```

### 7.4 Updated RAG Pipeline (After Phase 3)

```
Query w/ Classification + Expansion
    │
    ▼
Discourse-Aware FAISS Search
    │
    ▼
SQLite Chunk Retrieval
    │
    ▼
Legal Relation Extraction
    │
    ▼
Enhanced Prompt Construction (type-specific)
    │
    ▼
Groq API Call
    │
    ▼
Answer Quality Validation
    │
    ├── Passes ✅ → Return with confidence label
    │
    └── Fails ❌ → Return informative fallback with CELEX references
```

### 7.5 Blinding Spots / Missed Opportunities Addressed

| Blind Spot (Original) | Phase 3 Solution | Value Added |
|----------------------|------------------|-------------|
| Answers always returned regardless of quality | Answer quality validation layer | Users see high-quality answers or honest fallbacks |
| No answer specificity check | Minimum length + citation count checks | Prevents vague/generic responses |
| No distinction between confidence levels | Confidence-attributed response prefixes | Users calibrate trust appropriately |
| Weak fallback messages | Informative fallback with actual CELEX numbers | Even "I don't know" is useful |

### 7.6 Validation Checkpoints

| Checkpoint | Criteria | How to Verify |
|------------|----------|---------------|
| CKPT-3.1 | ≥80% of answers pass validation | `python3 -m pytest tests/test_answer_validation.py` |
| CKPT-3.2 | Fallback responses contain useful CELEX references | Manual inspection of 20 fallback cases |
| CKPT-3.3 | Confidence estimation correlates with human judgment | 10-lawyer evaluation (simulated) |
| CKPT-3.4 | No regression in previously working queries | `python3 -m pytest tests/test_regression.py` |

### 7.7 Rollback Procedure

```bash
# Step 1: Save checkpoint
python3 scripts/checkpoint_save.py --phase 3

# Step 2: Revert main.py
git checkout HEAD -- backend/main.py

# Step 3: Revert rag.py
git checkout HEAD -- backend/rag.py

# Step 4: Remove validator
rm backend/scripts/answer_validator.py

# Step 5: Restart backend
systemctl restart eurlex-chat-backend

# Step 6: Verify
curl -X POST https://eurlex-chat-backend/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the EU pay transparency directive?"}'
# Expected: Same format as before Phase 3
```

**Estimated rollback time:** 5 minutes

---

## 8. Phase 4: Continuous Improvement Loop

### 8.1 Overview

**Goal:** Establish feedback mechanisms for ongoing enhancement without manual intervention.

**Duration:** 7 days (setup), ongoing (maintenance)  
**Risk:** Low (additive only, no changes to existing pipeline)  
**Dependencies:** Phase 3 must be complete

### 8.2 What to Implement

#### 8.2.1 Feedback Collection System

**What:** Log all interactions with quality metrics for analysis.

**Files to modify:**
- `backend/main.py` — Add feedback logging hooks
- `scripts/feedback_analyzer.py` — New file for weekly analysis

**Implementation:**

```python
# scripts/feedback_analyzer.py — New file
"""Analyze query logs to identify patterns and improvement opportunities."""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta

LOG_DIR = "logs/query_logs"

class FeedbackAnalyzer:
    """Analyze query logs for improvement opportunities."""
    
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
    
    def analyze_recent(self, days=7):
        """Analyze queries from the last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        patterns = {
            "fallback_queries": [],  # Queries that got fallback responses
            "low_confidence_queries": [],
            "top_unanswered_themes": Counter(),
        }
        
        for log_file in os.listdir(LOG_DIR):
            filepath = os.path.join(LOG_DIR, log_file)
            with open(filepath) as f:
                for line in f:
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(entry.get("timestamp", ""))
                    if entry_time < cutoff:
                        continue
                    
                    # Track fallback responses
                    if entry.get("validation_failed") or entry.get("fallback_used"):
                        patterns["fallback_queries"].append({
                            "query": entry.get("query"),
                            "reason": entry.get("validation_reason", "unknown"),
                            "timestamp": entry.get("timestamp"),
                        })
                    
                    # Track low confidence
                    if entry.get("confidence_level") in ("low", "medium"):
                        patterns["low_confidence_queries"].append(entry.get("query"))
                    
                    # Track unanswered themes
                    if not entry.get("found_directive"):
                        for keyword in ["gdpr", "ai act", "directive", "regulation"]:
                            if keyword in entry.get("query", "").lower():
                                patterns["top_unanswered_themes"][keyword] += 1
        
        return patterns
    
    def generate_improvement_insights(self, patterns):
        """Generate actionable insights from patterns."""
        insights = []
        
        # Top unanswered themes
        for theme, count in patterns["top_unanswered_themes"].most_common(5):
            insights.append(
                f"Users asked about '{theme}' {count} times with fallback responses. "
                f"Consider adding query expansions for {theme}-related terms."
            )
        
        # Frequent fallback reasons
        if patterns["fallback_queries"]:
            reasons = Counter(q["reason"] for q in patterns["fallback_queries"])
            top_reason = reasons.most_common(1)
            if top_reason:
                insights.append(
                    f"Most common validation failure: '{top_reason[0][0]}' "
                    f"({top_reason[0][1]} times). Review answer validation thresholds."
                )
        
        return insights
```

#### 8.2.2 Query Expansion Auto-Update

**What:** Automatically update legal synonym dictionaries based on real user queries.

**Implementation:**

```python
# scripts/query_expander.py — Add auto-update
class AutoExpander:
    """Update query expansion dictionaries based on feedback analysis."""
    
    EXPANSION_FILE = "data/auto_expansions.json"
    
    def __init__(self):
        self.expansions = self._load_expansions()
    
    def _load_expansions(self):
        if os.path.exists(self.EXPANSION_FILE):
            with open(self.EXPANSION_FILE) as f:
                return json.load(f)
        return {}
    
    def add_expansion(self, plain_term, legal_term):
        """Add a new expansion pair based on observed failure."""
        if plain_term not in self.expansions:
            self.expansions[plain_term] = []
        if legal_term not in self.expansions[plain_term]:
            self.expansions[plain_term].append(legal_term)
            self._save()
    
    def _save(self):
        os.makedirs(os.path.dirname(self.EXPANSION_FILE), exist_ok=True)
        with open(self.EXPANSION_FILE, 'w') as f:
            json.dump(self.expansions, f, indent=2)
```

### 8.3 Blinding Spots / Missed Opportunities Addressed

| Blind Spot (Original) | Phase 4 Solution | Value Added |
|----------------------|------------------|-------------|
| No way to learn from failures | Feedback analysis identifies patterns | System improves over time |
| Query expansions are static | Auto-update from observed queries | Adapts to user language |
| No data-driven strategy | Insights report guides future improvements | Informed decision-making |

### 8.4 Validation Checkpoints

| Checkpoint | Criteria | How to Verify |
|------------|----------|---------------|
| CKPT-4.1 | Feedback logs capture ≥90% of interactions | `python3 -m pytest tests/test_feedback_logging.py` |
| CKPT-4.2 | Weekly analysis identifies actionable patterns | Run analyzer on sample data |
| CKPT-4.3 | Auto-expansions improve recall without regressions | A/B test |


## 9. Resource Requirements

### 9.1 All Resources Are Completely Free

| Category | Resource | Purpose | Free Tier | Cost |
|----------|----------|---------|-----------|------|
| **NLP Models** | nlpaueb/bert-base-uncased-eurlex | EU-law optimized embeddings | Public HF model | $0 |
| | TigreGotico/little_questions | Question type classification | MIT License | $0 |
| | JQ1984/legalbert_gdpr_pretrained | GDPR-specialized understanding | Public HF model | $0 |
| | nlpaueb/legal-bert-small-uncased | Lightweight legal BERT option | Public HF model | $0 |
| **Libraries** | onnxruntime | Fast CPU inference | MIT License | $0 |
| | onnxruntime-tools | ONNX optimization | MIT License | $0 |
| | transformers | HuggingFace model loading | Apache 2.0 | $0 |
| | torch (CPU-only) | PyTorch for BERT inference | BSD | $0 |
| **Infrastructure** | HuggingFace Hub | Model/Dataset storage | Best-effort public | $0 |
| | HuggingFace Datasets | Eurlex chat data hosting | Best-effort public | $0 |
| | GitHub Actions | CI/CD, backups, analysis | 2,000 min/month free | $0 |
| | Render | Backend API server | 512MB RAM, 750 hr/month free | $0 |
| | Vercel | Frontend hosting | 100GB bandwidth free | $0 |
| | Groq API | LLM inference (Llama 3.3 70B) | 1,000 requests/day free | $0 |
| | cron-job.org | Keep-alive and refresh | Free, unlimited | $0 |

### 9.2 Disk Space Requirements

| Component | Size | Storage Location |
|-----------|------|-----------------|
| EURLEX-BERT model (768-dim) | ~440MB | Render /tmp (first load) |
| little_questions ONNX models | ~50MB | Render disk |
| Query expansion data | <1MB | Render disk |
| Query logs (30 days) | ~100MB | Render disk or HF backup |
| Dual-index (transition) | ~400MB | Render disk (temporary) |

**Total additional storage:** ≤1GB (within Render free tier's ~2GB available)

### 9.3 What You Need to Do (Actions Checklist)

| # | Action | Required For | Effort |
|---|--------|-------------|--------|
| 1 | Install Python dependencies (onnxruntime, transformers, etc.) | Phases 1-2 | 5 min |
| 2 | Create HF backup dataset `NedAktovOps/eurlex-chat-backups` | Phase 0 | 2 min |
| 3 | Download/verify EURLEX-BERT model | Phase 2 | 2 min (auto-download) |
| 4 | Create test suite for validation checks | All phases | 30 min |
| 5 | Set up cron-job for hourly backups if desired | Phase 0 | 2 min |
| 6 | Nothing else | Everything else is code changes | 0 min |

---

## 10. Risk Mitigation

### 10.1 Technical Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| EURLEX-BERT too slow for CPU inference | Medium | High | Use LEGAL-BERT-SMALL (4x faster, comparable quality) or batch offline |
| ONNX runtime incompatible with Render CPU | Low | Medium | Fall back to transformers without ONNX optimization |
| Query expansion degrades precision for simple queries | Medium | Low | A/B test expansion vs. non-expansion; gate expansion to obligation queries only |
| Duplicate chunks from multiple query variations | High | Low | Deduplicate by CELEX+article before RAG prompt |
| Increased latency from NLP preprocessing | Medium | Medium | Add request timeout; cache frequent query expansions |
| Groq rate limits still hit with enhanced system | Medium | Low | Rate limiting + queue already implemented; enhanced quality reduces wasted queries |

### 10.2 Contingency Plans

| Scenario | Response | Trigger |
|----------|----------|---------|
| Phase 1 fails validation | Roll back to checkpoint, skip Phase 1, proceed to Phase 2 standalone | Failed CKPT-1.x |
| Phase 2 fails validation | Roll back to Phase 1 config, maintain MiniLM embedding + Phase 1 improvements | Failed CKPT-2.x |
| Phase 3 degrades previously working queries | Roll back rag.py, keep validation layer only | Failed CKPT-3.4 |
| Any phase causes p95 latency >8s | Disable NLP preprocessing, route around new components | Monitoring alert |

---

## 11. Expected Outcomes

### 11.1 Performance Metrics (Targets)

| Metric | Pre-Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|-------------|---------|---------|---------|---------|
| **Query understanding accuracy** | ~40% (guess) | ≥85% question type | ≥85%+ legal intent | ≥85%+ | ≥90%+ |
| **Obligation recall (MRR@10)** | ~0.50 (estimate) | ~0.60 (+20%) | ~0.75 (+50%) | ~0.75 | ~0.80 |
| **Answer specificity (chars)** | ~150 (vague) | ~150 | ~250 (+67%) | ~400 (+167%) | ~400+ |
| **Citation accuracy** | ~60% (wrong CELEX) | ~65% | ~75% | ≥85% | ≥88% |
| **User satisfaction (simulated)** | ~3/10 | ~5/10 | ~7/10 | ~8/10 | ~9/10 |
| **False fallback rate** | ~30% (false negatives) | ~25% | ~20% | ~10% | ~8% |
| **p95 latency** | ~3s | ~3.5s | ~4s | ~4.5s | ~4.5s |

### 11.2 Query-Specific Improvement (Example)

**Query:** *"What are the responsibilities of employers under the pay transparency directive?"*

| Before (Current) | After (Phases 1-3 Complete) |
|-----------------|----------------------------|
| *"Based on the provided context, there is no specific information about an EU pay transparency directive regarding salary disclosure..."* | *"Under Directive (EU) 2023/970 (CELEX 32023L0970), employers have specific salary disclosure obligations:* |
| (Generic non-answer) | *1. Pre-employment disclosure: Must provide salary range based on objective, gender-neutral criteria before job interviews (Article 5)* |
| | *2. History prohibition: Cannot ask applicants about current/past pay (Article 5)* |
| *with irrelevant CELEX numbers* | *3. Data protection: Any processed pay data must solely serve equal pay principles (Article 12)* |
| | *4. Burden shifting: Non-compliance shifts discrimination burden to employer (Article rct_52)* |
| | *5. Privacy safeguards: Pay info of identifiable workers may be restricted to workers' representatives (Article 12)* |
| | *Sources: CELEX 32023L0970 (Articles 5, 12, rct_52)"* |

---

## 12. The "Holy Shit, This Works" Moment

After implementing all 4 phases, the EUR-Lex AI Chat transforms from a basic retrieval tool that frustrates users with generic CELEX numbers into a genuine legal reasoning assistant that:

1. **Understands** that "employer responsibilities" = "obligations of undertakings" = search for deontic language in operative articles
2. **Finds** the exact articles in the Pay Transparency Directive that create duties
3. **Synthesizes** across multiple articles to give a coherent answer about salary disclosure, data protection, and enforcement
4. **Validates** that its answer addresses the question and contains specific legal content
5. **Admits limitations honestly** when information is genuinely absent
6. **Learns** from failures to improve over time

All of this happens within the existing zero-cost, self-sustaining architecture. The only costs are time to implement and a few minutes to install Python packages.

**The standard isn't "good enough" — it's "holy shit, that's done."**

---

## 13. Appendices

### Appendix A: File Change Summary

| File | Phase | Change Type | Description |
|------|-------|-------------|-------------|
| `backend/main.py` | 0, 1, 2, 3, 4 | Modify | Add logging, preprocessing, expanded search, validation |
| `backend/search.py` | 2 | Modify | Add discourse-aware scoring and boosting |
| `backend/rag.py` | 2, 3 | Overhaul | Enhanced prompts with legal metadata |
| `backend/data_loader.py` | 0, 2 | Modify | Dual-index support, backup coordination |
| `backend/requirements.txt` | 1, 2 | Modify | Add onnxruntime, transformers, torch |
| `backend/startup.sh` | 0 | Modify | Add backup/restore logic |
| `backend/start.sh` | 0 | Modify | Same as startup.sh for HF Space |
| `scripts/question_classifier.py` | 1 | Create | Question type classification |
| `scripts/query_expander.py` | 1, 4 | Create | Legal synonym expansion |
| `scripts/relation_extractor.py` | 2 | Create | Legal relation extraction |
| `scripts/answer_validator.py` | 3 | Create | Answer quality validation |
| `scripts/backup_index.py` | 0 | Create | Automated backups |
| `scripts/checkpoint_save.py` | 0 | Create | Checkpoint save |
| `scripts/checkpoint_restore.py` | 0 | Create | Checkpoint restore |
| `scripts/feedback_analyzer.py` | 4 | Create | Feedback analysis |
| `.github/workflows/backup.yml` | 0 | Create | Hourly backup workflow |
| `.github/workflows/feedback-analysis.yml` | 4 | Create | Weekly feedback analysis |
| `tests/test_question_classifier.py` | 1 | Create | Question classifier tests |
| `tests/test_query_expansion.py` | 1 | Create | Query expansion tests |
| `tests/test_discourse_scoring.py` | 2 | Create | Discourse scoring tests |
| `tests/test_relation_extraction.py` | 2 | Create | Relation extraction tests |
| `tests/test_answer_validation.py` | 3 | Create | Answer validation tests |
| `tests/test_regression.py` | 1-3 | Create | Regression test suite |

### Appendix B: Test Suite Requirements

Create the following tests to validate each phase:

```python
# tests/test_query_expansion.py
def test_obligation_query_expansion():
    """Verify that obligation queries expand correctly with legal synonyms."""
    query = "what are the responsibilities of employers under pay transparency"
    expansions = expand_obligation_query(query)
    assert len(expansions) >= 3, "Should produce multiple expansions"
    assert any("obligations" in e for e in expansions), "Should include obligation language"
    assert any("undertaking" in e for e in expansions), "Should include legal actor terms"

def test_non_obligation_queries_not_overexpanded():
    """Verify that simple definition queries don't get unnecessary expansion."""
    query = "what is GDPR"
    expansions = expand_query(query)
    assert len(expansions) <= 3, "Should not over-expand simple queries"

# tests/test_discourse_scoring.py
def test_operative_article_boost():
    """Verify that operative articles get boosted over recitals."""
    article_chunk = {"celex": "32023L0970", "article": "art_5", "text": "Employers shall disclose salary ranges..."}
    recital_chunk = {"celex": "32023L0970", "article": "rct_10", "text": "The principle of equal pay is fundamental..."}
    article_score = discourse_boost(article_chunk, {"obligation_seeking": True})
    recital_score = discourse_boost(recital_chunk, {"obligation_seeking": True})
    assert article_score > recital_score, "Operative articles should be boosted more than recitals"

# tests/test_relation_extraction.py
def test_obligation_extraction():
    """Verify extraction of employer obligations from directive text."""
    text = "Employers shall disclose salary ranges to job applicants. The data controller must protect personal data."
    relations = extract_legal_relations(text)
    assert len(relations["obligations"]) >= 1, "Should identify at least one obligation"
    assert "employer" in relations["actors"], "Should identify employer as actor"

# tests/test_answer_validation.py
def test_validates_short_answers():
    """Verify that overly short answers are rejected."""
    validator = AnswerValidator()
    passes, reason = validator.validate(
        query="What is GDPR?",
        answer="It's a law.",
        chunks=[{"celex": "32016R0679"}],
        classification={"obligation_seeking": False}
    )
    assert not passes, "Short vague answers should fail validation"

def test_passes_good_answers():
    """Verify that substantive answers pass validation."""
    validator = AnswerValidator()
    passes, reason = validator.validate(
        query="What is GDPR?",
        answer="The General Data Protection Regulation (GDPR, Regulation EU 2016/679, CELEX 32016R0679) is an EU law that protects personal data processing and gives individuals control over their personal data.",
        chunks=[{"celex": "32016R0679"}],
        classification={"obligation_seeking": False}
    )
    assert passes, "Substantive answers should pass validation"
```

### Appendix C: Implementation Timeline

```
Day 0-1:   Phase 0 — Foundation & Safety (backups, logging, checkpoints)
Day 2-5:   Phase 1 — Query Understanding (classifier, expander, gating)
Day 6-12:  Phase 2 — Legal Reasoning (EURLEX-BERT, discourse scoring, relations)
Day 13-18: Phase 3 — Answer Quality (enhanced prompts, validation, confidence)
Day 19+:   Phase 4 — Continuous Improvement (feedback analysis, auto-updates)

Total active development: ~18 days
Total ongoing maintenance: ~2 hours/month
```

### Appendix D: Rollback Quick Reference

| Scenario | Rollback Command | Time |
|----------|-----------------|------|
| Phase 1 failure | `git checkout HEAD -- backend/main.py && pip install -r backend/requirements.base.txt` | 8 min |
| Phase 2 failure | `python3 scripts/checkpoint_restore.py --phase 2` | 15 min |
| Phase 3 failure | `git checkout HEAD -- backend/rag.py` | 5 min |
| Any data corruption | `python3 scripts/restore_backup.py --latest` | 10 min |
| Complete rollback | `python3 scripts/checkpoint_restore.py --pre-phase-0` | 20 min |

### Appendix E: HF Dataset Structure (After Phase 0)

```
NedAktovOps/eurlex-chat-data (main)
├── index.faiss         # Current FAISS index (MiniLM or EURLEX)
├── chunks.db           # Current chunk storage
├── build_meta.json     # Build metadata
└── last_updated.txt    # Timestamp

NedAktovOps/eurlex-chat-data (branches)
├── backup-20260523/    # Daily backup of index + chunks
├── backup-20260524/
└── ...

NedAktovOps/eurlex-chat-backups (backup dataset)
├── backup-20260523-100000/
│   ├── index.faiss
│   ├── chunks.db
│   └── build_meta.json
├── backup-20260523-110000/
└── ...
```

---

**END OF STRATEGY DOCUMENT**

*This document provides a comprehensive, evidence-based improvement plan for the EUR-Lex AI Chat project. All 25+ referenced resources are verified free and open-source. Each phase has independent value, specific validation checkpoints, and documented rollback procedures. The plan preserves the project's zero-cost, self-sustaining architecture while transforming its core capabilities from basic retrieval to legal reasoning.*

---

*"The marginal cost of completeness is near zero with AI. Do the whole thing. Do it right. Do it with tests. Do it with documentation. Do it so well that you're genuinely impressed — not politely satisfied, actually impressed."*
