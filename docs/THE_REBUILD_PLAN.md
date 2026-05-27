# EUR-Lex AI Chat — Complete Rebuild Plan

> **Goal:** Fix the broken build pipeline, cover ALL EU legislation (not just REG+DIR 2004–2023), switch to structure-aware chunking, add hybrid search + reranking, and instrument with proper evaluation.

**Architecture:** 5 incremental phases, each independently verifiable. Phase 0 unblocks everything — the current `build_index.py` produces garbage or 404s for every modern document. Phases 1–4 can be done in any order after Phase 0.

**Tech Stack additions:**
- `eurlxp` (kevin91nl/eurlex) — Cellar RDF traversal, SPARQL, async HTML fetch
- `chunkweaver` (metawake/chunkweaver) — structure-aware chunking with `LEGAL_EU` preset
- `rank_bm25` (0.2.2) — BM25 keyword retrieval
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — neural reranker
- `ragas` (explodinggradients/ragas) — evaluation framework

---

## Phase 0: Unbreak the Build Pipeline

**Goal:** `build_index.py` can actually download real EU legislation instead of RDF metadata or 404.

**Current broken code:** `scripts/build_index.py:173-201` — `fetch_document_xhtml()` tries `{celex}.ENG.xhtml` which returns RDF metadata or 404 for every document. This broke in commit `9497876` when the URL changed from `eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}` (which worked at the time) to `publications.europa.eu/resource/celex/{celex}.ENG.xhtml` (which doesn't).

### Task 0.1: Replace fetch_document_xhtml() with Cellar RDF traversal

**Files:**
- Modify: `scripts/build_index.py:173-201` — replace `fetch_document_xhtml()`
- Add dep: add `eurlxp` to requirements

**Approach A (recommended): Use `eurlxp` library**

Library: https://github.com/kevin91nl/eurlex
Documentation: https://pypi.org/project/eurlxp/
Also useful: https://github.com/seljaseppala/eu_corpus_compiler (SPARQL→CELLAR download pipeline)

```python
from eurlxp import get_html_by_celex_id, parse_html, WAFChallengeError

def fetch_document_xhtml(doc):
    """Fetch XHTML from EUR-Lex via eurlxp (handles Cellar RDF traversal)."""
    celex = doc["celex"]
    try:
        html = get_html_by_celex_id(celex, language="en")
        if not html or len(html) < 500:
            logger.warning(f"  Empty content for {celex}")
            return None
        return html
    except WAFChallengeError:
        logger.warning(f"  WAF challenge for {celex}, trying SPARQL fallback...")
        # Fallback: use SPARQL to look up cellar URL
        try:
            from eurlxp import lookup_cellar_url
            cellar_url = lookup_cellar_url(celex)
            if cellar_url:
                from eurlxp import get_html_by_cellar_url
                return get_html_by_cellar_url(cellar_url)
        except Exception as e2:
            logger.debug(f"  SPARQL fallback also failed for {celex}: {e2}")
        return None
    except Exception as e:
        logger.debug(f"  Failed to fetch {celex}: {e}")
        return None
```

**Approach B (manual, no new deps):** Parse RDF metadata to find `DOC_1` → `owl:sameAs` URL

The verified working URL pattern (tested on 3 documents):
1. Fetch `https://publications.europa.eu/resource/celex/{CELEX}` (no `.xhtml`, no `.ENG`)
2. Parse RDF/XML to find the cellar `manifestation` → `manifestation_has_item` → `DOC_1`
3. Fetch `DOC_1`'s RDF to find its `owl:sameAs` URL (ends in `.doc.html` or `.html`)
4. Fetch that URL for the actual XHTML content

```python
def resolve_cellar_doc_url(celex):
    """Resolve CELEX to actual XHTML URL via CELLAR RDF traversal."""
    # Step 1: Fetch RDF metadata for the CELEX expression
    rdf_url = f"https://publications.europa.eu/resource/celex/{quote(celex, safe='')}"
    r = requests.get(rdf_url, headers={"Accept": "application/rdf+xml"}, timeout=15)
    r.raise_for_status()
    
    # Step 2: Parse RDF to find XHTML manifestation → DOC_1 → owl:sameAs
    import xml.etree.ElementTree as ET
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "cdm": "http://publications.europa.eu/ontology/cdm#",
        "cmr": "http://publications.europa.eu/ontology/cdm/cmr#",
        "owl": "http://www.w3.org/2002/07/owl#",
    }
    root = ET.fromstring(r.text)
    
    # Find XHTML manifestation: expression_manifested_by_manifestation with type=xhtml
    # Then follow manifestation_has_item → DOC_1 → owl:sameAs
    doc_url = None
    for desc in root.findall(".//rdf:Description", ns):
        type_el = desc.find("cdm:manifestation_type", ns)
        if type_el is not None and type_el.text == "xhtml":
            # Get the cellar URL from the manifestation
            has_item = desc.find("cdm:manifestation_has_item", ns)
            if has_item is not None:
                item_url = has_item.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
                # Fetch the item RDF to find owl:sameAs
                item_rdf = requests.get(item_url, timeout=15)
                item_root = ET.fromstring(item_rdf.text)
                same_as = item_root.find(".//owl:sameAs", ns)
                if same_as is not None:
                    doc_url = same_as.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
    
    return doc_url
```

Approach A is simpler and battle-tested. Approach B gives you zero-dependency control.

### Task 0.2: Expand SPARQL query to cover ALL resource types

**Files:**
- Modify: `scripts/build_index.py:119-170` — `query_all_documents()`
- Possibly: add `pyeurlex` for structured SPARQL query building

Library: https://pypi.org/project/pyeurlex/
Also: https://github.com/Kymylyy/cellar-wrapper (CLI + Python + MCP for CELLAR)

Current code limits to REG + DIR only and filters `FROM_DATE=2004-01-01`:

```python
DOC_TYPES = ["REG", "DIR"]  # MISSING: DEC, REC, OPIN, CJEU, etc.
FROM_DATE = "2004-01-01"    # MISSING: older documents, no upper bound
```

**EUR-Lex resource types from the authority taxonomy:**
- `REG` — Regulations
- `DIR` — Directives
- `DEC` — Decisions
- `REC` — Recommendations
- `OPIN` — Opinions
- `RES` — Resolutions
- `CONS` — Consolidated texts
- `INF` — Information
- `INT` — International agreements
- `CJEU` — Court of Justice case law
- `COM` — Commission documents
- And more: https://publications.europa.eu/resource/authority/resource-type

**New SPARQL query approach:**

```python
DOC_TYPES = [
    "REG", "DIR", "DEC", "REC", "OPIN",
    "RES", "CONS", "INF", "INT", "CJEU",
]

FROM_DATE = "1952-01-01"  # Treaty of Paris
# Remove TO_DATE entirely — let SPARQL return everything
```

Or better: use `pyeurlex` or `eurlxp`'s SPARQL builder:
```python
from eurlex import Eurlex  # if using pyeurlex
eur = Eurlex()
q = eur.make_query(resource_type="legislation", order=True)
```

Alternatively use the high-performance miner: https://github.com/do-me/eur-lex
- Custom SPARQL templates with J2
- Multi-threaded parsing
- Joblib caching
- Polars-based text cleaning
- HF dataset upload built-in

### Task 0.3: Benchmark existing coverage

**Before rebuilding**, measure exactly what's missing:

```python
# Count documents per year missing from current index vs SPARQL
current_celexes = set(row[0] for row in conn.execute("SELECT DISTINCT celex FROM chunks"))
sparql_results = query_all_documents()
missing = [d for d in sparql_results if d["celex"] not in current_celexes]
print(f"Current: {len(current_celexes)} CELEXes")
print(f"Available: {len(sparql_results)} CELEXes")
print(f"Missing: {len(missing)} ({100*len(missing)/len(sparql_results):.0f}%)")
```

### Task 0.4: Rebuild index with fixed pipeline

Run the full pipeline:
```bash
python3 scripts/build_index.py
```

Verify with spot checks on specific CELEXes that were previously missing (32024R1689, 32016R0679).

**Expected output:** every document that has an XHTML version on EUR-Lex gets properly chunked with correct `article` IDs (no more `article=NULL`).

---

## Phase 1: Structure-Aware Chunking

**Goal:** Replace the naive BeautifulSoup fallback strategies with `chunkweaver`'s `LEGAL_EU` preset for clean, coherent chunks.

**Tool:** https://github.com/metawake/chunkweaver
- GitHub: https://github.com/metawake/chunkweaver
- Preset: `LEGAL_EU` detects `Article N`, `CHAPTER`, `SECTION`, `(N)` recitals
- Leveled variant: `LEGAL_EU_LEVELED` for hierarchical: CHAPTER > SECTION > Article > recital
- Benchmark: tested on EU AI Act and GDPR with LLM-as-judge
- Zero dependencies (stdlib only)
- Integration support: LangChain, LlamaIndex

### Task 1.1: Replace parse_html_to_chunks() with chunkweaver

**Files:**
- Modify: `scripts/build_index.py:204-285` — replace `parse_html_to_chunks()`

```python
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU_LEVELED
from chunkweaver.detectors import HeadingDetector, TableDetector

def parse_html_to_chunks(html, celex_id, title):
    """Parse EUR-Lex HTML into chunks using structure-aware chunking."""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract title if not provided
    if not title:
        title_el = soup.select_one(".eli-main-title .oj-doc-ti")
        if title_el:
            title = title_el.get_text(strip=True)
        elif soup.title:
            title = soup.title.string or ""
    
    # Get the document text with structural markers
    # First try ELI subdivisions (modern documents)
    container = soup.select_one(".eli-container")
    if container:
        # Extract text while preserving structural markers
        text_parts = []
        current_article = None
        chunks = []
        
        for el in container.descendants:
            if el.name == "div" and el.get("id", "").startswith("art_"):
                current_article = el.get("id")
            # ... extract text with structural annotations
        
        # Use chunkweaver for structure-aware splitting
        chunker = Chunker(
            target_size=1024,
            overlap=2,
            overlap_unit="sentence",
            boundaries=LEGAL_EU_LEVELED,
            detectors=[HeadingDetector(), TableDetector()],
        )
        
        clean_text = container.get_text(separator=" ", strip=True)
        cw_chunks = chunker.chunk_with_metadata(clean_text)
        
        # Map chunkweaver output back to our chunk format
        result = []
        for c in cw_chunks:
            result.append({
                "text": c.text,
                "celex": celex_id,
                "title": title,
                "article": c.boundary_type if c.boundary_type else None,
                "type": _classify_chunk(c),
            })
        return result
    
    # Fallback for non-ELI documents
    text = soup.get_text(separator="\n", strip=True)
    paragraphs = extract_meaningful_paragraphs(text)
    if paragraphs:
        chunker = Chunker(target_size=1024, overlap=2, boundaries=[])
        cw_chunks = chunker.chunk_with_metadata(text)
        return [{"text": c.text, "celex": celex_id, "title": title,
                 "article": None, "type": "paragraph"} for c in cw_chunks]
    
    return []
```

This eliminates all 22K `article=NULL` chunks because chunkweaver respects structural boundaries and won't split mid-article.

### Task 1.2: Rebuild chunks with new chunking

Run with a subset first to verify quality, then full rebuild:
```bash
MAX_CHUNKS=100000 python3 scripts/build_index.py  # test with 100K chunks
```

Verify: sample chunks should not straddle article boundaries, and every chunk should have a valid `article` value.

---

## Phase 2: Hybrid Search + Reranking

**Goal:** Replace pure FAISS dense retrieval with BM25+FAISS hybrid (RRF fusion) + cross-encoder reranker. This fixes queries (like "obligations of employers") that fail because dense embeddings miss keyword overlap.

**Current gap:** `app/search.py:108-168` does pure FAISS cosine similarity with hand-written `discourse_boost()` (1.3× multiplier for articles). No BM25, no neural reranker.

### Task 2.1: Add BM25 index

**Files:**
- Modify: `app/search.py` — add `BM25Store` class
- Create: `app/bm25_store.py` — BM25 index with pickle persistence

**Tool:** `rank_bm25` — https://pypi.org/project/rank-bm25/
Reference implementation: https://github.com/im-anishraj/Hybrid-Search-RAG-Engine

```python
# app/bm25_store.py
import pickle
from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.chunk_ids = []
    
    def build(self, chunks):
        """Build BM25 index from list of chunks."""
        self.documents = [c["text"] for c in chunks]
        self.chunk_ids = [c["id"] for c in chunks]
        tokenized = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
    
    def search(self, query, top_k=10):
        tokenized = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk_id": self.chunk_ids[i], "score": float(scores[i]), "rank": idx + 1}
            for idx, i in enumerate(top_indices)
        ]
    
    def _tokenize(self, text):
        import re
        return [t.lower() for t in re.findall(r'\w{2,}', text)]
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids, "documents": self.documents}, f)
    
    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.documents = data["documents"]
```

### Task 2.2: Implement RRF Fusion

**Files:**
- Create: `app/hybrid_search.py` — RRF fusion + combined search interface

**Tool:** Reciprocal Rank Fusion with k=60 (standard value from benchmarks)
Reference: https://github.com/MudassarHakim/Advance-RAG-ReRanking-FusionRetreival-RRF-HyDe

```python
# app/hybrid_search.py
class HybridSearcher:
    def __init__(self, bm25_store, faiss_index, chunks_db, rrf_k=60, alpha=0.5):
        self.bm25 = bm25_store
        self.faiss = faiss_index
        self.chunks = chunks_db
        self.rrf_k = rrf_k
        self.alpha = alpha  # for weighted fusion
    
    def search_rrf(self, query_vector, query_text, top_k=10):
        """RRF fusion: combine BM25 + FAISS rankings."""
        # Get FAISS results
        faiss_scores, faiss_indices = self.faiss.search(query_vector, top_k * 3)
        
        # Get BM25 results
        bm25_results = self.bm25.search(query_text, top_k=top_k * 3)
        
        # RRF scoring
        rrf = {}
        for rank, idx in enumerate(faiss_indices[0]):
            chunk_id = int(self.chunks.execute(
                "SELECT id FROM chunks ORDER BY id LIMIT 1 OFFSET ?", (int(idx),)
            ).fetchone()[0])  # simplified — actual mapping needed
            rrf[chunk_id] = rrf.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank + 1)
        
        for result in bm25_results:
            chunk_id = result["chunk_id"]
            rrf[chunk_id] = rrf.get(chunk_id, 0) + 1.0 / (self.rrf_k + result["rank"])
        
        # Sort by RRF score
        sorted_ids = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Fetch chunk data
        results = []
        for chunk_id, score in sorted_ids:
            row = self.chunks.execute(
                "SELECT id, celex, title, article, type, text FROM chunks WHERE id = ?",
                (chunk_id,)
            ).fetchone()
            if row:
                results.append({...})
        
        return results
```

For simpler integration, start without FAISS mapping and just store chunk IDs in the FAISS index metadata.

### Task 2.3: Add cross-encoder reranker

**Files:**
- Modify: `app/hybrid_search.py` — add reranking step
- Or: `app/search.py` — replace `discourse_boost()` with neural reranker

**Tool:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
Reference: https://github.com/Eva-iq/E.V.A.-Cascading-Retrieval (proven 62% → 91% accuracy with full cascade)

Alternative (stronger): `BAAI/bge-reranker-v2-minicpm-layer` (multilingual)

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, candidates, top_k=5):
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [{"chunk": c, "rerank_score": float(s)} for c, s in scored[:top_k]]
```

**Pipeline integration:**
```
BM25 (k=30) ──┐
               ├── RRF (k=60) ──→ top-20 ──→ Cross-encoder ──→ top-5 ──→ LLM
FAISS (k=30) ─┘
```

This 3-stage cascade (BM25 + FAISS → RRF → cross-encoder) is the standard proven pattern. Reference implementations:
- https://github.com/im-anishraj/Hybrid-Search-RAG-Engine
- https://github.com/Eva-iq/E.V.A.-Cascading-Retrieval
- https://github.com/ara-5/Genai-rag-agent (includes Corrective RAG + web fallback)

---

## Phase 3: Evaluation Framework

**Goal:** Before making further changes, establish a baseline with RAGAS metrics. Then measure every change against it.

**Tool:** https://github.com/explodinggradients/ragas (now at vibrantlabsai/ragas)

Legal-specific RAG evaluation references:
- https://github.com/235471/rag-evaluation-contracts-ragas — composite score for legal RAG
- https://github.com/CSHaitao/LexRAG — multi-turn legal conversation evaluation
- https://github.com/hoorangyee/LRAGE — legal domain RAG evaluation toolkit
- https://github.com/AnimeshR22/GroundedLegal — legal RAG benchmark with bias testing
- https://github.com/isaacus-dev/legal-rag-bench — legal RAG evaluation benchmark

### Task 3.1: Create ground-truth QA dataset

Create 100+ question-answer pairs from EUR-Lex documents. Each entry:
```json
{
    "question": "What are the obligations of employers under GDPR?",
    "answer": "Employers must... (reference answer)",
    "contexts": [
        "GDPR Article 5: Personal data shall be...",
        "GDPR Article 6: Lawfulness of processing..."
    ],
    "reference": "32016R0679",
    "articles": ["art_5", "art_6"]
}
```

Can auto-generate from structured chunks:
```python
# For each document, create Q&A pairs from article headings
# E.g., "Article 5 — Principles relating to processing of personal data"
# → Question: "What are the principles of GDPR?"
```

### Task 3.2: Set up RAGAS evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Evaluate a run
results = evaluate(
    dataset=test_dataset,  # HuggingFace Dataset
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(results)
```

For legal domain, use composite scoring (from https://github.com/235471/rag-evaluation-contracts-ragas):
```python
COMPOSITE = 0.35 * faithfulness + 0.30 * context_recall + 0.20 * answer_correctness + 0.15 * context_precision
```

### Task 3.3: Add CI quality gate

```yaml
# .github/workflows/evaluate.yml
name: RAG Evaluation
on: [push]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ragas
      - run: python scripts/evaluate.py
      - name: Check quality gate
        run: |
          python -c "
          metrics = json.load(open('eval_results.json'))
          assert metrics['faithfulness'] > 0.7
          assert metrics['context_recall'] > 0.6
          print('Quality gate passed')
          "
```

Reference: https://github.com/ara-5/Genai-rag-agent for full CI evaluation pipeline.

---

## Phase 4: Frontend & Deployment

**Goal:** Wire up real feedback, add model selection, and handle edge cases.

- Wire `ChatWidget.jsx:177` feedback buttons to `/feedback` endpoint
- Add model selection (MiniLM default, EURLEX-BERT optional)
- Handle rate limiting and model failover gracefully
- Update docs to match current architecture

---

## Effort Estimate

| Phase | Tasks | Effort | Impact |
|-------|-------|--------|--------|
| 0: Fix build pipeline | 4 tasks | 1-2 days | 🔴 **Critical** — nothing else works without it |
| 1: Structure chunking | 2 tasks | 1 day | 🟡 High — eliminates 22K NULL articles |
| 2: Hybrid search | 3 tasks | 2 days | 🟡 High — fixes failed queries |
| 3: Evaluation | 3 tasks | 1-2 days | 🟢 Medium — prevents regressions |
| 4: Frontend | 3 tasks | 1 day | 🟢 Medium — polish |

**Total: ~6-8 days for full rebuild.**

---

## GitHub Links Summary

| Category | Tool | URL |
|----------|------|-----|
| **Cellar API** | eurlxp | https://github.com/kevin91nl/eurlex |
| **Cellar API** | pyeurlex | https://pypi.org/project/pyeurlex/ |
| **Cellar API** | cellar-wrapper | https://github.com/Kymylyy/cellar-wrapper |
| **Cellar API** | eu_corpus_compiler | https://github.com/seljaseppala/eu_corpus_compiler |
| **Cellar API** | do-me/eur-lex miner | https://github.com/do-me/eur-lex |
| **Cellar API** | maastrichtlawtech/cellar-extractor | https://github.com/maastrichtlawtech/cellar-extractor |
| **Chunking** | chunkweaver | https://github.com/metawake/chunkweaver |
| **Hybrid Search** | rank_bm25 | https://pypi.org/project/rank-bm25/ |
| **Hybrid Search** | Hybrid-Search-RAG-Engine | https://github.com/im-anishraj/Hybrid-Search-RAG-Engine |
| **Hybrid Search** | RRF example | https://github.com/MudassarHakim/Advance-RAG-ReRanking-FusionRetreival-RRF-HyDe |
| **Hybrid Search** | E.V.A. Cascading Retrieval | https://github.com/Eva-iq/E.V.A.-Cascading-Retrieval |
| **Hybrid Search** | Genai-rag-agent | https://github.com/ara-5/Genai-rag-agent |
| **Hybrid Search** | hybrid-rag-evaluation | https://github.com/anandsuraj/hybrid-rag-system-with-automated-evaluation |
| **Reranking** | cross-encoder models | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Evaluation** | RAGAS | https://github.com/explodinggradients/ragas |
| **Evaluation** | legal RAG eval with RAGAS | https://github.com/235471/rag-evaluation-contracts-ragas |
| **Evaluation** | LexRAG | https://github.com/CSHaitao/LexRAG |
| **Evaluation** | LRAGE | https://github.com/hoorangyee/LRAGE |
| **Evaluation** | GroundedLegal | https://github.com/AnimeshR22/GroundedLegal |
| **Evaluation** | Legal RAG Bench | https://github.com/isaacus-dev/legal-rag-bench |
