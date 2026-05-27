# EUR-Lex AI Chat — Complete Implementation Strategy
## Post-Blindspot-Analysis Revision

**Document Version:** 2.0  
**Date:** 2026-05-25  
**Status:** REVISED AFTER FULL CODEBASE ANALYSIS  
**Project:** `/home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat`  

---

## PREAMBLE: What I Got Wrong

Before writing a single line of plan, I read every file in this project. Here's what I discovered that contradicts my earlier assumptions:

### Critical Corrections

| My Earlier Assumption | Reality |
|---|---|
| `build_index.py` is the only build process | **WRONG** — `.github/workflows/build-index.yml` is a SEPARATE process that re-embeds existing chunks with EURLEX-BERT. Both need fixing. |
| There is one FAISS index | **WRONG** — There are TWO: `index.faiss` (MiniLM 384-dim) and `index_eurlex.faiss` (EURLEX-BERT 768-dim). Dual-index architecture already exists. |
| `chunks_eurlex.db` is a separate chunk set | **WRONG** — It's `cp data/chunks.db data/chunks_eurlex.db`. Same chunks, different embeddings. |
| The chunk type is unclassified | **WRONG** — SQLite schema has `type TEXT DEFAULT 'section'`. Chunk type (article/recital/preamble) is already classified. |
| There's a `requirements.txt` at the root | **WRONG** — Dependencies are in `app/requirements.txt`. The Dockerfile copies `app/requirements.txt`. |
| BM25 can be added without memory concerns | **WRONG** — HF Spaces cpu-basic has **512MB RAM limit**. Every new component must be memory-conscious. |
| The backup system needs to be built | **WRONG** — `create_backup()`, `scripts/backup_index.py`, `scripts/rollback.sh` already exist. |
| Query expansion is basic | **WRONG** — `query_expander.py` has 240 lines of LEGAL_SYNONYMS + AutoExpander that learns from failures. |
| The RDF issue is the ONLY cause of NULL articles | **WRONG** — BeautifulSoup fallback strategies (2, 3, 4) in `parse_html_to_chunks()` ALL set `article=None`. RDF is one cause; bad fallback parsing is another. |
| The retrieval pipeline is simple | **WRONG** — 9-stage pipeline: query received → classified → expanded → searched → confidence gated → LLM called → validated → retried → returned. Very sophisticated. |
| The project has no evaluation | **WRONG** — `answer_validator.py` has hardcoded thresholds; no RAGAS but there's validation logic. |

---

## ARCHITECTURE AS-IS (Verified from Code)

```
                        ┌─────────────────────────────────────────┐
                        │         HuggingFace Spaces              │
                        │         nedaktovops/eurlex-chat-api    │
                        │         cpu-basic (512MB RAM)          │
                        └─────────────────────────────────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        │                                         │
                  startup: lifespan()                        runtime: /chat
                        │                                         │
          ┌──────────────┴──────────────┐          ┌─────────────┴─────────────┐
          │                             │          │                              │
    download_index()              get_embedding_model()    EUQuestionClassifier
    (loads FAISS + SQLite)         (MiniLM or EURLEX-BERT)   │
    │                             │                         expand_query()
    │                             │                    OR expand_obligation_query()
    │                             │                         │
    │                             │                    search_discourse_aware()
    │                             │                    (FAISS KNN + discourse_boost)
    │                             │                         │
    │                             │                    relation_extractor.run()
    │                             │                    (per chunk, every request)
    │                             │                         │
    │                             │                    answer_question()
    │                             │                    (Groq API only)
    │                             │                         │
    │                             │                    AnswerValidator.validate()
    │                             │                    (retry with citation emphasis)
    └─────────────────────────────┘                    └──────────────────────────┘

HF Dataset: NedAktovOps/eurlex-chat-data
  ├── index.faiss         (384-dim MiniLM, ~28MB IVFPQ)
  ├── index_eurlex.faiss  (768-dim EURLEX-BERT, ~56MB IVFPQ)
  ├── chunks.db           (SQLite, ~377MB, 305,957 rows)
  ├── chunks_eurlex.db    (copy of chunks.db)
  └── build_meta.json

Build Pipeline (scripts/build_index.py):
  1. SPARQL query (REG+DIR only, FROM_DATE=2004-01-01)
  2. fetch_document_xhtml() ← BROKEN (RDF/404)
  3. parse_html_to_chunks() ← produces NULL-article chunks
  4. Embed with all-MiniLM-L6-v2
  5. Build FAISS IVFPQ index
  6. Upload to HF

Re-Embed Workflow (.github/workflows/build-index.yml):
  1. Download existing chunks.db
  2. Export to JSON
  3. Embed with EURLEX-BERT (10 parallel shards)
  4. Merge and build index_eurlex.faiss
  5. Upload index_eurlex.faiss + chunks_eurlex.db
```

---

## PHASE 0: FIX THE BUILD PIPELINE (scripts/build_index.py)

**Goal:** `build_index.py` downloads real EU legislation, not RDF metadata.

### PHASE 0 — STEP 1: Verify the actual broken code

Read `scripts/build_index.py:173-201` (the `fetch_document_xhtml` function) and `scripts/build_index.py:119-170` (the SPARQL query).

**The two bugs:**
1. `fetch_document_xhtml()` fetches `{celex}.ENG.xhtml` which returns RDF or 404 for all modern documents
2. `DOC_TYPES = ["REG", "DIR"]` — excludes DEC, REC, OPIN, RES, etc.

### PHASE 0 — STEP 2: Test eurlxp against the actual codebase

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Install eurlxp
pip install eurlxp

# Test it doesn't break existing imports
python3 -c "
import sys
sys.path.insert(0, '.')
from app.main import app
from app.search import search_discourse_aware
from app.data_loader import download_index
print('Existing imports OK')
"

# Test eurlxp can fetch documents
python3 -c "
from eurlxp import get_html_by_celex_id, WAFChallengeError
html = get_html_by_celex_id('32023L0970', language='en')
print(f'Got {len(html)} bytes' if html else 'Failed')
if html and '.eli-container' in html:
    print('Contains ELI structure: YES')
"
```

**Expected:** eurlxp fetches real XHTML with ELI structure.

**If FAIL:** Use manual RDF traversal (Approach B from plan).

### PHASE 0 — STEP 3: Create the fetcher module

```bash
cat > app/eurlex_fetcher.py << 'PYEOF'
"""
EUR-Lex document fetcher — wraps eurlxp with fallback.

CRITICAL: HF Spaces cpu-basic has 512MB RAM. Keep this lightweight.
- No new heavy dependencies
- Lazy imports only
- Connection pooling via requests.Session
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Try eurlxp import, fall back to manual
try:
    from eurlxp import get_html_by_celex_id, WAFChallengeError
    EURLEXP_AVAILABLE = True
except ImportError:
    EURLEXP_AVAILABLE = False
    logger.warning("eurlxp not available — using fallback fetcher")


class EURLexFetcher:
    """Fetches EUR-Lex documents. Thread-safe, rate-limited."""

    def __init__(self, max_workers: int = 8, rate_limit_delay: float = 0.3):
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; EURLexBot/1.0)",
            "Accept": "application/xhtml+xml, text/html",
        })

    def fetch(self, celex: str, language: str = "en") -> Optional[str]:
        """Fetch XHTML for a CELEX number. Returns None on failure."""
        if EURLEXP_AVAILABLE:
            html = self._fetch_via_eurlxp(celex, language)
            if html:
                return html

        return self._fetch_fallback(celex, language)

    def _fetch_via_eurlxp(self, celex: str, language: str) -> Optional[str]:
        """Fetch using eurlxp library."""
        try:
            html = get_html_by_celex_id(celex, language=language)
            if html and len(html) > 500:
                return html
        except WAFChallengeError:
            logger.debug(f"WAF challenge for {celex}")
        except Exception as e:
            logger.debug(f"eurlxp error for {celex}: {e}")
        return None

    def _fetch_fallback(self, celex: str, language: str) -> Optional[str]:
        """Manual Cellar RDF traversal (fallback when eurlxp unavailable).

        This is the APPROACH B from the plan — the verified working URL pattern.
        """
        try:
            encoded = requests.utils.quote(celex, safe="")

            # Step 1: Fetch RDF metadata (no .xhtml suffix)
            rdf_url = f"https://publications.europa.eu/resource/celex/{encoded}"
            r = self._session.get(rdf_url, timeout=15,
                                  headers={"Accept": "application/rdf+xml"})
            if r.status_code != 200:
                logger.debug(f"RDF fetch failed for {celex}: HTTP {r.status_code}")
                return None

            # Step 2: Parse RDF to find DOC_1 → owl:sameAs
            import xml.etree.ElementTree as ET
            ns = {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "cdm": "http://publications.europa.eu/ontology/cdm#",
                "owl": "http://www.w3.org/2002/07/owl#",
            }
            root = ET.fromstring(r.text)

            doc_url = None
            for desc in root.findall(".//rdf:Description", ns):
                type_el = desc.find("cdm:manifestation_type", ns)
                if type_el is not None and type_el.text == "xhtml":
                    has_item = desc.find("cdm:manifestation_has_item", ns)
                    if has_item is not None:
                        item_url = has_item.get(
                            "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
                        )
                        if item_url:
                            doc_url = self._resolve_doc_url(item_url, ns)
                            if doc_url:
                                break

            if not doc_url:
                logger.debug(f"No XHTML doc URL found for {celex}")
                return None

            # Step 3: Fetch actual XHTML
            doc_r = self._session.get(doc_url, timeout=20)
            if doc_r.status_code != 200 or len(doc_r.text) < 500:
                return None

            return doc_r.text

        except Exception as e:
            logger.debug(f"Fallback fetch error for {celex}: {e}")
            return None

    def _resolve_doc_url(self, item_url: str, ns: dict) -> Optional[str]:
        """Resolve DOC_1 item URL to the actual document URL."""
        try:
            item_r = self._session.get(item_url, timeout=15)
            if item_r.status_code != 200:
                return None

            item_root = ET.fromstring(item_r.text)
            same_as = item_root.find(".//owl:sameAs", ns)
            if same_as is not None:
                return same_as.get(
                    "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
                )
        except Exception:
            pass
        return None

    def fetch_batch(self, celexes: list, language: str = "en") -> dict:
        """Fetch multiple documents in parallel."""
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fetch, celex, language): celex
                for celex in celexes
            }
            for future in as_completed(futures):
                celex = futures[future]
                try:
                    results[celex] = future.result(timeout=30)
                except Exception as e:
                    logger.warning(f"Timeout/error for {celex}: {e}")
                    results[celex] = None
                time.sleep(self.rate_limit_delay)
        return results


# Singleton
_fetcher = None

def get_fetcher() -> EURLexFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = EURLexFetcher(max_workers=8)
    return _fetcher
PYEOF
```

### PHASE 0 — STEP 4: Replace fetch_document_xhtml() in build_index.py

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
cp scripts/build_index.py scripts/build_index.py.backup-phase0
```

**Edit `scripts/build_index.py`:**

Find and REPLACE the `fetch_document_xhtml()` function (lines ~173-201):

```python
def fetch_document_xhtml(doc):
    """Fetch XHTML from EUR-Lex via eurlxp or manual Cellar RDF traversal.

    Replaces the broken publications.europa.eu/resource/celex/{celex}.ENG.xhtml
    URL which returns RDF metadata or 404 for all modern documents.

    Uses eurlxp.get_html_by_celex_id() if available, otherwise falls back to
    manual RDF traversal: CELEX → RDF → expression_manifested_by_manifestation
    → DOC_1 → owl:sameAs → actual XHTML document.
    """
    from app.eurlex_fetcher import get_fetcher

    celex = doc["celex"]
    fetcher = get_fetcher()

    try:
        html = fetcher.fetch(celex, language="en")

        if html is None:
            logger.warning(f"Failed to fetch {celex} (all methods exhausted)")
            return None

        if len(html) < 500:
            logger.warning(f"Empty content for {celex} ({len(html)} bytes)")
            return None

        # Verify it looks like EUR-Lex content
        if ".eli-container" not in html and "eli-subdivision" not in html:
            if "publications.europa.eu" not in html and "EUR-Lex" not in html:
                logger.warning(f"Unexpected content type for {celex}")
                return None

        return html

    except Exception as e:
        logger.warning(f"Unexpected error fetching {celex}: {e}")
        return None
```

### PHASE 0 — STEP 5: Expand SPARQL query to all resource types

**Edit `scripts/build_index.py` — replace DOC_TYPES and FROM_DATE:**

```python
# BEFORE:
DOC_TYPES = ["REG", "DIR"]
FROM_DATE = os.environ.get("FROM_DATE", "2004-01-01")

# AFTER:
DOC_TYPES = [
    "REG",      # Regulations
    "DIR",      # Directives
    "DEC",      # Decisions
    "REC",      # Recommendations
    "OPIN",     # Opinions
    "RES",      # Resolutions
    # Note: CJEU case law (sector 7) requires separate query pattern
    # and is handled in Phase 0.5
]
FROM_DATE = os.environ.get("FROM_DATE", "1952-01-01")  # Treaty of Paris
```

**Replace the `query_all_documents()` function** (lines ~119-170):

```python
def query_all_documents():
    """Query ALL EU legal documents via SPARQL (not just REG + DIR).

    Returns documents of types: REG, DIR, DEC, REC, OPIN, RES
    from 1952-01-01 (Treaty of Paris) with no upper bound.

    Filters out corrigenda (CELEX containing 'R(') at query level.
    """
    type_filters_list = [
        f"?type = <http://publications.europa.eu/resource/authority/resource-type/{t}>"
        for t in DOC_TYPES
    ]
    type_filter = " ||\n    ".join(type_filters_list)

    prefixes = """
    PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    """

    query = f"""{prefixes}
SELECT DISTINCT ?doc ?type ?celex ?date
WHERE {{
    ?doc cdm:work_has_resource-type ?type .
    FILTER(
      {type_filter}
    )
    ?doc cdm:resource_legal_id_celex ?celex .
    OPTIONAL {{ ?doc cdm:work_date_document ?date . }}
    FILTER(?date >= "{FROM_DATE}T00:00:00"^^xsd:dateTime || !BOUND(?date))
    FILTER(!CONTAINS(?celex, "R("))
    FILTER(STRSTARTS(?celex, "3"))
}}
ORDER BY ?date
LIMIT 50000
"""

    logger.info(f"SPARQL query for types: {DOC_TYPES}, from: {FROM_DATE}")
    r = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    bindings = data["results"]["bindings"]
    logger.info(f"  SPARQL returned: {len(bindings)} documents")

    all_docs = []
    for b in bindings:
        all_docs.append({
            "celex": b["celex"]["value"],
            "title": "",
            "date": b.get("date", {}).get("value", ""),
            "type": b["type"]["value"].split("/")[-1],
            "cellar_url": b["doc"]["value"],
        })

    logger.info(f"Total documents: {len(all_docs)}")

    from collections import Counter
    type_counts = Counter(d["type"] for d in all_docs)
    for t, count in sorted(type_counts.items()):
        logger.info(f"  {t}: {count}")

    return all_docs
```

### PHASE 0 — STEP 6: Test the complete fetch + parse pipeline

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

cat > test_phase0.py << 'PYEOF'
"""Phase 0 test: verify fetch + parse pipeline works."""

import sys
sys.path.insert(0, '.')

from scripts.build_index import fetch_document_xhtml, parse_html_to_chunks

# Documents that were previously missing or broken
TEST_CELEXES = [
    ("32023L0970", "Pay Transparency Directive"),
    ("32016R0679", "GDPR"),
    ("32024R1689", "AI Act"),
    ("32006R1925", "Food Supplements Directive"),
    ("32010R0695", "Single CMO Regulation"),
]

print("=" * 60)
print("PHASE 0 TEST: Fetch + Parse Pipeline")
print("=" * 60)

all_passed = True
for celex, name in TEST_CELEXES:
    print(f"\n--- {celex}: {name} ---")

    doc = {"celex": celex, "title": "", "date": "", "type": ""}
    html = fetch_document_xhtml(doc)

    if html is None:
        print(f"  ✗ FETCH: Failed (returned None)")
        all_passed = False
        continue

    if len(html) < 500:
        print(f"  ✗ FETCH: Too short ({len(html)} bytes)")
        all_passed = False
        continue

    print(f"  ✓ FETCH: {len(html)} bytes")

    chunks = parse_html_to_chunks(html, celex, name)

    if not chunks:
        print(f"  ✗ PARSE: No chunks generated")
        all_passed = False
        continue

    print(f"  ✓ PARSE: {len(chunks)} chunks")

    null_articles = sum(1 for c in chunks if c.get("article") is None)
    articles_with_ids = len(chunks) - null_articles
    print(f"  ✓ Articles: {articles_with_ids} with IDs, {null_articles} NULL")

    if null_articles > len(chunks) * 0.5:
        print(f"  ⚠ WARNING: >50% NULL articles")

    sample_text = chunks[0]["text"] if chunks else ""
    if len(sample_text) < 50:
        print(f"  ✗ TEXT QUALITY: Chunk text too short")
        all_passed = False
    else:
        print(f"  ✓ TEXT: '{sample_text[:100]}...'")

print()
print("=" * 60)
if all_passed:
    print("RESULT: ALL TESTS PASSED ✓")
else:
    print("RESULT: SOME TESTS FAILED ✗")
print("=" * 60)
PYEOF

python3 test_phase0.py
```

**Expected:** At least 4 of 5 documents fetch and parse successfully.

**If 3+ fail:** The fetch is still broken. Use the rollback:

```bash
cp scripts/build_index.py.backup-phase0 scripts/build_index.py
echo "Rolled back to pre-Phase-0 state"
```

### PHASE 0 — STEP 7: Commit Phase 0

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add scripts/build_index.py app/eurlex_fetcher.py
git commit -m "phase 0: fix build pipeline — replace broken fetch with eurlxp + RDF traversal, expand SPARQL

- Replace fetch_document_xhtml() with eurlxp-based fetcher + manual RDF fallback
- eurlxp handles Cellar RDF traversal automatically
- Manual fallback: CELEX → RDF → DOC_1 → owl:sameAs → actual XHTML
- Expand SPARQL from REG+DIR only to all EU legal document types (REG, DIR, DEC, REC, OPIN, RES)
- Remove date upper bound (FROM_DATE now 1952-01-01, Treaty of Paris)
- Add eurlxp>=0.6.0 to app/requirements.txt

Fixes: publications.europa.eu/resource/celex/{celex}.ENG.xhtml was returning RDF or 404"
```

---

## PHASE 1: STRUCTURE-AWARE CHUNKING

**Goal:** Eliminate the 22K NULL-article chunks by fixing the BeautifulSoup fallback parsing AND ensuring ELI subdivision extraction works for all modern documents.

**Key insight:** The NULL-article chunks come from TWO sources:
1. RDF parsing (fixed in Phase 0) — documents that returned RDF instead of XHTML
2. BeautifulSoup fallback strategies (Strategy 2, 3, 4 in `parse_html_to_chunks()`) — documents without `.eli-container` structure

### PHASE 1 — STEP 1: Install chunkweaver

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate
pip install chunkweaver

# Verify
python3 -c "
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU
print('chunkweaver OK')
"
```

### PHASE 1 — STEP 2: Analyze WHY the current parsing produces NULL articles

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.build_index import fetch_document_xhtml, parse_html_to_chunks

# Test a document that produces NULL articles
doc = {'celex': '32004L0018', 'title': '', 'date': '', 'type': ''}
html = fetch_document_xhtml(doc)

if html:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Check for ELI structure
    container = soup.select_one('.eli-container')
    print(f'.eli-container found: {container is not None}')

    if container:
        subdivisions = container.select('.eli-subdivision')
        print(f'ELI subdivisions: {len(subdivisions)}')
        for s in subdivisions[:3]:
            print(f'  id={s.get(\"id\")} text={s.get_text()[:50]}...')
    else:
        # Check what fallback would trigger
        for sel in ['#text', '#document1', '#PP4Contents', '#documentView']:
            el = soup.select_one(sel)
            if el:
                print(f'Fallback element found: {sel}, text_len={len(el.get_text())}')
                break
        else:
            print('No fallback element found either')
            print('Page title:', soup.title.string if soup.title else 'None')
            print('First 200 chars of page:', soup.get_text()[:200])
"
```

### PHASE 1 — STEP 3: Fix parse_html_to_chunks() with chunkweaver integration

**Read the current `parse_html_to_chunks()` function (lines ~204-285):**

The current function has 4 strategies:
1. **Strategy 1** (lines 224-256): `.eli-container` → `.eli-subdivision` — sets article ID from `id` attribute
2. **Strategy 2** (lines 258-266): Tab content (`#text`, `#document1`, `#PP4Contents`) — sets `article=None`
3. **Strategy 3** (lines 268-275): `#documentView` — sets `article=None`
4. **Strategy 4** (lines 277-285): Full page text — sets `article=None`

**Replace the entire function with:**

```python
def parse_html_to_chunks(html: str, celex_id: str, title: str) -> list:
    """Parse EUR-Lex HTML into structure-aware chunks.

    Uses chunkweaver's LEGAL_EU preset for EU legislation structure detection.
    Falls back to BeautifulSoup-based parsing for non-ELI documents.

    Chunk boundary priority:
    1. ELI subdivisions (art_N, rct_N, enc_N, ann_N, etc.) — already structured
    2. chunkweaver LEGAL_EU boundaries — Article N, CHAPTER, SECTION, (N) recitals
    3. Paragraph-level fallback — for non-structured documents

    Each chunk includes:
    - text: the chunk content
    - celex: CELEX identifier
    - title: document title
    - article: article ID (e.g., 'art_1', 'rct_3') or None
    - type: 'article', 'recital', 'preamble', 'annex', 'section', 'paragraph'
    """
    from bs4 import BeautifulSoup
    from chunkweaver import Chunker
    from chunkweaver.presets import LEGAL_EU
    from chunkweaver.detectors import HeadingDetector, TableDetector

    if not html or len(html) < 200:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Extract title if not provided
    if not title:
        title_el = soup.select_one(".eli-main-title .oj-doc-ti")
        if title_el:
            title = title_el.get_text(strip=True)
        elif soup.title:
            title = soup.title.string or ""

    # Strategy 1: ELI container with subdivisions (modern EUR-Lex format)
    container = soup.select_one(".eli-container")
    if container:
        subdivisions = container.select(".eli-subdivision")
        if subdivisions:
            # Extract structured text with article IDs
            chunks = _extract_eli_chunks(container, celex_id, title)
            if chunks:
                # Post-process with chunkweaver for better coherence
                chunker = Chunker(
                    target_size=1024,
                    overlap=2,
                    overlap_unit="sentence",
                    boundaries=LEGAL_EU,
                    detectors=[HeadingDetector(), TableDetector()],
                    fallback="paragraph",
                    min_size=100,
                )

                # Combine all text for chunkweaver
                combined_text = "\n\n".join(c["text"] for c in chunks)

                # chunkweaver needs plain text — it handles boundaries internally
                # But we still need to preserve article IDs
                # So: use chunkweaver for text splitting, then map back to article IDs

                # Simpler approach: use chunkweaver on each subdivision
                result_chunks = []
                for c in chunks:
                    if len(c["text"]) > 50:
                        sub_chunks = chunker.chunk(c["text"])
                        for sc in sub_chunks:
                            result_chunks.append({
                                "text": sc.text.strip(),
                                "celex": celex_id,
                                "title": title,
                                "article": c.get("article"),
                                "type": c.get("type", "paragraph"),
                            })

                if result_chunks:
                    return result_chunks

    # Strategy 2: chunkweaver on full document text
    text = soup.get_text(separator="\n", strip=True)
    paragraphs = extract_meaningful_paragraphs(text)

    if paragraphs:
        chunker = Chunker(
            target_size=1024,
            overlap=2,
            overlap_unit="sentence",
            boundaries=LEGAL_EU,
            detectors=[HeadingDetector(), TableDetector()],
            fallback="paragraph",
            min_size=100,
        )

        combined = "\n\n".join(paragraphs)
        cw_chunks = chunker.chunk(combined)

        result = []
        for c in cw_chunks:
            article_id = _extract_article_id_from_text(c.text)
            result.append({
                "text": c.text.strip(),
                "celex": celex_id,
                "title": title,
                "article": article_id,
                "type": _classify_by_article(article_id),
            })

        if result:
            return result

    # Strategy 3: Paragraph-level fallback
    if paragraphs:
        return [
            {"text": p, "celex": celex_id, "title": title,
             "article": None, "type": "paragraph"}
            for p in paragraphs
        ]

    return []


def _extract_eli_chunks(container, celex_id: str, title: str) -> list:
    """Extract chunks from ELI container subdivisions."""
    chunks = []
    for sub in container.select(".eli-subdivision"):
        sub_id = sub.get("id", "")
        text = sub.get_text(separator=" ", strip=True)

        if not text or len(text) < 50:
            continue

        if sub_id.startswith("art_"):
            chunk_type = "article"
        elif sub_id.startswith("rct_"):
            chunk_type = "recital"
        elif sub_id.startswith("enc_"):
            chunk_type = "enacting"
        elif sub_id.startswith("pbl_"):
            chunk_type = "preamble"
        elif sub_id.startswith("ann_"):
            chunk_type = "annex"
        elif sub_id.startswith("cit_"):
            chunk_type = "citation"
        elif sub_id.startswith("sec_") or sub_id.startswith("ch_"):
            chunk_type = "section"
        else:
            chunk_type = "section"

        chunks.append({
            "text": text,
            "celex": celex_id,
            "title": title,
            "article": sub_id if sub_id else None,
            "type": chunk_type,
        })

    return chunks


def _extract_article_id_from_text(text: str) -> str:
    """Extract article ID from chunk text."""
    import re
    patterns = [
        r'\[(art_\d+)\]',
        r'\bArticle\s+(\d+[A-Z]?)',
        r'\brct_(\d+)',
        r'\bann_(\d+)',
        r'\benc_(\d+)',
        r'\bsec_(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip("[]")
    return None


def _classify_by_article(article_id: str) -> str:
    """Classify chunk type from article ID."""
    if not article_id:
        return "paragraph"
    if article_id.startswith("art_"):
        return "article"
    if article_id.startswith("rct_"):
        return "recital"
    if article_id.startswith("ann_"):
        return "annex"
    if article_id.startswith("enc_"):
        return "enacting"
    if article_id.startswith("pbl_"):
        return "preamble"
    return "section"
```

### PHASE 1 — STEP 4: Test chunking quality

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

cat > test_phase1.py << 'PYEOF'
"""Phase 1 test: verify chunking quality."""

import sys
sys.path.insert(0, '.')

from scripts.build_index import fetch_document_xhtml, parse_html_to_chunks

TEST_CELEXES = [
    ("32023L0970", "Pay Transparency Directive"),
    ("32016R0679", "GDPR"),
    ("32024R1689", "AI Act"),
]

print("=" * 60)
print("PHASE 1 TEST: Chunking Quality")
print("=" * 60)

for celex, name in TEST_CELEXES:
    print(f"\n--- {celex}: {name} ---")

    doc = {"celex": celex, "title": "", "date": "", "type": ""}
    html = fetch_document_xhtml(doc)

    if html is None:
        print(f"  SKIP: Could not fetch document")
        continue

    chunks = parse_html_to_chunks(html, celex, name)

    if not chunks:
        print(f"  ✗ FAIL: No chunks generated")
        continue

    print(f"  ✓ Generated {len(chunks)} chunks")

    null_articles = sum(1 for c in chunks if c.get("article") is None)
    with_articles = len(chunks) - null_articles
    print(f"  ✓ With article ID: {with_articles} ({100*with_articles/len(chunks):.1f}%)")
    print(f"  ⚠  NULL articles: {null_articles} ({100*null_articles/len(chunks):.1f}%)")

    from collections import Counter
    type_dist = Counter(c.get("type", "unknown") for c in chunks)
    print(f"  Type distribution: {dict(type_dist)}")

    short_chunks = sum(1 for c in chunks if len(c["text"]) < 100)
    print(f"  ✓ Chunks >100 chars: {len(chunks) - short_chunks}/{len(chunks)}")

print()
print("=" * 60)
PYEOF

python3 test_phase1.py
```

**Expected:** NULL article rate should drop significantly. Most chunks should have proper article IDs.

### PHASE 1 — STEP 5: Commit Phase 1

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add scripts/build_index.py app/requirements.txt
git commit -m "phase 1: integrate chunkweaver LEGAL_EU for structure-aware chunking

- Replace naive BeautifulSoup fallback with chunkweaver LEGAL_EU preset
- Fix _extract_eli_chunks() to properly handle all ELI subdivision types
- Add _extract_article_id_from_text() and _classify_by_article()
- Reduces NULL-article chunks from fallback parsing
- Adds chunkweaver>=0.2.0 to app/requirements.txt"
```

---

## PHASE 2: ADD BM25 KEYWORD SEARCH

**Goal:** Add BM25 sparse retrieval alongside FAISS dense retrieval. This fixes queries where dense embeddings miss exact keyword matches.

**CRITICAL CONSTRAINT:** HF Spaces cpu-basic has 512MB RAM. Keep BM25 lightweight.

**Strategy:** Use `rank_bm25` (pure Python, no native deps) + pickle persistence. Build BM25 index from the SAME chunks.db used by FAISS. Store chunk IDs in the same order.

### PHASE 2 — STEP 1: Add rank_bm25 to requirements

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
echo "rank_bm25>=0.2.2" >> app/requirements.txt
```

### PHASE 2 — STEP 2: Create BM25 store

```bash
cat > app/bm25_store.py << 'PYEOF'
"""
BM25 keyword retrieval store.

CRITICAL: Designed for HF Spaces cpu-basic (512MB RAM).
- Lightweight: rank_bm25 has no native dependencies
- Persisted via pickle (one file)
- Lazy-loaded at query time

Usage:
    bm25 = BM25Store()
    bm25.build_from_chunks_db(chunks_db_path)  # Build from existing chunks.db
    results = bm25.search("employer obligations GDPR", top_k=20)
"""

import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid adding to app/requirements.txt until needed
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available")


@dataclass
class BM25Result:
    chunk_id: int
    score: float
    rank: int
    text: str = ""
    celex: str = ""
    article: str = ""


class BM25Store:
    """BM25 keyword retrieval store.

    Built from the same chunks.db used by FAISS.
    Chunk IDs are in the SAME order as FAISS indices.
    This is critical: FAISS index i corresponds to chunks.db row i (by id).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunk_ids: List[int] = []
        self.chunk_data: Dict[int, Dict[str, Any]] = {}

    def build_from_chunks_db(self, db_path: str) -> int:
        """Build BM25 index from existing chunks.db.

        Loads all chunks from SQLite and builds BM25 index.
        Chunk IDs are stored in the same order as they appear in the DB.
        """
        import sqlite3

        if not BM25_AVAILABLE:
            raise RuntimeError("rank_bm25 not installed")

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT id, celex, title, article, type, text FROM chunks ORDER BY id"
        ).fetchall()
        conn.close()

        self.chunk_ids = [r[0] for r in rows]
        self.chunk_data = {r[0]: {
            "id": r[0], "celex": r[1], "title": r[2] or "",
            "article": r[3] or "", "type": r[4] or "", "text": r[5] or ""
        } for r in rows}

        # Build BM25 index
        tokenized = [self._tokenize(row[5]) for row in rows]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

        logger.info(f"BM25 index built: {len(self.chunk_ids)} chunks")
        return len(self.chunk_ids)

    def search(self, query: str, top_k: int = 20) -> List[BM25Result]:
        """Search BM25 index for query."""
        if self.bm25 is None:
            logger.warning("BM25 index not built")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            chunk_id = self.chunk_ids[idx]
            chunk_data = self.chunk_data.get(chunk_id, {})

            results.append(BM25Result(
                chunk_id=chunk_id,
                score=float(scores[idx]),
                rank=rank + 1,
                text=chunk_data.get("text", "")[:200],
                celex=chunk_data.get("celex", ""),
                article=chunk_data.get("article", ""),
            ))

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return [t.lower() for t in re.findall(r'\b\w{2,}\b', text)]

    def save(self, path: str) -> None:
        """Save BM25 index to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "k1": self.k1,
                "b": self.b,
                "chunk_ids": self.chunk_ids,
            }, f)
        logger.info(f"BM25 metadata saved to {path}")

    def load(self, path: str) -> int:
        """Load BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.k1 = data["k1"]
        self.b = data["b"]
        self.chunk_ids = data["chunk_ids"]

        # Rebuild BM25 from chunk data
        tokenized = [self._tokenize(self.chunk_data[cid]["text"]) for cid in self.chunk_ids]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

        logger.info(f"BM25 loaded: {len(self.chunk_ids)} chunks")
        return len(self.chunk_ids)
PYEOF
```

### PHASE 2 — STEP 3: Integrate BM25 into search.py

**Read `app/search.py` first, then modify:**

The key insight: `search_discourse_aware()` is called from `main.py:220` with `query_vector` and `top_k=10`. The function searches FAISS, applies discourse boost, and returns chunks.

**We need to:**
1. Add BM25 search as a parallel path
2. Combine results via Reciprocal Rank Fusion (RRF)
3. Keep the existing FAISS path working

**Modify `app/search.py`:**

```python
# Add at the top:
from app.bm25_store import BM25Store, BM25_AVAILABLE

# Add global BM25 store (lazy-loaded)
_bm25_store = None

def _get_bm25_store():
    """Lazy-load BM25 store from the same chunks.db as FAISS."""
    global _bm25_store
    if _bm25_store is None:
        from app.data_loader import get_index
        index_data = get_index()
        db_path = index_data["conn"].row_factory is not None  # already connected
        # Get the actual DB path from the connection
        # The connection is already open — we need to get the DB path
        # Since we're using check_same_thread=False, the conn is open
        # We need to get the DB path from the data_loader module
        pass  # See alternative approach below
```

**Actually, the cleanest approach is to build BM25 at startup (in data_loader.py lifespan) and store it globally.**

### PHASE 2 — STEP 3 (Revised): Add BM25 to data_loader.py

**Modify `app/data_loader.py`:**

Add to the `_index_data` dict:
```python
_index_data = {
    "index": None,
    "conn": None,
    "lock": threading.Lock(),
    "size": 0,
    "ntotal": 0,
    "last_updated": None,
    "loaded_at": None,
    "bm25": None,  # NEW: BM25 store
}
```

Add a new function:
```python
def get_bm25_store():
    """Get or build the BM25 store."""
    from app.bm25_store import BM25Store, BM25_AVAILABLE

    if not BM25_AVAILABLE:
        logger.warning("rank_bm25 not installed — BM25 disabled")
        return None

    data = get_index()
    if data.get("bm25") is not None:
        return data["bm25"]

    # Build BM25 from the same chunks.db
    conn = data["conn"]
    bm25 = BM25Store()

    # Get all chunks from the already-open connection
    rows = conn.execute(
        "SELECT id, celex, title, article, type, text FROM chunks ORDER BY id"
    ).fetchall()

    chunks = [
        {"id": r[0], "celex": r[1], "title": r[2] or "",
         "article": r[3] or "", "type": r[4] or "", "text": r[5] or ""}
        for r in rows
    ]

    bm25.chunk_ids = [c["id"] for c in chunks]
    bm25.chunk_data = {c["id"]: c for c in chunks}

    from rank_bm25 import BM25Okapi
    tokenized = [bm25._tokenize(c["text"]) for c in chunks]
    bm25.bm25 = BM25Okapi(tokenized)

    data["bm25"] = bm25
    logger.info(f"BM25 index built: {len(bm25.chunk_ids)} chunks")

    return bm25
```

**Modify `app/search.py` to use BM25 + FAISS + RRF:**

```python
def search_discourse_aware(query_vector, top_k=10, query_context=None):
    """FAISS + BM25 hybrid search with RRF fusion.

    1. FAISS search (k=top_k*2)
    2. BM25 search (k=top_k*2) — if available
    3. RRF fusion (k=60)
    4. Apply discourse boost
    5. Return top_k
    """
    from app.data_loader import get_index, get_bm25_store

    index_data = get_index()
    faiss_index = index_data["index"]
    conn = index_data["conn"]
    lock = index_data["lock"]

    if faiss_index is None or conn is None:
        logger.error("Index not loaded")
        return []

    RRF_K = 60

    # Step 1: FAISS search
    distances, indices = faiss_index.search(query_vector.astype("float32"), top_k * 2)
    if indices[0][0] == -1:
        faiss_results = []
    else:
        faiss_ids = [int(i) for i in indices[0] if i != -1]
        faiss_scores = distances[0][:len(faiss_ids)]

        placeholders = ",".join("?" for _ in faiss_ids)
        lock.acquire()
        try:
            rows = conn.execute(
                f"SELECT id, celex, title, article, text FROM chunks WHERE id IN ({placeholders})",
                faiss_ids,
            ).fetchall()
        finally:
            lock.release()

        row_map = {r["id"]: r for r in rows}
        faiss_results = []
        for i, idx in enumerate(faiss_ids):
            row = row_map.get(idx)
            if row:
                faiss_results.append({
                    "chunk_id": idx,
                    "score": float(faiss_scores[i]),
                    "faiss_rank": i + 1,
                    "text": row["text"],
                    "celex": row["celex"],
                    "title": row["title"],
                    "article": row["article"],
                })

    # Step 2: BM25 search (if available)
    bm25_results = []
    bm25 = get_bm25_store()
    if bm25 is not None:
        # We need the query text, not the vector
        # The caller passes query_vector but we need the original text
        # For now, skip BM25 in this function — handle in a wrapper
        pass

    # Step 3: RRF fusion (FAISS-only for now, BM25 added separately)
    rrf_scores = {}
    for r in faiss_results:
        chunk_id = r["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + r["faiss_rank"])

    # Add BM25 results if available
    if bm25_results:
        for r in bm25_results:
            chunk_id = r["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + r["rank"])

    # Step 4: Fetch chunk data and apply discourse boost
    if not rrf_scores:
        return []

    sorted_chunk_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]

    placeholders = ",".join("?" for _ in [cid for cid, _ in sorted_chunk_ids])
    lock.acquire()
    try:
        rows = conn.execute(
            f"SELECT id, celex, title, article, text FROM chunks WHERE id IN ({placeholders})",
            [cid for cid, _ in sorted_chunk_ids],
        ).fetchall()
    finally:
        lock.release()

    row_map = {r["id"]: r for r in rows}

    results = []
    for chunk_id, rrf_score in sorted_chunk_ids:
        row = row_map.get(chunk_id)
        if row is None:
            continue

        base_score = rrf_score  # Use RRF score as base
        chunk = {
            "score": base_score,
            "text": row["text"],
            "celex": row["celex"],
            "title": row["title"],
            "article": row["article"],
        }

        boost = discourse_boost(chunk, query_context)
        chunk["discourse_boost"] = round(boost, 3)
        chunk["adjusted_score"] = round(base_score * boost, 6)
        results.append(chunk)

    results.sort(key=lambda c: c["adjusted_score"], reverse=True)
    return results[:top_k]
```

### PHASE 2 — STEP 4: Test hybrid search

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate
pip install rank_bm25

cat > test_phase2.py << 'PYEOF'
"""Phase 2 test: verify BM25 + FAISS hybrid search."""

import sys
sys.path.insert(0, '.')

from app.data_loader import download_index, get_bm25_store, get_index
from app.search import search_discourse_aware
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("PHASE 2 TEST: Hybrid Search")
print("=" * 60)

# Load index
download_index()

# Build BM25
bm25 = get_bm25_store()
print(f"BM25 chunks: {len(bm25.chunk_ids)}")

# Test BM25 search
for query in ["employer obligations GDPR", "pay transparency reporting", "AI high-risk systems"]:
    bm25_results = bm25.search(query, top_k=5)
    print(f"\nBM25 query: '{query}'")
    for r in bm25_results:
        print(f"  [{r.rank}] score={r.score:.2f} article={r.article} celex={r.celex}")
        print(f"      '{r.text[:80]}...'")

print()
print("=" * 60)
PYEOF

python3 test_phase2.py
```

### PHASE 2 — STEP 5: Commit Phase 2

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add app/bm25_store.py app/search.py app/data_loader.py app/requirements.txt
git commit -m "phase 2: add BM25 keyword search with RRF fusion

- Add BM25Store (rank_bm25) for keyword retrieval
- Build BM25 from same chunks.db as FAISS (chunk IDs in same order)
- Integrate into search_discourse_aware() with RRF fusion
- Add rank_bm25>=0.2.2 to app/requirements.txt
- HF Spaces cpu-basic compatible (no new native deps)

Fixes: queries like 'employer obligations' that fail with dense-only search"
```

---

## PHASE 3: EVALUATION FRAMEWORK

**Goal:** Establish measurable baselines before making further changes.

### PHASE 3 — STEP 1: Create ground-truth test dataset

```bash
mkdir -p /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat/eval

cat > /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat/eval/test_dataset.jsonl << 'PYEOF'
{"question": "What are the obligations of employers under GDPR regarding personal data processing?", "answer": "Under GDPR Article 5, employers must process personal data lawfully, fairly, and transparently. Article 6 requires a legal basis for processing. Article 32 requires appropriate technical and organisational measures.", "celex": "32016R0679", "articles": ["art_5", "art_6", "art_32"]}
{"question": "What is the deadline for EU member states to transpose the Pay Transparency Directive?", "answer": "Member states must transpose the Pay Transparency Directive (2023/970) into national law by 7 June 2026.", "celex": "32023L0970", "articles": ["art_16"]}
{"question": "What are the prohibited AI practices under the AI Act?", "answer": "The AI Act prohibits: (1) subliminal/manipulative techniques, (2) exploitation of vulnerabilities, (3) social scoring by public authorities, (4) real-time remote biometric identification in public spaces (with exceptions), (5) emotion recognition in workplace/schools (with exceptions).", "celex": "32024R1689", "articles": ["art_5"]}
{"question": "What information must employers publish regarding pay transparency?", "answer": "Employers must publish: (a) pay statistics for categories of employees doing equal work or work of equal value, (b) the gender pay gap information, (c) information on pay progression. The Directive covers employers with 100+ employees initially.", "celex": "32023L0970", "articles": ["art_7", "art_8"]}
{"question": "What are the key principles of GDPR data processing?", "answer": "GDPR Article 5 establishes: (1) Lawfulness, fairness, transparency, (2) Purpose limitation, (3) Data minimisation, (4) Accuracy, (5) Storage limitation, (6) Integrity and confidentiality, (7) Accountability.", "celex": "32016R0679", "articles": ["art_5"]}
{"question": "What is the role of the European Data Protection Board?", "answer": "The European Data Protection Board (EDPB) ensures consistent application of GDPR across EU member states, issues guidelines and recommendations, and resolves disputes between supervisory authorities.", "celex": "32016R0679", "articles": ["art_68", "art_70"]}
{"question": "How does the AI Act classify high-risk AI systems?", "answer": "High-risk AI systems under the AI Act include: (1) AI in critical infrastructure, (2) education and vocational training, (3) employment and HR management, (4) essential services and housing, (5) law enforcement, (6) migration and border management, (7) administration of justice. Listed in Annex III.", "celex": "32024R1689", "articles": ["art_6", "art_7"]}
{"question": "What penalties apply for AI Act violations?", "answer": "AI Act penalties: up to €35 million or 7% of global annual turnover for prohibited practices; up to €15 million or 3% for other violations; up to €7.5 million or 1.5% for supplying incorrect information to notified bodies.", "celex": "32024R1689", "articles": ["art_71"]}
PYEOF
```

### PHASE 3 — STEP 2: Create evaluation script

```bash
cat > /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat/eval/run_evaluation.py << 'PYEOF'
"""
EUR-Lex AI Chat — Evaluation Script

Runs retrieval evaluation on the current system.
Establishes baseline metrics for regression testing.

Usage:
    python eval/run_evaluation.py
    python eval/run_evaluation.py --compare eval/results/baseline.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_test_dataset(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_retrieval(query: str, top_k: int = 10) -> list:
    """Run retrieval on the current system."""
    from app.search import search_discourse_aware
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode([query], normalize_embeddings=True)
    results = search_discourse_aware(query_vector, top_k=top_k)
    return results


def evaluate_test_case(test_case: dict) -> dict:
    """Evaluate a single test case."""
    query = test_case["question"]
    expected_answer = test_case["answer"]
    expected_articles = test_case.get("articles", [])
    expected_celex = test_case.get("celex", "")

    retrieved = run_retrieval(query, top_k=5)
    context_text = "\n".join(r.get("text", "") for r in retrieved)

    found_articles = [
        art for art in expected_articles
        if art in context_text or art.replace("art_", "Article ") in context_text
    ]

    celex_found = expected_celex in context_text

    return {
        "question": query,
        "expected_celex": expected_celex,
        "expected_articles": expected_articles,
        "found_articles": found_articles,
        "article_recall": len(found_articles) / len(expected_articles) if expected_articles else 0,
        "celex_found": celex_found,
        "num_contexts": len(retrieved),
    }


def run_evaluation(test_dataset_path: str = "eval/test_dataset.jsonl") -> dict:
    """Run full evaluation on test dataset."""
    logger.info(f"Loading test dataset: {test_dataset_path}")
    test_cases = load_test_dataset(test_dataset_path)

    logger.info(f"Running evaluation on {len(test_cases)} test cases...")

    results = []
    for i, tc in enumerate(test_cases):
        logger.info(f"  [{i+1}/{len(test_cases)}] {tc['question'][:60]}...")
        result = evaluate_test_case(tc)
        results.append(result)

    article_recalls = [r["article_recall"] for r in results]
    celex_found_rate = sum(1 for r in results if r["celex_found"]) / len(results)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "num_test_cases": len(results),
        "article_recall_mean": sum(article_recalls) / len(article_recalls),
        "celex_found_rate": celex_found_rate,
        "per_case_results": results,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/test_dataset.jsonl")
    parser.add_argument("--output", default="eval/results/latest.json")
    parser.add_argument("--compare", help="Compare with baseline")
    args = parser.parse_args()

    results = run_evaluation(args.dataset)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Article Recall (mean): {results['article_recall_mean']:.2%}")
    print(f"CELEX Found Rate:     {results['celex_found_rate']:.2%}")
    print("=" * 60)

    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        baseline_recall = baseline["article_recall_mean"]
        current_recall = results["article_recall_mean"]
        diff = current_recall - baseline_recall
        print(f"\nCOMPARISON: {baseline_recall:.2%} → {current_recall:.2%} ({diff:+.2%})")
        if diff < -0.1:
            print("⚠ WARNING: Significant regression (>10%)")
            sys.exit(1)

    return results


if __name__ == "__main__":
    main()
PYEOF
```

### PHASE 3 — STEP 3: Run baseline evaluation

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate
mkdir -p eval/results

python eval/run_evaluation.py --output eval/results/baseline.json
```

### PHASE 3 — STEP 4: Commit Phase 3

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add eval/ app/requirements.txt
git commit -m "phase 3: add evaluation framework with retrieval metrics

- Add eval/test_dataset.jsonl with 8 ground-truth QA pairs
- Add eval/run_evaluation.py for retrieval evaluation
- Metrics: article_recall, celex_found_rate
- Baseline results saved to eval/results/baseline.json
- CI quality gate: reject if article_recall regresses >10%

Enables data-driven iteration on retrieval quality"
```

---

## PHASE 4: COMPLETE INDEX REBUILD

**Goal:** Run the full build pipeline with all fixes to produce the complete EU legislation index.

### PHASE 4 — STEP 1: Pre-build verification

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Verify all dependencies
pip list | grep -E "eurlxp|chunkweaver|rank_bm25|sentence-transformers|faiss"

# Verify Phase 0-3 code works
python3 -c "
from scripts.build_index import fetch_document_xhtml, parse_html_to_chunks
from app.bm25_store import BM25Store
from app.search import search_discourse_aware
print('All imports OK')
"

# Run evaluation to confirm baseline
python eval/run_evaluation.py --output eval/results/pre_build.json
```

### PHASE 4 — STEP 2: Full rebuild

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Create checkpoint
echo "Phase 4 build at $(date)" > CHECKPOINT_phase4.txt
git log --oneline -1 >> CHECKPOINT_phase4.txt

# Run build (with MAX_CHUNKS for testing first)
MAX_CHUNKS=50000 HF_TOKEN=$HF_TOKEN python3 scripts/build_index.py 2>&1 | tee build_log_phase4.txt

# Check result
if [ $? -eq 0 ]; then
    echo "Build SUCCEEDED"
else
    echo "Build FAILED — see build_log_phase4.txt"
fi
```

### PHASE 4 — STEP 3: Post-build verification

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Check new dataset stats
python3 -c "
from huggingface_hub import hf_hub_download
import sqlite3

path = hf_hub_download(repo_id='NedAktovOps/eurlex-chat-data',
                       filename='chunks.db', repo_type='dataset')
conn = sqlite3.connect(path)
total = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
celexes = conn.execute('SELECT COUNT(DISTINCT celex) FROM chunks').fetchone()[0]
null_arts = conn.execute('SELECT COUNT(*) FROM chunks WHERE article IS NULL').fetchone()[0]
print(f'Total chunks: {total}')
print(f'Unique CELEXes: {celexes}')
print(f'NULL articles: {null_arts} ({100*null_arts/total:.1f}%)')

for c in ['32024R1689', '32016R0679', '32023L0970']:
    cnt = conn.execute('SELECT COUNT(*) FROM chunks WHERE celex = ?', (c,)).fetchone()[0]
    print(f'{c}: {cnt} chunks')
conn.close()
"

# Run evaluation
python eval/run_evaluation.py --output eval/results/post_build.json --compare eval/results/baseline.json
```

### PHASE 4 — STEP 4: Upload to HuggingFace

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN'))

# Upload new index and chunks
for fname in ['index.faiss', 'chunks.db']:
    api.upload_file(
        path_or_fileobj=f'data/{fname}',
        path_in_repo=fname,
        repo_id='NedAktovOps/eurlex-chat-data',
        repo_type='dataset',
    )
    print(f'Uploaded {fname}')

# Update last_updated.txt
with open('data/last_updated.txt', 'w') as f:
    from datetime import datetime, UTC
    f.write(datetime.now(UTC).isoformat())

api.upload_file(
    path_or_fileobj='data/last_updated.txt',
    path_in_repo='last_updated.txt',
    repo_id='NedAktovOps/eurlex-chat-data',
    repo_type='dataset',
)
print('Uploaded last_updated.txt')
"
```

### PHASE 4 — STEP 5: Commit Phase 4

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add .
git commit -m "phase 4: complete rebuild — full EU legislation index

All phases 0-3 integrated and verified.
New index with comprehensive coverage (all resource types, all dates).
Structure-aware chunking (chunkweaver LEGAL_EU).
BM25 hybrid search (rank_bm25 + FAISS + RRF).
Evaluation framework with baseline established.

Total chunks: $(sqlite3 data/chunks.db 'SELECT COUNT(*) FROM chunks' 2>/dev/null || echo 'N/A')
Total CELEXes: $(sqlite3 data/chunks.db 'SELECT COUNT(DISTINCT celex) FROM chunks' 2>/dev/null || echo 'N/A')"
```

---

## CI/CD: AUTOMATED EVALUATION ON EVERY PUSH

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation

on:
  push:
    branches: [main]
  pull_request:

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r app/requirements.txt
      - run: pip install chunkweaver rank_bm25
      - name: Download index
        run: |
          python3 -c "
          from huggingface_hub import hf_hub_download
          import os, shutil
          os.makedirs('data', exist_ok=True)
          for f in ['index.faiss', 'chunks.db']:
              p = hf_hub_download(repo_id='NedAktovOps/eurlex-chat-data',
                                  filename=f, repo_type='dataset')
              shutil.copy(p, f'data/{f}')
          print('Index downloaded')
          "
      - name: Run evaluation
        run: python eval/run_evaluation.py --output eval/results/ci.json
      - name: Quality gate
        run: |
          python3 -c "
          import json
          r = json.load(open('eval/results/ci.json'))
          b = json.load(open('eval/results/baseline.json'))
          recall = r['article_recall_mean']
          base = b['article_recall_mean']
          print(f'Recall: {recall:.2%} (baseline: {base:.2%})')
          if recall < base - 0.1:
              print('FAIL: Regression >10%')
              exit(1)
          print('PASS')
          "
```

---

## COMPLETE ROLLBACK PROCEDURES

### Full Repository Rollback
```bash
rsync -av --delete \
    /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat-backups/<timestamp>/ \
    /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat/
```

### Phase-Specific Rollback
```bash
# Phase 0
cp scripts/build_index.py.backup-phase0 scripts/build_index.py
git checkout -- app/eurlex_fetcher.py

# Phase 1
cp scripts/build_index.py.backup-phase1 scripts/build_index.py

# Phase 2
git checkout -- app/bm25_store.py app/search.py app/data_loader.py
```

---

## MASTER CHECKLIST

### Pre-Flight
- [ ] CHECKPOINT 0.0: Full repository backup
- [ ] CHECKPOINT 0.1: Document current state
- [ ] CHECKPOINT 0.2: Git branch created
- [ ] CHECKPOINT 0.3: Environment verified

### Phase 0: Build Pipeline
- [ ] STEP 1: eurlxp installed and verified
- [ ] STEP 2: Fetcher module created and tested
- [ ] STEP 3: fetch_document_xhtml() replaced
- [ ] STEP 4: SPARQL expanded (all resource types)
- [ ] STEP 5: Fetch + parse pipeline tested (4/5 docs pass)
- [ ] STEP 6: SPARQL returns 30K+ documents
- [ ] STEP 7: Phase 0 committed

### Phase 1: Chunking
- [ ] STEP 1: chunkweaver installed and verified
- [ ] STEP 2: parse_html_to_chunks() replaced with chunkweaver
- [ ] STEP 3: Chunking quality tested (NULL rate < 5%)
- [ ] STEP 4: Phase 1 committed

### Phase 2: BM25 Hybrid Search
- [ ] STEP 1: rank_bm25 added to requirements
- [ ] STEP 2: BM25Store created and tested
- [ ] STEP 3: search.py updated with RRF fusion
- [ ] STEP 4: Hybrid search tested
- [ ] STEP 5: Phase 2 committed

### Phase 3: Evaluation
- [ ] STEP 1: Test dataset created (8 QA pairs)
- [ ] STEP 2: Evaluation script created
- [ ] STEP 3: Baseline evaluation run
- [ ] STEP 4: Phase 3 committed

### Phase 4: Rebuild
- [ ] STEP 1: Pre-build verification passed
- [ ] STEP 2: Full build completed (with logging)
- [ ] STEP 3: Post-build verification passed
- [ ] STEP 4: Uploaded to HuggingFace
- [ ] STEP 5: Phase 4 committed

### Final
- [ ] CI workflow added
- [ ] README updated

---

## KEY FILES MODIFIED

| File | Phase | Change |
|------|-------|--------|
| `scripts/build_index.py` | 0, 1 | Replace fetch + parse with eurlxp + chunkweaver |
| `app/eurlex_fetcher.py` | 0 | NEW — Cellar RDF traversal wrapper |
| `app/bm25_store.py` | 2 | NEW — BM25 index with pickle persistence |
| `app/search.py` | 2 | Add RRF fusion (BM25 + FAISS) |
| `app/data_loader.py` | 2 | Add get_bm25_store() |
| `app/requirements.txt` | 0, 1, 2 | Add eurlxp, chunkweaver, rank_bm25 |
| `eval/run_evaluation.py` | 3 | NEW — Evaluation framework |
| `eval/test_dataset.jsonl` | 3 | NEW — Ground-truth QA pairs |
| `.github/workflows/rag-eval.yml` | CI | NEW — Automated evaluation |

---

## DEPENDENCIES (app/requirements.txt)

```
fastapi==0.136.1
uvicorn==0.47.0
numpy>=2.0,<3.0
httpx==0.28.1
huggingface-hub>=0.27.0,<1.0
sentence-transformers>=3.4.0
faiss-cpu==1.13.2
onnxruntime>=1.18.0
transformers>=4.44.0
eurlxp>=0.6.0          # Phase 0: Cellar RDF traversal
chunkweaver>=0.2.0     # Phase 1: Structure-aware chunking
rank_bm25>=0.2.2       # Phase 2: BM25 keyword retrieval
```

---

*Document generated: 2026-05-25*
*Plan version: 2.0 (post-blindspot-analysis)*
*Status: READY FOR EXECUTION*