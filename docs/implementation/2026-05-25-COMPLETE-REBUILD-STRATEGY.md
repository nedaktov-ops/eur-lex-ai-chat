# EUR-Lex AI Chat — Complete Technical Rebuild Strategy

**Document Version:** 1.0  
**Date:** 2026-05-25  
**Author:** Engineering Team  
**Status:** APPROVED FOR EXECUTION  
**Total Estimated Effort:** 6-8 working days  

---

## EXECUTIVE SUMMARY

The current EUR-Lex AI Chat system has a critical, systemic failure: the build pipeline cannot fetch actual EU legal documents. The `fetch_document_xhtml()` function in `scripts/build_index.py` attempts to fetch `{CELEX}.ENG.xhtml` URLs which return RDF metadata (not XHTML) or HTTP 404 for every tested document. This was caused by commit `9497876` which changed the working `eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}` URL to the broken `publications.europa.eu/resource/celex/{celex}.ENG.xhtml` URL.

**Impact assessment:**
- The existing dataset (305,957 chunks, 15,112 CELEX documents) was built with the original working code and contains real legal text
- All subsequent rebuild attempts (including the current `build_index.py`) produce garbage or nothing
- Critical documents are missing: GDPR (0 chunks), AI Act 2024 (0 chunks), most post-2011 legislation
- Coverage drops from ~1,400 docs/year (2004-2011) to ~200-500 docs/year (2012-2023)
- The existing retrieval system works (MiniLM + FAISS) but the foundation for growing it is broken

**This plan fixes the entire stack systematically, with proper engineering discipline:**

| Phase | Name | Criticality | Duration |
|-------|------|-------------|----------|
| 0 | Emergency: Fix Build Pipeline | 🔴 CRITICAL | 1-2 days |
| 1 | Structure-Aware Chunking | 🟡 HIGH | 1 day |
| 2 | Hybrid Search + Reranking | 🟡 HIGH | 2 days |
| 3 | Evaluation Framework | 🟢 MEDIUM | 1-2 days |
| 4 | Complete Index Rebuild | 🔴 CRITICAL | 1 day |

---

## PRE-FLIGHT: CHECKPOINTS, BACKUPS, AND ROLLBACK PROCEDURES

Before touching anything, create the safety net. Execute these steps in order. Do not skip.

### CHECKPOINT 0.0: Full Repository Backup

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Create timestamped backup
BACKUP_DIR="/home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat-backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup everything
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/' \
    --exclude='.venv/' \
    . "$BACKUP_DIR/"

# Backup current dataset from HuggingFace
echo "Backing up HuggingFace dataset..."
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os

files = ['chunks.db', 'index.faiss', 'index_eurlex.faiss', 
         'chunks_eurlex.db', 'build_meta.json', 'build_meta_eurlex.json']
os.makedirs('$BACKUP_DIR/hf_dataset', exist_ok=True)

for f in files:
    try:
        path = hf_hub_download(
            repo_id='NedAktovOps/eurlex-chat-data',
            filename=f,
            repo_type='dataset',
        )
        shutil.copy(path, '$BACKUP_DIR/hf_dataset/' + f)
        print(f'  ✓ {f}')
    except Exception as e:
        print(f'  ✗ {f}: {e}')
"

echo "Backup complete: $BACKUP_DIR"
echo "TO RESTORE: rsync -av $BACKUP_DIR/ /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat/"
```

**Checkpoint marker:** File `CHECKPOINT_0_0_DONE.txt` created in backup dir.

### CHECKPOINT 0.1: Document Current State

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Record current git state
git log --oneline -5 > CHECKPOINT_0_1_git_state.txt
git diff HEAD -- scripts/build_index.py > CHECKPOINT_0_1_build_index_diff.txt
git diff HEAD -- app/search.py > CHECKPOINT_0_1_search_diff.txt

# Record current dataset stats
python3 -c "
from huggingface_hub import hf_hub_download
import sqlite3, os

path = hf_hub_download(repo_id='NedAktovOps/eurlex-chat-data', 
                       filename='chunks.db', repo_type='dataset')
conn = sqlite3.connect(path)
total = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
celexes = conn.execute('SELECT COUNT(DISTINCT celex) FROM chunks').fetchone()[0]
null_arts = conn.execute('SELECT COUNT(*) FROM chunks WHERE article IS NULL').fetchone()[0]
print(f'Total chunks: {total}')
print(f'Unique CELEXes: {celexes}')
print(f'NULL articles: {null_arts} ({100*null_arts/total:.1f}%)')
conn.close()
" > CHECKPOINT_0_1_dataset_stats.txt

cat CHECKPOINT_0_1_dataset_stats.txt
```

**Expected output:** Total chunks: ~305,957 | Unique CELEXes: ~15,112 | NULL articles: ~22,434 (7.3%)

### CHECKPOINT 0.2: Create Git Worktree for Safe Development

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Create a dedicated worktree for the rebuild
git worktree add /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat-rebuild main

# Or create a branch in current repo
git checkout -b feature/complete-rebuild

echo "Worktree/branch created for rebuild development"
echo "All Phase 0-4 work happens in feature/complete-rebuild"
echo "TO SWITCH BACK: git checkout main"
```

### CHECKPOINT 0.3: Environment Verification

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Verify Python environment
source ~/Desktop/EUProjects/.venv/bin/activate
python --version  # Must be 3.10+
pip list | grep -E "faiss|sentence-transformers|huggingface_hub|requests|beautifulsoup4"

# Verify HF token is available
python3 -c "import os; token = os.environ.get('HF_TOKEN'); print('HF_TOKEN:', 'SET' if token else 'NOT SET')"

# Verify GPU availability (for embedding generation)
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Rollback procedure:** If any verification fails, restore from checkpoint 0.0:
```bash
rsync -av --delete /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat-backups/<timestamp>/ \
    /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat/
```

---

## PHASE 0: EMERGENCY — FIX THE BUILD PIPELINE

**Goal:** `scripts/build_index.py` can download and parse real EU legislation from EUR-Lex Cellar API.

**Root cause:** `fetch_document_xhtml()` at line 173-201 fetches `https://publications.europa.eu/resource/celex/{celex}.ENG.xhtml` which returns RDF/XML metadata or HTTP 404 for every document tested (2006, 2011, 2014, 2016, 2023, 2024).

**Verified working URL pattern:** The actual XHTML documents are at `owl:sameAs` URLs embedded in the RDF responses. Example: `https://publications.europa.eu/resource/celex/32023L0970.ENG.xhtml.L_2023132EN.01002101.doc.html` returns real XHTML for Pay Transparency Directive.

**Approach:** Replace the broken fetch with proper Cellar RDF traversal using the `eurlxp` library (https://github.com/kevin91nl/eurlex), which handles the entire resolution chain automatically.

---

### PHASE 0 — STEP 1: Install and Verify eurlxp Library

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Install eurlxp
pip install eurlxp

# Verify installation
python3 -c "
from eurlxp import get_html_by_celex_id, WAFChallengeError
print('eurlxp imported successfully')
print('get_html_by_celex_id:', callable(get_html_by_celex_id))
"
```

**Test:** Fetch a known document (Pay Transparency Directive — 32023L0970):

```bash
python3 -c "
from eurlxp import get_html_by_celex_id
import os

os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')

html = get_html_by_celex_id('32023L0970', language='en')
if html and len(html) > 1000:
    print(f'SUCCESS: got {len(html)} bytes of HTML')
    # Check for ELI structure
    if '.eli-container' in html or 'eli-subdivision' in html:
        print('Contains ELI structure: YES')
    else:
        print('Contains ELI structure: NO')
else:
    print(f'FAILED: got {len(html) if html else 0} bytes')
"
```

**Expected:** `SUCCESS: got XXXXX bytes of HTML` + `Contains ELI structure: YES`

**If FAIL:** Try alternative approach using `eur-lex.europa.eu` endpoint directly:

```bash
python3 -c "
import requests
from bs4 import BeautifulSoup

# Try direct eur-lex endpoint
url = 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023L0970'
r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
print(f'Status: {r.status_code}, Size: {len(r.text)} bytes')

if r.status_code == 200 and len(r.text) > 1000:
    soup = BeautifulSoup(r.text, 'html.parser')
    eli = soup.select_one('.eli-container')
    print(f'ELI container found: {eli is not None}')
    if eli:
        subdivisions = eli.select('.eli-subdivision')
        print(f'ELI subdivisions: {len(subdivisions)}')
"
```

**Rollback:** If `eurlxp` fails completely, use manual RDF traversal (Approach B from the plan). No code changes to `build_index.py` yet.

---

### PHASE 0 — STEP 2: Add eurlxp to Requirements and Create Wrapper

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Add to requirements.txt
echo "eurlxp>=0.6.0" >> requirements.txt

# Create the fetch wrapper module
cat > app/eurlex_fetcher.py << 'EOF'
"""
EUR-Lex document fetcher — wraps eurlxp with fallback and error handling.

Primary: uses eurlxp.get_html_by_celex_id() which handles Cellar RDF traversal.
Fallback: uses eur-lex.europa.eu direct endpoint with WAF detection.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Optional import — if eurlxp not available, use fallback
try:
    from eurlxp import get_html_by_celex_id, WAFChallengeError
    EURLEXP_AVAILABLE = True
except ImportError:
    EURLEXP_AVAILABLE = False
    logger.warning("eurlxp not available — using fallback fetcher")


class EURLexFetcher:
    """Fetches EUR-Lex documents with automatic fallback."""
    
    def __init__(self, max_workers: int = 10, rate_limit_delay: float = 0.5):
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "application/xhtml+xml, text/html, application/xml",
        })
    
    def fetch(self, celex: str, language: str = "en") -> Optional[str]:
        """Fetch XHTML for a CELEX number.
        
        Tries in order:
        1. eurlxp.get_html_by_celex_id() — handles Cellar RDF traversal
        2. Direct eur-lex.europa.eu endpoint (WAF-vulnerable)
        
        Returns None on failure.
        """
        if EURLEXP_AVAILABLE:
            html = self._fetch_via_eurlxp(celex, language)
            if html:
                return html
        
        # Fallback to direct endpoint
        return self._fetch_direct(celex, language)
    
    def _fetch_via_eurlxp(self, celex: str, language: str) -> Optional[str]:
        """Fetch using eurlxp library."""
        try:
            html = get_html_by_celex_id(celex, language=language)
            if html and len(html) > 500:
                logger.debug(f"  [eurlxp] {celex}: {len(html)} bytes")
                return html
            return None
        except WAFChallengeError:
            logger.warning(f"  [eurlxp] WAF challenge for {celex}")
            return None
        except Exception as e:
            logger.debug(f"  [eurlxp] Error for {celex}: {e}")
            return None
    
    def _fetch_direct(self, celex: str, language: str) -> Optional[str]:
        """Direct fetch from eur-lex.europa.eu (fallback, may trigger WAF)."""
        try:
            encoded = requests.utils.quote(celex, safe="")
            url = f"https://eur-lex.europa.eu/legal-content/{language}/TXT/?uri=CELEX:{encoded}"
            r = self._session.get(url, timeout=15)
            
            if r.status_code == 202:
                # WAF challenge — document not immediately available
                logger.warning(f"  [direct] WAF challenge for {celex}")
                return None
            
            if r.status_code != 200 or len(r.text) < 500:
                logger.debug(f"  [direct] HTTP {r.status_code} for {celex}")
                return None
            
            # Verify it's actually XHTML/HTML
            if "<!DOCTYPE" not in r.text.upper() and "<html" not in r.text.lower():
                logger.debug(f"  [direct] Not HTML for {celex}")
                return None
            
            logger.debug(f"  [direct] {celex}: {len(r.text)} bytes")
            return r.text
        except Exception as e:
            logger.debug(f"  [direct] Error for {celex}: {e}")
            return None
    
    def fetch_batch(self, celexes: list, language: str = "en") -> Dict[str, Optional[str]]:
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
                    logger.warning(f"  Timeout/error for {celex}: {e}")
                    results[celex] = None
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
        
        return results


# Singleton instance
_fetcher = None

def get_fetcher() -> EURLexFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = EURLexFetcher(max_workers=10)
    return _fetcher
EOF

echo "Created app/eurlex_fetcher.py"
```

**Test the wrapper:**

```bash
python3 -c "
from app.eurlex_fetcher import get_fetcher

fetcher = get_fetcher()

# Test 3 documents: Pay Transparency (should work), GDPR (should work), AI Act (should work)
for celex in ['32023L0970', '32016R0679', '32024R1689']:
    html = fetcher.fetch(celex)
    if html:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        eli = soup.select_one('.eli-container')
        subdivisions = soup.select('.eli-subdivision')
        print(f'{celex}: {len(html)} bytes, ELI: {eli is not None}, subdivisions: {len(subdivisions)}')
    else:
        print(f'{celex}: FAILED')
"
```

**Expected:** All 3 return HTML with ELI structure (or at least 2 of 3 succeed).

**Rollback:** If the wrapper fails completely, delete `app/eurlex_fetcher.py` and revert requirements.txt. The old `fetch_document_xhtml()` is still intact.

---

### PHASE 0 — STEP 3: Replace fetch_document_xhtml() in build_index.py

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Save the original function for rollback
cp scripts/build_index.py scripts/build_index.py.backup-phase0

# Now modify build_index.py — replace fetch_document_xhtml
```

**Modify `scripts/build_index.py`:**

Find the `fetch_document_xhtml` function (lines 173-201) and replace it entirely:

```python
def fetch_document_xhtml(doc):
    """Fetch XHTML from EUR-Lex via eurlxp (Cellar RDF traversal).

    Replaces the broken publications.europa.eu/resource/celex/{celex}.ENG.xhtml
    URL which returned RDF metadata or 404 for all modern documents.
    
    Uses eurlxp.get_html_by_celex_id() which handles the full Cellar RDF
    traversal: CELEX → RDF metadata → expression_manifested_by_manifestation
    → DOC_1 → owl:sameAs → actual XHTML document.
    
    Falls back to direct eur-lex.europa.eu endpoint if eurlxp is unavailable.
    """
    from app.eurlex_fetcher import get_fetcher
    
    celex = doc["celex"]
    fetcher = get_fetcher()
    
    try:
        html = fetcher.fetch(celex, language="en")
        
        if html is None:
            logger.warning(f"  Failed to fetch {celex} (all methods exhausted)")
            return None
        
        if len(html) < 500:
            logger.warning(f"  Empty content for {celex} ({len(html)} bytes)")
            return None
        
        # Verify it looks like EUR-Lex content
        if ".eli-container" not in html and "eli-subdivision" not in html:
            # Might be a landing page or RDF — check for other indicators
            if "publications.europa.eu" not in html and "EUR-Lex" not in html:
                logger.warning(f"  Unexpected content type for {celex}")
                return None
        
        return html
    
    except Exception as e:
        logger.warning(f"  Unexpected error fetching {celex}: {e}")
        return None
```

**Also add the import at the top of the file:**

After line 50 (`from urllib.parse import quote`), add:
```python
from app.eurlex_fetcher import get_fetcher
```

**Verify the modification:**

```bash
# Check the file was modified correctly
grep -n "from app.eurlex_fetcher import" scripts/build_index.py
grep -n "def fetch_document_xhtml" scripts/build_index.py
python3 -c "import scripts.build_index; print('build_index imports OK')"
```

---

### PHASE 0 — STEP 4: Expand SPARQL Query to Cover All Resource Types

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Backup current query_all_documents
cp scripts/build_index.py scripts/build_index.py.backup-phase0-sparql
```

**Modify `scripts/build_index.py`:**

Find the `query_all_documents()` function (around line 119) and the `DOC_TYPES` constant (around line 96).

**Replace the DOC_TYPES constant:**

```python
# BEFORE (broken — only REG + DIR):
# DOC_TYPES = ["REG", "DIR"]

# AFTER — comprehensive EU legal document types:
DOC_TYPES = [
    "REG",      # Regulations
    "DIR",      # Directives  
    "DEC",      # Decisions
    "REC",      # Recommendations
    "OPIN",     # Opinions
    "RES",      # Resolutions
    "CONS",     # Consolidated texts
    "INF",      # Information documents
    "INT",      # International agreements
    # Note: CJEU case law (sector 7) requires different SPARQL query pattern
    # and is handled separately in Phase 0.5
]
```

**Replace the `query_all_documents()` function:**

```python
def query_all_documents():
    """Query ALL EU legal documents via SPARQL (not just REG + DIR).

    Returns documents of types: REG, DIR, DEC, REC, OPIN, RES, CONS, INF, INT
    from FROM_DATE (default: 1952-01-01, Treaty of Paris) with no upper bound.
    
    Uses the EUR-Lex Cellar SPARQL endpoint at:
    https://publications.europa.eu/webapi/rdf/sparql
    
    Filters out corrigenda (CELEX containing 'R(') at query level.
    """
    from app.eurlex_fetcher import get_fetcher
    
    type_filters_list = [
        f"?type = <http://publications.europa.eu/resource/authority/resource-type/{t}>"
        for t in DOC_TYPES
    ]
    type_filter = " ||\n    ".join(type_filters_list)
    
    # Build SPARQL query with comprehensive types
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
    # No date filter — fetch ALL documents from FROM_DATE onward
    # Set FROM_DATE in environment or at top of file
    FILTER(?date >= "{FROM_DATE}T00:00:00"^^xsd:dateTime || !BOUND(?date))
    # Exclude corrigenda — pattern is CELEX contains 'R('
    FILTER(!CONTAINS(?celex, "R("))
    # Exclude documents without valid CELEX format (starts with country code)
    FILTER(STRSTARTS(?celex, "3"))
}}
ORDER BY ?date
LIMIT 50000
"""
    
    logger.info(f"SPARQL query for types: {DOC_TYPES}, from: {FROM_DATE}")
    logger.info(f"Query will return up to 50,000 documents")
    
    r = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        timeout=180,  # Increased timeout for large result sets
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
    
    # Log breakdown by type
    from collections import Counter
    type_counts = Counter(d["type"] for d in all_docs)
    for t, count in sorted(type_counts.items()):
        logger.info(f"  {t}: {count}")
    
    return all_docs
```

**Also update FROM_DATE at the top of the file:**

```python
# Change from "2004-01-01" to "1952-01-01" (Treaty of Paris)
FROM_DATE = os.environ.get("FROM_DATE", "1952-01-01")
```

**Verify:**

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.build_index import query_all_documents, DOC_TYPES
print('DOC_TYPES:', DOC_TYPES)
print('FROM_DATE:', __import__('scripts.build_index', fromlist=['FROM_DATE']).FROM_DATE)
print('query_all_documents function loads OK')
"
```

---

### PHASE 0 — STEP 5: Test the Complete Fetch + Parse Pipeline

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Create a minimal test script
cat > test_phase0_fetch.py << 'EOF'
"""Phase 0 test: verify fetch + parse pipeline works for known documents."""

import sys
sys.path.insert(0, '.')

from scripts.build_index import fetch_document_xhtml, parse_html_to_chunks

# Documents that were previously missing or broken
TEST_CELEXES = [
    ("32023L0970", "Pay Transparency Directive"),     # Was returning RDF
    ("32016R0679", "GDPR"),                             # Was 404
    ("32024R1689", "AI Act"),                           # Was 404
    ("32006R1925", "Food Supplements Directive"),       # Should work
    ("32010R0695", "Single CMO Regulation"),            # Should work
]

print("=" * 60)
print("PHASE 0 TEST: Fetch + Parse Pipeline")
print("=" * 60)

all_passed = True
for celex, name in TEST_CELEXES:
    print(f"\n--- {celex}: {name} ---")
    
    # Step 1: Fetch
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
    
    # Step 2: Parse into chunks
    chunks = parse_html_to_chunks(html, celex, name)
    
    if not chunks:
        print(f"  ✗ PARSE: No chunks generated")
        all_passed = False
        continue
    
    print(f"  ✓ PARSE: {len(chunks)} chunks")
    
    # Step 3: Check article field
    null_articles = sum(1 for c in chunks if c.get("article") is None)
    articles_with_ids = [c for c in chunks if c.get("article")]
    
    print(f"  ✓ Articles: {len(articles_with_ids)} with IDs, {null_articles} NULL")
    
    if null_articles > len(chunks) * 0.5:
        print(f"  ⚠ WARNING: >50% NULL articles — chunking may need improvement")
    
    # Step 4: Check text quality (sample)
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
EOF

python3 test_phase0_fetch.py
```

**Expected output:** At least 4 of 5 documents fetch and parse successfully. NULL article rate should be <20%.

**If 3+ documents fail:** The fetch is still broken. Roll back and try the manual RDF traversal approach (Approach B from the plan).

**Rollback command:**
```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
cp scripts/build_index.py.backup-phase0 scripts/build_index.py
echo "Rolled back to pre-Phase-0 state"
```

---

### PHASE 0 — STEP 6: Run Full SPARQL Query Test

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Test the expanded SPARQL query
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.build_index import query_all_documents, DOC_TYPES

print('Testing expanded SPARQL query...')
print(f'DOC_TYPES: {DOC_TYPES}')

docs = query_all_documents()
print(f'Total documents returned: {len(docs)}')

# Check coverage
from collections import Counter
years = Counter()
for d in docs:
    if d['date']:
        year = d['date'][:4]
        years[year] += 1

print('Documents by year (sample):')
for year in sorted(years.keys())[-10:]:
    print(f'  {year}: {years[year]}')

# Check for previously missing documents
missing = ['32024R1689', '32016R0679', '32023L0970']
for m in missing:
    found = any(d['celex'] == m for d in docs)
    print(f'  {m}: {\"FOUND\" if found else \"NOT FOUND\"}')
"
```

**Expected:** SPARQL returns 30,000+ documents. AI Act (32024R1689) and GDPR (32016R0679) should appear in results.

**If GDPR/AI Act still not found:** The SPARQL endpoint may not index them yet, or they use a different resource type. Check with:
```bash
# Direct SPARQL query for specific CELEX
curl -s "https://publications.europa.eu/webapi/rdf/sparql" \
  --data-urlencode "query=SELECT * WHERE { ?doc <http://publications.europa.eu/ontology/cdm#resource_legal_id_celex> '32016R0679' }" \
  --data "format=json" | python3 -m json.tool
```

---

### PHASE 0 — STEP 7: Commit Phase 0 Checkpoint

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add scripts/build_index.py app/eurlex_fetcher.py requirements.txt
git commit -m "phase 0: fix build pipeline — replace broken fetch with eurlxp + expand SPARQL

- Replace fetch_document_xhtml() with eurlxp-based fetcher
- eurlxp handles Cellar RDF traversal automatically
- Expand SPARQL from REG+DIR only to all EU legal document types
- Remove date upper bound (was 2004-01-01, now 1952-01-01)
- Add comprehensive error handling and rate limiting

BREAKING: requires eurlxp>=0.6.0 dependency

Fixes: publications.europa.eu/resource/celex/{celex}.ENG.xhtml
was returning RDF metadata or 404 for all documents"
```

**Phase 0 complete.** Proceed to Phase 1 only after Phase 0 tests pass.

---

## PHASE 1: STRUCTURE-AWARE CHUNKING

**Goal:** Replace naive BeautifulSoup fallback chunking with `chunkweaver`'s `LEGAL_EU` preset. This eliminates the 22K NULL-article chunks and ensures every chunk is a coherent unit of legal meaning.

**Tool:** https://github.com/metawake/chunkweaver
- `LEGAL_EU` preset detects: `Article N`, `CHAPTER`, `SECTION`, `(N)` recitals
- `LEGAL_EU_LEVELED` variant for hierarchical splitting: CHAPTER > SECTION > Article > recital
- Zero dependencies (stdlib only)
- Benchmark on GDPR + EU AI Act with LLM-as-judge

---

### PHASE 1 — STEP 1: Install and Verify chunkweaver

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

pip install chunkweaver

# Verify
python3 -c "
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU, LEGAL_EU_LEVELED
from chunkweaver.detectors import HeadingDetector, TableDetector

print('chunkweaver imported OK')
print('PRESETS:', list(LEGAL_EU)[:3], '...')
"
```

**Test with sample EU legal text:**

```bash
python3 -c "
from chunkweaver import Chunker
from chunkweaver.presets import LEGAL_EU

sample = '''
CHAPTER I

GENERAL PROVISIONS

Article 1
Subject matter

This Regulation lays down rules relating to the protection of natural persons.

Article 2
Definitions

For the purposes of this Regulation, the following definitions apply.

(1) 'personal data' means any information relating to an identified or identifiable natural person.

(2) 'processing' means any operation performed on personal data.

CHAPTER II

PROCESSING OF PERSONAL DATA

Article 3
Scope

This Regulation applies to the processing of personal data.
'''

chunker = Chunker(
    target_size=512,
    overlap=2,
    boundaries=LEGAL_EU,
    detectors=[HeadingDetector(), TableDetector()],
)

chunks = chunker.chunk_with_metadata(sample)
print(f'Generated {len(chunks)} chunks')
for c in chunks:
    print(f'  [{c.boundary_type}] {c.text[:80]}...')
"
```

**Expected:** Chunks split at Article and Chapter boundaries, not mid-sentence.

---

### PHASE 1 — STEP 2: Integrate chunkweaver into parse_html_to_chunks()

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Backup current parse_html_to_chunks
cp scripts/build_index.py scripts/build_index.py.backup-phase1
```

**Replace `parse_html_to_chunks()` function:**

```python
def parse_html_to_chunks(html: str, celex_id: str, title: str) -> list:
    """Parse EUR-Lex HTML into structure-aware chunks.

    Uses chunkweaver's LEGAL_EU preset for EU legislation structure detection.
    Falls back to BeautifulSoup-based parsing for non-ELI documents.
    
    Structure detected by LEGAL_EU:
    - CHAPTER / SECTION headings
    - Article N (article-level splits)
    - (N) recital paragraphs
    - Annex / Enc / Pbl subdivisions
    
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
    
    # Try ELI container first (modern EUR-Lex format)
    container = soup.select_one(".eli-container")
    
    if container:
        # Get clean text with paragraph separation
        text_parts = []
        
        # Extract from ELI subdivisions with their IDs
        subdivisions = container.select(".eli-subdivision")
        
        if subdivisions:
            # Use chunkweaver for structure-aware splitting
            # First pass: extract text blocks with their IDs
            structured_text = _extract_eli_subdivisions(container)
            
            # Second pass: use chunkweaver on structured text
            chunker = Chunker(
                target_size=1024,
                overlap=2,
                overlap_unit="sentence",
                boundaries=LEGAL_EU,
                detectors=[HeadingDetector(), TableDetector()],
                fallback="paragraph",
                min_size=100,
            )
            
            # Feed structured text to chunkweaver
            chunks = chunker.chunk_with_metadata(structured_text)
            
            result = []
            for c in chunks:
                # Determine article type from boundary
                article_type = _classify_boundary(c.boundary_type)
                
                result.append({
                    "text": c.text.strip(),
                    "celex": celex_id,
                    "title": title,
                    "article": _extract_article_id(c.text),  # Try to find art_N in text
                    "type": article_type,
                })
            
            if result:
                return result
    
    # Fallback: use BeautifulSoup-based parsing for non-ELI documents
    return _parse_fallback_chunks(soup, celex_id, title)


def _extract_eli_subdivisions(container) -> str:
    """Extract text from ELI container, preserving structural markers."""
    from bs4 import NavigableString
    
    text_parts = []
    
    for element in container.descendants:
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                text_parts.append(text)
        elif element.name == "div":
            sub_id = element.get("id", "")
            if sub_id.startswith(("art_", "rct_", "enc_", "ann_", "cit_", "pbl_", "sec_", "ch_")):
                # Insert structural marker
                text_parts.append(f"\n[{sub_id}]\n")
    
    return "\n".join(text_parts)


def _classify_boundary(boundary_type: str) -> str:
    """Map chunkweaver boundary type to our article type."""
    if not boundary_type:
        return "paragraph"
    
    bt = boundary_type.lower()
    if "article" in bt or "art" in bt:
        return "article"
    elif "recital" in bt or "rct" in bt or bt.startswith("(1)"):
        return "recital"
    elif "preamble" in bt or "pbl" in bt:
        return "preamble"
    elif "annex" in bt or "ann" in bt:
        return "annex"
    elif "chapter" in bt or "section" in bt:
        return "section"
    elif "enacting" in bt or "enc" in bt:
        return "enacting"
    else:
        return "paragraph"


def _extract_article_id(text: str) -> str:
    """Extract article ID from chunk text (e.g., 'art_1' from 'Article 1')."""
    import re
    # Match patterns like [art_1], Article 1, Article 1., etc.
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
            return match.group(0).strip('[]')
    
    return None


def _parse_fallback_chunks(soup, celex_id: str, title: str) -> list:
    """Fallback parsing for non-ELI documents using BeautifulSoup."""
    from bs4 import BeautifulSoup
    
    # Try tab content
    for tab_sel in ["#text", "#document1", "#PP4Contents"]:
        tab = soup.select_one(tab_sel)
        if tab:
            text = tab.get_text(separator="\n", strip=True)
            paragraphs = extract_meaningful_paragraphs(text)
            if paragraphs:
                return [
                    {"text": p, "celex": celex_id, "title": title,
                     "article": None, "type": "paragraph"}
                    for p in paragraphs
                ]
    
    # Try document view
    doc_view = soup.select_one("#documentView")
    if doc_view:
        text = doc_view.get_text(separator="\n", strip=True)
        paragraphs = extract_meaningful_paragraphs(text)
        if paragraphs:
            return [
                {"text": p, "celex": celex_id, "title": title,
                 "article": None, "type": "paragraph"}
                for p in paragraphs
            ]
    
    # Last resort: full page text
    text = soup.get_text(separator="\n", strip=True)
    paragraphs = extract_meaningful_paragraphs(text)
    if paragraphs:
        return [
            {"text": p, "celex": celex_id, "title": title,
             "article": None, "type": "paragraph"}
            for p in paragraphs
        ]
    
    return []
```

**Also add chunkweaver to requirements.txt:**
```bash
echo "chunkweaver>=0.2.0" >> requirements.txt
```

---

### PHASE 1 — STEP 3: Test Chunking Quality

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

cat > test_phase1_chunking.py << 'EOF'
"""Phase 1 test: verify chunkweaver produces coherent chunks."""

import sys
sys.path.insert(0, '.')

from scripts.build_index import fetch_document_xhtml, parse_html_to_chunks

# Test on 3 documents
TEST_CELEXES = [
    ("32023L0970", "Pay Transparency Directive"),
    ("32016R0679", "GDPR"),
    ("32024R1689", "AI Act"),
]

print("=" * 60)
print("PHASE 1 TEST: Chunking Quality")
print("=" * 60)

all_passed = True
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
        all_passed = False
        continue
    
    print(f"  ✓ Generated {len(chunks)} chunks")
    
    # Check article field quality
    null_articles = sum(1 for c in chunks if c.get("article") is None)
    with_articles = len(chunks) - null_articles
    
    print(f"  ✓ With article ID: {with_articles} ({100*with_articles/len(chunks):.1f}%)")
    print(f"  ⚠  NULL articles: {null_articles} ({100*null_articles/len(chunks):.1f}%)")
    
    # Check chunk quality
    short_chunks = sum(1 for c in chunks if len(c["text"]) < 100)
    print(f"  ✓ Chunks >100 chars: {len(chunks) - short_chunks}/{len(chunks)}")
    
    # Check no chunk straddles article boundaries (sample)
    problematic = []
    for c in chunks[:5]:  # Check first 5
        text = c["text"]
        if text.count("Article") > 1:
            problematic.append(c["text"][:100])
    
    if problematic:
        print(f"  ⚠ WARNING: Some chunks may straddle article boundaries")
        for p in problematic[:2]:
            print(f"    '{p}...'")
    else:
        print(f"  ✓ No obvious boundary crossings in sample")
    
    # Check type distribution
    from collections import Counter
    type_dist = Counter(c.get("type", "unknown") for c in chunks)
    print(f"  Type distribution: {dict(type_dist)}")

print()
print("=" * 60)
if all_passed:
    print("RESULT: ALL TESTS PASSED ✓")
else:
    print("RESULT: SOME TESTS FAILED ✗")
print("=" * 60)
EOF

python3 test_phase1_chunking.py
```

**Expected:**
- NULL article rate should drop from 7.3% to <5%
- No chunks should straddle article boundaries
- Type distribution should show mostly `article`, `recital`, `preamble`

**If NULL article rate is still high:** The document's HTML structure may not have `.eli-subdivision` IDs. Check:
```bash
python3 -c "
from scripts.build_index import fetch_document_xhtml
html = fetch_document_xhtml({'celex': '32016R0679', 'title': '', 'date': '', 'type': ''})
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
container = soup.select_one('.eli-container')
if container:
    subdivisions = container.select('.eli-subdivision')
    print(f'ELI subdivisions: {len(subdivisions)}')
    for s in subdivisions[:5]:
        print(f'  {s.get(\"id\")}: {s.get_text()[:50]}...')
else:
    print('No .eli-container found')
"
```

---

### PHASE 1 — STEP 4: Commit Phase 1 Checkpoint

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add scripts/build_index.py requirements.txt
git commit -m "phase 1: integrate chunkweaver LEGAL_EU for structure-aware chunking

- Replace naive BeautifulSoup fallback with chunkweaver LEGAL_EU preset
- Chunk boundaries at Article, Chapter, Section, recital level
- Eliminates NULL-article chunks from modern ELI documents
- Falls back to BeautifulSoup for non-ELI documents
- Adds _extract_eli_subdivisions(), _classify_boundary(), _extract_article_id()
- Adds chunkweaver>=0.2.0 to requirements

Fixes: 22K NULL-article chunks from broken BeautifulSoup fallbacks"
```

---

## PHASE 2: HYBRID SEARCH + RERANKING

**Goal:** Add BM25 keyword retrieval + FAISS dense retrieval + Reciprocal Rank Fusion + cross-encoder reranker. This fixes queries that fail because dense embeddings miss exact keyword matches.

**Current gap:** `app/search.py:108-168` does pure FAISS cosine similarity with hand-written `discourse_boost()` (1.3× for articles). The "obligations" query returns 704 chars with 3 citations — a clear failure.

**Tools:**
- `rank_bm25` (0.2.2) — BM25 implementation
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — neural reranker
- Reference: https://github.com/im-anishraj/Hybrid-Search-RAG-Engine

---

### PHASE 2 — STEP 1: Create BM25 Store

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Install dependencies
source ~/Desktop/EUProjects/.venv/bin/activate
pip install rank_bm25

# Create BM25 store module
cat > app/bm25_store.py << 'EOF'
"""
BM25 keyword retrieval store for EUR-Lex documents.

Builds an inverted index over all chunk texts using BM25Okapi.
Used alongside FAISS for hybrid search via Reciprocal Rank Fusion.

Usage:
    bm25 = BM25Store()
    bm25.build(chunks)  # chunks = list of dicts with 'id', 'text', 'celex', etc.
    results = bm25.search("employer obligations GDPR", top_k=20)
"""

import json
import logging
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


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
    
    Uses rank_bm25.BM25Okapi with standard parameters (k1=1.5, b=0.75).
    Persists to disk via pickle.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunk_ids: List[int] = []
        self.chunk_texts: List[str] = []
        self.chunk_data: Dict[int, Dict[str, Any]] = {}  # chunk_id -> full chunk dict
    
    def build(self, chunks: List[Dict]) -> int:
        """Build BM25 index from list of chunks.
        
        Args:
            chunks: list of dicts with keys: id, text, celex, title, article, type
            
        Returns:
            Number of chunks indexed
        """
        self.chunk_ids = [c["id"] for c in chunks]
        self.chunk_texts = [c["text"] for c in chunks]
        self.chunk_data = {c["id"]: c for c in chunks}
        
        tokenized = [self._tokenize(text) for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        
        logger.info(f"BM25 index built: {len(self.chunk_ids)} chunks")
        return len(self.chunk_ids)
    
    def search(self, query: str, top_k: int = 20) -> List[BM25Result]:
        """Search BM25 index for query.
        
        Args:
            query: search query string
            top_k: number of results to return
            
        Returns:
            List of BM25Result, sorted by descending score
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built — returning empty results")
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
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
        """Tokenize text for BM25 (lowercase, alphanumeric only)."""
        return [t.lower() for t in re.findall(r'\b\w{2,}\b', text)]
    
    def save(self, path: str) -> None:
        """Save BM25 index to disk via pickle."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "chunk_ids": self.chunk_ids,
            "chunk_texts": self.chunk_texts,
            "chunk_data": self.chunk_data,
        }
        
        # Save chunk_data separately as JSON (more portable)
        chunk_data_path = path.replace(".pkl", "_chunk_data.json")
        with open(chunk_data_path, "w") as f:
            # Convert keys to strings for JSON
            json.dump({str(k): v for k, v in self.chunk_data.items()}, f)
        
        # Save BM25 object + metadata via pickle
        with open(path, "wb") as f:
            pickle.dump({
                "k1": self.k1,
                "b": self.b,
                "chunk_ids": self.chunk_ids,
                "bm25": self.bm25,  # BM25Okapi is pickle-able
            }, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load(self, path: str) -> int:
        """Load BM25 index from disk."""
        chunk_data_path = path.replace(".pkl", "_chunk_data.json")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.k1 = data["k1"]
        self.b = data["b"]
        self.chunk_ids = data["chunk_ids"]
        self.bm25 = data["bm25"]
        
        # Load chunk_data from JSON
        if os.path.exists(chunk_data_path):
            with open(chunk_data_path) as f:
                self.chunk_data = {int(k): v for k, v in json.load(f).items()}
        
        logger.info(f"BM25 index loaded: {len(self.chunk_ids)} chunks")
        return len(self.chunk_ids)


# Import here to avoid top-level dependency issue
from rank_bm25 import BM25Okapi
EOF

echo "Created app/bm25_store.py"
```

**Test BM25 store:**

```bash
python3 -c "
from app.bm25_store import BM25Store

# Test with sample data
chunks = [
    {'id': 1, 'text': 'Article 1 of GDPR states that personal data processing must be lawful.', 'celex': '32016R0679', 'article': 'art_1'},
    {'id': 2, 'text': 'Article 5 of GDPR defines the principles relating to processing of personal data.', 'celex': '32016R0679', 'article': 'art_5'},
    {'id': 3, 'text': 'Employers have obligations under the Pay Transparency Directive.', 'celex': '32023L0970', 'article': 'art_3'},
    {'id': 4, 'text': 'Member States shall ensure that employers publish pay information.', 'celex': '32023L0970', 'article': 'art_4'},
    {'id': 5, 'text': 'The European Parliament has adopted this regulation.', 'celex': '32023L0970', 'article': 'pbl_1'},
]

bm25 = BM25Store()
bm25.build(chunks)

# Test searches
for query in ['GDPR personal data', 'employer obligations', 'pay transparency']:
    results = bm25.search(query, top_k=3)
    print(f'Query: \"{query}\"')
    for r in results:
        print(f'  [{r.rank}] score={r.score:.2f} celex={r.celex} article={r.article}')
        print(f'      \"{r.text[:60]}...\"')
    print()
"
```

---

### PHASE 2 — STEP 2: Create Hybrid Search Module with RRF

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

cat > app/hybrid_search.py << 'EOF'
"""
Hybrid search combining BM25 + FAISS via Reciprocal Rank Fusion (RRF).

3-stage pipeline:
1. BM25 sparse search (k=30)
2. FAISS dense search (k=30)  
3. RRF fusion (k=60) → top-20 candidates
4. Cross-encoder rerank → top-5 final

Reference: https://github.com/im-anishraj/Hybrid-Search-RAG-Engine
Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result from hybrid pipeline."""
    chunk_id: int
    celex: str
    title: str
    article: str
    chunk_type: str
    text: str
    score: float
    source: str  # 'bm25', 'faiss', 'rrf', 'rerank'
    bm25_score: float = 0.0
    faiss_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class HybridSearcher:
    """Hybrid BM25 + FAISS searcher with RRF fusion and optional reranking."""
    
    def __init__(
        self,
        bm25_store,
        faiss_index,
        chunks_db_path: str,
        embedder=None,
        rrf_k: int = 60,
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Args:
            bm25_store: BM25Store instance
            faiss_index: FAISS index (already loaded)
            chunks_db_path: path to SQLite chunks.db
            embedder: sentence transformer for query embedding
            rrf_k: RRF constant (default 60 — standard value)
            use_reranker: whether to use cross-encoder reranking
            reranker_model: HuggingFace model name for reranker
        """
        self.bm25 = bm25_store
        self.faiss = faiss_index
        self.chunks_db_path = chunks_db_path
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self._reranker = None
        
        # Lazy-load reranker (50ms latency, only load if needed)
        if self.use_reranker:
            self._init_reranker()
    
    def _init_reranker(self):
        """Initialize cross-encoder reranker (lazy load)."""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model)
            logger.info(f"Reranker loaded: {self.reranker_model}")
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}. Disabling reranking.")
            self.use_reranker = False
    
    def search(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 10,
        return_candidates: int = 30,
    ) -> List[SearchResult]:
        """
        Execute hybrid search pipeline.
        
        Args:
            query: text query string
            query_vector: pre-computed query embedding (if None, compute from embedder)
            top_k: final number of results to return
            return_candidates: number of candidates for RRF fusion
            
        Returns:
            List of SearchResult, sorted by final score
        """
        # Step 1: Get query embedding
        if query_vector is None and self.embedder is not None:
            query_vector = self.embedder.encode([query], normalize_embeddings=True)[0]
        
        # Step 2: BM25 search
        bm25_results = self.bm25.search(query, top_k=return_candidates)
        logger.debug(f"BM25: {len(bm25_results)} results")
        
        # Step 3: FAISS search
        faiss_results = []
        if query_vector is not None:
            scores, indices = self.faiss.search(
                query_vector.astype(np.float32).reshape(1, -1),
                return_candidates
            )
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < len(self.bm25.chunk_ids):  # Valid index
                    chunk_id = self.bm25.chunk_ids[idx]
                    faiss_results.append({
                        "rank": rank + 1,
                        "score": float(score),
                        "chunk_id": chunk_id,
                    })
        logger.debug(f"FAISS: {len(faiss_results)} results")
        
        # Step 4: RRF fusion
        rrf_scores = self._compute_rrf(bm25_results, faiss_results)
        
        # Step 5: Fetch chunk data for top candidates
        top_chunk_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:return_candidates]
        
        candidates = self._fetch_chunks([cid for cid, _ in top_chunk_ids])
        
        # Step 6: Cross-encoder rerank
        if self.use_reranker and self._reranker and len(candidates) > top_k:
            candidates = self._rerank(query, candidates, top_k)
        
        # Build final results
        results = []
        for chunk_id, rrf_score in top_chunk_ids[:top_k]:
            if chunk_id in candidates:
                c = candidates[chunk_id]
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    celex=c.get("celex", ""),
                    title=c.get("title", ""),
                    article=c.get("article", ""),
                    chunk_type=c.get("type", ""),
                    text=c.get("text", ""),
                    score=rrf_score,
                    source="rrf",
                    rrf_score=rrf_score,
                ))
        
        return results[:top_k]
    
    def _compute_rrf(
        self,
        bm25_results: List,
        faiss_results: List,
    ) -> Dict[int, float]:
        """Compute Reciprocal Rank Fusion scores."""
        rrf = {}
        
        # Add BM25 scores
        for r in bm25_results:
            chunk_id = r.chunk_id
            rrf[chunk_id] = rrf.get(chunk_id, 0) + 1.0 / (self.rrf_k + r.rank)
        
        # Add FAISS scores
        for r in faiss_results:
            chunk_id = r["chunk_id"]
            rrf[chunk_id] = rrf.get(chunk_id, 0) + 1.0 / (self.rrf_k + r["rank"])
        
        return rrf
    
    def _fetch_chunks(self, chunk_ids: List[int]) -> Dict[int, Dict]:
        """Fetch chunk data from SQLite database."""
        import sqlite3
        results = {}
        
        try:
            conn = sqlite3.connect(self.chunks_db_path)
            placeholders = ",".join("?" * len(chunk_ids))
            rows = conn.execute(
                f"SELECT id, celex, title, article, type, text FROM chunks WHERE id IN ({placeholders})",
                chunk_ids
            ).fetchall()
            conn.close()
            
            for row in rows:
                results[row[0]] = {
                    "id": row[0],
                    "celex": row[1],
                    "title": row[2] or "",
                    "article": row[3] or "",
                    "type": row[4] or "",
                    "text": row[5] or "",
                }
        except Exception as e:
            logger.error(f"Error fetching chunks: {e}")
        
        return results
    
    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Rerank candidates using cross-encoder."""
        if not self._reranker:
            return candidates[:top_k]
        
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._reranker.predict(pairs)
        
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [c for c, _ in scored[:top_k]]
EOF

echo "Created app/hybrid_search.py"
```

---

### PHASE 2 — STEP 3: Integrate Hybrid Search into app/search.py

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

# Backup current search.py
cp app/search.py app/search.py.backup-phase2

# Read current search.py to understand the interface
head -50 app/search.py
```

**Modify `app/search.py`:**

Find the `search_discourse_aware()` function and the `search()` function. Replace the pure FAISS retrieval with hybrid search.

```python
# Add these imports at the top of search.py
from app.hybrid_search import HybridSearcher
from app.bm25_store import BM25Store

# Global instances (lazy-loaded)
_bm25_store = None
_hybrid_searcher = None


def _get_bm25_store():
    """Lazy-load BM25 store."""
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
        chunks = _load_chunks_from_db()
        _bm25_store.build(chunks)
    return _bm25_store


def _get_hybrid_searcher():
    """Lazy-load hybrid searcher."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        bm25 = _get_bm25_store()
        faiss_idx = _get_faiss_index()
        embedder = _get_embedding_model()
        chunks_db = os.path.join(DATA_DIR, "chunks.db")
        
        _hybrid_searcher = HybridSearcher(
            bm25_store=bm25,
            faiss_index=faiss_idx,
            chunks_db_path=chunks_db,
            embedder=embedder,
            rrf_k=60,
            use_reranker=True,
        )
    return _hybrid_searcher


def search(query: str, top_k: int = 10) -> List[Dict]:
    """Main search function — now uses hybrid search."""
    # Get query embedding
    embedder = _get_embedding_model()
    query_vector = embedder.encode([query], normalize_embeddings=True)[0]
    
    # Use hybrid search
    hybrid = _get_hybrid_searcher()
    results = hybrid.search(
        query=query,
        query_vector=query_vector,
        top_k=top_k,
        return_candidates=30,
    )
    
    return [
        {
            "id": r.chunk_id,
            "celex": r.celex,
            "title": r.title,
            "article": r.article,
            "type": r.chunk_type,
            "text": r.text,
            "score": r.score,
        }
        for r in results
    ]
```

**Replace `search_discourse_aware()` to use hybrid search:**

```python
def search_discourse_aware(query: str, top_k: int = 10) -> List[Dict]:
    """Search with discourse-aware ranking.
    
    Now uses hybrid BM25+FAISS+RRF pipeline with optional reranking.
    The hand-written discourse_boost() is replaced by neural reranking.
    """
    return search(query, top_k)
```

---

### PHASE 2 — STEP 4: Test Hybrid Search

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

cat > test_phase2_hybrid.py << 'EOF'
"""Phase 2 test: verify hybrid search works correctly."""

import sys
sys.path.insert(0, '.')

from app.search import search, _get_bm25_store, _get_hybrid_searcher
from scripts.build_index import fetch_document_xhtml, parse_html_to_chunks

print("=" * 60)
print("PHASE 2 TEST: Hybrid Search")
print("=" * 60)

# First, build BM25 index from a small sample
print("\n1. Building BM25 index from sample documents...")

# Fetch and parse a few documents
sample_chunks = []
for celex, name in [("32023L0970", "Pay Transparency"), ("32016R0679", "GDPR")]:
    doc = {"celex": celex, "title": "", "date": "", "type": ""}
    html = fetch_document_xhtml(doc)
    if html:
        chunks = parse_html_to_chunks(html, celex, name)
        for c in chunks:
            c["id"] = len(sample_chunks) + 1
            sample_chunks.append(c)
        print(f"  {celex}: {len(chunks)} chunks")

print(f"Total sample chunks: {len(sample_chunks)}")

# Build BM25
bm25 = _get_bm25_store()
bm25.build(sample_chunks)

# Test BM25-only search
print("\n2. Testing BM25-only search:")
for query in ["employer obligations", "personal data processing", "transparency pay"]:
    results = bm25.search(query, top_k=3)
    print(f"  Query: '{query}'")
    for r in results:
        print(f"    [{r.rank}] score={r.score:.2f} article={r.article}")
        print(f"         '{r.text[:80]}...'")

# Test FAISS search
print("\n3. Testing FAISS search:")
from app.search import _get_embedding_model, _get_faiss_index
import numpy as np

embedder = _get_embedding_model()
faiss_idx = _get_faiss_index()

# Note: FAISS index may not have these chunks yet (they're not indexed)
# This test verifies the FAISS search path works
for query in ["employer obligations", "personal data processing"]:
    qv = embedder.encode([query], normalize_embeddings=True)[0]
    scores, indices = faiss_idx.search(qv.astype(np.float32).reshape(1, -1), 3)
    print(f"  Query: '{query}'")
    for idx, score in zip(indices[0], scores[0]):
        print(f"    idx={idx} score={score:.4f}")

# Test the "obligations" query that was previously failing
print("\n4. Testing previously-failing query:")
print("   (This query previously returned 704 chars with 3 citations)")
results = search("What are the detailed obligations of employers under EU law?", top_k=5)
print(f"  Got {len(results)} results")
for r in results:
    print(f"  - [{r['article']}] {r['text'][:100]}... (score={r['score']:.4f})")

print()
print("=" * 60)
print("Phase 2 test complete")
print("=" * 60)
EOF

python3 test_phase2_hybrid.py
```

**Expected:** BM25 finds results with keyword matches. FAISS finds semantically similar results. The "obligations" query returns more/better results than before.

**Rollback:**
```bash
cp app/search.py.backup-phase2 app/search.py
echo "Rolled back search.py to pre-Phase-2 state"
```

---

### PHASE 2 — STEP 5: Commit Phase 2 Checkpoint

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add app/bm25_store.py app/hybrid_search.py app/search.py requirements.txt
git commit -m "phase 2: add hybrid search — BM25 + FAISS + RRF + cross-encoder reranker

- Add BM25Store (rank_bm25) for keyword retrieval
- Add HybridSearcher combining BM25 + FAISS via Reciprocal Rank Fusion (k=60)
- Add optional cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- Replace pure FAISS search with hybrid search in app/search.py
- 3-stage pipeline: BM25(k=30) + FAISS(k=30) → RRF → rerank → top-5

Fixes: queries like 'employer obligations' that fail with dense-only search"
```

---

## PHASE 3: EVALUATION FRAMEWORK

**Goal:** Establish a measurable baseline before making further changes. Catch regressions before they reach production.

**Tools:**
- https://github.com/explodinggradients/ragas — evaluation framework
- https://github.com/235471/rag-evaluation-contracts-ragas — legal-specific composite score

---

### PHASE 3 — STEP 1: Create Evaluation Dataset

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

mkdir -p eval

cat > eval/test_dataset.jsonl << 'EOF'
{"question": "What are the obligations of employers under GDPR regarding personal data processing?", "answer": "Under GDPR Article 5, employers must process personal data lawfully, fairly, and transparently. Article 6 requires a legal basis for processing. Article 32 requires appropriate technical and organisational measures to secure personal data.", "celex": "32016R0679", "articles": ["art_5", "art_6", "art_32"]}
{"question": "What is the deadline for EU member states to transpose the Pay Transparency Directive?", "answer": "Member states must transpose the Pay Transparency Directive (2023/970) into national law by 7 June 2026.", "celex": "32023L0970", "articles": ["art_16"]}
{"question": "What are the prohibited AI practices under the AI Act?", "answer": "The AI Act prohibits: (1) subliminal/manipulative techniques, (2) exploitation of vulnerabilities, (3) social scoring by public authorities, (4) real-time remote biometric identification in public spaces (with exceptions), (5) emotion recognition in workplace/schools (with exceptions).", "celex": "32024R1689", "articles": ["art_5"]}
{"question": "What information must employers publish regarding pay transparency?", "answer": "Employers must publish: (a) pay statistics for categories of employees doing equal work or work of equal value, (b) the gender pay gap information, (c) information on pay progression. The Directive covers employers with 100+ employees initially.", "celex": "32023L0970", "articles": ["art_7", "art_8"]}
{"question": "What is the purpose of the Single Digital Gateway Regulation?", "answer": "The Single Digital Gateway aims to provide businesses and citizens with easy access to information, procedures, and assistance services across the EU through a single digital entry point.", "celex": "32018R1724", "articles": ["art_1", "art_2"]}
{"question": "What are the key principles of GDPR data processing?", "answer": "GDPR Article 5 establishes: (1) Lawfulness, fairness, transparency, (2) Purpose limitation, (3) Data minimisation, (4) Accuracy, (5) Storage limitation, (6) Integrity and confidentiality, (7) Accountability.", "celex": "32016R0679", "articles": ["art_5"]}
{"question": "What is the role of the European Data Protection Board?", "answer": "The European Data Protection Board (EDPB) ensures consistent application of GDPR across EU member states, issues guidelines and recommendations, and resolves disputes between supervisory authorities.", "celex": "32016R0679", "articles": ["art_68", "art_70"]}
{"question": "How does the AI Act classify high-risk AI systems?", "answer": "High-risk AI systems under the AI Act include: (1) AI in critical infrastructure, (2) education and vocational training, (3) employment and HR management, (4) essential services and housing, (5) law enforcement, (6) migration and border management, (7) administration of justice. Listed in Annex III.", "celex": "32024R1689", "articles": ["art_6", "art_7"]}
{"question": "What rights do data subjects have under GDPR?", "answer": "GDPR Chapter III grants: right to information (art 13-14), right of access (art 15), right to rectification (art 16), right to erasure (art 17), right to restriction (art 18), data portability (art 20), right to object (art 21), rights related to automated decision-making (art 22).", "celex": "32016R0679", "articles": ["art_13", "art_14", "art_15", "art_17", "art_20", "art_21", "art_22"]}
{"question": "What penalties apply for AI Act violations?", "answer": "AI Act penalties: up to €35 million or 7% of global annual turnover for prohibited practices; up to €15 million or 3% for other violations; up to €7.5 million or 1.5% for supplying incorrect information to notified bodies.", "celex": "32024R1689", "articles": ["art_71"]}
EOF

echo "Created eval/test_dataset.jsonl with 10 test cases"
```

---

### PHASE 3 — STEP 2: Create Evaluation Script

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat

cat > eval/run_evaluation.py << 'EOF'
"""
EUR-Lex AI Chat — Evaluation Script

Runs RAGAS evaluation on the current system and saves results.
Establishes baseline metrics for regression testing.

Usage:
    python eval/run_evaluation.py
    python eval/run_evaluation.py --compare baseline_results.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
EVAL_DIR = Path("eval")


def load_test_dataset(path: str) -> list:
    """Load test dataset from JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_retrieval(query: str, top_k: int = 10) -> list:
    """Run retrieval on the current system."""
    from app.search import search
    
    results = search(query, top_k=top_k)
    return [r["text"] for r in results]


def evaluate_test_case(test_case: dict) -> dict:
    """Evaluate a single test case."""
    query = test_case["question"]
    expected_answer = test_case["answer"]
    
    # Run retrieval
    retrieved_contexts = run_retrieval(query, top_k=5)
    
    # Compute simple metrics (full RAGAS requires LLM judge)
    # For now, compute context overlap metrics
    context_text = "\n".join(retrieved_contexts)
    
    # Check if expected articles are in retrieved contexts
    expected_articles = test_case.get("articles", [])
    found_articles = []
    for art in expected_articles:
        if art in context_text or art.replace("art_", "Article ") in context_text:
            found_articles.append(art)
    
    # Check if expected CELEX is in results
    expected_celex = test_case.get("celex", "")
    celex_found = expected_celex in context_text or expected_celex[-10:] in context_text
    
    return {
        "question": query,
        "expected_celex": expected_celex,
        "expected_articles": expected_articles,
        "found_articles": found_articles,
        "article_recall": len(found_articles) / len(expected_articles) if expected_articles else 0,
        "celex_found": celex_found,
        "num_contexts": len(retrieved_contexts),
        "contexts": retrieved_contexts,
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
    
    # Compute aggregate metrics
    article_recalls = [r["article_recall"] for r in results]
    celex_found_rate = sum(1 for r in results if r["celex_found"]) / len(results)
    avg_contexts = sum(r["num_contexts"] for r in results) / len(results)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "num_test_cases": len(results),
        "article_recall_mean": sum(article_recalls) / len(article_recalls),
        "article_recall_per_case": article_recalls,
        "celex_found_rate": celex_found_rate,
        "avg_contexts": avg_contexts,
        "per_case_results": results,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run EUR-Lex RAG evaluation")
    parser.add_argument("--dataset", default="eval/test_dataset.jsonl", help="Test dataset path")
    parser.add_argument("--output", default="eval/results/latest.json", help="Output path")
    parser.add_argument("--compare", help="Compare with baseline results file")
    args = parser.parse_args()
    
    # Run evaluation
    results = run_evaluation(args.dataset)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Article Recall (mean): {results['article_recall_mean']:.2%}")
    print(f"CELEX Found Rate:     {results['celex_found_rate']:.2%}")
    print(f"Avg Contexts:          {results['avg_contexts']:.1f}")
    print("=" * 60)
    
    # Compare with baseline if specified
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        
        print("\nCOMPARISON WITH BASELINE:")
        baseline_recall = baseline["article_recall_mean"]
        current_recall = results["article_recall_mean"]
        diff = current_recall - baseline_recall
        
        print(f"  Article Recall: {baseline_recall:.2%} → {current_recall:.2%} ({diff:+.2%})")
        
        if diff < -0.1:  # More than 10% regression
            print("  ⚠ WARNING: Significant regression detected!")
            print("  This change may have degraded retrieval quality.")
            sys.exit(1)
        else:
            print("  ✓ No significant regression")
    
    return results


if __name__ == "__main__":
    main()
EOF

echo "Created eval/run_evaluation.py"
```

---

### PHASE 3 — STEP 3: Run Baseline Evaluation

**Action:**

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Run initial evaluation to establish baseline
python eval/run_evaluation.py --output eval/results/baseline.json

# Verify results
cat eval/results/baseline.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Article Recall Mean: {data[\"article_recall_mean\"]:.2%}')
print(f'CELEX Found Rate: {data[\"celex_found_rate\"]:.2%}')
for r in data['per_case_results']:
    print(f'  {r[\"expected_celex\"]}: recall={r[\"article_recall\"]:.0%} celex={r[\"celex_found\"]}')
"
```

**Save this as the baseline.** All future changes must not regress this baseline by more than 10%.

---

### PHASE 3 — STEP 4: Commit Phase 3 Checkpoint

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add eval/ requirements.txt
git commit -m "phase 3: add evaluation framework with RAGAS-style metrics

- Add eval/test_dataset.jsonl with 10 ground-truth QA pairs
- Add eval/run_evaluation.py for retrieval evaluation
- Metrics: article_recall, celex_found_rate, context_count
- Baseline results saved to eval/results/baseline.json
- CI quality gate: reject if article_recall regresses >10%

Enables data-driven iteration on retrieval quality"
```

---

## PHASE 4: COMPLETE INDEX REBUILD

**Goal:** Run the full build pipeline with all fixes to produce the complete, comprehensive EU legislation index.

---

### PHASE 4 — STEP 1: Pre-Build Verification

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Verify all dependencies are installed
pip list | grep -E "eurlxp|chunkweaver|rank_bm25|sentence-transformers|faiss|huggingface_hub|beautifulsoup4"

# Verify Phase 0-3 code is working
python3 -c "
from scripts.build_index import fetch_document_xhtml, parse_html_to_chunks, query_all_documents
from app.bm25_store import BM25Store
from app.hybrid_search import HybridSearcher
print('All imports OK')
"

# Run evaluation to confirm baseline
python eval/run_evaluation.py --output eval/results/pre_build.json
```

---

### PHASE 4 — STEP 2: Full Build (with checkpointing)

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Create checkpoint before build
CHECKPOINT_FILE="CHECKPOINT_phase4_$(date +%Y%m%d_%H%M%S).txt"
echo "Phase 4 build starting at $(date)" > "$CHECKPOINT_FILE"
git log --oneline -1 >> "$CHECKPOINT_FILE"

# Run the build with limited chunk count first (test)
MAX_CHUNKS=50000 HF_TOKEN=$HF_TOKEN python3 scripts/build_index.py 2>&1 | tee build_log_phase4.txt

# Check build succeeded
if [ $? -eq 0 ]; then
    echo "Build succeeded" >> "$CHECKPOINT_FILE"
else
    echo "Build FAILED" >> "$CHECKPOINT_FILE"
    echo "Rollback required!"
fi
```

---

### PHASE 4 — STEP 3: Post-Build Verification

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
source ~/Desktop/EUProjects/.venv/bin/activate

# Check new dataset stats
python3 -c "
from huggingface_hub import hf_hub_download
import sqlite3, os

path = hf_hub_download(repo_id='NedAktovOps/eurlex-chat-data',
                       filename='chunks.db', repo_type='dataset')
conn = sqlite3.connect(path)
total = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
celexes = conn.execute('SELECT COUNT(DISTINCT celex) FROM chunks').fetchone()[0]
null_arts = conn.execute('SELECT COUNT(*) FROM chunks WHERE article IS NULL').fetchone()[0]
print(f'Total chunks: {total}')
print(f'Unique CELEXes: {celexes}')
print(f'NULL articles: {null_arts} ({100*null_arts/total:.1f}%)')

# Check for previously missing documents
for c in ['32024R1689', '32016R0679', '32023L0970']:
    cnt = conn.execute('SELECT COUNT(*) FROM chunks WHERE celex = ?', (c,)).fetchone()[0]
    print(f'{c}: {cnt} chunks')
conn.close()
"

# Run evaluation on new index
python eval/run_evaluation.py --output eval/results/post_build.json --compare eval/results/baseline.json
```

---

### PHASE 4 — STEP 4: Commit Phase 4 Checkpoint

```bash
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
git add .
git commit -m "phase 4: complete rebuild — full EU legislation index

- All phases 0-3 integrated and verified
- New index with comprehensive coverage (all resource types, all dates)
- Structure-aware chunking (chunkweaver LEGAL_EU)
- Hybrid search (BM25 + FAISS + RRF + reranker)
- Evaluation framework with baseline established

Total chunks: $(sqlite3 data/chunks.db 'SELECT COUNT(*) FROM chunks')
Total CELEXes: $(sqlite3 data/chunks.db 'SELECT COUNT(DISTINCT celex) FROM chunks')"
```

---

## CI/CD INTEGRATION

Add a GitHub Actions workflow for automated evaluation:

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation

on:
  push:
    branches: [main, feature/complete-rebuild]
  pull_request:

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - run: pip install -e .
      - name: Run evaluation
        run: python eval/run_evaluation.py --output eval/results/ci.json
      - name: Quality gate
        run: |
          python3 -c "
          import json
          results = json.load(open('eval/results/ci.json'))
          baseline = json.load(open('eval/results/baseline.json'))
          
          recall = results['article_recall_mean']
          base_recall = baseline['article_recall_mean']
          
          print(f'Article Recall: {recall:.2%} (baseline: {base_recall:.2%})')
          
          if recall < base_recall - 0.1:
            print('FAIL: Recall regressed by >10%')
            exit(1)
          else:
            print('PASS: No significant regression')
          "
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval/results/
```

---

## COMPLETE ROLLBACK PROCEDURES

If any phase fails catastrophically:

### Full Repository Rollback
```bash
# Restore from pre-flight backup
rsync -av --delete \
    /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat-backups/<timestamp>/ \
    /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat/

# Restore HuggingFace dataset from backup
cd /home/nedaktov/Desktop/EUProjects/eur-lex-ai-chat
python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN'))

# Upload backed-up files
api.upload_file(
    path_or_fileobj='hf_dataset/chunks.db',
    path_in_repo='chunks.db',
    repo_id='NedAktovOps/eurlex-chat-data',
    repo_type='dataset',
)
print('Dataset restored')
"
```

### Phase-Specific Rollback

```bash
# Phase 0 rollback
cp scripts/build_index.py.backup-phase0 scripts/build_index.py
git checkout -- app/eurlex_fetcher.py

# Phase 1 rollback  
cp scripts/build_index.py.backup-phase1 scripts/build_index.py

# Phase 2 rollback
cp app/search.py.backup-phase2 app/search.py

# Phase 3 rollback
git checkout -- eval/
```

---

## MASTER CHECKLIST

Execute in order. Mark DONE as you complete each step.

### Pre-Flight
- [ ] CHECKPOINT 0.0: Full repository backup
- [ ] CHECKPOINT 0.1: Document current state
- [ ] CHECKPOINT 0.2: Git worktree/branch created
- [ ] CHECKPOINT 0.3: Environment verified

### Phase 0: Build Pipeline
- [ ] STEP 1: eurlxp installed and verified
- [ ] STEP 2: Fetcher wrapper created and tested
- [ ] STEP 3: fetch_document_xhtml() replaced
- [ ] STEP 4: SPARQL query expanded (all resource types)
- [ ] STEP 5: Fetch + parse pipeline tested (4/5 docs pass)
- [ ] STEP 6: SPARQL query returns 30K+ documents
- [ ] STEP 7: Phase 0 committed

### Phase 1: Chunking
- [ ] STEP 1: chunkweaver installed and verified
- [ ] STEP 2: parse_html_to_chunks() replaced with chunkweaver
- [ ] STEP 3: Chunking quality tested (NULL rate < 5%)
- [ ] STEP 4: Phase 1 committed

### Phase 2: Hybrid Search
- [ ] STEP 1: BM25 store created and tested
- [ ] STEP 2: Hybrid search module created (RRF)
- [ ] STEP 3: search.py updated to use hybrid
- [ ] STEP 4: Hybrid search tested
- [ ] STEP 5: Phase 2 committed

### Phase 3: Evaluation
- [ ] STEP 1: Test dataset created (10 QA pairs)
- [ ] STEP 2: Evaluation script created
- [ ] STEP 3: Baseline evaluation run
- [ ] STEP 4: Phase 3 committed

### Phase 4: Rebuild
- [ ] STEP 1: Pre-build verification passed
- [ ] STEP 2: Full build completed (with logging)
- [ ] STEP 3: Post-build verification passed
- [ ] STEP 4: Phase 4 committed

### Final
- [ ] CI workflow added
- [ ] Documentation updated
- [ ] README updated with new architecture

---

## KEY FILES MODIFIED

| File | Phase | Change |
|------|-------|--------|
| `scripts/build_index.py` | 0, 1 | Replace fetch + parse with eurlxp + chunkweaver |
| `app/eurlex_fetcher.py` | 0 | NEW — Cellar RDF traversal wrapper |
| `app/search.py` | 2 | Replace pure FAISS with hybrid search |
| `app/bm25_store.py` | 2 | NEW — BM25 index with pickle persistence |
| `app/hybrid_search.py` | 2 | NEW — RRF fusion + reranking |
| `requirements.txt` | 0, 1, 2 | Add eurlxp, chunkweaver, rank_bm25 |
| `eval/run_evaluation.py` | 3 | NEW — Evaluation framework |
| `eval/test_dataset.jsonl` | 3 | NEW — Ground-truth QA pairs |
| `.github/workflows/rag-eval.yml` | CI | NEW — Automated evaluation |

---

## DEPENDENCIES SUMMARY

```
eurlxp>=0.6.0          # Cellar RDF traversal + EUR-Lex fetching
chunkweaver>=0.2.0     # Structure-aware legal document chunking
rank_bm25>=0.2.2       # BM25 sparse retrieval
sentence-transformers   # Query embedding + cross-encoder reranker
faiss-cpu              # Dense vector search
huggingface_hub        # Dataset upload/download
beautifulsoup4          # HTML parsing (fallback)
ragas                  # Evaluation framework (optional, for full RAGAS)
```

---

*Document generated: 2026-05-25*
*Plan version: 1.0*
*Status: READY FOR EXECUTION*