# EUR-Lex AI Chat — Full Implementation Plan

**Project:** EUR-Lex AI Chat — Ask questions about EU law in plain language, get answers with citations to real EUR-Lex documents.
**Build target:** ~15 hours total (one-time, laptop-based)
**Ongoing cost:** $0.00/month — runs 100% autonomously after deployment

---

## Overview

### Architecture

```
DATA FLOW (fully automated):

EUR-Lex Cellar SPARQL (public)
     │ GitHub Actions queries daily
     ▼
Cellar REST RDF → XHTML (public, no auth)
     │ GitHub Actions downloads via RDF graph traversal
     ▼
GitHub Actions Runner (free, 7GB RAM, 2-core CPU)
     │ pip install → chunk → embed (sentence-transformers ONNX) → merge
     ▼
HuggingFace Hub Dataset (public, free storage)
     │ vectors.npy + chunks.json
     │ ▲                        │
     │ │                        ▼
     │ │               Render (512MB, free)
     │ │               ├── startup: download from HF Hub
     │ │               ├── hourly: check HF Hub for updates via /refresh
     │ │               └── live: FastAPI + numpy KNN (brute-force cosine similarity)
     │ │                        │
     │ │                        ▼
     │ │              ┌─────────────────┐
     │ └──────────────┤  Groq API        │
     │                │  (Llama 3.3 70B) │
     │                └────────┬─────────┘
     │                         │
USER FLOW:                     │
[Google Search]                │
     │ user finds FAQ/blog post
     ▼
[Browser] ─► [Vercel: Astro site] ─► [Render: FastAPI API] ─► [Groq]
              Static HTML pages       /chat endpoint         answer
              + React chat island     numpy KNN + RAG

KEEP ALIVE:
[cron-job.org] ──► [Render /health] every 5 min
[cron-job.org] ──► [Render /refresh] every 60 min
```

### Files Layout

```
eur-lex-ai-chat/
├── backend/
│   ├── main.py                 # FastAPI: /chat, /health, /refresh
│   ├── search.py               # numpy KNN search over pre-loaded vectors
│   ├── rag.py                  # Build prompt, call Groq, parse citations
│   ├── data_loader.py          # Download index from HF Hub at startup
│   ├── rate_limit.py           # Per-IP + global rate limiting
│   ├── requirements.txt        # fastapi, uvicorn, numpy, huggingface_hub, httpx
│   └── startup.sh              # Render entry point
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── index.astro         # Landing page (SEO + chat island)
│   │   │   ├── faq.astro           # FAQ page (JSON-LD)
│   │   │   └── blog/
│   │   │       ├── index.astro     # Blog listing
│   │   │       └── posts/          # Markdown blog posts
│   │   ├── components/
│   │   │   ├── ChatWidget.jsx      # React chat island
│   │   │   └── SeoHead.astro      # JSON-LD + OG tags
│   │   └── layouts/
│   │       └── Base.astro          # Main layout
│   ├── astro.config.mjs
│   ├── tailwind.config.mjs
│   └── package.json
├── scripts/
│   ├── build_index.py          # Laptop: FULL build from scratch (one-time)
│   └── update_index.py         # GitHub Actions: INCREMENTAL update (daily)
├── .github/
│   └── workflows/
│       └── update-index.yml    # GitHub Actions workflow
├── proposal.md
├── implementation-plan.md
└── README.md
```

---

## Phase 0: Pre-flight Checklist

**Goal:** Verify every tool, API key, and service is working before we start building.

### Step 0.1 — Verify Node.js (needs >=22.12.0 for Astro 5)

```bash
source ~/.nvm/nvm.sh
nvm use v22.22.3
node --version   # must show v22.22.3
npm --version    # must show 10.x or higher
```

If nvm is not found, install it:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.5/install.sh | bash
source ~/.bashrc
nvm install v22.22.3
nvm use v22.22.3
```

### Step 0.2 — Verify Python venv + packages

```bash
source ~/Desktop/EUProjects/.venv/bin/activate
python3 --version   # must show 3.12.x
pip list | grep -E "fastapi|uvicorn|numpy|httpx|huggingface|polars|eurlxp|beautifulsoup4|pymupdf|tqdm|pydantic"
```

Expected: fastapi, uvicorn, numpy, httpx, huggingface_hub, polars, eurlxp, beautifulsoup4, pymupdf, tqdm, pydantic.

If any are missing:
```bash
pip install fastapi uvicorn numpy httpx huggingface_hub polars eurlxp[sparql] beautifulsoup4 pymupdf tqdm
```

### Step 0.3 — Install sentence-transformers (needed for Phase 1)

```bash
source ~/Desktop/EUProjects/.venv/bin/activate
pip install sentence-transformers
```

This installs torch (~2GB). It's a one-time install for the laptop build. The build script runs once.

### Step 0.4 — Verify GROQ_API_KEY

```bash
source ~/Desktop/EUProjects/.venv/bin/activate
python3 -c "
import os, httpx
key = os.environ.get('GROQ_API_KEY')
if not key: print('MISSING: GROQ_API_KEY not found in env')
else:
    r = httpx.get('https://api.groq.com/openai/v1/models',
        headers={'Authorization': f'Bearer {key}'}, timeout=10)
    assert r.status_code == 200, f'GROQ API returned {r.status_code}'
    models = [m['id'] for m in r.json()['data']]
    print(f'GROQ OK — {len(models)} models available')
    print(f'  Default: llama-3.3-70b-versatile')
"
```

If `GROQ_API_KEY` is missing, it should be in `/home/nedaktov/Desktop/NedCode3/.env`:
```bash
source /home/nedaktov/Desktop/NedCode3/.env
echo "GROQ_API_KEY is set: ${GROQ_API_KEY:0:10}..."
```

### Step 0.5 — Verify HuggingFace Hub access

```bash
source ~/Desktop/EUProjects/.venv/bin/activate
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
# Test anonymous read access
try:
    api.dataset_info('hf-internal-testing/dummy_dataset')
    print('HF Hub accessible (anonymous)')
except Exception as e:
    print(f'HF Hub error: {e}')
"
```

### Step 0.6 — Create HuggingFace token and log in

We need a write token to upload the vector index. Create one at: https://huggingface.co/settings/tokens

Create a token with "write" permissions. Then:
```bash
source ~/Desktop/EUProjects/.venv/bin/activate
huggingface-cli login --token YOUR_HF_TOKEN
# Or use the Python API:
python3 -c "
from huggingface_hub import login
login(token='YOUR_HF_TOKEN', add_to_git_credential=True)
print('Logged in to HuggingFace')
"
```

### Step 0.7 — Create project directories

```bash
mkdir -p ~/Desktop/EUProjects/eur-lex-ai-chat/{backend,frontend/src/{pages,components,layouts,pages/blog/posts},scripts,.github/workflows,data}
```

### Step 0.8 — Verify EUR-Lex SPARQL endpoint + eurlxp

```bash
source ~/Desktop/EUProjects/.venv/bin/activate
python3 -c "
from eurlxp import get_documents
docs = get_documents(types=['REG'], limit=3)
print(f'SPARQL OK — got {len(docs)} documents')
for d in docs:
    print(f'  {d[\"celex\"]} ({d[\"type\"]}) — {d[\"date\"]}')
"
```

Expected output:
```
SPARQL OK — got 3 documents
  32025R1355R(02) (REG) — 2026-03-27
  ...
```

### Step 0.9 — Verify EUR-Lex HTML fetch + parse

⚠️ **Known issue:** `eurlxp.parse_html()` has a Polars schema bug — the `modifier` field is conditionally included, causing schema mismatch. Use the internal parser directly with a fixed schema instead.

```bash
source ~/Desktop/EUProjects/.venv/bin/activate
python3 -c "
from eurlxp import get_html_by_celex_id
from eurlxp.parser import _parse_html_with_beautifulsoup as internal_parse
import polars as pl

html = get_html_by_celex_id('32019R0947', language='en')
print(f'HTML fetched: {len(html)} bytes')

results = internal_parse(html)
print(f'Parsed: {len(results)} elements')

# Build records with fixed schema (workaround for eurlxp Polars bug)
records = []
for r in results:
    records.append({
        'text': r.text,
        'type': r.item_type,
        'ref': str(r.ref),
        'modifier': r.modifier,
        'document': r.context.document,
        'article': r.context.article,
        'article_subtitle': r.context.article_subtitle,
        'paragraph': r.context.paragraph,
        'group': r.context.group,
        'section': r.context.section,
    })

df = pl.DataFrame(records, schema={
    'text': pl.Utf8, 'type': pl.Utf8, 'ref': pl.Utf8,
    'modifier': pl.Utf8, 'document': pl.Utf8, 'article': pl.Utf8,
    'article_subtitle': pl.Utf8, 'paragraph': pl.Utf8,
    'group': pl.Utf8, 'section': pl.Utf8,
})
print(f'DataFrame: {len(df)} rows')
texts = df.filter(pl.col('type') == 'text')
print(f'Text elements: {len(texts)}')
print(texts.head(5))
"
```

### Step 0.10 — Pre-flight checklist summary

| # | Check | Status | Fix if failing |
|---|-------|--------|----------------|
| 1 | Node.js >=22.12.0 | | `nvm install v22.22.3` |
| 2 | Python venv active | | `source ~/Desktop/EUProjects/.venv/bin/activate` |
| 3 | All Python packages installed | | `pip install -r requirements.txt` |
| 4 | sentence-transformers installed | | `pip install sentence-transformers` |
| 5 | GROQ_API_KEY in env | | Add to .env and source it |
| 6 | HF Hub accessible | | Check internet, try again |
| 7 | HF write token + login | | Create token at huggingface.co/settings/tokens |
| 8 | Project directories exist | | Run mkdir command above |
| 9 | EUR-Lex SPARQL works | | Check internet, try later |
| 10 | EUR-Lex HTML fetch works | | Check eurlxp version, try again |

---

## Phase 1: Build the Data Index

**Goal:** Create the vector index of all EU legislative acts. This runs ONCE on your laptop (~3-5 hours). Output: vectors.npy + chunks.json uploaded to HuggingFace Hub.

**Scope:** All regulations (REG), directives (DIR), implementing regulations (REG_IMPL), and implementing directives (DIR_IMPL) with English titles, published from 2000-01-01 onwards. This covers ~25 years of EU legislation — the laws people actually search for.

### Step 1.1 — Write `scripts/build_index.py`

This is the main build script. Save it as `~/Desktop/EUProjects/eur-lex-ai-chat/scripts/build_index.py`:

```python
#!/usr/bin/env python3
"""
build_index.py — One-time full build of the EUR-Lex vector index.

Pipeline:
  1. SPARQL query → list of CELEX IDs with metadata
  2. For each CELEX ID: RDF graph traversal → XHTML content
  3. Parse HTML → structured text via eurlxp parser
  4. Chunk into passages (by article/paragraph)
  5. Embed with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
  6. Upload vectors.npy + chunks.json to HuggingFace Hub

Usage:
  source ~/Desktop/EUProjects/.venv/bin/activate
  HF_TOKEN=hf_yourtoken python3 scripts/build_index.py

Output:
  data/vectors.npy     — numpy array of shape (N, 384), float32
  data/chunks.json     — list of dicts with text + metadata
  data/last_updated.txt — ISO timestamp of build
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

# Document types to include
DOC_TYPES = [
    "REG",       # Regulations
    "DIR",       # Directives
    "REG_IMPL",  # Implementing regulations
    "DIR_IMPL",  # Implementing directives
]

# Date range: 2000-01-01 onwards (covers 25+ years of EU law)
FROM_DATE = "2000-01-01"

# How many documents to fetch per SPARQL page
SPARQL_PAGE_SIZE = 1000

# How many parallel download workers
DOWNLOAD_WORKERS = 20

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, runs on CPU

# HuggingFace dataset name
HF_DATASET_NAME = "eurlex-chat-data"

# ── Step 1: SPARQL Query ───────────────────────────────────────────────────

def make_type_filter(types):
    """Build SPARQL FILTER for document types."""
    type_uris = [
        f"<http://publications.europa.eu/resource/authority/resource-type/{t}>"
        for t in types
    ]
    return " ||\n    ".join(f"?type = {uri}" for uri in type_uris)


def query_all_documents():
    """Query all documents matching our criteria via SPARQL with pagination."""
    logger.info("Querying SPARQL for documents...")

    all_docs = []
    offset = 0

    while True:
        type_filter = make_type_filter(DOC_TYPES)
        query = f"""
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>

SELECT DISTINCT ?doc ?type ?celex ?title ?date
WHERE {{
    ?doc cdm:work_has_resource-type ?type .
    ?doc cdm:resource_legal_id_celex ?celex .
    ?doc dc:title ?title .
    ?doc cdm:work_date_document ?date .
    FILTER({type_filter})
    FILTER(LANG(?title) = "en")
    FILTER(?date >= "{FROM_DATE}"^^xsd:date)
}}
ORDER BY DESC(?date)
OFFSET {offset}
LIMIT {SPARQL_PAGE_SIZE}
"""
        r = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        bindings = data["results"]["bindings"]

        if not bindings:
            break

        for b in bindings:
            all_docs.append({
                "celex": b["celex"]["value"],
                "title": b["title"]["value"],
                "date": b["date"]["value"],
                "type": b["type"]["value"].split("/")[-1],
                "cellar_url": b["doc"]["value"],
            })

        offset += SPARQL_PAGE_SIZE
        logger.info(f"  Fetched {len(all_docs)} documents so far...")

        if len(bindings) < SPARQL_PAGE_SIZE:
            break

    logger.info(f"Total documents found: {len(all_docs)}")
    return all_docs


# ── Step 2: Download XHTML Content ─────────────────────────────────────────

def fetch_document_xhtml(doc):
    """Fetch a document's XHTML content via Cellar RDF graph traversal.

    Uses the same approach as eurlxp's _fetch_html_via_sparql:
    1. Get work RDF → find English expression
    2. Get expression RDF → find XHTML manifestation
    3. Download XHTML
    """
    celex = doc["celex"]
    try:
        # Step 1: Get work RDF graph
        work_url = f"http://publications.europa.eu/resource/celex/{celex}?language=eng"
        r = requests.get(work_url, timeout=30)
        r.raise_for_status()

        # Parse RDF/XML to find English expression
        from xml.etree import ElementTree as ET
        root = ET.fromstring(r.content)

        # Namespaces
        ns = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "cdm": "http://publications.europa.eu/ontology/cdm#",
        }

        # Find the English expression (ENG)
        expressions = root.findall(".//cdm:work_has_expression", ns)
        expression_url = None
        for expr in expressions:
            resource = expr.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
            if resource and resource.endswith(".ENG"):
                expression_url = resource
                break

        if not expression_url and expressions:
            # Fallback to first expression
            resource = expressions[0].get(
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
            )
            if resource:
                expression_url = resource

        if not expression_url:
            logger.warning(f"  No expression found for {celex}")
            return None

        # Step 2: Get expression RDF to find XHTML manifestation
        r2 = requests.get(expression_url, timeout=30)
        r2.raise_for_status()
        expr_root = ET.fromstring(r2.content)

        manifestations = expr_root.findall(
            ".//cdm:expression_manifested_by_manifestation", ns
        )
        xhtml_url = None
        for manif in manifestations:
            resource = manif.get(
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
            )
            if resource and resource.endswith(".xhtml"):
                xhtml_url = resource
                break

        if not xhtml_url and manifestations:
            # Fallback to .fmx4 (Formex 4 format)
            for manif in manifestations:
                resource = manif.get(
                    "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
                )
                if resource and resource.endswith(".fmx4"):
                    xhtml_url = resource
                    break

        if not xhtml_url:
            logger.warning(f"  No XHTML manifestation for {celex}")
            return None

        # Step 3: Download the XHTML content
        r3 = requests.get(
            xhtml_url,
            headers={"Accept": "application/xhtml+xml, text/html"},
            timeout=30,
        )
        r3.raise_for_status()
        html = r3.text

        if len(html) < 100:
            logger.warning(f"  Empty content for {celex}")
            return None

        return html

    except requests.RequestException as e:
        logger.warning(f"  HTTP error for {celex}: {e}")
        return None
    except ET.ParseError as e:
        logger.warning(f"  XML parse error for {celex}: {e}")
        return None
    except Exception as e:
        logger.warning(f"  Unexpected error for {celex}: {e}")
        return None


# ── Step 3: Parse HTML → Structured Text ───────────────────────────────────

def parse_html_to_chunks(html, celex_id, title):
    """Parse EUR-Lex HTML into text chunks using eurlxp internal parser.

    ⚠️ Uses _parse_html_with_beautifulsoup directly instead of parse_html()
    because eurlxp's parse_html() has a Polars schema bug (conditional modifier
    field causes schema mismatch). This workaround uses a fixed schema.
    """
    from eurlxp.parser import _parse_html_with_beautifulsoup as internal_parse
    import polars as pl

    try:
        results = internal_parse(html)
    except Exception as e:
        logger.warning(f"  Parse error for {celex_id}: {e}")
        return []

    if not results:
        return []

    # Convert to DataFrame with fixed schema (bypasses eurlxp bug)
    records = []
    for r in results:
        records.append({
            'text': r.text,
            'type': r.item_type,
            'ref': str(r.ref),
            'modifier': r.modifier,
            'document': r.context.document,
            'article': r.context.article,
            'article_subtitle': r.context.article_subtitle,
            'paragraph': r.context.paragraph,
            'group': r.context.group,
            'section': r.context.section,
        })

    df = pl.DataFrame(records, schema={
        'text': pl.Utf8, 'type': pl.Utf8, 'ref': pl.Utf8,
        'modifier': pl.Utf8, 'document': pl.Utf8, 'article': pl.Utf8,
        'article_subtitle': pl.Utf8, 'paragraph': pl.Utf8,
        'group': pl.Utf8, 'section': pl.Utf8,
    })

    if len(df) == 0:
        return []

    chunks = []
    current_article = None
    current_text = []

    # Process rows in order
    for row in df.to_dicts():
        text = row.get("text", "").strip()
        row_type = row.get("type", "")
        article = row.get("article", None)

        if not text:
            continue

        # Skip very short fragments (titles, headers)
        if len(text) < 40 and row_type in ("doc-title", "art-title", "art-subtitle", "group-title"):
            if row_type == "doc-title":
                pass  # We already have the title from SPARQL
            elif row_type == "art-title" and article:
                # Finalize previous article chunk
                if current_text:
                    chunk_text = " ".join(current_text)
                    if len(chunk_text) > 50:
                        chunks.append({
                            "text": chunk_text,
                            "celex": celex_id,
                            "title": title,
                            "article": current_article,
                            "type": "article",
                        })
                    current_text = []
                current_article = article
            continue

        # Skip pure metadata rows
        if row_type in ("doc-title", "art-subtitle", "group-title", "section-title"):
            continue

        # Skip notice/note rows
        if row.get("modifier") in ("note", "signatory"):
            continue

        current_text.append(text)

        # If text is a full paragraph and we have enough context, finalize
        if len(text) > 100 and current_text:
            # Check if this is a natural break (new article, new section)
            pass

    # Flush remaining text
    if current_text:
        chunk_text = " ".join(current_text)
        if len(chunk_text) > 50:
            chunks.append({
                "text": chunk_text,
                "celex": celex_id,
                "title": title,
                "article": current_article,
                "type": "article",
            })

    # If no article-based chunks found, fall back to paragraph splitting
    if not chunks:
        # Try to split by paragraphs (double newlines)
        paragraphs = [
            p.strip() for p in html.replace("</p>", "\n\n")
            .replace("</div>", "\n\n")
            .replace("<br/>", "\n")
            .split("\n\n")
            if len(p.strip()) > 80
        ]
        for i, para in enumerate(paragraphs):
            chunks.append({
                "text": para,
                "celex": celex_id,
                "title": title,
                "article": None,
                "type": "paragraph",
            })

    return chunks


# ── Step 4: Embed ──────────────────────────────────────────────────────────

def embed_chunks(all_chunks, batch_size=128):
    """Embed all chunks using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model loaded")

    texts = [c["text"] for c in all_chunks]
    all_embeddings = []

    logger.info(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embeddings)

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embeddings).astype(np.float32)


# ── Step 5: Upload to HuggingFace Hub ──────────────────────────────────────

def upload_to_hub(vectors, chunks, dataset_name, token):
    """Upload vectors.npy + chunks.json + last_updated.txt to HF Hub."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{dataset_name}"

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        logger.info(f"HF dataset repo: {repo_id}")
    except Exception as e:
        logger.warning(f"Repo creation warning (may already exist): {e}")

    # Save vectors
    vectors_path = os.path.join(DATA_DIR, "vectors.npy")
    np.save(vectors_path, vectors)
    logger.info(f"Saved vectors: {vectors.shape} ({os.path.getsize(vectors_path) / 1e6:.1f} MB)")

    # Save chunks
    chunks_path = os.path.join(DATA_DIR, "chunks.json")
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)
    logger.info(f"Saved chunks: {len(chunks)} items ({os.path.getsize(chunks_path) / 1e6:.1f} MB)")

    # Save timestamp
    ts_path = os.path.join(DATA_DIR, "last_updated.txt")
    ts = datetime.now(timezone.utc).isoformat()
    with open(ts_path, "w") as f:
        f.write(ts)

    # Upload files
    api.upload_file(
        repo_id=repo_id,
        path_in_repo="vectors.npy",
        path_or_fileobj=vectors_path,
        repo_type="dataset",
        token=token,
    )
    api.upload_file(
        repo_id=repo_id,
        path_in_repo="chunks.json",
        path_or_fileobj=chunks_path,
        repo_type="dataset",
        token=token,
    )
    api.upload_file(
        repo_id=repo_id,
        path_in_repo="last_updated.txt",
        path_or_fileobj=ts_path,
        repo_type="dataset",
        token=token,
    )

    logger.info(f"Uploaded to HF Hub: {repo_id}")
    return repo_id


# ── Main Pipeline ──────────────────────────────────────────────────────────

def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable required")
        logger.error("Usage: HF_TOKEN=hf_yourtoken python3 scripts/build_index.py")
        return

    total_start = time.time()

    # Step 1: Query SPARQL
    docs = query_all_documents()
    if not docs:
        logger.error("No documents found — SPARQL query returned empty")
        return
    logger.info(f"Documents to process: {len(docs)}")

    # Step 2: Download all documents in parallel
    logger.info(f"Downloading documents ({DOWNLOAD_WORKERS} workers)...")
    html_results = {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_document_xhtml, doc): doc for doc in docs
        }
        for future in tqdm(as_completed(future_map), total=len(docs), desc="Downloading"):
            doc = future_map[future]
            try:
                html = future.result()
                if html:
                    html_results[doc["celex"]] = html
            except Exception as e:
                logger.debug(f"Failed {doc['celex']}: {e}")

    logger.info(f"Downloaded {len(html_results)}/{len(docs)} documents successfully")

    # Step 3: Parse all documents into chunks
    logger.info("Parsing HTML into chunks...")
    all_chunks = []
    for doc in tqdm(docs, desc="Parsing"):
        celex = doc["celex"]
        html = html_results.get(celex)
        if not html:
            continue
        chunks = parse_html_to_chunks(html, celex, doc["title"])
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks produced — check parsing")
        return

    # Step 4: Embed all chunks
    vectors = embed_chunks(all_chunks)
    logger.info(f"Embedding complete: {vectors.shape}")

    # Step 5: Upload to HF Hub
    repo_id = upload_to_hub(vectors, all_chunks, HF_DATASET_NAME, hf_token)

    total_time = time.time() - total_start
    logger.info(f"Build complete in {total_time / 60:.1f} minutes")
    logger.info(f"Dataset: {repo_id}")
    logger.info(f"Documents: {len(html_results)} | Chunks: {len(all_chunks)} | Dims: {vectors.shape[1]}")


if __name__ == "__main__":
    main()
```

### Step 1.2 — Run the build

> **⏱ Estimated runtime: 3-5 hours.** This downloads ~30K-80K documents (depending on date filter) and embeds ~100K-200K chunks. You can leave it running overnight.

```bash
source ~/Desktop/EUProjects/.venv/bin/activate
cd ~/Desktop/EUProjects/eur-lex-ai-chat
HF_TOKEN=hf_your_write_token python3 scripts/build_index.py 2>&1 | tee build_log.txt
```

Expected output milestones:
```
Querying SPARQL for documents...
  Fetched 1000 documents so far...
  Fetched 2000 documents so far...
  ...
Total documents found: XXXX

Downloading documents (20 workers)...
Downloading: 100%|████████████| XXXX/XXXX [XX:XX<00:00]
Downloaded XXXX/XXXX documents successfully

Parsing HTML into chunks...
Total chunks: XXXX

Loading embedding model: all-MiniLM-L6-v2
Embedding XXXX chunks in batches of 128...
Embedding complete: (XXXX, 384)

Saved vectors: (XXXX, 384) (XX.X MB)
Saved chunks: XXXX items (XX.X MB)
Uploaded to HF Hub: yourusername/eurlex-chat-data

Build complete in XXX.X minutes
```

### Step 1.3 — Verify the upload

```bash
source ~/Desktop/EUProjects/.venv/bin/activate
python3 -c "
from huggingface_hub import hf_hub_download
import numpy as np, json

# Download vectors
vectors_path = hf_hub_download('yourusername/eurlex-chat-data', 'vectors.npy')
vectors = np.load(vectors_path)
print(f'Vectors: {vectors.shape}, dtype={vectors.dtype}')

# Download chunks
chunks_path = hf_hub_download('yourusername/eurlex-chat-data', 'chunks.json')
with open(chunks_path) as f:
    chunks = json.load(f)
print(f'Chunks: {len(chunks)}')
print(f'Sample: {chunks[0][\"text\"][:100]}...')
print(f'Source: CELEX {chunks[0][\"celex\"]} — {chunks[0][\"title\"][:60]}...')
"
```

If verification passes, **Phase 1 is complete.** The vector index is live on HuggingFace Hub.

---

## Phase 2: FastAPI Backend

**Goal:** A FastAPI server that loads the vector index from HF Hub at startup, accepts chat queries, finds relevant chunks via numpy KNN, calls Groq to generate answers with citations, and provides /health and /refresh endpoints.

### Step 2.1 — Write `backend/requirements.txt`

```
fastapi==0.136.1
uvicorn==0.47.0
numpy==2.4.6
httpx==0.28.1
huggingface_hub==1.15.0
```

### Step 2.2 — Write `backend/data_loader.py`

```python
"""Download and manage the vector index from HuggingFace Hub."""

import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
from huggingface_hub import hf_hub_download, HfApi

logger = logging.getLogger(__name__)

# HF dataset configuration
HF_USERNAME = os.environ.get("HF_USERNAME", "yourusername")
HF_DATASET = os.environ.get("HF_DATASET", "eurlex-chat-data")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

REPO_ID = f"{HF_USERNAME}/{HF_DATASET}"

# Current index in memory
_index_data = {
    "vectors": None,
    "chunks": None,
    "last_updated": None,
    "loaded_at": None,
}


def download_index():
    """Download vectors.npy + chunks.json from HF Hub into memory."""
    logger.info(f"Downloading index from {REPO_ID}...")

    try:
        vectors_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="vectors.npy",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        chunks_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="chunks.json",
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except Exception as e:
        logger.error(f"Failed to download from HF Hub: {e}")
        raise

    vectors = np.load(vectors_path)
    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    _index_data["vectors"] = vectors
    _index_data["chunks"] = chunks
    _index_data["last_updated"] = _get_last_updated()
    _index_data["loaded_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"Index loaded: {vectors.shape[0]} vectors, {len(chunks)} chunks")
    return _index_data


def _get_last_updated():
    """Get the last_updated timestamp from HF Hub."""
    try:
        ts_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="last_updated.txt",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(ts_path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def check_for_updates():
    """Check if the index has been updated on HF Hub since we loaded it."""
    current_remote = _get_last_updated()
    if current_remote and current_remote != _index_data["last_updated"]:
        logger.info(f"Remote index updated: {current_remote}")
        return True
    return False


def reload_index():
    """Re-download and reload the index."""
    return download_index()


def get_index():
    """Get the current in-memory index."""
    return _index_data


def get_stats():
    """Get index statistics."""
    data = get_index()
    return {
        "vectors": data["vectors"].shape if data["vectors"] is not None else None,
        "chunks": len(data["chunks"]) if data["chunks"] is not None else 0,
        "last_updated": data["last_updated"],
        "loaded_at": data["loaded_at"],
    }
```

### Step 2.3 — Write `backend/search.py`

```python
"""numpy KNN search over pre-loaded vectors."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def search(query_vector, top_k=10):
    """Find top-k most similar chunks using brute-force cosine similarity.

    Uses numpy dot product since vectors are L2-normalized.
    Returns list of dicts with chunk data and similarity score.
    """
    from data_loader import get_index

    index = get_index()
    vectors = index["vectors"]
    chunks = index["chunks"]

    if vectors is None or chunks is None:
        logger.error("Index not loaded")
        return []

    # Cosine similarity = dot product (vectors are normalized)
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    similarities = np.dot(vectors, query_vector.T).flatten()

    # Get top-k indices
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(similarities[idx]),
            "text": chunks[idx]["text"],
            "celex": chunks[idx]["celex"],
            "title": chunks[idx]["title"],
            "article": chunks[idx].get("article"),
        })

    return results
```

### Step 2.4 — Write `backend/rag.py`

```python
"""Build prompts and call Groq API for RAG."""

import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a legal AI assistant specialized in EU law. You help users understand EU legislation by answering their questions based on provided context from EUR-Lex documents.

Guidelines:
1. Answer based ONLY on the provided context. If the context doesn't contain enough information, say so.
2. Always cite the specific EUR-Lex document(s) you used with their CELEX numbers.
3. When citing articles, include the article number and CELEX number.
4. Keep answers clear and accessible — explain legal concepts in plain language.
5. If the user asks in a non-English language, respond in that language.
6. Do not make up legal citations or references. Only cite what's in the context.
7. Be honest about limitations — if you're unsure, say so."""


def build_prompt(query, context_chunks):
    """Build the prompt with context chunks and user query."""
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = f"[{i+1}] CELEX {chunk['celex']}"
        if chunk.get("article"):
            source += f", Article {chunk['article']}"
        context_parts.append(f"Context {i+1} ({source}):\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""Here are relevant excerpts from EU law documents:

{context_str}

Based on the above legal texts, please answer the following question:

{query}"""
    return prompt


def call_groq(prompt, max_retries=3):
    """Call Groq API with the prompt and return the response."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set")
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    for attempt in range(max_retries):
        try:
            r = httpx.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if r.status_code == 429:
                logger.warning(f"Groq rate limited (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
                continue
            logger.error(f"Groq API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None

    return None


def extract_citations(text):
    """Extract CELEX citations from the response text."""
    celex_pattern = r"CELEX\s+(\d{2,4}[A-Z0-9]+(?:\([A-Z0-9]+\))?(?:\([0-9]+\))?)"
    return re.findall(celex_pattern, text)


def answer_question(query, context_chunks):
    """Full RAG pipeline: build prompt → call Groq → return answer with citations."""
    prompt = build_prompt(query, context_chunks)
    answer = call_groq(prompt)

    if not answer:
        return {
            "answer": "Sorry, I couldn't generate an answer right now. Please try again.",
            "citations": [],
        }

    citations = extract_citations(answer)

    # Also add sources from our context
    source_citations = []
    seen = set()
    for chunk in context_chunks:
        if chunk["celex"] not in seen:
            seen.add(chunk["celex"])
            source_citations.append({
                "celex": chunk["celex"],
                "title": chunk["title"],
                "article": chunk.get("article"),
                "score": chunk["score"],
            })

    return {
        "answer": answer,
        "citations": citations,
        "sources": source_citations,
    }
```

### Step 2.5 — Write `backend/rate_limit.py`

```python
"""Per-IP rate limiting for the /chat endpoint."""

import time
from collections import defaultdict

# Rate limit configuration
MAX_REQUESTS_PER_IP = 20  # Max requests per window
WINDOW_SECONDS = 60       # Window size in seconds
MAX_GLOBAL_PER_MINUTE = 100  # Global max

_ip_counters = defaultdict(list)
_global_counters = []


def is_rate_limited(client_ip):
    """Check if a client IP has exceeded the rate limit.

    Returns True if rate limited, False if request is allowed.
    """
    now = time.time()
    window_start = now - WINDOW_SECONDS

    # Clean old entries
    _ip_counters[client_ip] = [
        t for t in _ip_counters[client_ip] if t > window_start
    ]

    # Check IP limit
    if len(_ip_counters[client_ip]) >= MAX_REQUESTS_PER_IP:
        return True

    # Check global limit
    global _global_counters
    _global_counters = [t for t in _global_counters if t > window_start]
    if len(_global_counters) >= MAX_GLOBAL_PER_MINUTE:
        return True

    # Record this request
    _ip_counters[client_ip].append(now)
    _global_counters.append(now)

    return False
```

### Step 2.6 — Write `backend/main.py`

```python
"""FastAPI application for EUR-Lex AI Chat."""

import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Global model instance — loaded once at startup, reused for all requests
_embedding_model = None


def get_embedding_model():
    """Get the singleton embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector index and embedding model on startup."""
    from data_loader import download_index

    logger.info("Starting up — loading index...")
    try:
        download_index()
        logger.info("Index loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")

    # Pre-load embedding model
    logger.info("Pre-loading embedding model...")
    get_embedding_model()
    logger.info("Embedding model loaded")

    yield
    logger.info("Shutting down")


app = FastAPI(title="EUR-Lex AI Chat API", version="1.0.0", lifespan=lifespan)

# CORS — allow frontend on Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://eurlex-chat.vercel.app",
        "http://localhost:4321",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint — used by cron-job.org keepalive."""
    from data_loader import get_stats

    stats = get_stats()
    return {
        "status": "ok",
        "index_loaded": stats["vectors"] is not None,
        "vector_count": stats["vectors"][0] if stats["vectors"] else 0,
        "chunk_count": stats["chunks"],
        "last_updated": stats["last_updated"],
        "loaded_at": stats["loaded_at"],
    }


@app.get("/refresh")
async def refresh():
    """Check for index updates on HF Hub and reload if newer."""
    from data_loader import check_for_updates, reload_index, get_stats

    try:
        has_updates = check_for_updates()
        if has_updates:
            logger.info("New index available, reloading...")
            reload_index()
            return {"status": "reloaded", "message": "Index updated successfully"}
        else:
            stats = get_stats()
            return {"status": "current", "message": "Index is up to date"}
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/chat")
async def chat(request: Request):
    """Main chat endpoint. Accepts {query: string}, returns {answer, sources}."""
    from rate_limit import is_rate_limited
    from search import search
    from rag import answer_question

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit check
    if is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 20 requests per minute per IP.",
        )

    # Parse request
    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    if len(query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 chars)")

    # Embed query using the singleton model
    model = get_embedding_model()
    query_vector = model.encode(query, normalize_embeddings=True)

    # Search
    chunks = search(query_vector, top_k=10)
    if not chunks:
        return {
            "answer": "I don't have enough information to answer that question. Try asking about a specific EU regulation or directive.",
            "citations": [],
            "sources": [],
        }

    # Generate answer via RAG
    result = answer_question(query, chunks)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
```

### Step 2.7 — Write `backend/startup.sh`

```bash
#!/bin/bash
# Render entry point — downloads index and starts uvicorn
set -e

cd "$(dirname "$0")"

echo "=== EUR-Lex AI Chat Backend Startup ==="
echo "Python: $(python3 --version)"

# The FastAPI app loads the index at startup via lifespan
echo "Starting uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
```

Make it executable:
```bash
chmod +x ~/Desktop/EUProjects/eur-lex-ai-chat/backend/startup.sh
```

### Step 2.8 — Test the backend locally

```bash
source ~/Desktop/EUProjects/.venv/bin/activate
cd ~/Desktop/EUProjects/eur-lex-ai-chat/backend

# Start server in background
uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 5

# Test health endpoint
curl -s http://localhost:8000/health | python3 -m json.tool

# Test chat endpoint
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements of the GDPR?"}' \
  | python3 -m json.tool

# Kill the server
kill %1 2>/dev/null
```

Expected `/health` response:
```json
{
  "status": "ok",
  "index_loaded": true,
  "vector_count": 123456,
  "chunk_count": 123456,
  "last_updated": "2026-05-20T12:00:00+00:00",
  "loaded_at": "2026-05-20T12:00:00+00:00"
}
```

### Step 2.9 — Create `backend/render.yaml` for Render deployment

```yaml
# render.yaml — Render Blueprint for EUR-Lex AI Chat backend
services:
  - type: web
    name: eurlex-chat-api
    runtime: python
    repo: https://github.com/yourusername/eur-lex-ai-chat
    branch: main
    buildCommand: pip install -r backend/requirements.txt
    startCommand: cd backend && ./startup.sh
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: HF_TOKEN
        sync: false
      - key: HF_USERNAME
        value: yourusername
      - key: HF_DATASET
        value: eurlex-chat-data
      - key: PYTHON_VERSION
        value: 3.12.3
    healthCheckPath: /health
```

We'll set the environment variables in the Render dashboard (not in the yaml for security).

---

## Phase 3: Astro Frontend

**Goal:** A fully SEO-optimized Astro website with a React chat island. Ships zero-JS HTML by default. Chat widget is only interactive JS on the page.

### Step 3.1 — Scaffold Astro project

```bash
source ~/.nvm/nvm.sh
nvm use v22.22.3
cd ~/Desktop/EUProjects/eur-lex-ai-chat

# Create Astro project in frontend/ directory
npm create astro@latest frontend -- --template basics --typescript --no-install --no-git

cd frontend

# Install dependencies
npm install astro @astrojs/react @astrojs/tailwind @astrojs/sitemap tailwindcss react react-dom

# Add integrations
npx astro add react --yes
npx astro add tailwind --yes
npx astro add sitemap --yes
```

### Step 3.2 — Write `frontend/astro.config.mjs`

```javascript
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwind from "@astrojs/tailwind";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://eurlex-chat.vercel.app",
  integrations: [
    react(),
    tailwind(),
    sitemap({
      changefreq: "weekly",
      priority: 0.7,
      lastmod: new Date(),
    }),
  ],
});
```

### Step 3.3 — Write `frontend/tailwind.config.mjs`

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,mdx,tsx}"],
  theme: {
    extend: {
      colors: {
        eu: {
          blue: "#003399",
          gold: "#FFCC00",
          navy: "#002266",
          light: "#F0F4FF",
        },
      },
    },
  },
  plugins: [],
};
```

### Step 3.4 — Write `frontend/src/layouts/Base.astro`

```astro
---
// Base layout — used by all pages
// Includes: SEO meta tags, JSON-LD, nav, footer
export interface Props {
  title: string;
  description: string;
  canonical?: string;
  ogType?: string;
  jsonLd?: Record<string, any>;
}

const {
  title,
  description,
  canonical = "https://eurlex-chat.vercel.app",
  ogType = "website",
  jsonLd,
} = Astro.props;
---

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title} — EUR-Lex AI Chat</title>
    <meta name="description" content={description} />
    <link rel="canonical" href={canonical} />

    <!-- Open Graph -->
    <meta property="og:title" content={title} />
    <meta property="og:description" content={description} />
    <meta property="og:url" content={canonical} />
    <meta property="og:type" content={ogType} />
    <meta property="og:site_name" content="EUR-Lex AI Chat" />
    <meta property="og:locale" content="en_US" />

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content={title} />
    <meta name="twitter:description" content={description} />

    {/* JSON-LD */}
    {jsonLd && <script type="application/ld+json" set:html={JSON.stringify(jsonLd)} />}

    {/* Google Search Console verification */}
    <meta name="google-site-verification" content="YOUR_VERIFICATION_CODE" />
  </head>
  <body class="bg-white text-gray-900 min-h-screen flex flex-col">
    <!-- Navigation -->
    <nav class="bg-eu-blue text-white shadow-lg">
      <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <a href="/" class="text-xl font-bold tracking-tight">
          EUR-Lex AI Chat
        </a>
        <div class="flex gap-4 text-sm">
          <a href="/" class="hover:text-eu-gold transition">Home</a>
          <a href="/faq" class="hover:text-eu-gold transition">FAQ</a>
          <a href="/blog" class="hover:text-eu-gold transition">Blog</a>
        </div>
      </div>
    </nav>

    <!-- Main content -->
    <main class="flex-1">
      <slot />
    </main>

    <!-- Footer -->
    <footer class="bg-gray-50 border-t py-8 mt-12">
      <div class="max-w-6xl mx-auto px-4 text-center text-sm text-gray-600">
        <p class="mb-2">
          EUR-Lex AI Chat — An open-source tool for exploring EU law.
          Uses data from
          <a href="https://eur-lex.europa.eu/" class="text-eu-blue underline" target="_blank" rel="noopener">EUR-Lex</a>
          via the Cellar API.
        </p>
        <p>
          <a href="https://github.com/yourusername/eur-lex-ai-chat" class="text-eu-blue underline" target="_blank" rel="noopener">Source code (MIT)</a>
          &middot;
          <a href="/faq" class="text-eu-blue underline">FAQ</a>
          &middot;
          <a href="/blog" class="text-eu-blue underline">Blog</a>
        </p>
      </div>
    </footer>
  </body>
</html>
```

### Step 3.5 — Write `frontend/src/components/SeoHead.astro`

```astro
---
// Helper component for page-specific SEO metadata
export interface Props {
  pageTitle: string;
  pageDescription: string;
  pageUrl?: string;
  jsonLd?: Record<string, any>;
}

const {
  pageTitle,
  pageDescription,
  pageUrl = "https://eurlex-chat.vercel.app",
  jsonLd,
} = Astro.props;
---

<!-- Page title suffix -->
{pageTitle && <title>{pageTitle} — EUR-Lex AI Chat</title>}
{pageDescription && <meta name="description" content={pageDescription} />}
<link rel="canonical" href={pageUrl} />
<meta property="og:title" content={pageTitle} />
<meta property="og:description" content={pageDescription} />
<meta property="og:url" content={pageUrl} />

{jsonLd && <script type="application/ld+json" set:html={JSON.stringify(jsonLd)} />}
```

### Step 3.6 — Write `frontend/src/components/ChatWidget.jsx`

```jsx
import { useState, useRef, useEffect } from "react";

const API_URL = import.meta.env.PUBLIC_API_URL || "http://localhost:8000";

export default function ChatWidget() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi! I'm an AI assistant specialized in EU law. Ask me anything about EU regulations, directives, or legislation.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const query = input.trim();
    if (!query || loading) return;

    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || `Error ${res.status}`);
      }

      const data = await res.json();

      // Format answer with sources
      let answer = data.answer;
      if (data.sources && data.sources.length > 0) {
        answer += "\n\n**Sources:**";
        for (const s of data.sources) {
          answer += `\n- CELEX ${s.celex}: ${s.title}`;
          if (s.article) answer += ` (Article ${s.article})`;
        }
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: answer },
      ]);
    } catch (err) {
      setError(err.message);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Sorry, I encountered an error: ${err.message}. Please try again.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden max-w-2xl mx-auto">
      {/* Chat header */}
      <div className="bg-eu-blue text-white px-4 py-3 font-semibold">
        Ask about EU Law
      </div>

      {/* Messages */}
      <div className="h-96 overflow-y-auto p-4 space-y-3 bg-gray-50">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 text-sm whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-eu-blue text-white"
                  : "bg-white border border-gray-200 text-gray-800"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-lg px-4 py-2 text-sm text-gray-500 italic">
              Thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-3 flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about EU law..."
          rows={1}
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-eu-blue resize-none"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="bg-eu-blue text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-eu-navy transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </div>

      {error && (
        <div className="bg-red-50 text-red-700 text-xs px-4 py-2 border-t border-red-200">
          {error}
        </div>
      )}
    </div>
  );
}
```

### Step 3.7 — Write `frontend/src/pages/index.astro`

```astro
---
import Base from "../layouts/Base.astro";
import ChatWidget from "../components/ChatWidget";

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "EUR-Lex AI Chat",
  applicationCategory: "WebApplication",
  operatingSystem: "Web",
  description:
    "Chat about EU law in plain English. Get answers with citations to real EUR-Lex documents.",
  url: "https://eurlex-chat.vercel.app",
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "EUR",
  },
};
---

<Base
  title="EUR-Lex AI Chat — Ask EU Law Questions in Plain English"
  description="Free AI-powered chat over EU legislation. Ask questions about GDPR, AI Act, Digital Markets Act, and thousands of other EU regulations and directives. Get answers with citations."
  jsonLd={jsonLd}
>
  <!-- Hero section -->
  <section class="bg-gradient-to-br from-eu-blue to-eu-navy text-white">
    <div class="max-w-4xl mx-auto px-4 py-16 text-center">
      <h1 class="text-3xl md:text-4xl font-bold mb-4">
        Ask EU Law Questions in Plain English
      </h1>
      <p class="text-lg text-blue-200 max-w-2xl mx-auto mb-8">
        EUR-Lex AI Chat helps you understand EU regulations, directives, and legislation.
        Ask any question and get answers with citations to real EUR-Lex documents.
      </p>
      <div class="flex justify-center gap-3 flex-wrap">
        <span class="bg-white/10 rounded-full px-4 py-1.5 text-sm">Free</span>
        <span class="bg-white/10 rounded-full px-4 py-1.5 text-sm">No account needed</span>
        <span class="bg-white/10 rounded-full px-4 py-1.5 text-sm">20 queries/day</span>
        <span class="bg-white/10 rounded-full px-4 py-1.5 text-sm">Open source</span>
      </div>
    </div>
  </section>

  <!-- Chat section -->
  <section class="max-w-4xl mx-auto px-4 py-12">
    <h2 class="text-2xl font-bold text-center mb-8">Try It Now</h2>
    <ChatWidget client:load />
  </section>

  <!-- Features section -->
  <section class="bg-gray-50 py-12">
    <div class="max-w-4xl mx-auto px-4">
      <h2 class="text-2xl font-bold text-center mb-8">How It Works</h2>
      <div class="grid md:grid-cols-3 gap-6">
        <div class="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <div class="text-eu-blue text-2xl font-bold mb-2">1</div>
          <h3 class="font-semibold mb-2">Ask a Question</h3>
          <p class="text-sm text-gray-600">
            Type any question about EU law in plain language. No legal knowledge needed.
          </p>
        </div>
        <div class="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <div class="text-eu-blue text-2xl font-bold mb-2">2</div>
          <h3 class="font-semibold mb-2">AI Searches EU Law</h3>
          <p class="text-sm text-gray-600">
            We search over 100,000 passages from EUR-Lex documents to find relevant legal texts.
          </p>
        </div>
        <div class="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <div class="text-eu-blue text-2xl font-bold mb-2">3</div>
          <h3 class="font-semibold mb-2">Get Cited Answers</h3>
          <p class="text-sm text-gray-600">
            Receive clear answers with CELEX numbers and article references so you can verify the source.
          </p>
        </div>
      </div>
    </div>
  </section>

  <!-- Example questions section -->
  <section class="py-12">
    <div class="max-w-4xl mx-auto px-4">
      <h2 class="text-2xl font-bold text-center mb-8">Example Questions You Can Ask</h2>
      <div class="grid md:grid-cols-2 gap-4">
        <div class="bg-white border border-gray-200 rounded-lg p-4 text-sm text-gray-700">
          "What are the data protection requirements under the GDPR?"
        </div>
        <div class="bg-white border border-gray-200 rounded-lg p-4 text-sm text-gray-700">
          "When does the EU AI Act come into force?"
        </div>
        <div class="bg-white border border-gray-200 rounded-lg p-4 text-sm text-gray-700">
          "What are the obligations for gatekeepers under the DMA?"
        </div>
        <div class="bg-white border border-gray-200 rounded-lg p-4 text-sm text-gray-700">
          "What is a CELEX number and how do I use it?"
        </div>
      </div>
    </div>
  </section>
</Base>
```

### Step 3.8 — Write `frontend/src/pages/faq.astro`

```astro
---
import Base from "../layouts/Base.astro";

const faqQuestions = [
  {
    question: "What is EUR-Lex?",
    answer:
      "EUR-Lex is the official online portal for European Union law. It provides access to the Official Journal of the EU, treaties, legislation, case law, and other legal documents. EUR-Lex AI Chat uses data from EUR-Lex via the Cellar API.",
  },
  {
    question: "What is a CELEX number?",
    answer:
      "A CELEX number is a unique identifier assigned to every document in EUR-Lex. It follows a specific format: for example, 32019R0947 breaks down as 3 (sector: legislation), 2019 (year), R (regulation), 0947 (document number). CELEX numbers are used to cite and locate EU legal documents precisely.",
  },
  {
    question: "How accurate is the AI?",
    answer:
      "The AI searches over actual EUR-Lex documents and cites its sources with CELEX numbers. However, you should always verify important legal information by checking the official documents on EUR-Lex. The AI can make mistakes and should not be used as a substitute for professional legal advice.",
  },
  {
    question: "Is this service free?",
    answer:
      "Yes, completely free. You can make up to 20 queries per day per IP address. There are no accounts, no subscriptions, and no hidden costs. This is an open-source project built with free cloud services.",
  },
  {
    question: "What documents are included?",
    answer:
      "The index includes EU regulations, directives, and implementing acts published since 2000, available in English. This covers over 25 years of EU legislation, including major laws like the GDPR, AI Act, Digital Markets Act, and thousands more.",
  },
  {
    question: "How often is the data updated?",
    answer:
      "The index is refreshed daily via automated GitHub Actions workflows. New regulations and amendments are picked up within 24 hours of publication in the Official Journal.",
  },
  {
    question: "What is the Cellar API?",
    answer:
      "The Cellar API is the official Semantic Web interface for EUR-Lex. It provides structured access to EU legal documents via SPARQL queries and REST endpoints. This project uses Cellar APIs exclusively, not web scraping.",
  },
  {
    question: "What does a CELEX number look like?",
    answer:
      "CELEX numbers have a standard format: 32019R0947 (a regulation from 2019), 32016R0679 (the GDPR), or 32024R1689 (the AI Act). The first digit indicates the sector, the next four digits are the year, followed by a document type code and number.",
  },
  {
    question: "Can I cite answers from this tool in legal documents?",
    answer:
      "No. This tool is for informational purposes only. Always verify legal information against the official text published in the Official Journal of the EU (OJEU). The AI-generated answers should not be used as legal authority.",
  },
  {
    question: "What is the difference between a regulation and a directive?",
    answer:
      "A regulation is a binding legislative act that must be applied in its entirety across all EU member states. A directive is a legislative act that sets a goal that all EU countries must achieve, but it's up to the individual countries to devise their own laws to reach that goal.",
  },
  {
    question: "What is the GDPR?",
    answer:
      "The General Data Protection Regulation (GDPR, Regulation 2016/679) is the EU's primary data protection law. It governs how personal data must be collected, processed, stored, and protected by organizations operating in the EU.",
  },
  {
    question: "What is the EU AI Act?",
    answer:
      "The EU AI Act (Regulation 2024/1689) is the world's first comprehensive legal framework for artificial intelligence. It categorizes AI systems by risk level and imposes requirements ranging from transparency obligations for minimal-risk AI to bans on unacceptable-risk AI.",
  },
  {
    question: "What is the Digital Markets Act (DMA)?",
    answer:
      "The Digital Markets Act (Regulation 2022/1925) is an EU law that aims to make digital markets fairer and more contestable. It sets obligations for large online platforms designated as 'gatekeepers' to prevent anti-competitive practices.",
  },
  {
    question: "What is the Digital Services Act (DSA)?",
    answer:
      "The Digital Services Act (Regulation 2022/2065) is an EU law that regulates online intermediary platforms. It sets rules for content moderation, transparency, and accountability of platforms from the largest social networks to small online marketplaces.",
  },
  {
    question: "Where can I find the official text of an EU regulation?",
    answer:
      "The official text is published in the Official Journal of the EU (OJEU) and is available on EUR-Lex at eur-lex.europa.eu. You can search by CELEX number, title, or date. The OJ L series contains legislation, while the OJ C series contains information and notices.",
  },
  {
    question: "Can I search for documents from a specific year?",
    answer:
      "Yes. The index covers documents from 2000 onwards. You can ask questions like 'What regulations were passed in 2023 regarding digital platforms?' and the AI will search the relevant documents.",
  },
  {
    question: "How do I find the CELEX number for a specific law?",
    answer:
      "You can search directly on this chat by asking 'What is the CELEX number for the GDPR?' or 'Find CELEX for the AI Act.' The AI will return the CELEX number along with the title and other metadata.",
  },
  {
    question: "What languages are supported?",
    answer:
      "The index currently contains documents in English. However, you can ask questions in any language, and the AI will respond in the language you used. The underlying legal texts are in English.",
  },
  {
    question: "Can I download or export the results?",
    answer:
      "Currently, the chat provides answers with citations on screen. You can copy the text directly. We're working on adding export functionality in future updates.",
  },
  {
    question: "Is this project open source?",
    answer:
      "Yes! The full source code is available on GitHub under the MIT license. You can view, fork, and contribute to the project. The data pipeline, AI backend, and frontend are all open source.",
  },
  {
    question: "What happens if I reach the daily limit?",
    answer:
      "After 20 queries in a minute, you'll see a rate limit message. Wait a minute and try again. This limit helps keep the service free and available for everyone.",
  },
];

const faqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: faqQuestions.map((q) => ({
    "@type": "Question",
    name: q.question,
    acceptedAnswer: {
      "@type": "Answer",
      text: q.answer,
    },
  })),
};
---

<Base
  title="Frequently Asked Questions"
  description="Answers to common questions about EUR-Lex AI Chat, CELEX numbers, EU regulations, directives, GDPR, AI Act, DMA, DSA, and how to use the service."
  canonical="https://eurlex-chat.vercel.app/faq"
  jsonLd={faqJsonLd}
>
  <div class="max-w-3xl mx-auto px-4 py-12">
    <h1 class="text-3xl font-bold mb-2">Frequently Asked Questions</h1>
    <p class="text-gray-600 mb-8">
      Everything you need to know about EUR-Lex AI Chat and EU law.
    </p>

    <div class="space-y-4">
      {faqQuestions.map((item) => (
        <details class="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <summary class="px-4 py-3 cursor-pointer font-medium hover:bg-gray-50 transition">
            {item.question}
          </summary>
          <div class="px-4 py-3 text-sm text-gray-700 border-t border-gray-100">
            {item.answer}
          </div>
        </details>
      ))}
    </div>
  </div>
</Base>
```

### Step 3.9 — Write blog listing page `frontend/src/pages/blog/index.astro`

```astro
---
import Base from "../../layouts/Base.astro";

const posts = await Astro.glob("./posts/*.md");
// Sort by date descending
posts.sort((a, b) => new Date(b.frontmatter.date) - new Date(a.frontmatter.date));
---

<Base
  title="Blog"
  description="Articles about EU law, including guides to GDPR, AI Act, CELEX numbers, and other European legislation topics."
  canonical="https://eurlex-chat.vercel.app/blog"
>
  <div class="max-w-3xl mx-auto px-4 py-12">
    <h1 class="text-3xl font-bold mb-2">Blog</h1>
    <p class="text-gray-600 mb-8">Articles about EU law and how to use EUR-Lex AI Chat.</p>

    <div class="space-y-6">
      {posts.map((post) => (
        <article class="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition">
          <time class="text-xs text-gray-500">{post.frontmatter.date}</time>
          <h2 class="text-xl font-semibold mt-1 mb-2">
            <a href={post.url} class="text-eu-blue hover:underline">{post.frontmatter.title}</a>
          </h2>
          <p class="text-sm text-gray-600">{post.frontmatter.description}</p>
        </article>
      ))}
    </div>
  </div>
</Base>
```

### Step 3.10 — Create blog post directory and write first post

```bash
mkdir -p ~/Desktop/EUProjects/eur-lex-ai-chat/frontend/src/pages/blog/posts
```

Write blog posts in Markdown. Save each as `.md` in `frontend/src/pages/blog/posts/`. Each needs frontmatter with `title`, `description`, `date`, and optionally `tags`.

### Step 3.11 — Test the frontend locally

```bash
source ~/.nvm/nvm.sh
nvm use v22.22.3
cd ~/Desktop/EUProjects/eur-lex-ai-chat/frontend
npm run dev
```

Open http://localhost:4321 in a browser. Verify:
- Landing page renders with chat widget
- FAQ page loads with all 20 questions expandable
- Blog index lists posts
- Test the chat widget (it will try to connect to local backend)

### Step 3.12 — Create `vercel.json` (optional)

For custom Vercel config:
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "astro"
}
```

---

## Phase 4: SEO Content

**Goal:** Write the FAQ content (already done in Phase 3 — 20 questions) and 3 blog posts. This is pure content writing with Markdown.

### Step 4.1 — Blog Post 1: eu-ai-act-explained.md

```markdown
---
title: "EU AI Act Explained: What You Need to Know"
description: "A comprehensive guide to the EU AI Act (Regulation 2024/1689) — the world's first legal framework for artificial intelligence. Learn about risk categories, requirements, and timelines."
date: "2026-05-20"
tags: ["AI Act", "artificial intelligence", "regulation"]
---

The EU AI Act (Regulation 2024/1689) entered into force on August 1, 2024, marking a historic moment in technology regulation. It is the world's first comprehensive legal framework for artificial intelligence.

## What Is the AI Act?

The AI Act establishes a risk-based classification system for AI systems:

- **Unacceptable risk** — AI systems that pose a clear threat to safety, livelihoods, or rights (banned)
- **High risk** — AI systems in critical areas like healthcare, law enforcement, employment (strict requirements)
- **Limited risk** — AI systems with specific transparency obligations (e.g., chatbots must disclose they are AI)
- **Minimal risk** — All other AI systems (no additional obligations)

## Key Requirements

High-risk AI systems must comply with:

1. **Risk management system** (Article 9)
2. **Data governance** — training data must be relevant, representative, and free from errors (Article 10)
3. **Technical documentation** — detailed description of the system's development and capabilities (Article 11)
4. **Record-keeping** — automatic logging of system events (Article 12)
5. **Transparency** — users must be informed they are interacting with AI (Article 13)
6. **Human oversight** — humans must be able to override or stop the system (Article 14)
7. **Accuracy, robustness, and cybersecurity** (Article 15)

## Timeline

- **August 2024**: AI Act enters into force
- **February 2025**: Prohibitions on unacceptable risk AI apply
- **August 2025**: Rules for general-purpose AI models apply
- **August 2026**: Most rules apply (including high-risk AI systems)
- **August 2027**: Rules for high-risk AI systems in certain product categories apply

## CELEX Number

The AI Act has CELEX number **32024R1689**. You can find the full text on EUR-Lex at this CELEX identifier.

## How EUR-Lex AI Chat Can Help

Use EUR-Lex AI Chat to ask specific questions about the AI Act, such as:
- "What are the transparency obligations for chatbots under the AI Act?"
- "Which AI systems are banned under the AI Act?"
- "What are the penalties for non-compliance with the AI Act?"
```

### Step 4.2 — Blog Post 2: gdpr-guide.md

```markdown
---
title: "GDPR Explained: A Practical Guide to EU Data Protection Law"
description: "Everything you need to know about the General Data Protection Regulation (GDPR, Regulation 2016/679). Covers data subject rights, obligations for controllers and processors, and key requirements."
date: "2026-05-20"
tags: ["GDPR", "data protection", "privacy", "regulation"]
---

The General Data Protection Regulation (GDPR, Regulation 2016/679) is the European Union's flagship data protection law. Since its application on May 25, 2018, it has become the global benchmark for privacy regulation.

## Scope

The GDPR applies to:

- Organizations established in the EU that process personal data
- Organizations outside the EU that offer goods or services to individuals in the EU
- Organizations outside the EU that monitor the behavior of individuals in the EU

## Key Principles (Article 5)

Personal data must be:

1. **Lawfulness, fairness, and transparency** — processed lawfully, fairly, and transparently
2. **Purpose limitation** — collected for specified, explicit, and legitimate purposes
3. **Data minimization** — adequate, relevant, and limited to what is necessary
4. **Accuracy** — accurate and kept up to date
5. **Storage limitation** — kept no longer than necessary
6. **Integrity and confidentiality** — processed securely
7. **Accountability** — the controller is responsible for compliance

## Data Subject Rights (Chapter III)

Individuals have the right to:

- **Access** their personal data (Article 15)
- **Rectification** of inaccurate data (Article 16)
- **Erasure** ('right to be forgotten') (Article 17)
- **Restriction of processing** (Article 18)
- **Data portability** (Article 20)
- **Object** to processing (Article 21)

## CELEX Number

The GDPR has CELEX number **32016R0679**. You can find the full text on EUR-Lex at this CELEX identifier.

## Use EUR-Lex AI Chat

Ask questions like:
- "What are the requirements for data protection impact assessments under the GDPR?"
- "When can personal data be transferred outside the EU?"
- "What are the penalties for GDPR violations?"
```

### Step 4.3 — Blog Post 3: what-is-celex.md

```markdown
---
title: "What Is a CELEX Number? A Complete Guide to EUR-Lex Document Identifiers"
description: "Learn what CELEX numbers are, how to read them, and how to use them to find EU legal documents on EUR-Lex. Includes examples for GDPR, AI Act, and more."
date: "2026-05-20"
tags: ["CELEX", "EUR-Lex", "document identifiers", "guide"]
---

If you've ever tried to cite an EU legal document precisely, you've probably encountered CELEX numbers. This guide explains what they are, how they work, and how to use them.

## What Is a CELEX Number?

A CELEX number is a unique identifier assigned to every document in EUR-Lex, the official online portal for EU law. Think of it as a digital fingerprint for EU legal documents — no two documents share the same CELEX number.

## How to Read a CELEX Number

A CELEX number follows a specific format. Let's break down the GDPR's CELEX number: **32016R0679**

| Digit(s) | Meaning | Example |
|----------|---------|---------|
| 1st digit | Sector | 3 = Legislation |
| 2nd-5th digits | Year | 2016 |
| Letter(s) | Document type | R = Regulation |
| Remaining digits | Document number | 0679 |

### Sector Codes

- **1**: Treaties
- **2**: International agreements
- **3**: Legislation (most common for our purposes)
- **4**: Complementary legislation
- **5**: Preparatory acts
- **6**: Case law
- **7**: National transposition measures
- **8**: References
- **9**: Parliamentary questions
- **0**: Consolidated texts

### Document Type Letters

- **R**: Regulation
- **L**: Directive
- **D**: Decision
- **H**: Recommendation
- **A**: Opinion
- **C**: Cost of the Treaties

## Examples of Common CELEX Numbers

| Law | CELEX Number |
|-----|-------------|
| GDPR | 32016R0679 |
| EU AI Act | 32024R1689 |
| Digital Markets Act | 32022R1925 |
| Digital Services Act | 32022R2065 |
| Uber ruling (ECJ) | 62017CJ0434 |

## How to Use CELEX Numbers

1. **Search**: Go to eur-lex.europa.eu and paste a CELEX number into the search box
2. **Cite**: Use CELEX numbers in legal citations for precision
3. **API**: Developers can use CELEX numbers to retrieve documents programmatically via the Cellar API

## Use EUR-Lex AI Chat

Ask questions like:
- "What is the CELEX number for the GDPR?"
- "Find the CELEX number for the Digital Markets Act"
- "How do I search for documents by CELEX number?"
```

### Step 4.4 — Create `frontend/public/robots.txt`

```
User-agent: *
Allow: /
Sitemap: https://eurlex-chat.vercel.app/sitemap-index.xml
```

---

## Phase 5: GitHub Actions Automation

**Goal:** A daily GitHub Actions workflow that scrapes EUR-Lex for new/changed documents, processes them incrementally, and uploads the updated index to HuggingFace Hub.

### Step 5.1 — Write `scripts/update_index.py`

This incremental update script:
1. Gets the current last_updated timestamp from HF Hub
2. Queries SPARQL for documents modified since that timestamp
3. Downloads and processes only new/changed documents
4. Merges with existing index
5. Re-embeds (only new vectors)
6. Uploads updated index

```python
#!/usr/bin/env python3
"""
update_index.py — Incremental daily update of the EUR-Lex vector index.
Runs in GitHub Actions on a schedule.

Pipeline:
  1. Download current index from HF Hub
  2. Query SPARQL for documents modified since last update
  3. Only process new/changed documents
  4. Merge with existing index (replace changed, append new)
  5. Upload updated index to HF Hub

Usage:
  HF_TOKEN=hf_yourtoken python3 scripts/update_index.py
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import requests
from huggingface_hub import hf_hub_download, HfApi, upload_file
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
DOWNLOAD_WORKERS = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HF_USERNAME = os.environ.get("HF_USERNAME", "yourusername")
HF_DATASET = os.environ.get("HF_DATASET", "eurlex-chat-data")
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = f"{HF_USERNAME}/{HF_DATASET}"


def download_current_index():
    """Download the current vectors.npy and chunks.json from HF Hub."""
    logger.info("Downloading current index from HF Hub...")
    vectors_path = hf_hub_download(REPO_ID, "vectors.npy", repo_type="dataset", token=HF_TOKEN)
    chunks_path = hf_hub_download(REPO_ID, "chunks.json", repo_type="dataset", token=HF_TOKEN)
    ts_path = hf_hub_download(REPO_ID, "last_updated.txt", repo_type="dataset", token=HF_TOKEN)

    vectors = np.load(vectors_path)
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    with open(ts_path, "r") as f:
        last_updated = f.read().strip()

    logger.info(f"Current index: {vectors.shape[0]} vectors, {len(chunks)} chunks")
    logger.info(f"Last updated: {last_updated}")
    return vectors, chunks, last_updated


def query_modified_documents(since_date):
    """Query SPARQL for documents modified since the given date."""
    query = f"""
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>

SELECT DISTINCT ?doc ?type ?celex ?title ?date
WHERE {{
    ?doc cdm:work_has_resource-type ?type .
    ?doc cdm:resource_legal_id_celex ?celex .
    ?doc dc:title ?title .
    ?doc cdm:work_date_lastUpdate ?date .
    FILTER(LANG(?title) = "en")
    FILTER(?date >= "{since_date}"^^xsd:dateTime)
}}
ORDER BY DESC(?date)
"""
    r = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    docs = []
    for b in data["results"]["bindings"]:
        docs.append({
            "celex": b["celex"]["value"],
            "title": b["title"]["value"],
            "date": b["date"]["value"],
            "type": b["type"]["value"].split("/")[-1],
        })
    logger.info(f"Modified documents since {since_date}: {len(docs)}")
    return docs


def fetch_document_html(celex):
    """Fetch XHTML content via Cellar RDF graph traversal."""
    try:
        from xml.etree import ElementTree as ET
        ns = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "cdm": "http://publications.europa.eu/ontology/cdm#",
        }

        work_url = f"http://publications.europa.eu/resource/celex/{celex}?language=eng"
        r = requests.get(work_url, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        expressions = root.findall(".//cdm:work_has_expression", ns)
        expression_url = None
        for expr in expressions:
            resource = expr.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
            if resource and resource.endswith(".ENG"):
                expression_url = resource
                break
        if not expression_url and expressions:
            resource = expressions[0].get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
            if resource:
                expression_url = resource
        if not expression_url:
            return None

        r2 = requests.get(expression_url, timeout=30)
        r2.raise_for_status()
        expr_root = ET.fromstring(r2.content)

        manifestations = expr_root.findall(".//cdm:expression_manifested_by_manifestation", ns)
        xhtml_url = None
        for manif in manifestations:
            resource = manif.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
            if resource and (resource.endswith(".xhtml") or resource.endswith(".fmx4")):
                xhtml_url = resource
                break

        if not xhtml_url:
            return None

        r3 = requests.get(xhtml_url, headers={"Accept": "application/xhtml+xml, text/html"}, timeout=30)
        r3.raise_for_status()
        return r3.text
    except Exception as e:
        logger.warning(f"Failed to fetch {celex}: {e}")
        return None


def parse_to_chunks(html, celex, title):
    """Parse HTML into text chunks.

    ⚠️ Uses _parse_html_with_beautifulsoup directly (eurlxp parse_html has
    a Polars schema bug with conditional modifier field).
    """
    try:
        from eurlxp.parser import _parse_html_with_beautifulsoup as internal_parse
        import polars as pl
        results = internal_parse(html)
    except Exception:
        return []

    if not results:
        return []

    # Fixed-schema conversion (bypasses eurlxp parse_html bug)
    records = []
    for r in results:
        records.append({
            'text': r.text, 'type': r.item_type, 'ref': str(r.ref),
            'modifier': r.modifier, 'document': r.context.document,
            'article': r.context.article, 'article_subtitle': r.context.article_subtitle,
            'paragraph': r.context.paragraph, 'group': r.context.group,
            'section': r.context.section,
        })

    df = pl.DataFrame(records, schema={
        'text': pl.Utf8, 'type': pl.Utf8, 'ref': pl.Utf8,
        'modifier': pl.Utf8, 'document': pl.Utf8, 'article': pl.Utf8,
        'article_subtitle': pl.Utf8, 'paragraph': pl.Utf8,
        'group': pl.Utf8, 'section': pl.Utf8,
    })

    if len(df) == 0:
        return []

    chunks = []
    current_article = None
    current_text = []

    for row in df.to_dicts():
        text = row.get("text", "").strip()
        row_type = row.get("type", "")
        article = row.get("article")

        if not text or len(text) < 40:
            continue

        if row_type in ("doc-title", "art-subtitle", "group-title", "section-title"):
            continue
        if row.get("modifier") in ("note", "signatory"):
            continue

        current_text.append(text)

    if current_text:
        chunk_text = " ".join(current_text)
        if len(chunk_text) > 50:
            chunks.append({
                "text": chunk_text,
                "celex": celex,
                "title": title,
                "article": current_article,
                "type": "article",
            })

    if not chunks:
        paragraphs = [p.strip() for p in html.split("\n\n") if len(p.strip()) > 80]
        for para in paragraphs:
            chunks.append({
                "text": para,
                "celex": celex,
                "title": title,
                "article": None,
                "type": "paragraph",
            })

    return chunks


def embed_chunks(chunks, batch_size=128):
    """Embed chunks using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings).astype(np.float32)


def upload_index(vectors, chunks):
    """Upload updated index to HF Hub."""
    api = HfApi()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as vf:
        np.save(vf.name, vectors)
        api.upload_file(
            repo_id=REPO_ID,
            path_in_repo="vectors.npy",
            path_or_fileobj=vf.name,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as cf:
        json.dump(chunks, cf, indent=2)
        cf.flush()
        api.upload_file(
            repo_id=REPO_ID,
            path_in_repo="chunks.json",
            path_or_fileobj=cf.name,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tf:
        tf.write(datetime.now(timezone.utc).isoformat())
        tf.flush()
        api.upload_file(
            repo_id=REPO_ID,
            path_in_repo="last_updated.txt",
            path_or_fileobj=tf.name,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    logger.info("Index uploaded to HF Hub")


def main():
    if not HF_TOKEN:
        logger.error("HF_TOKEN environment variable required")
        return

    # Step 1: Download current index
    vectors, chunks, last_updated = download_current_index()

    # Build CELEX → chunk mapping for dedup
    existing_celex_set = set(c["celex"] for c in chunks)

    # Step 2: Query modified documents
    docs = query_modified_documents(last_updated)
    if not docs:
        logger.info("No new/changed documents")
        return

    # Step 3: Download and process only new docs
    logger.info("Downloading new/changed documents...")
    html_map = {}
    new_celex_ids = [d["celex"] for d in docs if d["celex"] not in existing_celex_set]
    changed_celex_ids = [d["celex"] for d in docs if d["celex"] in existing_celex_set]

    logger.info(f"New: {len(new_celex_ids)}, Changed: {len(changed_celex_ids)}")

    all_to_process = docs
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(fetch_document_html, d["celex"]): d for d in all_to_process}
        for future in tqdm(as_completed(future_map), total=len(all_to_process), desc="Downloading"):
            doc = future_map[future]
            try:
                html = future.result()
                if html:
                    html_map[doc["celex"]] = html
            except Exception:
                pass

    # Step 4: Parse and chunk new docs
    logger.info("Parsing new documents...")
    new_chunks = []
    for doc in tqdm(all_to_process, desc="Parsing"):
        html = html_map.get(doc["celex"])
        if html:
            parsed = parse_to_chunks(html, doc["celex"], doc["title"])
            new_chunks.extend(parsed)

    if not new_chunks:
        logger.info("No new chunks to add")
        return

    # Step 5: Remove changed docs from existing, add new ones
    celexes_to_remove = set(changed_celex_ids)
    filtered_chunks = [c for c in chunks if c["celex"] not in celexes_to_remove]
    updated_chunks = filtered_chunks + new_chunks

    # Step 6: Embed new chunks only
    logger.info(f"Embedding {len(new_chunks)} new chunks...")
    new_vectors = embed_chunks(new_chunks)

    # Step 7: Merge vectors
    # For changed docs: need to figure out which vector indices to remove
    # Simple approach: re-embed everything (safer but slower for large changes)
    # For small daily updates: just append new vectors
    if celexes_to_remove:
        # Re-embed all chunks to keep vectors in sync
        logger.info("Changed docs detected — re-embedding all chunks...")
        updated_vectors = embed_chunks(updated_chunks)
    else:
        # Just append new vectors
        updated_vectors = np.vstack([vectors, new_vectors])

    logger.info(f"Updated index: {updated_vectors.shape[0]} vectors, {len(updated_chunks)} chunks")

    # Step 8: Upload
    upload_index(updated_vectors, updated_chunks)
    logger.info("Daily update complete")


if __name__ == "__main__":
    main()
```

### Step 5.2 — Write `.github/workflows/update-index.yml`

```yaml
name: Daily EUR-Lex Index Update

on:
  schedule:
    - cron: "0 6 * * *"  # Every day at 06:00 UTC
  workflow_dispatch:       # Allow manual trigger

jobs:
  update:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install numpy huggingface_hub sentence-transformers requests tqdm eurlxp[sparql] polars

      - name: Run incremental update
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: ${{ secrets.HF_USERNAME }}
          HF_DATASET: "eurlex-chat-data"
        run: python scripts/update_index.py
```

### Step 5.3 — Create the `.github/workflows/` directory

```bash
mkdir -p ~/Desktop/EUProjects/eur-lex-ai-chat/.github/workflows
```

### Step 5.4 — Set up GitHub secrets

After pushing to GitHub, add these secrets in the repository Settings → Secrets and variables → Actions:

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | Your HuggingFace write token |
| `HF_USERNAME` | Your HuggingFace username |

---

## Phase 6: Deployment

**Goal:** Deploy the backend to Render, frontend to Vercel, set up cron-job.org keepalives, and submit to Google Search Console.

### Step 6.1 — Push code to GitHub

```bash
cd ~/Desktop/EUProjects/eur-lex-ai-chat

# Initialize git repo
git init
git add -A
git commit -m "Initial commit: EUR-Lex AI Chat — full implementation"

# Create GitHub repo and push
gh repo create eur-lex-ai-chat --public --push --source=.
# If gh isn't authenticated:
# 1. Go to https://github.com/new
# 2. Create repo named "eur-lex-ai-chat"
# 3. Run:
#    git remote add origin https://github.com/YOUR_USERNAME/eur-lex-ai-chat.git
#    git branch -M main
#    git push -u origin main
```

### Step 6.2 — Deploy backend to Render

1. Go to https://dashboard.render.com
2. Click **New +** → **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `eurlex-chat-api`
   - **Runtime**: Python 3
   - **Region**: Frankfurt (EU) — for low latency in Europe
   - **Branch**: `main`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && ./startup.sh`
   - **Instance Type**: Free ($0/month, 512MB RAM)
5. Add environment variables:
   - `GROQ_API_KEY`: your_groq_key
   - `HF_TOKEN`: your_hf_token
   - `HF_USERNAME`: your_hf_username
   - `HF_DATASET`: eurlex-chat-data
6. Click **Deploy Web Service**

The deploy takes ~2-3 minutes. Render will:
- Clone the repo
- Install dependencies
- Start the FastAPI server
- Load the vector index from HF Hub at startup

### Step 6.3 — Note the Render URL

After deployment, Render assigns a URL like: `https://eurlex-chat-api.onrender.com`

Test it:
```bash
curl https://eurlex-chat-api.onrender.com/health
```

Expected response:
```json
{"status": "ok", "index_loaded": true, "vector_count": ..., "chunk_count": ...}
```

### Step 6.4 — Configure frontend API URL

Before deploying frontend, set the API URL:

```bash
cd ~/Desktop/EUProjects/eur-lex-ai-chat/frontend

# Set the production API URL
echo "PUBLIC_API_URL=https://eurlex-chat-api.onrender.com" > .env.production
```

### Step 6.5 — Deploy frontend to Vercel

1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Configure:
   - **Framework Preset**: Astro
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `dist` (default)
4. Add environment variables:
   - `PUBLIC_API_URL`: `https://eurlex-chat-api.onrender.com`
5. Click **Deploy**

Vercel auto-deploys on every push to `main`.

### Step 6.6 — Configure custom domain (optional)

1. Buy a domain (e.g., `eurlex-chat.com` from Namecheap, Cloudflare, etc.) — this is the ONLY potential cost
   - OR use Vercel's free subdomain: `eurlex-chat.vercel.app` (free, no domain needed)
2. In Vercel dashboard → Project → Settings → Domains → Add your domain
3. Add the DNS records provided by Vercel to your domain registrar
4. Wait for DNS propagation (5 min to 48 hours)

For $0 hosting, skip the custom domain. Use `eurlex-chat.vercel.app` — it's free, HTTPS by default, and performs equally well in SEO.

### Step 6.7 — Update the sitemap URL

If using a custom domain, update `astro.config.mjs`:
```javascript
site: "https://yourdomain.com",
```

Also update `robots.txt` and `Base.astro` canonical URLs.

### Step 6.8 — Set up cron-job.org

1. Go to https://cron-job.org
2. Create account (free, no credit card)
3. Create two cron jobs:

**Job 1: Keep-alive** (prevents Render from sleeping)
- **URL**: `https://eurlex-chat-api.onrender.com/health`
- **Interval**: Every 5 minutes
- **Method**: GET

**Job 2: Data refresh** (hourly check for new index)
- **URL**: `https://eurlex-chat-api.onrender.com/refresh`
- **Interval**: Every 60 minutes
- **Method**: GET

Render free tier sleeps after 15 minutes of inactivity. Cron-job.org pings every 5 minutes to prevent sleep. The /refresh endpoint checks HF Hub for a newer index and reloads if available.

### Step 6.9 — Submit to Google Search Console

1. Go to https://search.google.com/search-console
2. Add your property:
   - URL prefix: `https://eurlex-chat.vercel.app`
3. Verify ownership:
   - **Recommended**: Add DNS TXT record at your domain registrar (if using custom domain)
   - **Alternative**: HTML file upload — download the verification HTML file and add it to `frontend/public/`
4. After verification, submit sitemap:
   - Go to Sitemaps section
   - Enter: `https://eurlex-chat.vercel.app/sitemap-index.xml`
   - Click Submit

The sitemap tells Google about all pages on the site:
- `/` (landing page)
- `/faq` (FAQ with 20 questions)
- `/blog` (blog listing)
- `/blog/eu-ai-act-explained`
- `/blog/gdpr-guide`
- `/blog/what-is-celex`

### Step 6.10 — Verify deployments

**Frontend check:**
```bash
curl -s https://eurlex-chat.vercel.app | head -50
```
Should return HTML content (not blank JS shell).

**FAQ rich snippets check:**
Go to https://search.google.com/test/rich-results
Enter: `https://eurlex-chat.vercel.app/faq`
Should detect FAQPage schema with 20 questions.

**Backend health check:**
```bash
curl -s https://eurlex-chat-api.onrender.com/health
```

**Chat endpoint check:**
```bash
curl -s -X POST https://eurlex-chat-api.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the GDPR?"}'
```

---

## Phase 7: Go Live Checklist

**Goal:** Verify everything works end-to-end and the system runs autonomously.

### Step 7.1 — Run SEO audit with seofor.dev

```bash
# Start the dev server
cd ~/Desktop/EUProjects/eur-lex-ai-chat/frontend
npm run dev &

# Run SEO audit
seo audit run --port 4321

# Or audit the live site
seo audit run --url https://eurlex-chat.vercel.app
```

Check for:
- ✅ All pages have `<title>` and `<meta description>`
- ✅ No broken links
- ✅ Proper heading hierarchy (H1 → H2 → H3)
- ✅ Canonical URLs set correctly
- ✅ Open Graph + Twitter Card tags present
- ✅ Sitemap is valid and reachable
- ✅ robots.txt is valid

### Step 7.2 — Verify cron-job.org keepalive

Wait 5 minutes after setting up cron-job.org jobs. Check Render logs:
```bash
curl -s https://eurlex-chat-api.onrender.com/health
```

The `loaded_at` field should show the server has been up continuously (not sleeping).

### Step 7.3 — Test full user flow

1. Open `https://eurlex-chat.vercel.app` in a browser
2. Verify landing page loads with chat widget visible
3. Send a test query: "What is the GDPR?"
4. Verify response includes CELEX number citations
5. Navigate to FAQ page — verify all 20 questions expand
6. Navigate to Blog — verify all 3 posts listed
7. Open blog post — verify Article JSON-LD renders
8. Test on mobile — verify responsive design

### Step 7.4 — Verify Google Search Console indexing

1. In Search Console, go to **URL Inspection**
2. Enter `https://eurlex-chat.vercel.app/`
3. Click **Request Indexing**
4. Repeat for `/faq` and `/blog`
5. Check back in 24-48 hours for index status

### Step 7.5 — Verify GitHub Actions runs

1. Go to GitHub → Actions tab
2. Verify the first scheduled run happens at 06:00 UTC next day
3. Or trigger manually: click **Run workflow** → **Daily EUR-Lex Index Update**
4. Monitor the run — check logs for errors

### Step 7.6 — Final autonomy checklist

| # | Check | Verified |
|---|-------|----------|
| 1 | Frontend serves from Vercel (no laptop needed) | |
| 2 | Backend serves from Render (no laptop needed) | |
| 3 | cron-job.org pings /health every 5 min | |
| 4 | cron-job.org pings /refresh every 60 min | |
| 5 | GitHub Actions runs daily at 06:00 UTC | |
| 6 | Google Search Console has sitemap submitted | |
| 7 | SEO structured data validates (JSON-LD) | |
| 8 | Chat widget works with production backend | |
| 9 | All pages return 200, no broken links | |
| 10 | Rate limiting active (20 req/min/IP) | |

When all 10 are verified, **the system is fully autonomous.**

**Your laptop can power off. The project runs itself forever at $0/month.**

### Step 7.7 — What to monitor

| What | How | Frequency |
|------|-----|-----------|
| Render logs | Render dashboard → Logs | Weekly (5 min) |
| GitHub Actions runs | GitHub → Actions tab | Daily (2 min) |
| Google Search Console | Search Console → Performance | Weekly (5 min) |
| cron-job.org stats | cron-job.org dashboard | Monthly (2 min) |
| Groq API usage | Groq console | Monthly (2 min) |
| HF Hub dataset size | HF Hub → dataset | Monthly (2 min) |

### Step 7.8 — Future improvements

After the system is live, possible enhancements:

1. **More document types**: Add decisions, international agreements, case law
2. **More languages**: Process documents in French, German, Spanish, etc.
3. **Multi-query expansion**: Send 3-5 related queries to retrieve more relevant context
4. **Citation highlighting**: Highlight exact cited paragraphs in the answer
5. **Conversation history**: Multi-turn chat with context
6. **Related documents**: "What other regulations are related to this topic?"
7. **PDF export**: Download chat conversations as PDF
8. **Advanced search**: Filter by year, document type, CELEX number
9. **Rate limit per user**: Token bucket algorithm instead of fixed window
10. **Backlinks**: List on awesome-opensource, GitHub topics, product hunt

---

## Appendix: Troubleshooting

### SPARQL query returns 0 results

If the SPARQL query in build_index.py returns 0 documents:
```bash
# Test with a minimal query
python3 -c "
import requests
r = requests.get('https://publications.europa.eu/webapi/rdf/sparql',
    params={'query': 'SELECT * WHERE { ?s ?p ?o } LIMIT 1', 'format': 'json'},
    timeout=30)
print(r.status_code, len(r.json()['results']['bindings']))
"
```
If this also fails, the endpoint might be temporarily down. Retry later.

### eurlxp HTML fetch fails with WAF

The build_index.py uses Cellar RDF graph traversal (not the EUR-Lex website), so WAF should not be triggered. If it fails:
```bash
# Test with eurlxp directly (which has SPARQL fallback)
python3 -c "
from eurlxp import get_html_by_celex_id
html = get_html_by_celex_id('32019R0947')
print('Success:', len(html), 'bytes')
"
```

### Render won't start / out of memory

The index size should fit in 512MB RAM (~150MB vectors + ~50MB chunks.json + ~100MB overhead = ~300MB). If Render runs out of memory:
1. Reduce index size by filtering fewer document types
2. Or upgrade to Render's $7/month Starter plan (512MB → 1GB RAM)

### Frontend can't reach backend (CORS error)

Check:
1. `PUBLIC_API_URL` env var is set correctly in Vercel
2. Backend CORS `allow_origins` includes the Vercel domain
3. Backend is not sleeping (cron-job.org /health ping working)

### Groq API returns 429 rate limited

The backend has 20 req/min/IP limit. If Groq itself rate-limits:
```bash
# Check remaining rate limit
curl -s -I https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  | grep -i ratelimit
```
Free tier: 1,000 requests/day, 20 requests/min. Backend rate limiting prevents hitting Groq limits.

---

## Appendix: Key Links

| Service | URL | Purpose |
|---------|-----|---------|
| Render dashboard | https://dashboard.render.com | Backend hosting |
| Vercel dashboard | https://vercel.com | Frontend hosting |
| cron-job.org | https://cron-job.org | Keepalive + refresh |
| Google Search Console | https://search.google.com/search-console | SEO monitoring |
| HuggingFace Hub | https://huggingface.co/ | Vector index storage |
| Groq Console | https://console.groq.com | AI API key + usage |
| GitHub repo | https://github.com/YOUR_USERNAME/eur-lex-ai-chat | Source code |
| seofor.dev | npx seo audit | SEO auditing |
| EUR-Lex SPARQL | https://publications.europa.eu/webapi/rdf/sparql | Document querying |
| EUR-Lex Cellar | https://publications.europa.eu/resource/ | Document fetching |
| Rich Results Test | https://search.google.com/test/rich-results | JSON-LD validation |
| PageSpeed Insights | https://pagespeed.web.dev/ | Performance testing |
