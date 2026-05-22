#!/usr/bin/env python3
"""
build_index.py — Full build of the EUR-Lex vector index.

Pipeline:
  1. SPARQL query -> DIR + REG docs with dates from 2004-01-01 onward
  2. Fetch XHTML from Cellar CELEX endpoint (parallel, 20 workers)
  3. Parse XHTML via BeautifulSoup into chunks (articles, preamble, annex)
  4. Embed with sentence-transformers (all-MiniLM-L6-v2, 384-dim, normalized)
  5. Build FAISS IVF+PCA+PQ index (~32MB for 500K vectors, vs 768MB float32)
  6. Build SQLite DB of chunks (~35MB, on-disk, zero RAM at query time)
  7. Upload index.faiss + chunks.db + last_updated.txt to HuggingFace Hub
  8. Delete old vectors.npy + chunks.json from Hub

Memory optimization:
  - Streams through download+parse (no html_results dict → saves ~7.6GB)
  - FAISS IVFPQ index: ~32MB vs 768MB for raw float32
  - SQLite chunks: on-disk, only loads matched rows (top-10 per query)
  - Total RAM at query time: ~50MB (index) + model + overhead ≈ ~350MB

Usage:
  source ~/Desktop/EUProjects/.venv/bin/activate
  HF_TOKEN=hf_yourtoken python3 scripts/build_index.py

Output:
  data/index.faiss       - FAISS IVFPQ index (~32MB)
  data/chunks.db         - SQLite database of chunks (~35MB)
  data/last_updated.txt  - ISO timestamp of build
"""

import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import quote

import numpy as np
import requests
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None  # will be checked before use

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

DOC_TYPES = [
    "REG",
    "DIR",
]

FROM_DATE = "2004-01-01"
DOWNLOAD_WORKERS = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HF_DATASET_NAME = "eurlex-chat-data"


def query_all_documents():
    """Query documents via SPARQL with date filter.

    Returns DIR + REG documents with dates from FROM_DATE onward.
    Uses FILTER on xsd:dateTime so the SPARQL endpoint does the date
    filtering server-side, not in Python.
    """

    type_filters_list = [
        f"?type = <http://publications.europa.eu/resource/authority/resource-type/{t}>"
        for t in DOC_TYPES
    ]
    type_filter = " ||\n    ".join(type_filters_list)

    prefixes = "PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>"
    query = f"""{prefixes}
SELECT DISTINCT ?doc ?type ?celex ?date
WHERE {{
    ?doc cdm:work_has_resource-type ?type .
    FILTER(
      {type_filter}
    )
    ?doc cdm:resource_legal_id_celex ?celex .
    OPTIONAL {{ ?doc cdm:work_date_document ?date . }}
    FILTER(?date >= "{FROM_DATE}T00:00:00"^^xsd:dateTime)
}}
"""
    logger.info(f"SPARQL query for types: {DOC_TYPES}, from: {FROM_DATE}")
    r = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    bindings = data["results"]["bindings"]
    logger.info(f"  Raw results: {len(bindings)} documents")

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
    return all_docs


def fetch_document_xhtml(doc):
    """Fetch XHTML from Cellar CELEX endpoint.

    Uses publications.europa.eu (no WAF) with proper content negotiation.
    """
    celex = doc["celex"]
    try:
        url = (
            f"https://publications.europa.eu/resource/celex/"
            f"{quote(celex, safe='')}.ENG.xhtml"
        )
        r = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "application/xhtml+xml",
        })
        r.raise_for_status()

        if len(r.content) < 500:
            logger.warning(f"  Empty content for {celex}")
            return None

        return r.text

    except requests.RequestException as e:
        logger.debug(f"  HTTP error for {celex}: {e}")
        return None
    except Exception as e:
        logger.warning(f"  Unexpected error for {celex}: {e}")
        return None


def parse_html_to_chunks(html, celex_id, title):
    """Parse EUR-Lex HTML into chunks.

    Strategy:
      1. Try .eli-container → .eli-subdivision (structured articles)
      2. Try #text or #document1 (tab content)
      3. Try #documentView (fallback container)
      4. Last resort: extract meaningful paragraphs
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    if not title:
        title_el = soup.select_one(".eli-main-title .oj-doc-ti")
        if title_el:
            title = title_el.get_text(strip=True)
        elif soup.title:
            title = soup.title.string or ""

    # Strategy 1: ELI subdivisions (modern documents)
    container = soup.select_one(".eli-container")
    if container:
        subdivisions = container.select(".eli-subdivision")
        if subdivisions:
            chunks = []
            for sub in subdivisions:
                sub_id = sub.get("id", "")
                text = sub.get_text(separator=" ", strip=True)
                if not text or len(text) < 50:
                    continue

                if sub_id.startswith("art_"):
                    chunk_type = "article"
                elif sub_id.startswith("enc_"):
                    chunk_type = "enacting"
                elif sub_id.startswith("pbl_"):
                    chunk_type = "preamble"
                elif sub_id.startswith("ann_"):
                    chunk_type = "annex"
                else:
                    chunk_type = "section"

                chunks.append({
                    "text": text,
                    "celex": celex_id,
                    "title": title,
                    "article": sub_id,
                    "type": chunk_type,
                })

            if chunks:
                return chunks

    # Strategy 2: Tab content (corrigenda, some older docs)
    for tab_sel in ["#text", "#document1", "#PP4Contents"]:
        tab = soup.select_one(tab_sel)
        if tab:
            text = tab.get_text(separator="\n", strip=True)
            paragraphs = extract_meaningful_paragraphs(text)
            if paragraphs:
                return [{"text": p, "celex": celex_id, "title": title,
                         "article": None, "type": "paragraph"} for p in paragraphs]

    # Strategy 3: Document view (any page)
    doc_view = soup.select_one("#documentView")
    if doc_view:
        text = doc_view.get_text(separator="\n", strip=True)
        paragraphs = extract_meaningful_paragraphs(text)
        if paragraphs:
            return [{"text": p, "celex": celex_id, "title": title,
                     "article": None, "type": "paragraph"} for p in paragraphs]

    # Strategy 4: Full page text
    text = soup.get_text(separator="\n", strip=True)
    paragraphs = extract_meaningful_paragraphs(text)
    if paragraphs:
        return [{"text": p, "celex": celex_id, "title": title,
                 "article": None, "type": "paragraph"} for p in paragraphs]

    logger.warning(f"  No content could be extracted for {celex_id}")
    return []


def extract_meaningful_paragraphs(text):
    """Extract meaningful paragraphs, filtering out navigation and feature garbage."""
    lines = [p.strip() for p in text.split("\n") if p.strip()]
    meaningful = []
    for line in lines:
        line = line.strip()
        if len(line) < 40:
            continue
        if any(skip in line.lower() for skip in [
            "experimental feature", "deep linking", "visualisation of document",
            "replacement of celex identifiers", "do you want to help improving",
            "skip to main content", "you are here", "multilingual display",
            "toggle dropdown", "select your language", "more about this",
            "choose the experimental features", "official eu languages",
            "europa eur-lex home", "help print text", "document information",
            "up-to-date link", "permanent link", "download notice", "save to my items",
            "create an email alert", "create an rss", "log in", "sign in", "register",
            "my recent searches", "my eur-lex", "ecl-site-header",
        ]):
            continue
        if line.startswith("http://") or line.startswith("https://") or line.startswith("<"):
            continue
        meaningful.append(line)
    return meaningful


def embed_chunks(all_chunks, batch_size=128):
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


def build_faiss_index(vectors):
    """Build FAISS IVFPQ index and return the path."""
    if faiss is None:
        raise RuntimeError("faiss not installed. Run: pip install faiss-cpu")

    n_vectors, dim = vectors.shape
    n_centroids = min(int(4 * np.sqrt(n_vectors)), max(n_vectors // 40, 1))
    n_centroids = max(n_centroids, 1)

    index = faiss.index_factory(
        dim, f"IVF{n_centroids},PQ48x8", faiss.METRIC_INNER_PRODUCT
    )
    index.nprobe = min(8, n_centroids)

    logger.info(f"Training FAISS index: {n_centroids} centroids on {n_vectors} vectors...")
    index.train(vectors.astype(np.float32))
    logger.info("Adding vectors to index...")
    index.add(vectors.astype(np.float32))

    index_path = os.path.join(DATA_DIR, "index.faiss")
    faiss.write_index(index, index_path)
    index_size = os.path.getsize(index_path) / 1e6
    logger.info(f"FAISS index saved: {index_size:.1f} MB")
    return index_path


def build_chunks_db(all_chunks):
    """Build SQLite database of chunks and return the path."""
    db_path = os.path.join(DATA_DIR, "chunks.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            celex TEXT NOT NULL,
            title TEXT DEFAULT '',
            article TEXT DEFAULT NULL,
            type TEXT DEFAULT 'section',
            text TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_celex ON chunks(celex)")

    rows = [
        (i, c["celex"], c.get("title", ""), c.get("article"), c.get("type", "section"), c["text"])
        for i, c in enumerate(all_chunks)
    ]
    conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.execute("PRAGMA optimize")
    conn.close()

    db_size = os.path.getsize(db_path) / 1e6
    logger.info(f"SQLite DB saved: {len(rows)} chunks ({db_size:.1f} MB)")
    return db_path


def upload_to_hub(index_path, db_path, dataset_name, token):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    who = api.whoami()["name"]
    repo_id = f"{who}/{dataset_name}"

    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        logger.info(f"HF dataset repo: {repo_id}")
    except Exception as e:
        logger.warning(f"Repo creation warning (may already exist): {e}")

    ts_path = os.path.join(DATA_DIR, "last_updated.txt")
    ts = datetime.now(timezone.utc).isoformat()
    with open(ts_path, "w") as f:
        f.write(ts)

    api.upload_file(
        repo_id=repo_id,
        path_in_repo="index.faiss",
        path_or_fileobj=index_path,
        repo_type="dataset",
        token=token,
    )
    api.upload_file(
        repo_id=repo_id,
        path_in_repo="chunks.db",
        path_or_fileobj=db_path,
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

    for old_file in ["vectors.npy", "chunks.json", "vectors.npy", "test_vectors.npy", "test_chunks.json"]:
        try:
            api.delete_file(
                repo_id=repo_id,
                path_in_repo=old_file,
                repo_type="dataset",
                token=token,
            )
            logger.info(f"Deleted old file: {old_file}")
        except Exception:
            pass

    logger.info(f"Uploaded to HF Hub: {repo_id}")
    return repo_id


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable required")
        logger.error("Usage: HF_TOKEN=hf_yourtoken python3 scripts/build_index.py")
        return

    if faiss is None:
        logger.error("faiss not installed. Run: pip install faiss-cpu")
        return

    total_start = time.time()

    docs = query_all_documents()
    if not docs:
        logger.error("No documents found - SPARQL query returned empty")
        return
    logger.info(f"Documents to process: {len(docs)}")

    logger.info(f"Downloading and parsing documents ({DOWNLOAD_WORKERS} workers)...")
    all_chunks = []
    success_count = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        # Submit in batches so completed futures' results are freed promptly
        # (never hold more than BATCH_SIZE futures in memory)
        BATCH_SIZE = DOWNLOAD_WORKERS * 3  # 60
        for batch_start in range(0, len(docs), BATCH_SIZE):
            batch = docs[batch_start:batch_start + BATCH_SIZE]
            future_map = {
                executor.submit(fetch_document_xhtml, doc): doc for doc in batch
            }
            for future in tqdm(as_completed(future_map), total=len(future_map),
                              desc=f"Fetching {batch_start}-{batch_start + len(batch)}", leave=False):
                doc = future_map.pop(future)  # pop immediately to free the future's result
                try:
                    html = future.result()
                    if html:
                        chunks = parse_html_to_chunks(html, doc["celex"], "")
                        all_chunks.extend(chunks)
                        success_count += 1
                except Exception as e:
                    logger.debug(f"Failed {doc['celex']}: {e}")

            # Batch done — future_map is empty, all batch futures freed
            logger.info(f"  Batch done: {success_count}/{batch_start + len(batch)} docs, "
                        f"{len(all_chunks)} chunks so far")

    logger.info(f"Downloaded {success_count}/{len(docs)} documents successfully")
    logger.info(f"Total chunks: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks produced - check Cellar XHTML endpoint")
        return

    vectors = embed_chunks(all_chunks)
    dim = vectors.shape[1]
    logger.info(f"Embedding complete: {vectors.shape}")

    index_path = build_faiss_index(vectors)
    # vectors no longer needed — free ~700MB
    del vectors

    db_path = build_chunks_db(all_chunks)
    # all_chunks no longer needed — free ~456MB
    chunk_count = len(all_chunks)
    del all_chunks

    repo_id = upload_to_hub(index_path, db_path, HF_DATASET_NAME, hf_token)

    total_time = time.time() - total_start
    logger.info(f"Build complete in {total_time / 60:.1f} minutes")
    logger.info(f"Dataset: {repo_id}")
    logger.info(f"Documents: {success_count} | Chunks: {chunk_count} | Dims: {dim}")


if __name__ == "__main__":
    main()
