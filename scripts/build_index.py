#!/usr/bin/env python3
"""
build_index.py — One-time full build of the EUR-Lex vector index.

Pipeline:
  1. SPARQL query -> list of CELEX IDs with metadata
  2. Download HTML from EUR-Lex directly (1 HTTP request, ~0.6s avg)
  3. Parse HTML via BeautifulSoup into chunks (articles, preamble, etc.)
  4. Embed with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
  5. Upload vectors.npy + chunks.json to HuggingFace Hub

Usage:
  source ~/Desktop/EUProjects/.venv/bin/activate
  HF_TOKEN=hf_yourtoken python3 scripts/build_index.py

Output:
  data/vectors.npy       - numpy array of shape (N, 384), float32
  data/chunks.json       - list of dicts with text + metadata
  data/last_updated.txt  - ISO timestamp of build
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

# Configuration

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

DOC_TYPES = [
    "REG",
    "DIR",
    "REG_IMPL",
    "DIR_IMPL",
]

FROM_DATE = "2000-01-01"
DOWNLOAD_WORKERS = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HF_DATASET_NAME = "eurlex-chat-data"


def query_all_documents():
    """Query all documents via SPARQL with date filter at query level.

    Builds the same query pattern as eurlxp.get_documents() but adds date filtering
    to avoid fetching all 143K documents and filtering in Python.
    """
    import requests as req

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
    r = req.get(
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
    """Fetch document HTML directly from EUR-Lex (1 HTTP request vs 3 for Cellar RDF).

    EUR-Lex HTML endpoint is ~10x faster than Cellar RDF traversal.
    Content extracted from .eli-subdivision divs using BeautifulSoup.
    """
    celex = doc["celex"]
    try:
        from urllib.parse import quote

        encoded = quote(celex, safe="")
        url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{encoded}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        if len(r.text) < 500:
            logger.warning(f"  Empty content for {celex}")
            return None

        return r.text

    except requests.RequestException as e:
        logger.warning(f"  HTTP error for {celex}: {e}")
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


def upload_to_hub(vectors, chunks, dataset_name, token):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{dataset_name}"

    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        logger.info(f"HF dataset repo: {repo_id}")
    except Exception as e:
        logger.warning(f"Repo creation warning (may already exist): {e}")

    vectors_path = os.path.join(DATA_DIR, "vectors.npy")
    np.save(vectors_path, vectors)
    logger.info(f"Saved vectors: {vectors.shape} ({os.path.getsize(vectors_path) / 1e6:.1f} MB)")

    chunks_path = os.path.join(DATA_DIR, "chunks.json")
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)
    logger.info(f"Saved chunks: {len(chunks)} items ({os.path.getsize(chunks_path) / 1e6:.1f} MB)")

    ts_path = os.path.join(DATA_DIR, "last_updated.txt")
    ts = datetime.now(timezone.utc).isoformat()
    with open(ts_path, "w") as f:
        f.write(ts)

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


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable required")
        logger.error("Usage: HF_TOKEN=hf_yourtoken python3 scripts/build_index.py")
        return

    total_start = time.time()

    docs = query_all_documents()
    if not docs:
        logger.error("No documents found - SPARQL query returned empty")
        return
    logger.info(f"Documents to process: {len(docs)}")

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
        logger.error("No chunks produced - check parsing")
        return

    vectors = embed_chunks(all_chunks)
    logger.info(f"Embedding complete: {vectors.shape}")

    repo_id = upload_to_hub(vectors, all_chunks, HF_DATASET_NAME, hf_token)

    total_time = time.time() - total_start
    logger.info(f"Build complete in {total_time / 60:.1f} minutes")
    logger.info(f"Dataset: {repo_id}")
    logger.info(f"Documents: {len(html_results)} | Chunks: {len(all_chunks)} | Dims: {vectors.shape[1]}")


if __name__ == "__main__":
    main()
