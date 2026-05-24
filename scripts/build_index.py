#!/usr/bin/env python3
"""
build_index.py — Full build of the EUR-Lex vector index.

Pipeline:
  1. SPARQL query -> DIR + REG docs (non-corrigenda) from 2004-01-01 onward
  2. Fetch XHTML from Cellar CELEX endpoint (parallel, 20 workers)
  3. Parse XHTML via BeautifulSoup into chunks (articles, preamble, annex)
  4. Embed with sentence-transformers (all-MiniLM-L6-v2, 384-dim, normalized)
  5. Build FAISS IVFPQ index (~28MB for 500K vectors, vs 768MB float32)
  6. Build SQLite DB of chunks (~35MB, on-disk, zero RAM at query time)
  7. Upload index.faiss + chunks.db + last_updated.txt to HuggingFace Hub
  8. Delete old vectors.npy + chunks.json from Hub

Memory optimizations:
  - ThreadPoolExecutor batches with immediate future.pop() per future (CPython
    bpo-27144: as_completed no longer keeps references to yielded objects)
  - Corrigenda filtered at SPARQL level (~15-20% fewer downloads)
  - Explicit del + gc.collect() after embedding and chunk building
  - FAISS IVFPQ index: ~28MB vs 768MB for raw float32 (PQ48x8 = 48 bytes/vector)
  - use_precomputed_table=-1 saves ~127MB of precomputed distance tables
  - SQLite chunks: on-disk, only loads matched rows (top-10 per query)
  - Total RAM at query time: ~363MB (well under 512MB HF Space limit)

Usage:
  source ~/Desktop/EUProjects/.venv/bin/activate
  HF_TOKEN=hf_yourtoken python3 scripts/build_index.py

Output:
  data/index.faiss       - FAISS IVFPQ index (~28MB)
  data/chunks.db         - SQLite database of chunks (~35MB)
  data/last_updated.txt  - ISO timestamp of build

References:
  - CPython bpo-27144: concurrent.futures memory leak fix
  - FAISS guidelines: nlist ≈ 4*sqrt(N) for IVF (issue #2692)
  - EUR-Lex corrigendum pattern: R(xx) suffix on CELEX number
"""

import gc
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
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

# Capture git revision for traceability
GIT_REV = ""
try:
    GIT_REV = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, timeout=5,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    ).stdout.strip()
except Exception:
    pass

# Configuration

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_SUFFIX = os.environ.get("INDEX_SUFFIX", "")  # e.g., "_eurlex" for EURLEX-BERT

# Model-dependent defaults
_IS_EURLEX = "eurlex" in EMBEDDING_MODEL.lower()
EMBED_DIM = 768 if _IS_EURLEX else 384
# PQ config: 384/48 = 8 sub-quantizers, 768/48 = 16 sub-quantizers
PQ_BITS = 48
FAISS_INDEX_FACTORY = f"IVF{{nlist}},PQ{PQ_BITS}x{{dim//PQ_BITS}}"

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

DOC_TYPES = [
    "REG",
    "DIR",
]

FROM_DATE = os.environ.get("FROM_DATE", "2004-01-01")
MAX_CHUNKS = os.environ.get("MAX_CHUNKS")
if MAX_CHUNKS is not None:
    MAX_CHUNKS = int(MAX_CHUNKS)
DOWNLOAD_WORKERS = 20
HF_DATASET_NAME = "eurlex-chat-data"
HF_BACKUP_DATASET = "NedAktovOps/eurlex-chat-backups"


def _cap_chunks(all_chunks, success_count):
    """Truncate chunks to MAX_CHUNKS limit if set."""
    if MAX_CHUNKS and len(all_chunks) > MAX_CHUNKS:
        logger.warning("Capping chunks from %d to %d (MAX_CHUNKS=%d)",
                       len(all_chunks), MAX_CHUNKS, MAX_CHUNKS)
        # Keep only first MAX_CHUNKS chunks
        truncated = all_chunks[:MAX_CHUNKS]
        # Recompute success_count from unique CELEXes in remaining chunks
        success_count = len(set(c["celex"] for c in truncated))
        return truncated, success_count
    return all_chunks, success_count


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
    # Exclude corrigenda — EUR-Lex docs confirm R(xx) suffix pattern
    FILTER(!CONTAINS(?celex, "R("))
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


def embed_chunks(all_chunks, batch_size=None):
    """Embed all chunks using sentence-transformers or EURLEX-BERT.

    Uses EMBEDDING_MODEL from environment (default: all-MiniLM-L6-v2).
    Memory-efficient batching: embed BATCH_SIZE chunks at a time.
    """
    if batch_size is None:
        batch_size = 64 if _IS_EURLEX else 1024
    if _IS_EURLEX:
        logger.info(f"Loading EURLEX-BERT model: {EMBEDDING_MODEL} (768-dim)")
        import torch
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        model.eval()

    texts = [c["text"] for c in all_chunks]
    all_embeddings = []

    logger.info(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1

        if _IS_EURLEX:
            import torch
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        else:
            embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(embeddings)

        if batch_num % 5 == 0 or batch_num == total_batches:
            mem_mb = _get_memory_mb()
            logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} "
                        f"(batch {batch_num}/{total_batches}) [mem: {mem_mb:.0f}MB]")

    return np.vstack(all_embeddings).astype(np.float32)


def _get_memory_mb():
    """Get current process memory usage in MB (best effort)."""
    try:
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / 1e6
    except ImportError:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # /proc reports in kB
                        return float(line.split()[1]) / 1024
        except Exception:
            pass
        return 0.0


def build_faiss_index(vectors):
    """Build FAISS IVFPQ index and return the path.

    Uses IVF(nlist) + PQ48x8 for memory-efficient approximate search.
    Evidence:
      - nlist = 4*sqrt(N) per FAISS issue #2692
      - PQ48x8 = 48 bytes/vector (384/48 = 8 dims per sub-quantizer)
      - use_precomputed_table=-1 to save ~127MB of precomputed tables
      - nprobe ~ sqrt(nlist) for optimal recall/speed tradeoff
    """
    if faiss is None:
        raise RuntimeError("faiss not installed. Run: pip install faiss-cpu")

    n_vectors, dim = vectors.shape
    # nlist = 4*sqrt(N) as recommended by FAISS guidelines (issue #2692)
    n_centroids = min(int(4 * np.sqrt(n_vectors)), max(n_vectors // 40, 1))
    n_centroids = max(n_centroids, 1)

    index = faiss.index_factory(
        dim, f"IVF{n_centroids},PQ48x8", faiss.METRIC_INNER_PRODUCT
    )

    logger.info(f"Training FAISS index: {n_centroids} centroids on {n_vectors} vectors...")
    index.train(vectors.astype(np.float32))

    logger.info("Adding vectors to index...")
    index.add(vectors.astype(np.float32))

    # Disable precomputed table to save ~127MB (nlist × M × ksub × 4 bytes).
    # Must be set AFTER add() because add() internally resets it.
    # FAISS IndexIVFPQ.h: precomputed_table size = nlist * pq.M * pq.ksub
    index.use_precomputed_table = -1
    logger.info("Precomputed distance table disabled (saves ~127MB)")

    # nprobe ~ sqrt(nlist) per FAISS wiki: recall knee near sqrt(nlist)
    nprobe = min(50, n_centroids)
    index.nprobe = nprobe
    logger.info(f"nprobe set to {nprobe}")

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


def upload_to_hub(index_path, db_path, dataset_name, token, success_count=0, chunk_count=0):
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
    ts = datetime.now(UTC).isoformat()
    build_meta = {
        "timestamp": ts,
        "git_rev": GIT_REV,
        "n_documents": success_count,
        "n_chunks": chunk_count,
    }
    with open(ts_path, "w") as f:
        f.write(ts)
    # Also write structured metadata
    meta_path = os.path.join(DATA_DIR, "build_meta.json")
    with open(meta_path, "w") as f:
        json.dump(build_meta, f, indent=2)

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
    api.upload_file(
        repo_id=repo_id,
        path_in_repo="build_meta.json",
        path_or_fileobj=meta_path,
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


def _load_chunks_from_db(db_path):
    """Load chunks from a local chunks.db SQLite file.

    Returns list of chunk dicts or empty list if file missing.
    """
    if not os.path.exists(db_path):
        logger.warning("chunks.db not found at %s", db_path)
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT celex, title, article, type, text FROM chunks ORDER BY id").fetchall()
    conn.close()
    chunks = [
        {"celex": r[0], "title": r[1] or "", "article": r[2], "type": r[3], "text": r[4]}
        for r in rows
    ]
    logger.info("Loaded %d chunks from %s", len(chunks), db_path)
    return chunks


def load_chunks_from_backup(token):
    """Load chunks from the latest HF backup dataset, skipping SPARQL crawl.

    Returns list of chunk dicts with keys: celex, title, article, type, text.
    Returns empty list if no backup found.
    """
    import shutil
    import tempfile

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        refs = api.list_repo_refs(HF_BACKUP_DATASET, repo_type="dataset")
        backup_branches = [b for b in (refs.branches or []) if b.name.startswith("backup-")]
        if not backup_branches:
            logger.warning("No backup branches found in %s", HF_BACKUP_DATASET)
            return []
        latest = sorted(backup_branches, key=lambda b: b.name, reverse=True)[0]
        logger.info("Loading chunks from backup: %s@%s", HF_BACKUP_DATASET, latest.name)
    except Exception:
        logger.warning("Could not list backup branches (dataset may not exist yet)")
        return []

    restore_dir = tempfile.mkdtemp(prefix="backup-restore-")
    try:
        api.snapshot_download(
            repo_id=HF_BACKUP_DATASET,
            repo_type="dataset",
            revision=latest.name,
            local_dir=restore_dir,
            local_dir_use_symlinks=False,
        )
        db_path = os.path.join(restore_dir, "chunks.db")
        return _load_chunks_from_db(db_path)
    finally:
        shutil.rmtree(restore_dir, ignore_errors=True)


def _download_and_chunk(docs):
    """Download documents from SPARQL, parse XHTML, return (chunks, success_count)."""
    logger.info(f"Downloading and parsing documents ({DOWNLOAD_WORKERS} workers)...")
    all_chunks = []
    success_count = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        batch_size = DOWNLOAD_WORKERS * 3  # 60
        for batch_start in range(0, len(docs), batch_size):
            batch = docs[batch_start:batch_start + batch_size]
            future_map = {
                executor.submit(fetch_document_xhtml, doc): doc for doc in batch
            }
            for future in tqdm(as_completed(future_map), total=len(future_map),
                               desc=f"Fetching {batch_start}-{batch_start + len(batch)}", leave=False):
                doc = future_map.pop(future)
                try:
                    html = future.result()
                    if html:
                        chunks = parse_html_to_chunks(html, doc["celex"], "")
                        all_chunks.extend(chunks)
                        success_count += 1
                except Exception as e:
                    logger.debug(f"Failed {doc['celex']}: {e}")

            logger.info(f"  Batch done: {success_count}/{batch_start + len(batch)} docs, "
                        f"{len(all_chunks)} chunks so far")

    logger.info(f"Downloaded {success_count}/{len(docs)} documents successfully")
    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks, success_count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build EUR-Lex vector index")
    parser.add_argument("--from-backup", action="store_true",
                        help="Load chunks from HF backup instead of SPARQL crawl")
    parser.add_argument("--from-artifact", action="store_true",
                        help="Load chunks from local data/chunks.db (GitHub artifact)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download + chunk, save data/chunks.db, then exit")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable required")
        logger.error("Usage: HF_TOKEN=hf_yourtoken python3 scripts/build_index.py")
        return

    total_start = time.time()

    # Mode 1: --download-only (SPARQL crawl → chunks.db, for Job 1)
    if args.download_only:
        if faiss is None:
            logger.warning("faiss not installed (not needed for download-only)")
        docs = query_all_documents()
        if not docs:
            logger.error("No documents found")
            return
        logger.info(f"Documents to process: {len(docs)}")
        all_chunks, success_count = _download_and_chunk(docs)
        if not all_chunks:
            logger.error("No chunks produced")
            return
        build_chunks_db(all_chunks)
        logger.info("Download phase complete. data/chunks.db ready for artifact.")
        total_time = time.time() - total_start
        logger.info(f"  Duration: {total_time / 60:.1f} minutes")
        logger.info(f"  Documents: {success_count}")
        logger.info(f"  Chunks: {len(all_chunks)}")
        return

    # Mode 2: --from-artifact (load chunks from local file, for Job 2)
    if args.from_artifact:
        db_path = os.path.join(DATA_DIR, "chunks.db")
        all_chunks = _load_chunks_from_db(db_path)
        if not all_chunks:
            logger.error("No chunks found in data/chunks.db")
            return
        success_count = len(set(c["celex"] for c in all_chunks))
        logger.info(f"Loaded {len(all_chunks)} chunks from artifact "
                    f"({success_count} documents)")
        _build_from_chunks(all_chunks, success_count, hf_token, total_start)
        return

    if faiss is None:
        logger.error("faiss not installed. Run: pip install faiss-cpu")
        return

    # Mode 3: --from-backup (load from HF backup, existing behavior)
    if args.from_backup:
        all_chunks = load_chunks_from_backup(hf_token)
        if not all_chunks:
            logger.error("No backup chunks available")
            return
        success_count = len(set(c["celex"] for c in all_chunks))
        logger.info(f"Loaded {len(all_chunks)} chunks from backup "
                    f"({success_count} documents)")
        _build_from_chunks(all_chunks, success_count, hf_token, total_start)
        return

    # Mode 4: Full pipeline (SPARQL → download → embed → FAISS → upload)
    docs = query_all_documents()
    if not docs:
        logger.error("No documents found - SPARQL query returned empty")
        return
    logger.info(f"Documents to process: {len(docs)}")

    all_chunks, success_count = _download_and_chunk(docs)

    if not all_chunks:
        logger.error("No chunks produced - check Cellar XHTML endpoint")
        return

    _build_from_chunks(all_chunks, success_count, hf_token, total_start)


def _build_from_chunks(all_chunks, success_count, hf_token, total_start):
    """Embed chunks, build FAISS index + SQLite DB, and upload to HF Hub."""
    vectors = embed_chunks(all_chunks)
    dim = vectors.shape[1]
    logger.info(f"Embedding complete: {vectors.shape}")

    index_path = build_faiss_index(vectors)
    # vectors no longer needed — free ~700MB
    del vectors
    gc.collect()
    logger.info("Memory freed: embeddings (~700MB)")

    db_path = build_chunks_db(all_chunks)
    # all_chunks no longer needed — free ~456MB
    chunk_count = len(all_chunks)
    del all_chunks
    gc.collect()
    logger.info("Memory freed: chunks (~456MB)")

    repo_id = upload_to_hub(index_path, db_path, HF_DATASET_NAME, hf_token,
                            success_count=success_count, chunk_count=chunk_count)

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Git revision:  {GIT_REV}")
    logger.info(f"  Duration:      {total_time / 60:.1f} minutes")
    logger.info(f"  Documents:     {success_count}")
    logger.info(f"  Chunks:        {chunk_count}")
    logger.info(f"  Dimensions:    {dim}")
    logger.info(f"  Dataset:       {repo_id}")
    logger.info("  Index:         data/index.faiss")
    logger.info("  Database:      data/chunks.db")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Build interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Build failed: {e}")
        sys.exit(1)
