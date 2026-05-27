"""FAISS KNN search over IVFPQ index + SQLite chunk lookup."""

import logging

from .data_loader import get_bm25_store, get_index

logger = logging.getLogger(__name__)

# Optional hybrid search with RRF fusion
try:
    from .hybrid_search import HybridSearcher
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# Discourse weights for different article types
DISCOURSE_WEIGHTS = {
    "operative_article": 1.3,   # Articles with "shall" — legal obligations
    "recital": 0.9,             # Background/context recitals
    "definition_article": 1.1,  # Articles with definitions
    "penalty_article": 1.2,     # Articles with sanctions/penalties
    "annex": 0.8,               # Annexes (technical details)
}

# Deontic language patterns for obligation detection
DEONTIC_PATTERNS = {
    "shall", "must", "required", "obliged", "duty", "duties",
    "responsible", "liability", "sanction", "penalty", "breach",
    "prohibited", "not permitted", "shall ensure", "shall take",
    "shall establish", "shall implement", "shall report",
    "obligation", "obligations", "responsibility", "responsibilities",
    "mandatory", "compliance", "compliant", "bound to", "obliged to",
}


def discourse_boost(chunk: dict, query_context: dict = None) -> float:
    """Calculate discourse-aware boost factor for a chunk.

    Args:
        chunk: A search result dict with text, celex, article, score keys.
        query_context: Optional dict with classification info (obligation_seeking, etc.).

    Returns:
        Boost multiplier (1.0 = no change, >1.0 = boost, <1.0 = demote).
    """
    boost = 1.0
    text_lower = chunk.get("text", "").lower()
    article_code = chunk.get("article", "")

    # Boost chunks with deontic language for obligation queries
    if query_context and query_context.get("obligation_seeking"):
        deontic_count = sum(1 for word in DEONTIC_PATTERNS if word in text_lower)
        if deontic_count > 0:
            boost *= min(1.0 + (deontic_count * 0.08), 1.5)

    # Boost/demote based on article type
    if article_code and article_code.startswith("art_"):
        boost *= DISCOURSE_WEIGHTS["operative_article"]
    elif article_code and article_code.startswith("rct_"):
        boost *= DISCOURSE_WEIGHTS["recital"]
    elif article_code and article_code.startswith("anx_"):
        boost *= DISCOURSE_WEIGHTS["annex"]

    return boost


def search(query_vector, top_k=10, model_name=None):
    """Basic FAISS KNN search (original, kept for backwards compatibility)."""
    from .data_loader import get_index

    index_data = get_index(model_name=model_name)
    faiss_index = index_data["index"]
    conn = index_data["conn"]
    lock = index_data["lock"]

    if faiss_index is None or conn is None:
        logger.error("Index not loaded")
        return []

    distances, indices = faiss_index.search(query_vector.astype("float32"), top_k)

    if indices[0][0] == -1:
        return []

    ids = [int(i) for i in indices[0] if i != -1]
    if not ids:
        return []

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

    results = []
    for i, idx in enumerate(ids):
        row = row_map.get(idx)
        if row is None:
            continue
        results.append({
            "score": float(distances[0][i]),
            "text": row["text"],
            "celex": row["celex"],
            "title": row["title"],
            "article": row["article"],
            "chunk_id": int(row["id"]),
        })

    return results


def search_discourse_aware(query_vector, top_k=10, query_context=None, model_name=None):
    """FAISS search with discourse-aware re-ranking and boosting.

    Retrieves more candidates (2x top_k) then re-ranks using discourse boost.
    """
    from .data_loader import get_index

    index_data = get_index(model_name=model_name)
    faiss_index = index_data["index"]
    conn = index_data["conn"]
    lock = index_data["lock"]

    if faiss_index is None or conn is None:
        logger.error("Index not loaded")
        return []

    # Retrieve more candidates for re-ranking
    distances, indices = faiss_index.search(query_vector.astype("float32"), top_k * 2)

    if indices[0][0] == -1:
        return []

    ids = [int(i) for i in indices[0] if i != -1]
    if not ids:
        return []

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
        boost = discourse_boost(chunk, query_context)
        chunk["discourse_boost"] = round(boost, 3)
        chunk["adjusted_score"] = round(base_score * boost, 6)
        results.append(chunk)

    # Re-rank by adjusted score
    results.sort(key=lambda c: c["adjusted_score"], reverse=True)

    return results[:top_k]


def get_bm25_results(query_text: str, top_k: int = 20, model_name=None, suffix=None):
    """Get BM25 search results with full chunk data.

    Args:
        query_text: The query string.
        top_k: Number of top results to return.
        model_name: Model name to select the appropriate BM25 store and index.
        suffix: Direct index suffix (overrides model_name if provided).

    Returns:
        List of result dicts with keys: score, text, celex, title, article, chunk_id.
    """
    store = get_bm25_store(model_name=model_name, suffix=suffix)
    raw_results = store.search(query_text, top_k=top_k)
    if not raw_results:
        return []

    index_data = get_index(model_name=model_name, suffix=suffix)
    conn = index_data["conn"]
    lock = index_data["lock"]

    # Extract chunk IDs (as integers)
    chunk_ids = [int(item["chunk_id"]) for item in raw_results]
    placeholders = ",".join("?" for _ in chunk_ids)

    lock.acquire()
    try:
        rows = conn.execute(
            f"SELECT id, celex, title, article, text FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
    finally:
        lock.release()

    row_map = {r["id"]: r for r in rows}

    results = []
    for raw in raw_results:
        cid = int(raw["chunk_id"])
        row = row_map.get(cid)
        if row is None:
            continue
        results.append({
            "score": raw["score"],
            "text": row["text"],
            "celex": row["celex"],
            "title": row["title"],
            "article": row["article"],
            "chunk_id": cid,
        })

    return results


def rrf_fuse(result_lists: list, k: int = 60, top_n: int = 20) -> list:
    """Fuse multiple result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of result lists, each a list of dicts with 'chunk_id'.
        k: Constant to avoid division by zero and control influence of top ranks.
        top_n: Number of top results to return.

    Returns:
        List of candidate dicts (from the first occurrence) with added 'rrf_score',
        sorted descending, limited to top_n.
    """
    scores = {}
    item_map = {}

    for lst in result_lists:
        for rank, item in enumerate(lst, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
            if cid not in item_map:
                item_map[cid] = item

    sorted_cids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

    top_items = []
    for cid in sorted_cids[:top_n]:
        item = item_map[cid].copy()
        item["rrf_score"] = scores[cid]
        top_items.append(item)

    return top_items


def search_hybrid(query_vector, query_text: str, top_k: int = 10, model_name=None):
    """Hybrid search using RRF fusion of FAISS and BM25 if available.

    Falls back to FAISS-only search if HybridSearcher is not available.

    Args:
        query_vector: Dense embedding vector for the query
        query_text: Original query string for BM25
        top_k: Number of results to return
        model_name: Model name to select the appropriate index and BM25 store.

    Returns:
        List of chunk result dicts with keys: id, celex, title, article, type, text, score
    """
    if not HYBRID_AVAILABLE:
        logger.warning("HybridSearcher not available, falling back to FAISS search")
        return search(query_vector, top_k, model_name=model_name)

    from .data_loader import get_index, get_bm25_store
    index_data = get_index(model_name=model_name)
    faiss_index = index_data["index"]
    conn = index_data["conn"]

    if faiss_index is None or conn is None:
        logger.error("Index not loaded")
        return []

    # Get BM25 store if available
    bm25_store = None
    try:
        bm25_store = get_bm25_store(model_name=model_name)
    except Exception as e:
        logger.info(f"BM25 store not available: {e}")

    searcher = HybridSearcher(
        bm25_store=bm25_store,
        faiss_index=faiss_index,
        chunks_db=conn,
        rrf_k=60
    )
    return searcher.search_rrf(query_vector, query_text, top_k)
