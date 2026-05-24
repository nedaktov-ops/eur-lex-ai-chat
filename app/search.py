"""FAISS KNN search over IVFPQ index + SQLite chunk lookup."""

import logging

logger = logging.getLogger(__name__)

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


def search(query_vector, top_k=10):
    """Basic FAISS KNN search (original, kept for backwards compatibility)."""
    from .data_loader import get_index

    index_data = get_index()
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
        })

    return results


def search_discourse_aware(query_vector, top_k=10, query_context=None):
    """FAISS search with discourse-aware re-ranking and boosting.

    Retrieves more candidates (2x top_k) then re-ranks using discourse boost.
    """
    from .data_loader import get_index

    index_data = get_index()
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
