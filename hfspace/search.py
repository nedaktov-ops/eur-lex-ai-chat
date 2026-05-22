"""FAISS KNN search over IVFPQ index + SQLite chunk lookup."""

import logging

logger = logging.getLogger(__name__)


def search(query_vector, top_k=10):
    from data_loader import get_index

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
