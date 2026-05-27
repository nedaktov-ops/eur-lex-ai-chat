"""Hybrid search using RRF (Reciprocal Rank Fusion) to combine FAISS and BM25 results."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Combines dense (FAISS) and sparse (BM25) retrieval using RRF fusion.

    Attributes:
        bm25_store: BM25Store instance or None (optional)
        faiss_index: FAISS index object with search method
        chunks_db: SQLite connection to chunks database
        rrf_k: RRF constant (default 60)
    """

    def __init__(self, bm25_store, faiss_index, chunks_db, rrf_k: int = 60):
        self.bm25_store = bm25_store
        self.faiss_index = faiss_index
        self.chunks_db = chunks_db
        self.rrf_k = rrf_k

    def search_rrf(self, query_vector, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search with RRF fusion.

        Retrieves top 3*top_k from each retriever (FAISS and BM25 if available),
        then combines using Reciprocal Rank Fusion: score = 1/(k + rank).

        Args:
            query_vector: Dense embedding vector (numpy array)
            query_text: Original query text for BM25
            top_k: Number of final results to return

        Returns:
            List of chunk result dicts with keys: id, celex, title, article, type, text, score
        """
        fetch_k = top_k * 3

        # Step 1: FAISS search
        faiss_hits = []
        try:
            distances, indices = self.faiss_index.search(query_vector.astype("float32"), fetch_k)
            for rank, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                chunk_id = int(idx)
                faiss_hits.append({
                    "id": chunk_id,
                    "faiss_rank": rank,
                    "faiss_distance": float(distances[0][rank]),
                })
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

        # Step 2: BM25 search (if available)
        bm25_hits = []
        if self.bm25_store is not None:
            try:
                bm25_results = self.bm25_store.search(query_text, top_k=fetch_k)
                for rank, hit in enumerate(bm25_results):
                    # Support both object with .chunk_id and dict with 'chunk_id' or 'id'
                    if hasattr(hit, "chunk_id"):
                        cid = hit.chunk_id
                    elif isinstance(hit, dict):
                        cid = hit.get("chunk_id", hit.get("id"))
                    else:
                        cid = getattr(hit, "id", None)
                    if cid is None:
                        continue
                    bm25_hits.append({
                        "id": int(cid),
                        "bm25_rank": rank,
                        "bm25_score": getattr(hit, "score", hit.get("score", 0.0)),
                    })
            except Exception as e:
                logger.warning(f"BM25 search skipped: {e}")

        # Step 3: Collect all unique chunk IDs
        all_ids = list(set(hit["id"] for hit in faiss_hits + bm25_hits))
        if not all_ids:
            return []

        # Step 4: Fetch chunk data from database
        placeholders = ",".join("?" for _ in all_ids)
        try:
            cursor = self.chunks_db.execute(
                f"SELECT id, celex, title, article, type, text FROM chunks WHERE id IN ({placeholders})",
                all_ids,
            )
            rows = cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {e}")
            return []

        row_map = {row["id"]: dict(row) for row in rows}

        # Step 5: RRF fusion
        results_map: Dict[int, Dict[str, Any]] = {}
        for hit in faiss_hits:
            cid = hit["id"]
            if cid not in row_map:
                continue
            rrf_score = 1.0 / (self.rrf_k + hit["faiss_rank"])
            row = row_map[cid]
            results_map[cid] = {
                "id": cid,
                "celex": row["celex"],
                "title": row["title"],
                "article": row["article"],
                "type": row.get("type"),
                "text": row["text"],
                "score": rrf_score,
            }

        for hit in bm25_hits:
            cid = hit["id"]
            if cid not in row_map:
                continue
            rrf_score = 1.0 / (self.rrf_k + hit["bm25_rank"])
            if cid in results_map:
                results_map[cid]["score"] += rrf_score
            else:
                row = row_map[cid]
                results_map[cid] = {
                    "id": cid,
                    "celex": row["celex"],
                    "title": row["title"],
                    "article": row["article"],
                    "type": row.get("type"),
                    "text": row["text"],
                    "score": rrf_score,
                }

        # Step 6: Sort by RRF score and return top_k
        results = list(results_map.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
