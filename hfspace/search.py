"""numpy KNN search over pre-loaded vectors."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def search(query_vector, top_k=10):
    from data_loader import get_index

    index = get_index()
    vectors = index["vectors"]
    chunks = index["chunks"]

    if vectors is None or chunks is None:
        logger.error("Index not loaded")
        return []

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    similarities = np.dot(vectors, query_vector.T).flatten()

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
