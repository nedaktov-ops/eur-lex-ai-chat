"""Cross-encoder reranker for refining search results."""

from sentence_transformers import CrossEncoder


class Reranker:
    """Reranker using a cross-encoder model for precise query-document relevance scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder model.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        """Rerank candidate documents using the cross-encoder.

        Args:
            query: The user query string.
            candidates: List of candidate dicts, each must contain a 'text' field.
            top_k: Number of top results to return.

        Returns:
            List of candidate dicts with an added 'rerank_score' field, sorted descending,
            limited to top_k.
        """
        if not candidates:
            return []

        # Prepare query-passage pairs
        pairs = [(query, cand["text"]) for cand in candidates]

        # Predict scores
        scores = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        # Convert to Python floats if numpy array
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        else:
            scores = list(scores)

        # Attach scores to candidates
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        # Sort by rerank_score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]
