import pickle
import re
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi


class BM25Store:
    """BM25 index for document retrieval with persistence support."""
    
    def __init__(self):
        self.index = None
        self.chunk_ids: List[str] = []
        self.documents: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        r"""Tokenize text using regex \w{2,} in lowercase."""
        return re.findall(r'\w{2,}', text.lower())
    
    def build(self, chunks: Dict[str, str]) -> None:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: Dictionary mapping chunk_id to text content
        """
        self.chunk_ids = list(chunks.keys())
        self.documents = [self._tokenize(text) for text in chunks.values()]
        self.index = BM25Okapi(self.documents, k1=1.5, b=0.75)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for top_k most relevant chunks.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of dicts with chunk_id, score, and rank (1-indexed)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        query_tokens = self._tokenize(query)
        scores = self.index.get_scores(query_tokens)
        
        # Get top_k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(scores[idx]),
                    "rank": rank
                })
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save index to disk using pickle.
        
        Args:
            path: File path to save to
        """
        if self.index is None:
            raise ValueError("No index to save. Call build() first.")
        
        data = {
            'chunk_ids': self.chunk_ids,
            'documents': self.documents,
            'index': self.index
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """
        Load index from disk.
        
        Args:
            path: File path to load from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.chunk_ids = data['chunk_ids']
        self.documents = data['documents']
        self.index = data['index']
