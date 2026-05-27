#!/usr/bin/env python3
"""
RAGAS evaluation pipeline for EUR-Lex AI Chat.

Evaluates faithfulness, answer_relevancy, context_precision, and context_recall.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Patch missing langchain_community vertexai module (sunset in newer versions)
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    mock_vertexai = MagicMock()
    mock_vertexai.ChatVertexAI = MagicMock
    sys.modules['langchain_community.chat_models.vertexai'] = mock_vertexai

import numpy as np
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Patch missing FeedbackRequest in app module to avoid NameError on import
try:
    import app as _app_mod
    if not hasattr(_app_mod, 'FeedbackRequest'):
        _app_mod.FeedbackRequest = type('FeedbackRequest', (), {})
except Exception:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = project_root / "data"
DATASET_PATH = DATA_DIR / "qa_dataset.json"
DOCS_DIR = project_root / "docs"
OUTPUT_DIR = DOCS_DIR / "eval_results"
BATCH_SIZE = 5  # number of samples to evaluate


def load_dataset():
    """Load evaluation dataset from JSON."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {DATASET_PATH}")
    return data[:BATCH_SIZE]


def init_rag_system():
    """Initialize the RAG pipeline components."""
    logger.info("Initializing RAG system...")
    from app.data_loader import download_index, get_bm25_store
    from app.main import get_embedding_model

    # Download and load index if not present
    download_index()
    logger.info("Index loaded")

    # Load embedding model
    model = get_embedding_model()
    logger.info("Embedding model loaded")

    # Load or build BM25 store
    bm25_store = get_bm25_store()
    logger.info("BM25 store ready")

    return model, bm25_store


def run_rag(query: str):
    """Run the RAG pipeline for a single query and return answer and context chunks."""
    from app.search import rrf_fuse, search as faiss_search, get_bm25_results
    from app.reranker import Reranker
    from app.rag import answer_question

    # Encode query
    q_vec = model.encode([query], normalize_embeddings=True)

    # Retrieve candidates
    bm25_res = get_bm25_results(query, top_k=20)
    faiss_res = faiss_search(q_vec, top_k=20)

    # Reciprocal Rank Fusion
    candidates = rrf_fuse([bm25_res, faiss_res], k=60, top_n=20)

    # Rerank
    reranker = Reranker()
    chunks = reranker.rerank(query, candidates, top_k=10)

    # Generate answer
    result = answer_question(query, chunks)
    return result, chunks


def compute_metrics(records):
    """Compute RAGAS metrics for the evaluation records."""
    # Convert to Dataset format expected by RAGAS
    dataset = Dataset.from_list(records)

    # Use metric classes (not instantiated) as per RAGAS API
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    logger.info("Evaluating metrics with RAGAS...")
    evaluation = evaluate(dataset, metrics=metrics)
    return evaluation.to_pandas() if hasattr(evaluation, 'to_pandas') else evaluation


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset()

    # Initialize RAG system once
    global model
    model, _ = init_rag_system()

    # Run RAG for each sample and collect data
    records = []
    for idx, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        logger.info(f"Processing sample {idx+1}/{len(dataset)}: {question[:80]}...")
        try:
            result, chunks = run_rag(question)
            answer = result.get("answer", "")
            contexts = [chunk["text"] for chunk in chunks]
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            answer = ""
            contexts = []

        record = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        }
        records.append(record)

    # Save raw results before metrics
    raw_path = OUTPUT_DIR / f"raw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"Raw results saved to {raw_path}")

    # Compute metrics
    try:
        metrics_df = compute_metrics(records)
        # Save metrics
        results_path = OUTPUT_DIR / f"eval_results_{datetime.now().strftime('%Y%m%d')}.json"
        metrics_df.to_json(results_path, orient="records", indent=2, force_ascii=False)
        logger.info(f"Metrics saved to {results_path}")
        # Also print summary
        print("\n=== RAGAS Evaluation Results ===")
        print(metrics_df.to_string())
    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")
        # Save a simple error report
        error_path = OUTPUT_DIR / f"eval_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_path, "w") as f:
            f.write(f"Metrics computation failed: {e}\n")
        logger.info(f"Error details saved to {error_path}")

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
