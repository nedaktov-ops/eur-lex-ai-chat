"""FastAPI application for EUR-Lex AI Chat."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .logging_middleware import (
    PipelineLoggingMiddleware,
    log_answer_generated,
    log_query_processed,
    log_query_received,
    log_response_returned,
    log_search_performed,
)
from .search import get_bm25_results, search as faiss_search, rrf_fuse
from .reranker import Reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_embedding_models = {}


class FeedbackRequest(BaseModel):
    """Model for user feedback on answers."""
    pass


def get_embedding_model(index_suffix=""):
    """Get embedding model for a given suffix (cached)."""
    global _embedding_models
    if index_suffix in _embedding_models:
        return _embedding_models[index_suffix]

    # Determine which model to load based on suffix
    if "eurlex" in index_suffix.lower():
        from .data_loader import EURLEXEmbedder
        logger.info("Loading EURLEX-BERT embedding model (768-dim)")
        model = EURLEXEmbedder()
    else:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: all-MiniLM-L6-v2 (384-dim)")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    _embedding_models[index_suffix] = model
    return model


_reranker = None

def get_reranker():
    """Get the global reranker instance, initializing if needed."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .data_loader import download_index

    index_suffix = os.environ.get("INDEX_SUFFIX", "")
    logger.info(f"Starting up — loading index (suffix='{index_suffix}')...")
    try:
        download_index(index_suffix=index_suffix)
        logger.info("Index loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")

    logger.info("Pre-loading embedding model...")
    get_embedding_model(index_suffix=index_suffix)
    logger.info("Embedding model loaded")

    yield
    logger.info("Shutting down")


app = FastAPI(title="EUR-Lex AI Chat API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://eurlex-chat.vercel.app",
        "https://frontend-ruddy-zeta-40.vercel.app",
        "http://localhost:4321",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PipelineLoggingMiddleware)


@app.get("/health")
async def health():
    from .data_loader import get_stats

    stats = get_stats()
    return {
        "status": "ok",
        "index_loaded": stats["vectors"] > 0,
        "ntotal": stats["vectors"],
        "size": stats["size"],
        "last_updated": stats["last_updated"],
        "loaded_at": stats["loaded_at"],
    }

@app.get("/models")
async def list_models():
    return {"models": ["minilm", "eurlex-bert"]}

@app.get("/refresh")
async def refresh():
    from .data_loader import check_for_updates, reload_index

    try:
        has_updates = check_for_updates()
        if has_updates:
            logger.info("New index available, reloading...")
            reload_index()
            return {"status": "reloaded", "message": "Index updated successfully"}
        else:
            return {"status": "current", "message": "Index is up to date"}
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/backup")
async def backup():
    from .data_loader import create_backup

    try:
        backup_dir = create_backup()
        if backup_dir:
            return {
                "status": "ok",
                "message": f"Backup created at {backup_dir}",
                "backup_path": str(backup_dir),
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Backup failed"},
            )
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/feedback")
async def feedback(fb: FeedbackRequest):
    import json
    import os

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/feedback.jsonl"

    # Append feedback to JSONL file
    with open(log_path, "a") as f:
        f.write(json.dumps(fb.dict()) + "\n")

    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request):
    import json
    import time
    import traceback

    from .answer_validator import AnswerValidator, estimate_confidence
    from .query_expander import AutoExpander, expand_obligation_query, expand_query
    from .question_classifier import EUQuestionClassifier
    from .rag import answer_question
    from .rate_limit import is_rate_limited

    client_ip = request.client.host if request.client else "unknown"

    if is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 20 requests per minute per IP.",
        )

    try:
        body = await request.json()
        query = body.get("query", "").strip()
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    if len(query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 chars)")

    # Get model selection from query parameter
    model_name = request.query_params.get("model", "minilm")
    if model_name not in ["minilm", "eurlex-bert"]:
        raise HTTPException(status_code=400, detail="Invalid model. Choose 'minilm' or 'eurlex-bert'.")

    # Determine index suffix for this model
    model_to_suffix = {"minilm": "", "eurlex-bert": "eurlex"}
    suffix = model_to_suffix[model_name]

    # === Stage 1: Query Received ===
    request_id = getattr(request.state, "request_id", None)
    log_query_received(request_id, query, client_ip)

    try:
        # === Stage 2: Query Classification ===
        classifier = EUQuestionClassifier()
        classification = classifier.classify(query)

        # Check if we need clarification
        if classifier.needs_clarification(classification):
            log_query_processed(request_id, classification=classification)
            log_answer_generated(request_id, 0, 0, 0, False)
            log_response_returned(request_id, 200, 72, True)
            return {
                "answer": "I can help with EU law topics including GDPR, the AI Act, "
                          "the Pay Transparency Directive, and more. Could you please "
                          "rephrase your question to be more specific?",
                "citations": [],
                "sources": [],
            }

        # === Stage 3: Query Expansion ===
        if classification.get("obligation_seeking"):
            query_variations = expand_obligation_query(query)
        else:
            query_variations = expand_query(query)

        # Limit to 5 variations to control latency
        query_variations = query_variations[:5]

        log_query_processed(
            request_id,
            classification=classification,
            expanded_queries=query_variations[1:],
        )

        # === Stage 4: Hybrid Search with RRF & Cross-Encoder Reranking ===
        embedder = get_embedding_model(index_suffix=suffix)
        search_start = time.time()

        # Collect BM25 and FAISS results for all query variations
        bm25_lists = []
        faiss_lists = []
        for q_variant in query_variations:
            q_vec = embedder.encode([q_variant], normalize_embeddings=True)
            bm25_res = get_bm25_results(q_variant, top_k=20, model_name=model_name)
            bm25_lists.append(bm25_res)
            faiss_res = faiss_search(q_vec, top_k=20, model_name=model_name)
            faiss_lists.append(faiss_res)

        # Reciprocal Rank Fusion across all retrievers
        all_retrievals = bm25_lists + faiss_lists
        rrf_candidates = rrf_fuse(all_retrievals, k=60, top_n=20)

        # Cross-encoder reranking using the original query
        reranker = get_reranker()
        chunks = reranker.rerank(query, rrf_candidates, top_k=10)

        search_duration_ms = (time.time() - search_start) * 1000

        # Log search results (using rerank_score)
        top_scores = [c.get("rerank_score", 0) for c in chunks[:3]] if chunks else []
        query_vector_shape = (1, q_vec.shape[-1]) if hasattr(q_vec, "shape") else (1, 0)
        log_search_performed(
            request_id, query_vector_shape, len(chunks), top_scores, search_duration_ms,
        )

        if not chunks:
            log_answer_generated(request_id, 0, 0, 0, False)
            log_response_returned(request_id, 200, 0, True)
            return {
                "answer": "I don't have enough information to answer that question. Try asking about a specific EU regulation or directive.",
                "citations": [],
                "sources": [],
            }

        # === Stage 5: Confidence Gating ===
        should_answer, reason = classifier.should_answer(classification, chunks)
        if not should_answer:
            log_answer_generated(request_id, 0, len(chunks), 0, False)
            log_response_returned(request_id, 200, 0, True)
            if "obligation" in reason and classification.get("obligation_seeking"):
                return {
                    "answer": f"I found documents mentioning '{query}' but couldn't identify "
                              f"specific employer obligations in the retrieved texts. The relevant "
                              f"articles may need more specific terms. Try asking about a particular "
                              f"aspect of employer responsibilities under this directive.",
                    "citations": [c.get("celex") for c in chunks[:3]],
                    "sources": [{
                        "celex": c.get("celex"),
                        "title": c.get("title"),
                        "article": c.get("article"),
                        "score": c.get("score"),
                    } for c in chunks[:3]],
                }
            else:
                return {
                    "answer": "I don't have enough information to answer that question. "
                             "Try asking about a specific EU regulation or directive.",
                    "citations": [],
                    "sources": [],
                }

        # === Stage 6: Answer Generation (with classification context) ===
        result = answer_question(query, chunks, request_id=request_id, classification=classification)

        # === Stage 7: Answer Validation ===
        validator = AnswerValidator()
        passes_validation, validation_reason = validator.validate(
            query=query, answer=result.get("answer", ""),
            chunks=chunks, classification=classification,
        )

        if not passes_validation:
            logger.warning(f"Answer validation failed: {validation_reason} | query: {query[:80]}")
            # Retry once with CELEX citation emphasis
            logger.info(f"Retrying with citation emphasis for {request_id}")
            result2 = answer_question(
                query, chunks, request_id=request_id,
                classification=classification, retry_with_citation_emphasis=True,
            )
            passes_v2, reason_v2 = validator.validate(
                query=query, answer=result2.get("answer", ""),
                chunks=chunks, classification=classification,
            )
            if passes_v2:
                logger.info(f"Retry succeeded for {request_id}")
                result = result2
                passes_validation = True
                validation_reason = reason_v2
            else:
                logger.warning(f"Retry also failed: {reason_v2} for {request_id}")
                AutoExpander().record_failure(query, reason_v2)
                fallback = validator.make_fallback_answer(
                    query, chunks, classification=classification,
                    validation_reason=validation_reason,
                )
                fallback_result = {
                    "answer": fallback,
                    "citations": list(dict.fromkeys(c.get("celex") for c in chunks[:5] if c.get("celex"))),
                    "sources": [{
                        "celex": c.get("celex"), "title": c.get("title"),
                        "article": c.get("article"), "score": c.get("score"),
                    } for c in chunks[:5]],
                }
                log_answer_generated(
                    request_id, len(fallback), len(fallback_result.get("citations", [])),
                    len(fallback_result.get("sources", [])),
                    validation_passed=False,
                )
                log_response_returned(request_id, 200, len(json.dumps(fallback_result)))
                return fallback_result

        # === Stage 8: Confidence Estimation ===
        confidence = estimate_confidence(chunks, classification=classification)
        answer_text = result.get("answer", "") or ""

        log_answer_generated(
            request_id, len(answer_text), len(result.get("citations", [])),
            len(result.get("sources", [])), validation_passed=True,
            confidence=confidence,
        )

        # === Stage 9: Response Returned ===
        result["_confidence"] = confidence["level"]
        response_json = json.dumps(result)
        log_response_returned(request_id, 200, len(response_json))

        return result

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Unexpected error in /chat (query='{query}'): {e}\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": f"Internal error: {type(e).__name__}: {e}"},
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
