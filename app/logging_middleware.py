"""Structured JSON logging middleware for EUR-Lex AI Chat pipeline."""

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configure structured logging
logger = logging.getLogger("eurlex-chat.pipeline")
logger.setLevel(logging.INFO)

# Create handler if not exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


class PipelineLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all stages of the RAG pipeline with structured JSON."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Add request ID to request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log request/response metadata (basic)
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "client_host": request.client.host if request.client else None,
        }

        logger.info(json.dumps(log_entry))

        return response


def log_pipeline_stage(stage: str, request_id: str, data: dict, level: str = "INFO"):
    """Log a structured pipeline stage event.

    Args:
        stage: Name of the pipeline stage (e.g., "query_received", "search_performed")
        request_id: Unique request identifier
        data: Dictionary of stage-specific data to log
        level: Logging level (INFO, WARNING, ERROR)
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": request_id,
        "stage": stage,
        "data": data,
    }

    log_json = json.dumps(log_entry)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, log_json)


# Convenience functions for common pipeline stages
def log_query_received(request_id: str, query: str, client_host: str = None):
    log_pipeline_stage(
        "query_received",
        request_id,
        {
            "query": query[:500],  # Truncate very long queries
            "query_length": len(query),
            "client_host": client_host,
        }
    )


def log_query_processed(request_id: str, classification: dict = None, expanded_queries: list = None):
    log_pipeline_stage(
        "query_processed",
        request_id,
        {
            "classification": classification,
            "expanded_queries_count": len(expanded_queries) if expanded_queries else 0,
            "expansion_ratios": [len(eq) for eq in expanded_queries] if expanded_queries else [],
        }
    )


def log_search_performed(request_id: str, query_vector_shape: tuple, results_count: int,
                        top_scores: list, search_duration_ms: float):
    log_pipeline_stage(
        "search_performed",
        request_id,
        {
            "query_vector_shape": query_vector_shape,
            "results_count": results_count,
            "top_3_scores": [round(s, 4) for s in top_scores[:3]] if top_scores else [],
            "search_duration_ms": round(search_duration_ms, 2),
        }
    )


def log_prompt_built(request_id: str, prompt_length: int, context_chunks_count: int,
                    classification: dict = None):
    log_pipeline_stage(
        "prompt_built",
        request_id,
        {
            "prompt_length": prompt_length,
            "context_chunks_count": context_chunks_count,
            "classification": classification,
        }
    )


def log_llm_call(request_id: str, model_name: str, prompt_tokens: int = None,
                completion_tokens: int = None, duration_ms: float = None):
    log_pipeline_stage(
        "llm_call",
        request_id,
        {
            "model_name": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
            "duration_ms": round(duration_ms, 2) if duration_ms else None,
        }
    )


def log_answer_generated(request_id: str, answer_length: int, citations_count: int,
                        sources_count: int, validation_passed: bool = None,
                        confidence: dict = None):
    data = {
        "answer_length": answer_length,
        "citations_count": citations_count,
        "sources_count": sources_count,
        "validation_passed": validation_passed,
    }
    if confidence:
        data["confidence_level"] = confidence.get("level")
        data["confidence_score"] = confidence.get("overall_score")
    log_pipeline_stage("answer_generated", request_id, data)


def log_response_returned(request_id: str, status_code: int, response_size: int,
                         fallback_used: bool = False):
    log_pipeline_stage(
        "response_returned",
        request_id,
        {
            "status_code": status_code,
            "response_size_bytes": response_size,
            "fallback_used": fallback_used,
        }
    )
