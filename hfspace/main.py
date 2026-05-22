"""FastAPI application for EUR-Lex AI Chat."""

import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    from data_loader import download_index

    logger.info("Starting up — loading index...")
    try:
        download_index()
        logger.info("Index loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")

    logger.info("Pre-loading embedding model...")
    get_embedding_model()
    logger.info("Embedding model loaded")

    yield
    logger.info("Shutting down")


app = FastAPI(title="EUR-Lex AI Chat API", version="1.0.0", lifespan=lifespan)

_frontend_url = os.environ.get("FRONTEND_URL", "https://eurlex-chat.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        _frontend_url,
        "https://eurlex-chat.vercel.app",
        "https://frontend-ruddy-zeta-40.vercel.app",
        "http://localhost:4321",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    from data_loader import get_stats

    stats = get_stats()
    return {
        "status": "ok",
        "index_loaded": stats["vectors"] > 0,
        "ntotal": stats["vectors"],
        "size": stats["size"],
        "last_updated": stats["last_updated"],
        "loaded_at": stats["loaded_at"],
    }


@app.get("/refresh")
async def refresh():
    from data_loader import check_for_updates, reload_index, get_stats

    try:
        has_updates = check_for_updates()
        if has_updates:
            logger.info("New index available, reloading...")
            reload_index()
            return {"status": "reloaded", "message": "Index updated successfully"}
        else:
            stats = get_stats()
            return {"status": "current", "message": "Index is up to date"}
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/chat")
async def chat(request: Request):
    from rate_limit import is_rate_limited
    from search import search
    from rag import answer_question

    client_ip = request.client.host if request.client else "unknown"

    if is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 20 requests per minute per IP.",
        )

    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    if len(query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 chars)")

    model = get_embedding_model()
    query_vector = model.encode(query, normalize_embeddings=True)

    chunks = search(query_vector, top_k=10)
    if not chunks:
        return {
            "answer": "I don't have enough information to answer that question. Try asking about a specific EU regulation or directive.",
            "citations": [],
            "sources": [],
        }

    result = answer_question(query, chunks)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
