"""Download and manage the vector index from HuggingFace Hub."""

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_USERNAME = os.environ.get("HF_USERNAME", "NedAktovOps")
HF_DATASET = os.environ.get("HF_DATASET", "eurlex-chat-data")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

REPO_ID = f"{HF_USERNAME}/{HF_DATASET}"

_index_data = {
    "vectors": None,
    "chunks": None,
    "last_updated": None,
    "loaded_at": None,
}


def download_index():
    logger.info(f"Downloading index from {REPO_ID}...")
    try:
        vectors_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="vectors.npy",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        chunks_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="chunks.json",
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except Exception as e:
        logger.error(f"Failed to download from HF Hub: {e}")
        raise

    vectors = np.load(vectors_path)
    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    _index_data["vectors"] = vectors
    _index_data["chunks"] = chunks
    _index_data["last_updated"] = _get_last_updated()
    _index_data["loaded_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"Index loaded: {vectors.shape[0]} vectors, {len(chunks)} chunks")
    return _index_data


def _get_last_updated():
    try:
        ts_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="last_updated.txt",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(ts_path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def check_for_updates():
    current_remote = _get_last_updated()
    if current_remote and current_remote != _index_data["last_updated"]:
        logger.info(f"Remote index updated: {current_remote}")
        return True
    return False


def reload_index():
    return download_index()


def get_index():
    return _index_data


def get_stats():
    data = get_index()
    return {
        "vectors": data["vectors"].shape if data["vectors"] is not None else None,
        "chunks": len(data["chunks"]) if data["chunks"] is not None else 0,
        "last_updated": data["last_updated"],
        "loaded_at": data["loaded_at"],
    }
