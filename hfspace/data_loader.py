"""Download and manage the FAISS index + SQLite chunk storage from HuggingFace Hub."""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

import faiss
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_USERNAME = os.environ.get("HF_USERNAME", "NedAktovOps")
HF_DATASET = os.environ.get("HF_DATASET", "eurlex-chat-data")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

REPO_ID = f"{HF_USERNAME}/{HF_DATASET}"

_index_data = {
    "index": None,
    "conn": None,
    "lock": threading.Lock(),
    "size": 0,
    "ntotal": 0,
    "last_updated": None,
    "loaded_at": None,
}


def download_index():
    logger.info(f"Downloading index from {REPO_ID}...")
    try:
        index_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="index.faiss",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        db_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="chunks.db",
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except Exception as e:
        logger.error(f"Failed to download from HF Hub: {e}")
        raise

    index = faiss.read_index(index_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1")
    conn.execute("PRAGMA temp_store = MEMORY")

    cursor = conn.execute("SELECT COUNT(*) AS cnt FROM chunks")
    size = cursor.fetchone()["cnt"]

    _index_data["index"] = index
    _index_data["conn"] = conn
    _index_data["lock"] = threading.Lock()
    _index_data["size"] = size
    _index_data["ntotal"] = index.ntotal
    _index_data["last_updated"] = _get_last_updated()
    _index_data["loaded_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"Index loaded: {index.ntotal} vectors, {size} chunks")
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
    conn = _index_data.get("conn")
    if conn:
        conn.close()
    return download_index()


def get_index():
    return _index_data


def get_stats():
    data = get_index()
    return {
        "vectors": data["ntotal"],
        "size": data["size"],
        "last_updated": data["last_updated"],
        "loaded_at": data["loaded_at"],
    }
