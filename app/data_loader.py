"""Download and manage the FAISS index + SQLite chunk storage from HuggingFace Hub."""

import logging
import os
import shutil
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import faiss
from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)

HF_USERNAME = os.environ.get("HF_USERNAME", "NedAktovOps")
HF_DATASET = os.environ.get("HF_DATASET", "eurlex-chat-data")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
BACKUP_DATASET = f"{HF_USERNAME}/eurlex-chat-backups"

REPO_ID = f"{HF_USERNAME}/{HF_DATASET}"

BACKUP_FILES = ["index.faiss", "chunks.db", "build_meta.json", "last_updated.txt"]
DATA_DIR = Path(__file__).parent.parent / "data"

_index_data = {
    "index": None,
    "conn": None,
    "lock": threading.Lock(),
    "size": 0,
    "ntotal": 0,
    "last_updated": None,
    "loaded_at": None,
}


def download_index(index_suffix=""):
    """Download index files from HF Hub. Supports suffix for EURLEX-BERT (set INDEX_SUFFIX env var)."""
    suffix = index_suffix or os.environ.get("INDEX_SUFFIX", "")
    index_file = f"index{suffix}.faiss"
    db_file = f"chunks{suffix}.db"
    logger.info(f"Downloading index from {REPO_ID} (files: {index_file}, {db_file})...")
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
    _index_data["loaded_at"] = datetime.now(UTC).isoformat()

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
        with open(ts_path) as f:
            return f.read().strip()
    except Exception:
        return None


def check_for_updates():
    current_remote = _get_last_updated()
    if current_remote and current_remote != _index_data["last_updated"]:
        logger.info(f"Remote index updated: {current_remote}")
        return True
    return False


def create_backup():
    """Create a local backup of current index data before refreshing.

    Copies current data files to a timestamped backup directory.
    If HF_TOKEN is set, also uploads to HuggingFace Hub backup dataset.

    Returns:
        Path to the backup directory, or None if backup failed.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup_dir = DATA_DIR / f"backup-{timestamp}"

    try:
        os.makedirs(backup_dir, exist_ok=True)

        copied = []
        for f in BACKUP_FILES:
            src = DATA_DIR / f
            if src.exists():
                shutil.copy2(src, backup_dir / f)
                copied.append(f)

        if not copied:
            logger.warning("No data files found to backup")
            shutil.rmtree(backup_dir, ignore_errors=True)
            return None

        logger.info(f"Local backup created at {backup_dir}: {', '.join(copied)}")

        # Attempt to upload to HuggingFace Hub if token is available
        if HF_TOKEN:
            try:
                branch = f"backup-{datetime.now(UTC).strftime('%Y%m%d')}"
                api = HfApi(token=HF_TOKEN)
                api.upload_folder(
                    folder_path=str(backup_dir),
                    repo_id=BACKUP_DATASET,
                    repo_type="dataset",
                    revision=branch,
                    create_pr=False,
                )
                logger.info(f"Remote backup saved to {BACKUP_DATASET}@{branch}")
            except Exception as e:
                logger.warning(f"Remote backup failed (local backup still exists): {e}")

        return backup_dir

    except Exception as e:
        logger.error(f"Backup failed: {e}")
        # Clean up partial backup
        shutil.rmtree(backup_dir, ignore_errors=True)
        return None


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
