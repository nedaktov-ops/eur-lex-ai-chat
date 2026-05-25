"""Download and manage the FAISS index + SQLite chunk storage from HuggingFace Hub."""

import logging
import os
import shutil
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
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


class EURLEXEmbedder:
    """768-dim legal embeddings via ONNX Runtime for EURLEX-BERT.

    Loads quantized ONNX model + tokenizer on first use.
    Used for query encoding at runtime (not for bulk index building).
    """

    def __init__(self, model_name: str = "nlpaueb/bert-base-uncased-eurlex"):
        self.model_name = model_name
        self._tokenizer = None
        self._session = None
        self._dim = 768

    def _load(self):
        """Lazy-load tokenizer and ONNX session."""
        if self._session is not None:
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Try local ONNX model first, fall back to PyTorch
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "eurlex-bert-onnx", "model.quant.onnx")

        if os.path.exists(local_path):
            model_path = local_path
        else:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="NedAktovOps/eurlex-chat-data",
                filename="onnx_models/eurlex-bert/model.quant.onnx",
                repo_type="dataset",
                token=HF_TOKEN,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        logger.info(f"EURLEXEmbedder loaded: {self.model_name} ({self._dim}-dim, model={model_path})")

    def encode(self, texts: list[str], batch_size: int = 32) -> list:
        """Encode texts to 768-dim embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Inference batch size (default 32)

        Returns:
            List of embedding vectors (list of lists)
        """
        self._load()

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="np",
            )

            feed = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
            if "token_type_ids" in encoded:
                feed["token_type_ids"] = encoded["token_type_ids"]

            outputs = self._session.run(None, feed)[0]

            # Mean pooling
            mask = encoded["attention_mask"][:, :, None].astype(outputs.dtype)
            mask_sum = mask.sum(axis=1)
            embeddings = (outputs * mask).sum(axis=1) / np.maximum(mask_sum, 1e-9)

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).tolist()


def download_index(index_suffix=""):
    """Download index files from HF Hub. Supports suffix for EURLEX-BERT (set INDEX_SUFFIX env var)."""
    suffix = index_suffix or os.environ.get("INDEX_SUFFIX", "")
    index_file = f"index{suffix}.faiss"
    db_file = f"chunks{suffix}.db"
    logger.info(f"Downloading index from {REPO_ID} (files: {index_file}, {db_file})...")
    try:
        index_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=index_file,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        db_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=db_file,
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
