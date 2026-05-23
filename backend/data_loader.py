"""Download and manage the FAISS index + SQLite chunk storage from HuggingFace Hub."""

import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import faiss
from huggingface_hub import hf_hub_download, HfApi

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


def create_backup():
    """Create a local backup of current index data before refreshing.
    
    Copies current data files to a timestamped backup directory.
    If HF_TOKEN is set, also uploads to HuggingFace Hub backup dataset.
    
    Returns:
        Path to the backup directory, or None if backup failed.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
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
                branch = f"backup-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
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
    
    suffix = os.environ.get("INDEX_SUFFIX", "")
    # Create backup before reloading
    logger.info("Creating backup before index reload...")
    backup_dir = create_backup()
    if backup_dir:
        logger.info(f"Pre-refresh backup saved: {backup_dir}")
    else:
        logger.warning("Pre-refresh backup not available, proceeding without backup")
    
    return download_index(index_suffix=suffix)


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


# ─── Phase 2: Dual-Index Support ───────────────────────────────────────────


class DualIndexManager:
    """Manages both old (MiniLM_384) and new (EURLEX_768) indexes during transition."""

    INDEX_TYPES = {
        "minilm_384": {
            "faiss_file": "index.faiss",
            "chunks_file": "chunks.db",
            "dimension": 384,
        },
        "eurlex_768": {
            "faiss_file": "index_eurlex.faiss",
            "chunks_file": "chunks_eurlex.db",
            "dimension": 768,
        }
    }

    def __init__(self):
        self.current_index_type = "minilm_384"
        self.available = {"minilm_384": False, "eurlex_768": False}
        self._check_available()

    def _check_available(self):
        for idx_type, config in self.INDEX_TYPES.items():
            faiss_path = DATA_DIR / config["faiss_file"]
            chunks_path = DATA_DIR / config["chunks_file"]
            self.available[idx_type] = faiss_path.exists() and chunks_path.exists()
        logger.info(f"DualIndexManager: available={self.available}")

    def switch_to(self, index_type: str) -> bool:
        """Switch to a specific index type if available."""
        if index_type not in self.INDEX_TYPES:
            logger.error(f"Unknown index type: {index_type}")
            return False
        if not self.available.get(index_type, False):
            logger.warning(f"Index type '{index_type}' not available")
            return False
        self.current_index_type = index_type
        logger.info(f"Switched to {index_type} index ({self.INDEX_TYPES[index_type]['dimension']}-dim)")
        return True

    def switch_to_eurlex(self) -> bool:
        """Switch to EURLEX-BERT index if available."""
        return self.switch_to("eurlex_768")

    def rollback_to_minilm(self) -> bool:
        """Rollback to MiniLM index."""
        return self.switch_to("minilm_384")

    def get_current_dimension(self) -> int:
        """Get the embedding dimension for the current index."""
        return self.INDEX_TYPES[self.current_index_type]["dimension"]

    def get_status(self) -> dict:
        """Get status of all index types."""
        return {
            "current": self.current_index_type,
            "available": dict(self.available),
            "dimensions": {
                k: v["dimension"] for k, v in self.INDEX_TYPES.items()
            },
        }


class EURLEXEmbedder:
    """Embedding model specifically trained on EU legislation.

    Uses nlpaueb/bert-base-uncased-eurlex which was pre-trained on 116,062
    EU legislation documents from EUR-LEX. Produces 768-dim embeddings.
    """

    MODEL_NAME = "nlpaueb/bert-base-uncased-eurlex"
    DIMENSION = 768

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._tokenizer = None
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            logger.info(f"Loading EURLEX-BERT embedder ({self.MODEL_NAME})...")
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModel.from_pretrained(self.MODEL_NAME)
            self._model.to(self.device)
            self._model.eval()
            logger.info("EURLEX-BERT embedder ready (768-dim)")

    def encode(self, texts, normalize_embeddings=True):
        """Encode texts using mean pooling of EURLEX-BERT token embeddings.

        Args:
            texts: String or list of strings to encode.
            normalize_embeddings: If True, L2-normalize the embeddings.

        Returns:
            Numpy array of shape (n_texts, 768).
        """
        import numpy as np
        import torch

        self._lazy_load()

        if isinstance(texts, str):
            texts = [texts]

        encoded_input = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            model_output = self._model(**encoded_input)

        # Mean pooling
        token_embeddings = model_output[0]
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        # Normalize
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()
