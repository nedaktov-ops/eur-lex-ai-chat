#!/usr/bin/env python3
"""Upload chunks.db to HuggingFace Hub so Colab notebook can access it.

Usage: HF_TOKEN=hf_yourtoken python3 scripts/upload_chunks_to_hub.py
"""
import os, sys, json, time
from datetime import UTC, datetime
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable required")
    sys.exit(1)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "chunks.db")
if not os.path.exists(DB_PATH):
    print(f"ERROR: {DB_PATH} not found. Run build_index.py --download-only first.")
    sys.exit(1)

api = HfApi()
who = api.whoami(token=HF_TOKEN)["name"]
REPO_ID = f"{who}/eurlex-chat-data"
create_repo(REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
print(f"Repo: {REPO_ID}")

start = time.time()
api.upload_file(
    repo_id=REPO_ID,
    path_in_repo="chunks_raw.db",
    path_or_fileobj=DB_PATH,
    repo_type="dataset",
    token=HF_TOKEN,
)
size_gb = os.path.getsize(DB_PATH) / 1e9
elapsed = time.time() - start
print(f"Uploaded {size_gb:.1f} GB in {elapsed/60:.1f} min")
print(f"Now open colab_embed.ipynb in Google Colab and run it!")
