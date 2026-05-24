#!/usr/bin/env python3
"""Save a checkpoint of critical project files before making changes.

Checkpoints store snapshots of key backend files, data files, and metadata
to enable point-in-time recovery. Compatible with the multi-phase improvement plan.

Usage:
    python3 scripts/checkpoint_save.py --phase 0
    python3 scripts/checkpoint_save.py --phase "phase-0-complete" --message "Logging + backup scripts"
    python3 scripts/checkpoint_save.py --list
    python3 scripts/checkpoint_save.py --show <checkpoint_id>

Checkpoints stored in: .checkpoints/<checkpoint_id>/
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / ".checkpoints"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"

# Files to include in every checkpoint
DEFAULT_FILES = [
    "app/main.py",
    "app/rag.py",
    "app/search.py",
    "app/data_loader.py",
    "app/rate_limit.py",
    "app/logging_middleware.py",
    "app/question_classifier.py",
    "app/query_expander.py",
    "app/relation_extractor.py",
    "app/requirements.txt",
    "scripts/backup_index.py",
    "scripts/checkpoint_save.py",
    "scripts/checkpoint_restore.py",
]


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_checkpoint(phase: str, message: str = "", extra_files: list = None) -> str:
    """Save a checkpoint of current project state.

    Args:
        phase: Phase identifier (e.g., "0", "1", "pre-phase-0")
        message: Optional human-readable message describing the checkpoint
        extra_files: Additional file paths to include (relative to project root)

    Returns:
        Checkpoint ID string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    checkpoint_id = f"ckpt-{timestamp}"
    checkpoint_path = CHECKPOINT_DIR / checkpoint_id

    # Create checkpoint directory
    os.makedirs(checkpoint_path, exist_ok=True)

    files_to_backup = list(DEFAULT_FILES)
    if extra_files:
        files_to_backup.extend(extra_files)

    backed_up = []
    missing = []
    file_hashes = {}

    for rel_path in files_to_backup:
        src = PROJECT_ROOT / rel_path
        if src.exists():
            # Preserve directory structure within checkpoint
            dst = checkpoint_path / rel_path
            os.makedirs(dst.parent, exist_ok=True)
            shutil.copy2(src, dst)
            file_hash = compute_file_hash(src)
            file_hashes[rel_path] = file_hash
            backed_up.append(rel_path)
        else:
            missing.append(rel_path)

    # Also backup data files if they exist (they might be large, skip chunks.db if too big)
    for data_file in ["index.faiss", "build_meta.json", "last_updated.txt"]:
        src = DATA_DIR / data_file
        if src.exists():
            dst = checkpoint_path / "data" / data_file
            os.makedirs(dst.parent, exist_ok=True)
            shutil.copy2(src, dst)
            file_hash = compute_file_hash(src)
            file_hashes[f"data/{data_file}"] = file_hash
            backed_up.append(f"data/{data_file}")

    # Create checkpoint metadata
    meta = {
        "checkpoint_id": checkpoint_id,
        "phase": phase,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "message": message,
        "files_backed_up": backed_up,
        "files_missing": missing,
        "file_hashes": file_hashes,
        "rollback_command": f"python3 scripts/checkpoint_restore.py --id {checkpoint_id}",
    }

    meta_path = checkpoint_path / "checkpoint.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Update .checkpoints/index.json for quick listing
    index_path = CHECKPOINT_DIR / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {"checkpoints": []}

    index["checkpoints"].append({
        "id": checkpoint_id,
        "phase": phase,
        "timestamp": meta["timestamp"],
        "message": message,
        "files_count": len(backed_up),
    })
    # Keep sorted by timestamp (newest first)
    index["checkpoints"].sort(key=lambda c: c["timestamp"], reverse=True)

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Print summary
    print(f"✓ Checkpoint saved: {checkpoint_id}")
    print(f"  Phase: {phase}")
    print(f"  Time: {meta['timestamp']}")
    print(f"  Files backed up: {len(backed_up)}")
    for f in backed_up:
        print(f"    • {f}")
    if missing:
        print(f"  Files missing (skipped): {len(missing)}")
        for f in missing:
            print(f"    • {f}")
    print(f"\n  Rollback: {meta['rollback_command']}")

    return checkpoint_id


def list_checkpoints():
    """List all saved checkpoints."""
    index_path = CHECKPOINT_DIR / "index.json"
    if not index_path.exists():
        print("No checkpoints found")
        return

    with open(index_path) as f:
        index = json.load(f)

    checkpoints = index.get("checkpoints", [])
    if not checkpoints:
        print("No checkpoints found")
        return

    print(f"Checkpoints ({len(checkpoints)}):")
    for c in checkpoints:
        print(f"  {c['id']} | Phase: {c['phase']} | {c.get('message', '')} | {c['timestamp']}")


def show_checkpoint(checkpoint_id: str):
    """Show details of a specific checkpoint."""
    ckpt_path = CHECKPOINT_DIR / checkpoint_id
    meta_path = ckpt_path / "checkpoint.json"

    if not meta_path.exists():
        print(f"ERROR: Checkpoint '{checkpoint_id}' not found", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Checkpoint: {meta['checkpoint_id']}")
    print(f"  Phase: {meta['phase']}")
    print(f"  Timestamp: {meta['timestamp']}")
    print(f"  Message: {meta.get('message', '')}")
    print(f"  Files backed up ({len(meta['files_backed_up'])}):")
    for f in meta["files_backed_up"]:
        status = "✓" if (ckpt_path / f).exists() else "✗"
        hash_val = meta.get("file_hashes", {}).get(f, "?")[:16]
        size = os.path.getsize(ckpt_path / f) if (ckpt_path / f).exists() else 0
        print(f"    {status} {f} ({size} bytes, hash: {hash_val}...)")
    if meta.get("files_missing"):
        print(f"  Missing (at save time):")
        for f in meta["files_missing"]:
            print(f"    • {f}")
    print(f"\n  Rollback: {meta['rollback_command']}")


def main():
    parser = argparse.ArgumentParser(
        description="Save/restore checkpoints for EUR-Lex AI Chat project"
    )
    parser.add_argument("--phase", default="0", help="Phase identifier (default: '0')")
    parser.add_argument("--message", default="", help="Human-readable checkpoint description")
    parser.add_argument("--extra-files", nargs="*", help="Additional files to include")
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument("--show", metavar="ID", help="Show checkpoint details")

    args = parser.parse_args()

    if args.list:
        list_checkpoints()
    elif args.show:
        show_checkpoint(args.show)
    else:
        save_checkpoint(args.phase, args.message, args.extra_files)


if __name__ == "__main__":
    main()
