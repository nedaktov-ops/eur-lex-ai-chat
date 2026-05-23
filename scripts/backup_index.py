#!/usr/bin/env python3
"""Backup the current FAISS index + SQLite DB to HuggingFace Hub backup dataset.

Usage:
    export HF_TOKEN=hf_xxx
    python3 scripts/backup_index.py                        # Create backup with auto-timestamp
    python3 scripts/backup_index.py --date 2026-05-23      # Create backup for specific date
    python3 scripts/backup_index.py --latest                # Restore from latest backup
    python3 scripts/backup_index.py --date 2026-05-23       # Restore from specific date

Requires:
    - huggingface_hub
    - HF_TOKEN environment variable
    - Data files in data/ directory (index.faiss, chunks.db, build_meta.json, last_updated.txt)
"""

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime

from huggingface_hub import HfApi

BACKUP_DATASET = "NedAktovOps/eurlex-chat-backups"
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BACKUP_FILES = ["index.faiss", "chunks.db", "build_meta.json", "last_updated.txt"]


def get_api() -> HfApi:
    """Initialize HuggingFace Hub API with token."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)
    return HfApi(token=token)


def create_backup(date_prefix: str = None) -> str:
    """Create a backup of current data to HuggingFace Hub.

    Args:
        date_prefix: Optional date string for the branch (e.g., "2026-05-23").
                     If None, uses today's date.

    Returns:
        The branch name used for the backup.
    """
    api = get_api()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    date_str = date_prefix or datetime.utcnow().strftime("%Y%m%d")
    branch = f"backup-{date_str}"

    print(f"Creating backup in {BACKUP_DATASET}@{branch} ...")

    # Verify source files exist
    missing = []
    for f in BACKUP_FILES:
        src = os.path.join(LOCAL_DATA_DIR, f)
        if not os.path.exists(src):
            missing.append(f)

    if missing:
        print(f"WARNING: Missing data files: {missing}", file=sys.stderr)
        print("Proceeding with available files only...")

    # Create temp directory and copy files
    backup_dir = tempfile.mkdtemp(prefix=f"backup-{timestamp}-")
    try:
        for f in BACKUP_FILES:
            src = os.path.join(LOCAL_DATA_DIR, f)
            if os.path.exists(src):
                dst = os.path.join(backup_dir, f)
                shutil.copy2(src, dst)
                print(f"  ✓ {f} ({os.path.getsize(dst)} bytes)")
            else:
                print(f"  ✗ {f} (not found, skipped)")

        # Create backup metadata
        meta = {
            "backup_timestamp": timestamp,
            "source_branch": branch,
            "files": [f for f in BACKUP_FILES if os.path.exists(os.path.join(LOCAL_DATA_DIR, f))],
            "description": f"Backup created by backup_index.py at {timestamp}",
        }

        # Save metadata
        import json
        meta_path = os.path.join(backup_dir, "backup_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  ✓ backup_meta.json")

        # Upload to HuggingFace Hub
        print(f"\nUploading to {BACKUP_DATASET}@{branch} ...")
        api.upload_folder(
            folder_path=backup_dir,
            repo_id=BACKUP_DATASET,
            repo_type="dataset",
            revision=branch,
            create_pr=False,
        )

        print(f"\n✓ Backup saved to {BACKUP_DATASET}@{branch}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Restore with: python3 scripts/backup_index.py --restore --date {date_str}")

    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)

    return branch


def restore_backup(date_str: str = None, latest: bool = False) -> None:
    """Restore data from a HuggingFace Hub backup.

    Args:
        date_str: Date string (e.g., "2026-05-23") for the backup branch.
        latest: If True, find and use the most recent backup branch.
    """
    api = get_api()

    if latest:
        # List all backup branches and find the latest
        try:
            branches = api.list_repo_refs(BACKUP_DATASET, repo_type="dataset")
            backup_branches = [b for b in (branches.branches or []) if b.name.startswith("backup-")]
            if not backup_branches:
                print("ERROR: No backup branches found", file=sys.stderr)
                sys.exit(1)
            # Sort by name descending (backup-YYYYMMDD format sorts naturally)
            latest_branch = sorted(backup_branches, key=lambda b: b.name, reverse=True)[0]
            branch = latest_branch.name
            print(f"Latest backup branch: {branch}")
        except Exception as e:
            print(f"ERROR: Failed to list backup branches: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        date_val = date_str or datetime.utcnow().strftime("%Y%m%d")
        branch = f"backup-{date_val}"

    print(f"Restoring from {BACKUP_DATASET}@{branch} ...")

    # Download all files from the branch
    try:
        # Create temp directory
        restore_dir = tempfile.mkdtemp(prefix="restore-")
        try:
            api.snapshot_download(
                repo_id=BACKUP_DATASET,
                repo_type="dataset",
                revision=branch,
                local_dir=restore_dir,
                local_dir_use_symlinks=False,
            )

            # Copy files to data directory
            os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
            restored = []
            for f in BACKUP_FILES:
                src = os.path.join(restore_dir, f)
                if os.path.exists(src):
                    dst = os.path.join(LOCAL_DATA_DIR, f)
                    shutil.copy2(src, dst)
                    restored.append(f)
                    print(f"  ✓ {f} restored ({os.path.getsize(dst)} bytes)")

            # Also restore metadata if present
            meta_src = os.path.join(restore_dir, "backup_meta.json")
            if os.path.exists(meta_src):
                import json
                with open(meta_src) as f:
                    meta = json.load(f)
                print(f"\nBackup metadata: {meta.get('description', 'N/A')}")
                print(f"  Original backup time: {meta.get('backup_timestamp', 'unknown')}")

            if not restored:
                print("ERROR: No files found in backup branch", file=sys.stderr)
                sys.exit(1)

            print(f"\n✓ Restored {len(restored)} files from {branch}")

        finally:
            shutil.rmtree(restore_dir, ignore_errors=True)

    except Exception as e:
        print(f"ERROR: Failed to restore: {e}", file=sys.stderr)
        sys.exit(1)


def list_backups() -> None:
    """List all available backup branches."""
    api = get_api()
    try:
        branches = api.list_repo_refs(BACKUP_DATASET, repo_type="dataset")
        backup_branches = [b for b in (branches.branches or []) if b.name.startswith("backup-")]
        if not backup_branches:
            print("No backup branches found")
            return

        print(f"Available backups ({len(backup_branches)}):")
        for b in sorted(backup_branches, key=lambda x: x.name, reverse=True):
            print(f"  {b.name}")
    except Exception as e:
        print(f"ERROR: Failed to list backups: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Backup/restore EUR-Lex AI Chat index data to/from HuggingFace Hub",
    )
    parser.add_argument("--date", help="Date string for backup branch (e.g., '2026-05-23')")
    parser.add_argument("--restore", action="store_true", help="Restore from backup instead of creating")
    parser.add_argument("--latest", action="store_true", help="Use latest backup (with --restore)")
    parser.add_argument("--list", action="store_true", help="List available backups")

    args = parser.parse_args()

    if args.list:
        list_backups()
    elif args.restore:
        restore_backup(date_str=args.date, latest=args.latest)
    else:
        create_backup(date_prefix=args.date)


if __name__ == "__main__":
    main()
