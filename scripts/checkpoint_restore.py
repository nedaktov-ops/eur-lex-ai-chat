#!/usr/bin/env python3
"""Restore project files from a saved checkpoint.

Supports restoring individual checkpoints, rolling back to the latest checkpoint
for a specific phase, or listing available checkpoints.

Usage:
    python3 scripts/checkpoint_restore.py --id ckpt-20260523-220000      # Restore specific checkpoint
    python3 scripts/checkpoint_restore.py --latest                        # Restore latest checkpoint
    python3 scripts/checkpoint_restore.py --phase 0                       # Restore latest Phase 0 checkpoint
    python3 scripts/checkpoint_restore.py --list                          # List available checkpoints
    python3 scripts/checkpoint_restore.py --id ckpt-xxx --dry-run         # Preview without restoring

Checkpoints stored in: .checkpoints/<checkpoint_id>/
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / ".checkpoints"


def load_index() -> dict:
    """Load the checkpoint index file."""
    index_path = CHECKPOINT_DIR / "index.json"
    if not index_path.exists():
        print("ERROR: No checkpoints found (index.json missing)", file=sys.stderr)
        sys.exit(1)
    with open(index_path) as f:
        return json.load(f)


def load_checkpoint_meta(checkpoint_id: str) -> dict:
    """Load metadata for a specific checkpoint."""
    meta_path = CHECKPOINT_DIR / checkpoint_id / "checkpoint.json"
    if not meta_path.exists():
        print(f"ERROR: Checkpoint '{checkpoint_id}' not found", file=sys.stderr)
        sys.exit(1)
    with open(meta_path) as f:
        return json.load(f)


def list_checkpoints():
    """List all checkpoints with details."""
    index = load_index()
    ckpts = index.get("checkpoints", [])
    if not ckpts:
        print("No checkpoints found")
        return

    print(f"Available checkpoints ({len(ckpts)}):")
    print(f"{'ID':<30} {'Phase':<12} {'Files':<6} {'Message'}")
    print("-" * 80)
    for c in ckpts:
        cid = c["id"]
        phase = c.get("phase", "?")
        count = c.get("files_count", "?")
        msg = c.get("message", "")[:40]
        print(f"{cid:<30} {phase:<12} {count:<6} {msg}")


def find_checkpoint(checkpoint_id: str = None, phase: str = None, latest: bool = False) -> str:
    """Find a checkpoint ID based on search criteria."""
    index = load_index()
    ckpts = index.get("checkpoints", [])

    if checkpoint_id:
        return checkpoint_id

    if latest:
        if not ckpts:
            print("ERROR: No checkpoints available", file=sys.stderr)
            sys.exit(1)
        return ckpts[0]["id"]

    if phase:
        phase_ckpts = [c for c in ckpts if c.get("phase") == phase]
        if not phase_ckpts:
            print(f"ERROR: No checkpoints found for phase '{phase}'", file=sys.stderr)
            sys.exit(1)
        return phase_ckpts[0]["id"]

    return ckpts[0]["id"]


def restore_checkpoint(checkpoint_id: str, dry_run: bool = False):
    """Restore files from a checkpoint.

    Args:
        checkpoint_id: The checkpoint ID to restore from.
        dry_run: If True, only show what would be restored without copying.
    """
    meta = load_checkpoint_meta(checkpoint_id)
    ckpt_dir = CHECKPOINT_DIR / checkpoint_id

    print(f"Checkpoint: {checkpoint_id}")
    print(f"  Phase: {meta['phase']}")
    print(f"  Timestamp: {meta['timestamp']}")
    print(f"  Message: {meta.get('message', '')}")
    print()

    if dry_run:
        print("DRY RUN — No files will be modified")
        print()

    files_to_restore = meta.get("files_backed_up", [])
    restored = []
    skipped = []

    for rel_path in files_to_restore:
        src = ckpt_dir / rel_path
        dst = PROJECT_ROOT / rel_path

        if not src.exists():
            skipped.append((rel_path, "source missing in checkpoint"))
            continue

        if dry_run:
            dst_exists = dst.exists()
            dst_size = os.path.getsize(dst) if dst_exists else 0
            src_size = os.path.getsize(src)
            print(f"  Would restore: {rel_path}")
            print(f"    From: {src_size} bytes (checkpoint)")
            print(f"    To:   {dst_size} bytes {'(existing)' if dst_exists else '(new file)'}")
            restored.append(rel_path)
            continue

        # Make backup of current file before overwriting? 
        # No — this is a restore, the checkpoint is the reference.
        # If user wants to keep current state, they should have saved a checkpoint first.
        os.makedirs(dst.parent, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            restored.append(rel_path)
            print(f"  ✓ {rel_path}")
        except Exception as e:
            skipped.append((rel_path, str(e)))

    if not dry_run:
        print()
        print(f"Restore complete: {len(restored)} files restored")
        if skipped:
            print(f"  {len(skipped)} files skipped:")
            for f, reason in skipped:
                print(f"    ✗ {f} ({reason})")

        # Update current checkpoint metadata
        print()
        print(f"Restored from checkpoint: {checkpoint_id}")
        print(f"  Phase: {meta['phase']}")
        print(f"  Original timestamp: {meta['timestamp']}")

        # If there are app files, suggest restart
        has_app = any(f.startswith("app/") for f in restored)
        if has_app:
            print()
            print("⚠️  App files were restored. Restart the API server:")
    else:
        print()
        print(f"Dry run complete: {len(restored)} files would be restored")
        if skipped:
            print(f"  {len(skipped)} files would be skipped")


def main():
    parser = argparse.ArgumentParser(
        description="Restore EUR-Lex AI Chat project from a saved checkpoint"
    )
    parser.add_argument("--id", help="Checkpoint ID to restore")
    parser.add_argument("--latest", action="store_true", help="Restore latest checkpoint")
    parser.add_argument("--phase", help="Restore latest checkpoint for a specific phase")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Preview without restoring")

    args = parser.parse_args()

    if args.list:
        list_checkpoints()
        return

    checkpoint_id = find_checkpoint(
        checkpoint_id=args.id,
        phase=args.phase,
        latest=args.latest,
    )

    restore_checkpoint(checkpoint_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
