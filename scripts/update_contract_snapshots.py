#!/usr/bin/env python3
"""
Contract Snapshot Updater - CLI tool for updating contract snapshots.

Build-time tooling for CI/PR workflows. NOT for runtime use.

Usage:
    # Check mode (CI) - validates snapshots match runtime hashes
    python scripts/update_contract_snapshots.py --check

    # Dry run - shows what would change without writing
    python scripts/update_contract_snapshots.py --dry-run

    # Update mode (default) - writes updated snapshots
    python scripts/update_contract_snapshots.py

Environment:
    CONTRACTS_DIR - Override contracts directory path
    MEDICALIZATION_GLOSSARY_PATH - Path to glossary JSON (required for medicalization hash)

Exit codes:
    0 - Success (or no changes needed)
    1 - Drift detected (--check mode) or validation error
    2 - Hash unavailable (empty string from runtime)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_contracts_dir(override: Optional[str] = None) -> Path:
    """
    Resolve contracts directory path.

    Priority:
    1. --contracts-dir argument
    2. CONTRACTS_DIR env var
    3. Default: app/contracts relative to project root
    """
    if override:
        return Path(override)

    env_dir = os.environ.get("CONTRACTS_DIR")
    if env_dir:
        return Path(env_dir)

    return PROJECT_ROOT / "app" / "contracts"


def get_medicalization_hash_and_version() -> Tuple[str, str]:
    """
    Get current medicalization glossary hash and version.

    Returns:
        Tuple of (hash, version). Hash may be empty string if unavailable.
    """
    try:
        from app.services.medicalization.medicalization_glossary import (
            get_glossary_hash,
            get_glossary_version
        )
        return get_glossary_hash(), get_glossary_version()
    except Exception as e:
        print(f"[WARN] Failed to get medicalization hash: {e}", file=sys.stderr)
        return "", "v1"


def get_normalization_hash_and_version() -> Tuple[str, str]:
    """
    Get current normalization rules hash and version.

    Returns:
        Tuple of (hash, version). Hash may be empty string if unavailable.
    """
    try:
        from app.services.normalization.normalization_contract import (
            get_normalization_hash,
            get_normalization_version
        )
        return get_normalization_hash(), get_normalization_version()
    except Exception as e:
        print(f"[WARN] Failed to get normalization hash: {e}", file=sys.stderr)
        return "", "v1"


def load_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    """Load existing snapshot JSON, or None if not found."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_snapshot(path: Path, data: Dict[str, Any]) -> None:
    """Write snapshot JSON with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")  # Trailing newline


def build_snapshot(
    current_hash: str,
    version: str,
    description: str,
    existing: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a new snapshot dict, preserving extra fields from existing.
    """
    snapshot = {
        "version": version,
        "expectedHash": current_hash,
        "description": description,
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    # Preserve any extra fields from existing snapshot
    if existing:
        for key in existing:
            if key not in snapshot:
                snapshot[key] = existing[key]

    return snapshot


def truncate_hash(h: str, length: int = 12) -> str:
    """Truncate hash for safe logging."""
    if len(h) <= length:
        return h
    return h[:length] + "..."


def run_check(contracts_dir: Path) -> int:
    """
    Check mode: validate that snapshots match runtime hashes.

    Returns:
        0 if all match, 1 if drift detected.
    """
    print(f"[CHECK] Contracts dir: {contracts_dir}")

    has_drift = False

    # Medicalization
    med_path = contracts_dir / "medicalization_contract.json"
    med_hash, med_version = get_medicalization_hash_and_version()
    med_snapshot = load_snapshot(med_path)

    if not med_hash:
        print(f"[WARN] Medicalization hash unavailable (empty)")
    elif med_snapshot is None:
        print(f"[DRIFT] Medicalization snapshot missing: {med_path}")
        has_drift = True
    else:
        expected = med_snapshot.get("expectedHash", "")
        if med_hash != expected:
            print(f"[DRIFT] Medicalization hash mismatch")
            print(f"        Expected: {truncate_hash(expected)}")
            print(f"        Actual:   {truncate_hash(med_hash)}")
            has_drift = True
        else:
            print(f"[OK] Medicalization: {truncate_hash(med_hash)}")

    # Normalization
    norm_path = contracts_dir / "normalization_contract.json"
    norm_hash, norm_version = get_normalization_hash_and_version()
    norm_snapshot = load_snapshot(norm_path)

    if not norm_hash:
        print(f"[WARN] Normalization hash unavailable (empty)")
    elif norm_snapshot is None:
        print(f"[DRIFT] Normalization snapshot missing: {norm_path}")
        has_drift = True
    else:
        expected = norm_snapshot.get("expectedHash", "")
        if norm_hash != expected:
            print(f"[DRIFT] Normalization hash mismatch")
            print(f"        Expected: {truncate_hash(expected)}")
            print(f"        Actual:   {truncate_hash(norm_hash)}")
            has_drift = True
        else:
            print(f"[OK] Normalization: {truncate_hash(norm_hash)}")

    if has_drift:
        print("\n[FAIL] Contract drift detected. Run without --check to update snapshots.")
        return 1

    print("\n[PASS] All contracts match.")
    return 0


def run_update(contracts_dir: Path, dry_run: bool = False) -> int:
    """
    Update mode: write updated snapshots.

    Args:
        contracts_dir: Path to contracts directory
        dry_run: If True, don't write, just show what would change

    Returns:
        0 if success, 2 if hash unavailable
    """
    mode = "DRY-RUN" if dry_run else "UPDATE"
    print(f"[{mode}] Contracts dir: {contracts_dir}")

    has_error = False
    changes = []

    # Medicalization
    med_path = contracts_dir / "medicalization_contract.json"
    med_hash, med_version = get_medicalization_hash_and_version()
    med_snapshot = load_snapshot(med_path)

    if not med_hash:
        print(f"[ERROR] Medicalization hash unavailable - cannot update snapshot")
        has_error = True
    else:
        med_expected = med_snapshot.get("expectedHash", "") if med_snapshot else ""
        if med_hash != med_expected:
            new_snapshot = build_snapshot(
                med_hash,
                med_version,
                "Medicalization glossary contract - colloquial_to_clinical_es.json",
                med_snapshot
            )
            changes.append(("medicalization", med_path, new_snapshot, med_expected, med_hash))
            print(f"[CHANGE] Medicalization: {truncate_hash(med_expected or '(none)')} -> {truncate_hash(med_hash)}")
        else:
            print(f"[UNCHANGED] Medicalization: {truncate_hash(med_hash)}")

    # Normalization
    norm_path = contracts_dir / "normalization_contract.json"
    norm_hash, norm_version = get_normalization_hash_and_version()
    norm_snapshot = load_snapshot(norm_path)

    if not norm_hash:
        print(f"[ERROR] Normalization hash unavailable - cannot update snapshot")
        has_error = True
    else:
        norm_expected = norm_snapshot.get("expectedHash", "") if norm_snapshot else ""
        if norm_hash != norm_expected:
            new_snapshot = build_snapshot(
                norm_hash,
                norm_version,
                "Normalization rules contract - ORL_STT_WHITELIST canonical hash",
                norm_snapshot
            )
            changes.append(("normalization", norm_path, new_snapshot, norm_expected, norm_hash))
            print(f"[CHANGE] Normalization: {truncate_hash(norm_expected or '(none)')} -> {truncate_hash(norm_hash)}")
        else:
            print(f"[UNCHANGED] Normalization: {truncate_hash(norm_hash)}")

    if has_error:
        print(f"\n[FAIL] Cannot update snapshots - hash(es) unavailable.")
        return 2

    if not changes:
        print(f"\n[OK] No changes needed.")
        return 0

    if dry_run:
        print(f"\n[DRY-RUN] Would update {len(changes)} snapshot(s). Use without --dry-run to write.")
        return 0

    # Write changes
    for name, path, snapshot, old_hash, new_hash in changes:
        write_snapshot(path, snapshot)
        print(f"[WRITTEN] {path}")

    print(f"\n[OK] Updated {len(changes)} snapshot(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update or check contract snapshots for drift detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: validate snapshots match runtime hashes (exit 1 if drift)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: show what would change without writing"
    )

    parser.add_argument(
        "--contracts-dir",
        type=str,
        default=None,
        help="Override contracts directory path (default: app/contracts or CONTRACTS_DIR env)"
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.check and args.dry_run:
        print("[ERROR] --check and --dry-run are mutually exclusive", file=sys.stderr)
        return 1

    contracts_dir = get_contracts_dir(args.contracts_dir)

    if args.check:
        return run_check(contracts_dir)
    else:
        return run_update(contracts_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
