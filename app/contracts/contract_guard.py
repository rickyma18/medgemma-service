"""
Contract Guard - Drift detection for medicalization and normalization contracts.

Compares current runtime hashes against expected snapshots to detect
unauthorized or uncoordinated changes between app and backend.

PHI-safe: Only exposes hashes and boolean flags, never content.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Contract snapshot filenames
_MEDICALIZATION_CONTRACT_FILE = "medicalization_contract.json"
_NORMALIZATION_CONTRACT_FILE = "normalization_contract.json"


def _get_contracts_dir() -> Path:
    """Returns the path to the contracts directory."""
    return Path(__file__).resolve().parent


def _load_contract_snapshot(filename: str) -> Optional[Dict[str, Any]]:
    """
    Loads a contract snapshot JSON file.

    Args:
        filename: Name of the contract file (e.g., 'medicalization_contract.json')

    Returns:
        Dict with contract data, or None if file not found/invalid.
    """
    try:
        contract_path = _get_contracts_dir() / filename
        if not contract_path.exists():
            return None
        with open(contract_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_medicalization_hash() -> str:
    """Gets current medicalization hash from glossary module."""
    try:
        from app.services.medicalization.medicalization_glossary import get_glossary_hash
        return get_glossary_hash()
    except Exception:
        return ""


def _get_normalization_hash() -> str:
    """Gets current normalization hash from contract module."""
    try:
        from app.services.normalization.normalization_contract import get_normalization_hash
        return get_normalization_hash()
    except Exception:
        return ""


def _check_single_contract(
    contract_name: str,
    snapshot_file: str,
    get_hash_fn
) -> Dict[str, Any]:
    """
    Check a single contract for drift.

    Drift logic:
    - drift = True ONLY when actual hash is non-empty AND does not match expected
    - If actual hash is "" → no drift, warning "*_hash_unavailable"
    - If snapshot missing → no drift, warning "*_snapshot_missing"
    - If snapshot exists but expectedHash missing/empty → no drift, warning "*_snapshot_invalid"

    Returns:
        Dict with keys: drift (bool), warning (Optional[str]), details (dict)
    """
    snapshot = _load_contract_snapshot(snapshot_file)
    actual = get_hash_fn()

    # Case 1: Snapshot file missing
    if snapshot is None:
        logger.warning(
            "Contract snapshot missing",
            extra={"contract": contract_name, "file": snapshot_file}
        )
        return {
            "drift": False,
            "warning": f"{contract_name}_snapshot_missing",
            "details": {
                "expected": None,
                "actual": actual,
                "match": False,
                "snapshotMissing": True,
                "snapshotInvalid": False
            }
        }

    expected = snapshot.get("expectedHash")
    # Normalize empty string to None for consistency
    if expected == "":
        expected = None

    # Case 2: Snapshot exists but expectedHash missing or empty
    if expected is None:
        logger.warning(
            "Contract snapshot invalid (missing expectedHash)",
            extra={"contract": contract_name, "file": snapshot_file}
        )
        return {
            "drift": False,
            "warning": f"{contract_name}_snapshot_invalid",
            "details": {
                "expected": None,
                "actual": actual,
                "match": False,
                "snapshotMissing": False,
                "snapshotInvalid": True
            }
        }

    # Case 3: Actual hash unavailable (empty string)
    if not actual:
        logger.warning(
            "Runtime hash unavailable",
            extra={"contract": contract_name}
        )
        return {
            "drift": False,
            "warning": f"{contract_name}_hash_unavailable",
            "details": {
                "expected": expected,
                "actual": actual,
                "match": False,
                "snapshotMissing": False,
                "snapshotInvalid": False
            }
        }

    # Case 4: Both hashes available - compare
    is_match = (actual == expected)

    if is_match:
        # No drift, no warning
        return {
            "drift": False,
            "warning": None,
            "details": {
                "expected": expected,
                "actual": actual,
                "match": True,
                "snapshotMissing": False,
                "snapshotInvalid": False
            }
        }
    else:
        # Real drift detected
        logger.warning(
            "Contract drift detected",
            extra={"contract": contract_name, "expected": expected[:8] + "...", "actual": actual[:8] + "..."}
        )
        return {
            "drift": True,
            "warning": f"DRIFT:{contract_name}_drift",
            "details": {
                "expected": expected,
                "actual": actual,
                "match": False,
                "snapshotMissing": False,
                "snapshotInvalid": False
            }
        }


def check_contracts() -> Dict[str, Any]:
    """
    Validates current hashes against expected contract snapshots.

    Drift logic (per contract):
    - drift = True ONLY when actual hash is non-empty AND does not match expected
    - If actual hash is "" → no drift, warning "*_hash_unavailable"
    - If snapshot missing → no drift, warning "*_snapshot_missing"
    - If snapshot invalid (no expectedHash) → no drift, warning "*_snapshot_invalid"

    Returns:
        Dict with structure:
        {
            "medicalizationDrift": bool,
            "normalizationDrift": bool,
            "warnings": ["medicalization_drift", ...],
            "details": {
                "medicalization": {
                    "expected": Optional[str],
                    "actual": str,
                    "match": bool,
                    "snapshotMissing": bool,
                    "snapshotInvalid": bool
                },
                "normalization": { ... }
            }
        }

    PHI-safe: Only hashes (truncated in logs) and flags exposed.
    """
    result: Dict[str, Any] = {
        "medicalizationDrift": False,
        "normalizationDrift": False,
        "warnings": [],
        "details": {}
    }

    # Check medicalization
    med_result = _check_single_contract(
        "medicalization",
        _MEDICALIZATION_CONTRACT_FILE,
        _get_medicalization_hash
    )
    result["medicalizationDrift"] = med_result["drift"]
    result["details"]["medicalization"] = med_result["details"]
    if med_result["warning"]:
        result["warnings"].append(med_result["warning"])

    # Check normalization
    norm_result = _check_single_contract(
        "normalization",
        _NORMALIZATION_CONTRACT_FILE,
        _get_normalization_hash
    )
    result["normalizationDrift"] = norm_result["drift"]
    result["details"]["normalization"] = norm_result["details"]
    if norm_result["warning"]:
        result["warnings"].append(norm_result["warning"])

    return result


def get_contract_warnings() -> List[str]:
    """
    Convenience function to get just the warnings list.

    Returns:
        List of warning strings (empty if no drift detected).
    """
    return check_contracts()["warnings"]


def has_drift() -> bool:
    """
    Quick check if any contract drift is detected.

    Returns:
        True if medicalization or normalization drift detected.
    """
    result = check_contracts()
    return result["medicalizationDrift"] or result["normalizationDrift"]
