"""
Contract module for normalization rules versioning and hashing.

Provides deterministic hash of normalization rules for anti-drift detection
between app and backend.

PHI-safe: Never logs rule content or transcript data.
"""
import hashlib
import re
from typing import List, Optional, Tuple

# Priority constants for rule families (stable ordering)
PRIORITY_ORL_STT_WHITELIST = 300

# Global cache
_CACHED_HASH: Optional[str] = None

# Versioning - static version identifier for contract tracking
NORMALIZATION_VERSION: str = "v1"

# Hash - computed lazily on first access, "" if unavailable
# Contract: always str, never None. Empty string = not yet computed or unavailable.
NORMALIZATION_HASH: str = ""


def _get_orl_stt_whitelist() -> List[Tuple[re.Pattern, str]]:
    """
    Imports and returns the ORL STT whitelist rules.
    Separated for testability and to avoid circular imports.
    """
    from app.services.text_normalizer_orl import ORL_STT_WHITELIST
    return ORL_STT_WHITELIST


def _build_canonical_rules() -> List[str]:
    """
    Builds a canonical list of rule strings for deterministic hashing.

    Format per rule: "{priority}|{pattern}|{replacement}"
    - pattern: regex pattern string (from compiled pattern)
    - replacement: literal replacement string
    - priority: numeric priority for stable ordering

    Returns:
        List of canonical rule strings, sorted deterministically.
    """
    canonical: List[str] = []

    # ORL STT Whitelist rules
    try:
        orl_rules = _get_orl_stt_whitelist()
        for pattern, replacement in orl_rules:
            # Extract pattern string from compiled regex
            pattern_str = pattern.pattern if hasattr(pattern, 'pattern') else str(pattern)
            canonical.append(f"{PRIORITY_ORL_STT_WHITELIST}|{pattern_str}|{replacement}")
    except Exception:
        # Soft-fail: if rules can't be loaded, return empty
        pass

    # Sort deterministically: priority DESC, pattern ASC, replacement ASC
    # We use negative priority for DESC sort, then pattern and replacement ASC
    canonical.sort(key=lambda x: (-int(x.split('|')[0]), x.split('|')[1], x.split('|')[2]))

    return canonical


def _compute_hash() -> str:
    """
    Computes SHA256 hash of canonical rules.

    Returns:
        64-char lowercase hex string, or "" if computation fails.
    """
    try:
        canonical_rules = _build_canonical_rules()
        if not canonical_rules:
            return ""

        canonical_blob = "\n".join(canonical_rules).encode("utf-8")
        return hashlib.sha256(canonical_blob).hexdigest()
    except Exception:
        return ""


def get_normalization_hash() -> str:
    """
    Returns the SHA256 hash of the normalization rules.

    Lazy-computes the hash on first access.
    PHI-safe: Only returns hash, never logs rule content.

    Returns:
        str: 64-char hex hash if rules available, "" if unavailable.
    """
    global _CACHED_HASH, NORMALIZATION_HASH

    if _CACHED_HASH is None:
        _CACHED_HASH = _compute_hash()

    # Sync module-level variable with cached value
    NORMALIZATION_HASH = _CACHED_HASH if _CACHED_HASH else ""
    return NORMALIZATION_HASH


def get_normalization_version() -> str:
    """Returns the normalization version identifier."""
    return NORMALIZATION_VERSION


def clear_cache() -> None:
    """
    Clears the hash cache. Useful for testing.
    """
    global _CACHED_HASH, NORMALIZATION_HASH
    _CACHED_HASH = None
    NORMALIZATION_HASH = ""
