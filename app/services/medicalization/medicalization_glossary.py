"""
Loader for the medicalization glossary from JSON.

Priority for glossary resolution:
1. MEDICALIZATION_GLOSSARY_PATH env var (if set and file exists)
2. Packaged resources in backend repo (app/resources/medical_lexicon/...)
3. Soft-fail (empty list, empty hash)

PHI-safe: Never logs glossary content.
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Constants for priorities
PRIORITY_SYMPTOMS = 300
PRIORITY_PHRASES = 200
PRIORITY_VOICE_TRANSFORMS = 100

# Packaged glossary filename
_GLOSSARY_FILENAME = "colloquial_to_clinical_es.json"


def _get_packaged_glossary_path() -> Path:
    """
    Returns the path to the packaged glossary in app/resources/.
    Uses Path relative to this module file for robustness.
    """
    # This file: app/services/medicalization/medicalization_glossary.py
    # Target: app/resources/medical_lexicon/colloquial_to_clinical_es.json
    this_file = Path(__file__).resolve()
    # Go up: medicalization -> services -> app
    app_dir = this_file.parent.parent.parent
    return app_dir / "resources" / "medical_lexicon" / _GLOSSARY_FILENAME


def resolve_glossary_path() -> Optional[Path]:
    """
    Resolves the glossary path with priority:
    1. MEDICALIZATION_GLOSSARY_PATH env var (if set and file exists)
    2. Packaged resources (app/resources/medical_lexicon/...)
    3. None (soft-fail)

    Returns:
        Path to glossary file, or None if not found.
    """
    # 1. Env var takes priority (for dev/testing with Flutter glossary)
    env_path = os.getenv("MEDICALIZATION_GLOSSARY_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        # PHI-safe: only log that path was not found, not the content
        logger.warning("MEDICALIZATION_GLOSSARY_PATH set but file not found")

    # 2. Packaged resources (prod default)
    packaged = _get_packaged_glossary_path()
    if packaged.exists():
        return packaged

    # 3. Soft-fail
    return None


class GlossaryEntry:
    """Represents a single glossary mapping entry."""

    def __init__(self, term: str, replacement: str, priority: int, category: str):
        self.term = term
        self.replacement = replacement
        self.priority = priority
        self.category = category


# Global cache for mappings and hash
_CACHED_MAPPINGS: Optional[List[GlossaryEntry]] = None
_CACHED_HASH: Optional[str] = None
_CACHED_PATH_USED: Optional[str] = None

# Versioning - static version identifier for contract tracking
MEDICALIZATION_GLOSSARY_VERSION: str = "v1"

# Hash - computed lazily on first access, "" if glossary unavailable
# Contract: always str, never None. Empty string = not yet computed or unavailable.
MEDICALIZATION_GLOSSARY_HASH: str = ""


def get_glossary_hash() -> str:
    """
    Returns the SHA256 hash of the currently loaded glossary.

    Lazy-loads the glossary if not already cached.
    PHI-safe: Only returns hash, never logs glossary content.

    Returns:
        str: 64-char hex hash if glossary loaded, "" if unavailable.
    """
    global MEDICALIZATION_GLOSSARY_HASH
    if _CACHED_HASH is None:
        load_glossary_mappings()
    # Sync module-level variable with cached value (or "" if still None)
    MEDICALIZATION_GLOSSARY_HASH = _CACHED_HASH if _CACHED_HASH else ""
    return MEDICALIZATION_GLOSSARY_HASH


def get_glossary_version() -> str:
    """Returns the glossary version identifier."""
    return MEDICALIZATION_GLOSSARY_VERSION


def load_glossary_mappings() -> List[GlossaryEntry]:
    """
    Loads and parses the glossary.

    Returns a flat list of entries sorted by priority (desc) then length (desc).
    Uses resolve_glossary_path() for path resolution.
    Caches result; invalidates cache if path changes.
    """
    global _CACHED_MAPPINGS, _CACHED_HASH, _CACHED_PATH_USED, MEDICALIZATION_GLOSSARY_HASH

    path = resolve_glossary_path()
    path_str = str(path) if path else None

    # Return cached if path hasn't changed and we have data
    if _CACHED_MAPPINGS is not None and path_str == _CACHED_PATH_USED:
        return _CACHED_MAPPINGS

    if not path:
        # Soft-fail: no glossary available
        logger.warning("Medicalization glossary not found (no env var, no packaged file)")
        _CACHED_MAPPINGS = []
        _CACHED_HASH = None
        MEDICALIZATION_GLOSSARY_HASH = ""
        return []

    try:
        # Read raw bytes for hash computation (PHI-safe: only hash, never log content)
        content_bytes = path.read_bytes()
        _CACHED_HASH = hashlib.sha256(content_bytes).hexdigest()
        MEDICALIZATION_GLOSSARY_HASH = _CACHED_HASH
        data = json.loads(content_bytes.decode("utf-8"))
        _CACHED_PATH_USED = path_str
    except Exception:
        # Soft-fail on read/parse error
        logger.warning("Failed to load medicalization glossary")
        _CACHED_MAPPINGS = []
        _CACHED_HASH = None
        MEDICALIZATION_GLOSSARY_HASH = ""
        return []

    entries: List[GlossaryEntry] = []

    # Map categories to priorities
    category_map = {
        "symptoms": PRIORITY_SYMPTOMS,
        "symptoms_orl": PRIORITY_SYMPTOMS,
        "antecedentes": PRIORITY_PHRASES,
        "habits": PRIORITY_PHRASES,
        "voice_transforms": PRIORITY_VOICE_TRANSFORMS
    }

    for cat, priority in category_map.items():
        section = data.get(cat, {})
        if not isinstance(section, dict):
            continue

        for term, info in section.items():
            if not isinstance(info, dict):
                continue

            replacement = info.get("clinical")
            if replacement:
                entries.append(GlossaryEntry(term, replacement, priority, cat))

    # Sort:
    # 1. Priority DESC
    # 2. Length of term DESC (longer matches first)
    entries.sort(key=lambda x: (x.priority, len(x.term)), reverse=True)

    _CACHED_MAPPINGS = entries
    return entries


def clear_cache() -> None:
    """
    Clears the glossary cache. Useful for testing.
    """
    global _CACHED_MAPPINGS, _CACHED_HASH, _CACHED_PATH_USED, MEDICALIZATION_GLOSSARY_HASH
    _CACHED_MAPPINGS = None
    _CACHED_HASH = None
    _CACHED_PATH_USED = None
    MEDICALIZATION_GLOSSARY_HASH = ""
