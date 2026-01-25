"""
Tests for medicalization glossary version and hash contract freeze.

ÉPICA 13.5: Anti-drift between app and backend.
Validates that glossary hash is computed correctly for contract tracking.

This test uses MEDICALIZATION_GLOSSARY_PATH pointing to the Flutter app glossary.
"""
import os
import re
import pytest


# Path to the real glossary from Flutter app
GLOSSARY_PATH = r"C:\dev\ent-voice-notes-app\lib\src\features\medical_notes\resources\medical_lexicon\colloquial_to_clinical_es.json"


@pytest.fixture(autouse=True)
def setup_glossary_env(monkeypatch):
    """Set MEDICALIZATION_GLOSSARY_PATH for all tests in this module."""
    monkeypatch.setenv("MEDICALIZATION_GLOSSARY_PATH", GLOSSARY_PATH)
    # Clear cache before each test
    from app.services.medicalization.medicalization_glossary import clear_cache
    clear_cache()
    yield
    # Clear cache after test
    clear_cache()


@pytest.fixture
def glossary_module():
    """Import and return the glossary module after env is set."""
    import app.services.medicalization.medicalization_glossary as mod
    return mod


class TestGlossaryHash:
    """Test suite for glossary hash computation."""

    def test_glossary_file_exists(self):
        """Verify the glossary file exists at the expected path."""
        assert os.path.exists(GLOSSARY_PATH), (
            f"Glossary file not found at {GLOSSARY_PATH}. "
            "Ensure ent-voice-notes-app repo is cloned at C:\\dev\\"
        )

    def test_hash_not_empty(self, glossary_module):
        """Hash should be computed and not empty (contract: str, never None)."""
        hash_value = glossary_module.get_glossary_hash()
        # Contract: always str, "" means unavailable
        assert isinstance(hash_value, str), f"Hash must be str, got {type(hash_value)}"
        assert hash_value != "", "Glossary hash is empty string (glossary not loaded?)"

    def test_hash_length_64(self, glossary_module):
        """SHA256 hex digest should be exactly 64 characters."""
        hash_value = glossary_module.get_glossary_hash()
        assert len(hash_value) == 64, (
            f"Expected hash length 64, got {len(hash_value)}"
        )

    def test_hash_only_hex_chars(self, glossary_module):
        """Hash should contain only lowercase hex characters [0-9a-f]."""
        hash_value = glossary_module.get_glossary_hash()
        hex_pattern = re.compile(r"^[0-9a-f]+$")
        assert hex_pattern.match(hash_value), (
            f"Hash contains non-hex characters: {hash_value}"
        )

    def test_version_not_empty(self, glossary_module):
        """Version should be set and not empty (contract: str, never None)."""
        version = glossary_module.get_glossary_version()
        assert isinstance(version, str), f"Version must be str, got {type(version)}"
        assert version != "", "Glossary version is empty string"

    def test_version_constant_available(self, glossary_module):
        """MEDICALIZATION_GLOSSARY_VERSION constant should be accessible."""
        assert hasattr(glossary_module, "MEDICALIZATION_GLOSSARY_VERSION")
        assert glossary_module.MEDICALIZATION_GLOSSARY_VERSION == "v1"

    def test_hash_constant_is_str_type(self, glossary_module):
        """MEDICALIZATION_GLOSSARY_HASH must be str type (contract: never None)."""
        assert hasattr(glossary_module, "MEDICALIZATION_GLOSSARY_HASH")
        # Before loading, should be "" (empty string), not None
        assert isinstance(glossary_module.MEDICALIZATION_GLOSSARY_HASH, str), (
            f"MEDICALIZATION_GLOSSARY_HASH must be str, got {type(glossary_module.MEDICALIZATION_GLOSSARY_HASH)}"
        )

    def test_hash_matches_manual_computation(self, glossary_module):
        """Hash should match independent SHA256 computation of file bytes."""
        import hashlib

        with open(GLOSSARY_PATH, "rb") as f:
            expected_hash = hashlib.sha256(f.read()).hexdigest()

        actual_hash = glossary_module.get_glossary_hash()
        assert actual_hash == expected_hash, (
            f"Hash mismatch!\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    def test_module_level_hash_synced_after_call(self, glossary_module):
        """MEDICALIZATION_GLOSSARY_HASH module var should sync after get_glossary_hash()."""
        # Call the getter which should sync the module-level variable
        hash_from_getter = glossary_module.get_glossary_hash()

        # Now check the module-level variable
        hash_from_module = glossary_module.MEDICALIZATION_GLOSSARY_HASH

        assert hash_from_module == hash_from_getter, (
            "Module-level MEDICALIZATION_GLOSSARY_HASH not synced with getter"
        )

    def test_hash_cached_after_load(self, glossary_module):
        """Hash should be cached after first load."""
        # First call loads
        hash1 = glossary_module.get_glossary_hash()

        # Second call should return same cached value
        hash2 = glossary_module.get_glossary_hash()

        assert hash1 == hash2, "Hash not consistently cached"

    def test_env_var_takes_priority(self, glossary_module):
        """Env var path should be used when set."""
        path = glossary_module.resolve_glossary_path()
        assert path is not None
        assert str(path) == GLOSSARY_PATH, (
            f"Expected env var path {GLOSSARY_PATH}, got {path}"
        )
