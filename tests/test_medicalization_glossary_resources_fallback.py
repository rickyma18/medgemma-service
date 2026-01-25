"""
Tests for medicalization glossary resources fallback.

ÉPICA 13.5: Anti-drift between app and backend.
Validates that the packaged glossary in resources/ is used when no env var is set.

This test runs WITHOUT MEDICALIZATION_GLOSSARY_PATH to verify prod-like behavior.
"""
import hashlib
import re
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clear_env_and_cache(monkeypatch):
    """Ensure no env var is set and cache is cleared."""
    # Remove env var if set
    monkeypatch.delenv("MEDICALIZATION_GLOSSARY_PATH", raising=False)
    # Clear cache before test
    from app.services.medicalization.medicalization_glossary import clear_cache
    clear_cache()
    yield
    # Clear cache after test
    clear_cache()


@pytest.fixture
def glossary_module():
    """Import and return the glossary module."""
    import app.services.medicalization.medicalization_glossary as mod
    return mod


@pytest.fixture
def packaged_glossary_path():
    """Returns the expected path to the packaged glossary."""
    # Use the same logic as the module to find the path
    this_file = Path(__file__).resolve()
    # tests/ -> medgemma-service/ -> app/resources/medical_lexicon/
    repo_root = this_file.parent.parent
    return repo_root / "app" / "resources" / "medical_lexicon" / "colloquial_to_clinical_es.json"


class TestResourcesFallback:
    """Test suite for packaged resources fallback."""

    def test_packaged_glossary_exists(self, packaged_glossary_path):
        """Verify the packaged glossary file exists in resources/."""
        assert packaged_glossary_path.exists(), (
            f"Packaged glossary not found at {packaged_glossary_path}. "
            "Ensure app/resources/medical_lexicon/colloquial_to_clinical_es.json exists."
        )

    def test_resolve_uses_packaged_when_no_env_var(self, glossary_module, packaged_glossary_path):
        """Without env var, resolve_glossary_path() should return packaged path."""
        resolved = glossary_module.resolve_glossary_path()
        assert resolved is not None, "resolve_glossary_path() returned None"
        # Compare resolved paths
        assert resolved.resolve() == packaged_glossary_path.resolve(), (
            f"Expected packaged path {packaged_glossary_path}, got {resolved}"
        )

    def test_hash_from_packaged_not_empty(self, glossary_module):
        """Hash from packaged glossary should be valid."""
        hash_value = glossary_module.get_glossary_hash()
        assert isinstance(hash_value, str), f"Hash must be str, got {type(hash_value)}"
        assert hash_value != "", "Glossary hash is empty (packaged file not loaded?)"

    def test_hash_from_packaged_length_64(self, glossary_module):
        """SHA256 hex digest should be exactly 64 characters."""
        hash_value = glossary_module.get_glossary_hash()
        assert len(hash_value) == 64, f"Expected hash length 64, got {len(hash_value)}"

    def test_hash_from_packaged_only_hex(self, glossary_module):
        """Hash should contain only lowercase hex characters [0-9a-f]."""
        hash_value = glossary_module.get_glossary_hash()
        hex_pattern = re.compile(r"^[0-9a-f]+$")
        assert hex_pattern.match(hash_value), f"Hash contains non-hex: {hash_value}"

    def test_hash_matches_packaged_file(self, glossary_module, packaged_glossary_path):
        """Hash should match SHA256 of the packaged file bytes."""
        expected_hash = hashlib.sha256(packaged_glossary_path.read_bytes()).hexdigest()
        actual_hash = glossary_module.get_glossary_hash()
        assert actual_hash == expected_hash, (
            f"Hash mismatch!\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )

    def test_load_mappings_not_empty(self, glossary_module):
        """load_glossary_mappings() should return entries from packaged glossary."""
        entries = glossary_module.load_glossary_mappings()
        assert isinstance(entries, list), f"Expected list, got {type(entries)}"
        assert len(entries) > 0, "Packaged glossary returned empty entries"

    def test_load_mappings_has_expected_categories(self, glossary_module):
        """Loaded mappings should include entries from multiple categories."""
        entries = glossary_module.load_glossary_mappings()
        categories = {e.category for e in entries}
        # Should have at least symptoms and voice_transforms
        assert "symptoms" in categories or "symptoms_orl" in categories, (
            f"Missing symptoms category. Found: {categories}"
        )
        assert "voice_transforms" in categories, (
            f"Missing voice_transforms category. Found: {categories}"
        )

    def test_version_available(self, glossary_module):
        """Version should be accessible."""
        version = glossary_module.get_glossary_version()
        assert isinstance(version, str)
        assert version != ""
        assert version == "v1"

    def test_module_hash_synced(self, glossary_module):
        """MEDICALIZATION_GLOSSARY_HASH should sync after get_glossary_hash()."""
        hash_from_getter = glossary_module.get_glossary_hash()
        hash_from_module = glossary_module.MEDICALIZATION_GLOSSARY_HASH
        assert hash_from_module == hash_from_getter


class TestEnvVarPriority:
    """Test that env var takes priority over packaged resources."""

    def test_env_var_overrides_packaged(self, glossary_module, monkeypatch, tmp_path):
        """When env var is set to a valid file, it should be used over packaged."""
        # Create a temp glossary file
        temp_glossary = tmp_path / "test_glossary.json"
        temp_glossary.write_text('{"symptoms": {}, "voice_transforms": {}}')

        # Set env var
        monkeypatch.setenv("MEDICALIZATION_GLOSSARY_PATH", str(temp_glossary))
        glossary_module.clear_cache()

        # Resolve should return the env var path
        resolved = glossary_module.resolve_glossary_path()
        assert resolved is not None
        assert resolved.resolve() == temp_glossary.resolve()

    def test_invalid_env_var_falls_back_to_packaged(
        self, glossary_module, monkeypatch, packaged_glossary_path
    ):
        """When env var points to non-existent file, should fall back to packaged."""
        # Set env var to non-existent path
        monkeypatch.setenv("MEDICALIZATION_GLOSSARY_PATH", "/nonexistent/path.json")
        glossary_module.clear_cache()

        # Should fall back to packaged
        resolved = glossary_module.resolve_glossary_path()
        assert resolved is not None
        assert resolved.resolve() == packaged_glossary_path.resolve()
