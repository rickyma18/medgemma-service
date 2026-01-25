"""
Tests for normalization rules contract versioning and hashing.

ÉPICA 13.6: Anti-drift between app and backend normalization rules.
Validates that normalization hash is computed correctly for contract tracking.

This test does NOT require env var since rules are hardcoded in code.
"""
import re
import pytest


@pytest.fixture(autouse=True)
def clear_cache_before_test():
    """Clear cache before each test."""
    from app.services.normalization.normalization_contract import clear_cache
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def contract_module():
    """Import and return the contract module."""
    import app.services.normalization.normalization_contract as mod
    return mod


class TestNormalizationContractHash:
    """Test suite for normalization contract hash computation."""

    def test_version_not_empty(self, contract_module):
        """NORMALIZATION_VERSION should not be empty."""
        version = contract_module.get_normalization_version()
        assert isinstance(version, str), f"Version must be str, got {type(version)}"
        assert version != "", "Normalization version is empty string"

    def test_version_constant_available(self, contract_module):
        """NORMALIZATION_VERSION constant should be accessible and correct."""
        assert hasattr(contract_module, "NORMALIZATION_VERSION")
        assert contract_module.NORMALIZATION_VERSION == "v1"

    def test_hash_is_string(self, contract_module):
        """Hash should always be a string (contract: never None)."""
        hash_value = contract_module.get_normalization_hash()
        assert isinstance(hash_value, str), f"Hash must be str, got {type(hash_value)}"

    def test_hash_not_empty(self, contract_module):
        """Hash should be computed and not empty."""
        hash_value = contract_module.get_normalization_hash()
        assert hash_value != "", "Normalization hash is empty string (no rules loaded?)"

    def test_hash_length_64(self, contract_module):
        """SHA256 hex digest should be exactly 64 characters."""
        hash_value = contract_module.get_normalization_hash()
        assert len(hash_value) == 64, (
            f"Expected hash length 64, got {len(hash_value)}"
        )

    def test_hash_only_hex_chars(self, contract_module):
        """Hash should contain only lowercase hex characters [0-9a-f]."""
        hash_value = contract_module.get_normalization_hash()
        hex_pattern = re.compile(r"^[0-9a-f]+$")
        assert hex_pattern.match(hash_value), (
            f"Hash contains non-hex characters: {hash_value}"
        )

    def test_module_level_hash_synced_after_call(self, contract_module):
        """NORMALIZATION_HASH module var should sync after get_normalization_hash()."""
        hash_from_getter = contract_module.get_normalization_hash()
        hash_from_module = contract_module.NORMALIZATION_HASH
        assert hash_from_module == hash_from_getter, (
            "Module-level NORMALIZATION_HASH not synced with getter"
        )

    def test_hash_cached_after_first_call(self, contract_module):
        """Hash should be cached after first computation."""
        hash1 = contract_module.get_normalization_hash()
        hash2 = contract_module.get_normalization_hash()
        assert hash1 == hash2, "Hash not consistently cached"

    def test_hash_deterministic(self, contract_module):
        """Hash should be deterministic (same rules = same hash)."""
        contract_module.clear_cache()
        hash1 = contract_module.get_normalization_hash()

        contract_module.clear_cache()
        hash2 = contract_module.get_normalization_hash()

        assert hash1 == hash2, "Hash is not deterministic"

    def test_clear_cache_resets_hash(self, contract_module):
        """clear_cache() should reset NORMALIZATION_HASH to empty string."""
        # First compute hash
        contract_module.get_normalization_hash()
        assert contract_module.NORMALIZATION_HASH != ""

        # Clear cache
        contract_module.clear_cache()
        assert contract_module.NORMALIZATION_HASH == ""

    def test_canonical_rules_not_empty(self, contract_module):
        """Internal _build_canonical_rules should return non-empty list."""
        canonical = contract_module._build_canonical_rules()
        assert isinstance(canonical, list)
        assert len(canonical) > 0, "Canonical rules list is empty"

    def test_canonical_rules_format(self, contract_module):
        """Each canonical rule should have format 'priority|pattern|replacement'."""
        canonical = contract_module._build_canonical_rules()
        for rule in canonical:
            parts = rule.split('|')
            assert len(parts) == 3, f"Rule format invalid: {rule}"
            priority, pattern, replacement = parts
            assert priority.isdigit(), f"Priority not numeric: {priority}"
            assert pattern, f"Pattern is empty in rule: {rule}"
            assert replacement, f"Replacement is empty in rule: {rule}"


class TestNormalizationContractIntegration:
    """Integration tests with actual normalization rules."""

    def test_rules_loaded_from_normalizer(self, contract_module):
        """Should load rules from text_normalizer_orl.ORL_STT_WHITELIST."""
        from app.services.text_normalizer_orl import ORL_STT_WHITELIST

        canonical = contract_module._build_canonical_rules()

        # Should have same count as whitelist
        assert len(canonical) == len(ORL_STT_WHITELIST), (
            f"Expected {len(ORL_STT_WHITELIST)} rules, got {len(canonical)}"
        )

    def test_hash_changes_if_rules_change(self, contract_module, monkeypatch):
        """Hash should change if underlying rules change."""
        original_hash = contract_module.get_normalization_hash()

        # Mock _get_orl_stt_whitelist to return different rules
        import re as re_module
        mock_rules = [(re_module.compile(r"\btest\b"), "TEST")]

        monkeypatch.setattr(
            contract_module,
            "_get_orl_stt_whitelist",
            lambda: mock_rules
        )
        contract_module.clear_cache()

        new_hash = contract_module.get_normalization_hash()

        assert new_hash != original_hash, (
            "Hash should change when rules change"
        )
