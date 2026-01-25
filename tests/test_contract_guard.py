"""
Tests for Contract Guard - Drift detection for medicalization and normalization contracts.

ÉPICA 13.7: Contract Snapshot & Drift Guard.
Validates that contract guard correctly detects drift between expected and actual hashes.

Test cases:
1. hash expected == actual → no warnings
2. hash modified → warning present
3. snapshot missing → warning soft
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def contract_guard_module():
    """Import and return the contract guard module."""
    import app.contracts.contract_guard as mod
    return mod


@pytest.fixture
def contracts_dir(contract_guard_module):
    """Get the contracts directory path."""
    return contract_guard_module._get_contracts_dir()


class TestContractGuardBasic:
    """Basic tests for contract guard module structure."""

    def test_module_exports_check_contracts(self, contract_guard_module):
        """check_contracts function should be available."""
        assert hasattr(contract_guard_module, "check_contracts")
        assert callable(contract_guard_module.check_contracts)

    def test_module_exports_get_contract_warnings(self, contract_guard_module):
        """get_contract_warnings function should be available."""
        assert hasattr(contract_guard_module, "get_contract_warnings")
        assert callable(contract_guard_module.get_contract_warnings)

    def test_module_exports_has_drift(self, contract_guard_module):
        """has_drift function should be available."""
        assert hasattr(contract_guard_module, "has_drift")
        assert callable(contract_guard_module.has_drift)

    def test_contracts_dir_exists(self, contracts_dir):
        """Contracts directory should exist."""
        assert contracts_dir.exists(), f"Contracts dir not found: {contracts_dir}"
        assert contracts_dir.is_dir()


class TestCheckContractsStructure:
    """Test check_contracts() return structure."""

    def test_returns_dict(self, contract_guard_module):
        """check_contracts should return a dict."""
        result = contract_guard_module.check_contracts()
        assert isinstance(result, dict)

    def test_has_required_keys(self, contract_guard_module):
        """Result should have all required keys."""
        result = contract_guard_module.check_contracts()
        required_keys = ["medicalizationDrift", "normalizationDrift", "warnings", "details"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_drift_flags_are_bool(self, contract_guard_module):
        """Drift flags should be boolean."""
        result = contract_guard_module.check_contracts()
        assert isinstance(result["medicalizationDrift"], bool)
        assert isinstance(result["normalizationDrift"], bool)

    def test_warnings_is_list(self, contract_guard_module):
        """warnings should be a list."""
        result = contract_guard_module.check_contracts()
        assert isinstance(result["warnings"], list)

    def test_details_is_dict(self, contract_guard_module):
        """details should be a dict."""
        result = contract_guard_module.check_contracts()
        assert isinstance(result["details"], dict)

    def test_details_has_medicalization(self, contract_guard_module):
        """details should have medicalization entry."""
        result = contract_guard_module.check_contracts()
        assert "medicalization" in result["details"]

    def test_details_has_normalization(self, contract_guard_module):
        """details should have normalization entry."""
        result = contract_guard_module.check_contracts()
        assert "normalization" in result["details"]


class TestHashMatch:
    """Test case: hash expected == actual → no warnings."""

    def test_no_drift_when_hashes_match(self, contract_guard_module):
        """When actual hash matches expected, no drift should be detected."""
        # Mock both hash getters to return the expected hashes from snapshots
        med_snapshot = contract_guard_module._load_contract_snapshot("medicalization_contract.json")
        norm_snapshot = contract_guard_module._load_contract_snapshot("normalization_contract.json")

        if med_snapshot is None or norm_snapshot is None:
            pytest.skip("Contract snapshots not found")

        med_expected = med_snapshot.get("expectedHash", "")
        norm_expected = norm_snapshot.get("expectedHash", "")

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=med_expected):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=norm_expected):
                result = contract_guard_module.check_contracts()

                assert result["medicalizationDrift"] is False
                assert result["normalizationDrift"] is False
                assert len(result["warnings"]) == 0

    def test_no_warnings_list_when_match(self, contract_guard_module):
        """warnings list should be empty when hashes match."""
        med_snapshot = contract_guard_module._load_contract_snapshot("medicalization_contract.json")
        norm_snapshot = contract_guard_module._load_contract_snapshot("normalization_contract.json")

        if med_snapshot is None or norm_snapshot is None:
            pytest.skip("Contract snapshots not found")

        med_expected = med_snapshot.get("expectedHash", "")
        norm_expected = norm_snapshot.get("expectedHash", "")

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=med_expected):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=norm_expected):
                warnings = contract_guard_module.get_contract_warnings()
                assert warnings == []

    def test_has_drift_false_when_match(self, contract_guard_module):
        """has_drift() should return False when hashes match."""
        med_snapshot = contract_guard_module._load_contract_snapshot("medicalization_contract.json")
        norm_snapshot = contract_guard_module._load_contract_snapshot("normalization_contract.json")

        if med_snapshot is None or norm_snapshot is None:
            pytest.skip("Contract snapshots not found")

        med_expected = med_snapshot.get("expectedHash", "")
        norm_expected = norm_snapshot.get("expectedHash", "")

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=med_expected):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=norm_expected):
                assert contract_guard_module.has_drift() is False

    def test_details_show_match_true(self, contract_guard_module):
        """details.medicalization.match and details.normalization.match should be True."""
        med_snapshot = contract_guard_module._load_contract_snapshot("medicalization_contract.json")
        norm_snapshot = contract_guard_module._load_contract_snapshot("normalization_contract.json")

        if med_snapshot is None or norm_snapshot is None:
            pytest.skip("Contract snapshots not found")

        med_expected = med_snapshot.get("expectedHash", "")
        norm_expected = norm_snapshot.get("expectedHash", "")

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=med_expected):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=norm_expected):
                result = contract_guard_module.check_contracts()

                assert result["details"]["medicalization"]["match"] is True
                assert result["details"]["normalization"]["match"] is True


class TestHashMismatch:
    """Test case: hash modified → warning present."""

    def test_medicalization_drift_when_hash_differs(self, contract_guard_module):
        """When medicalization hash differs, drift should be detected."""
        norm_snapshot = contract_guard_module._load_contract_snapshot("normalization_contract.json")
        if norm_snapshot is None:
            pytest.skip("Normalization contract snapshot not found")

        norm_expected = norm_snapshot.get("expectedHash", "")

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="different_hash_12345"):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=norm_expected):
                result = contract_guard_module.check_contracts()

                assert result["medicalizationDrift"] is True
                assert "medicalization_drift" in result["warnings"]

    def test_normalization_drift_when_hash_differs(self, contract_guard_module):
        """When normalization hash differs, drift should be detected."""
        med_snapshot = contract_guard_module._load_contract_snapshot("medicalization_contract.json")
        if med_snapshot is None:
            pytest.skip("Medicalization contract snapshot not found")

        med_expected = med_snapshot.get("expectedHash", "")

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=med_expected):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value="different_hash_67890"):
                result = contract_guard_module.check_contracts()

                assert result["normalizationDrift"] is True
                assert "normalization_drift" in result["warnings"]

    def test_both_drift_when_both_hashes_differ(self, contract_guard_module):
        """When both hashes differ, both drifts should be detected."""
        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="wrong_med_hash"):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value="wrong_norm_hash"):
                result = contract_guard_module.check_contracts()

                assert result["medicalizationDrift"] is True
                assert result["normalizationDrift"] is True
                assert "medicalization_drift" in result["warnings"]
                assert "normalization_drift" in result["warnings"]

    def test_has_drift_true_when_medicalization_differs(self, contract_guard_module):
        """has_drift() should return True when medicalization hash differs."""
        norm_snapshot = contract_guard_module._load_contract_snapshot("normalization_contract.json")
        norm_expected = norm_snapshot.get("expectedHash", "") if norm_snapshot else ""

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="different_hash"):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=norm_expected):
                assert contract_guard_module.has_drift() is True

    def test_has_drift_true_when_normalization_differs(self, contract_guard_module):
        """has_drift() should return True when normalization hash differs."""
        med_snapshot = contract_guard_module._load_contract_snapshot("medicalization_contract.json")
        med_expected = med_snapshot.get("expectedHash", "") if med_snapshot else ""

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=med_expected):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value="different_hash"):
                assert contract_guard_module.has_drift() is True

    def test_details_show_expected_and_actual(self, contract_guard_module):
        """details should show both expected and actual hashes for debugging."""
        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="actual_med_hash"):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value="actual_norm_hash"):
                result = contract_guard_module.check_contracts()

                med_details = result["details"]["medicalization"]
                assert "expected" in med_details
                assert "actual" in med_details
                assert med_details["actual"] == "actual_med_hash"

                norm_details = result["details"]["normalization"]
                assert "expected" in norm_details
                assert "actual" in norm_details
                assert norm_details["actual"] == "actual_norm_hash"


class TestSnapshotMissing:
    """Test case: snapshot missing → warning soft."""

    def test_medicalization_snapshot_missing_warning(self, contract_guard_module):
        """When medicalization snapshot is missing, soft warning should appear."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            # Return None for medicalization, valid for normalization
            def side_effect(filename):
                if "medicalization" in filename:
                    return None
                return {"expectedHash": "some_hash", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="actual_hash"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="some_hash"):
                    result = contract_guard_module.check_contracts()

                    assert "medicalization_snapshot_missing" in result["warnings"]
                    assert result["details"]["medicalization"]["snapshotMissing"] is True

    def test_normalization_snapshot_missing_warning(self, contract_guard_module):
        """When normalization snapshot is missing, soft warning should appear."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            # Return valid for medicalization, None for normalization
            def side_effect(filename):
                if "normalization" in filename:
                    return None
                return {"expectedHash": "some_hash", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="some_hash"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="actual_hash"):
                    result = contract_guard_module.check_contracts()

                    assert "normalization_snapshot_missing" in result["warnings"]
                    assert result["details"]["normalization"]["snapshotMissing"] is True

    def test_both_snapshots_missing(self, contract_guard_module):
        """When both snapshots are missing, both warnings should appear."""
        with patch.object(contract_guard_module, "_load_contract_snapshot", return_value=None):
            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="actual"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="actual"):
                    result = contract_guard_module.check_contracts()

                    assert "medicalization_snapshot_missing" in result["warnings"]
                    assert "normalization_snapshot_missing" in result["warnings"]

    def test_snapshot_missing_does_not_crash(self, contract_guard_module):
        """Missing snapshots should not cause exceptions."""
        with patch.object(contract_guard_module, "_load_contract_snapshot", return_value=None):
            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=""):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value=""):
                    # Should not raise
                    result = contract_guard_module.check_contracts()
                    assert result is not None


class TestHashUnavailable:
    """Test case: hash unavailable (empty string from getter) → NO drift, just warning."""

    def test_medicalization_hash_unavailable_no_drift(self, contract_guard_module):
        """When medicalization hash is empty, drift should be FALSE (not a real drift)."""
        norm_snapshot = contract_guard_module._load_contract_snapshot("normalization_contract.json")
        norm_expected = norm_snapshot.get("expectedHash", "") if norm_snapshot else "some_hash"

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=""):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=norm_expected):
                result = contract_guard_module.check_contracts()

                # Key change: drift is FALSE when hash unavailable
                assert result["medicalizationDrift"] is False
                assert "medicalization_hash_unavailable" in result["warnings"]

    def test_normalization_hash_unavailable_no_drift(self, contract_guard_module):
        """When normalization hash is empty, drift should be FALSE (not a real drift)."""
        med_snapshot = contract_guard_module._load_contract_snapshot("medicalization_contract.json")
        med_expected = med_snapshot.get("expectedHash", "") if med_snapshot else "some_hash"

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=med_expected):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=""):
                result = contract_guard_module.check_contracts()

                # Key change: drift is FALSE when hash unavailable
                assert result["normalizationDrift"] is False
                assert "normalization_hash_unavailable" in result["warnings"]

    def test_both_hashes_unavailable_no_drift(self, contract_guard_module):
        """When both hashes are empty, no drift should be detected."""
        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=""):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=""):
                result = contract_guard_module.check_contracts()

                assert result["medicalizationDrift"] is False
                assert result["normalizationDrift"] is False
                assert "medicalization_hash_unavailable" in result["warnings"]
                assert "normalization_hash_unavailable" in result["warnings"]


class TestGetContractWarnings:
    """Tests for get_contract_warnings() convenience function."""

    def test_returns_list(self, contract_guard_module):
        """get_contract_warnings should return a list."""
        result = contract_guard_module.get_contract_warnings()
        assert isinstance(result, list)

    def test_empty_list_when_no_drift(self, contract_guard_module):
        """Should return empty list when no drift detected."""
        med_snapshot = contract_guard_module._load_contract_snapshot("medicalization_contract.json")
        norm_snapshot = contract_guard_module._load_contract_snapshot("normalization_contract.json")

        if med_snapshot is None or norm_snapshot is None:
            pytest.skip("Contract snapshots not found")

        med_expected = med_snapshot.get("expectedHash", "")
        norm_expected = norm_snapshot.get("expectedHash", "")

        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value=med_expected):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value=norm_expected):
                warnings = contract_guard_module.get_contract_warnings()
                assert warnings == []

    def test_contains_drift_warnings_when_mismatch(self, contract_guard_module):
        """Should contain drift warnings when hashes mismatch."""
        with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="wrong"):
            with patch.object(contract_guard_module, "_get_normalization_hash", return_value="wrong"):
                warnings = contract_guard_module.get_contract_warnings()
                assert len(warnings) > 0


class TestPHISafety:
    """Tests to ensure PHI safety - no content leakage."""

    def test_no_transcript_content_in_result(self, contract_guard_module):
        """Result should never contain transcript content."""
        result = contract_guard_module.check_contracts()

        # Convert to string and check for common PHI patterns
        result_str = json.dumps(result)

        # Should only contain hashes, bools, and warning strings
        # No patient data, names, diagnoses, etc.
        assert "patient" not in result_str.lower()
        assert "diagnosis" not in result_str.lower()

    def test_details_only_contains_hashes(self, contract_guard_module):
        """details should only contain hash values and flags."""
        result = contract_guard_module.check_contracts()

        for contract_type in ["medicalization", "normalization"]:
            if contract_type in result["details"]:
                details = result["details"][contract_type]
                allowed_keys = {"expected", "actual", "match", "snapshotMissing", "snapshotInvalid"}
                assert set(details.keys()) <= allowed_keys, (
                    f"Unexpected keys in {contract_type} details: {set(details.keys()) - allowed_keys}"
                )


class TestSnapshotInvalid:
    """Test case: snapshot exists but expectedHash missing/empty → NO drift, warning *_snapshot_invalid."""

    def test_medicalization_snapshot_invalid_no_drift(self, contract_guard_module):
        """When medicalization snapshot has no expectedHash, drift should be FALSE."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            def side_effect(filename):
                if "medicalization" in filename:
                    return {"version": "v1"}  # No expectedHash
                return {"expectedHash": "some_hash", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="actual_hash"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="some_hash"):
                    result = contract_guard_module.check_contracts()

                    assert result["medicalizationDrift"] is False
                    assert "medicalization_snapshot_invalid" in result["warnings"]
                    assert result["details"]["medicalization"]["snapshotInvalid"] is True
                    assert result["details"]["medicalization"]["snapshotMissing"] is False

    def test_normalization_snapshot_invalid_no_drift(self, contract_guard_module):
        """When normalization snapshot has no expectedHash, drift should be FALSE."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            def side_effect(filename):
                if "normalization" in filename:
                    return {"version": "v1"}  # No expectedHash
                return {"expectedHash": "some_hash", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="some_hash"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="actual_hash"):
                    result = contract_guard_module.check_contracts()

                    assert result["normalizationDrift"] is False
                    assert "normalization_snapshot_invalid" in result["warnings"]
                    assert result["details"]["normalization"]["snapshotInvalid"] is True
                    assert result["details"]["normalization"]["snapshotMissing"] is False

    def test_snapshot_with_empty_expected_hash_is_invalid(self, contract_guard_module):
        """When expectedHash is empty string, treat as invalid (not drift)."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            def side_effect(filename):
                if "medicalization" in filename:
                    return {"expectedHash": "", "version": "v1"}  # Empty expectedHash
                return {"expectedHash": "some_hash", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="actual_hash"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="some_hash"):
                    result = contract_guard_module.check_contracts()

                    assert result["medicalizationDrift"] is False
                    assert "medicalization_snapshot_invalid" in result["warnings"]
                    assert result["details"]["medicalization"]["expected"] is None

    def test_both_snapshots_invalid(self, contract_guard_module):
        """When both snapshots have no expectedHash, both warnings should appear."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            mock_load.return_value = {"version": "v1"}  # No expectedHash

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="actual"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="actual"):
                    result = contract_guard_module.check_contracts()

                    assert result["medicalizationDrift"] is False
                    assert result["normalizationDrift"] is False
                    assert "medicalization_snapshot_invalid" in result["warnings"]
                    assert "normalization_snapshot_invalid" in result["warnings"]


class TestSnapshotMissingNoDrift:
    """Test case: snapshot missing → NO drift, just warning."""

    def test_medicalization_snapshot_missing_no_drift(self, contract_guard_module):
        """When medicalization snapshot missing, drift should be FALSE."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            def side_effect(filename):
                if "medicalization" in filename:
                    return None
                return {"expectedHash": "some_hash", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="actual_hash"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="some_hash"):
                    result = contract_guard_module.check_contracts()

                    assert result["medicalizationDrift"] is False
                    assert "medicalization_snapshot_missing" in result["warnings"]

    def test_normalization_snapshot_missing_no_drift(self, contract_guard_module):
        """When normalization snapshot missing, drift should be FALSE."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            def side_effect(filename):
                if "normalization" in filename:
                    return None
                return {"expectedHash": "some_hash", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="some_hash"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="actual_hash"):
                    result = contract_guard_module.check_contracts()

                    assert result["normalizationDrift"] is False
                    assert "normalization_snapshot_missing" in result["warnings"]

    def test_both_snapshots_missing_no_drift(self, contract_guard_module):
        """When both snapshots missing, no drift should be detected."""
        with patch.object(contract_guard_module, "_load_contract_snapshot", return_value=None):
            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="actual"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="actual"):
                    result = contract_guard_module.check_contracts()

                    assert result["medicalizationDrift"] is False
                    assert result["normalizationDrift"] is False


class TestRealDrift:
    """Test case: actual != expected (both non-empty) → TRUE drift."""

    def test_medicalization_real_drift(self, contract_guard_module):
        """When actual hash differs from expected, drift should be TRUE."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            def side_effect(filename):
                if "medicalization" in filename:
                    return {"expectedHash": "expected_hash_abc123", "version": "v1"}
                return {"expectedHash": "norm_expected", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="different_hash_xyz"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="norm_expected"):
                    result = contract_guard_module.check_contracts()

                    assert result["medicalizationDrift"] is True
                    assert result["normalizationDrift"] is False
                    assert "medicalization_drift" in result["warnings"]
                    assert result["details"]["medicalization"]["match"] is False

    def test_normalization_real_drift(self, contract_guard_module):
        """When normalization actual hash differs from expected, drift should be TRUE."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            def side_effect(filename):
                if "normalization" in filename:
                    return {"expectedHash": "expected_norm_hash", "version": "v1"}
                return {"expectedHash": "med_expected", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="med_expected"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="different_norm_hash"):
                    result = contract_guard_module.check_contracts()

                    assert result["medicalizationDrift"] is False
                    assert result["normalizationDrift"] is True
                    assert "normalization_drift" in result["warnings"]

    def test_both_real_drift(self, contract_guard_module):
        """When both hashes differ, both drifts should be TRUE."""
        with patch.object(contract_guard_module, "_load_contract_snapshot") as mock_load:
            def side_effect(filename):
                if "medicalization" in filename:
                    return {"expectedHash": "expected_med", "version": "v1"}
                return {"expectedHash": "expected_norm", "version": "v1"}

            mock_load.side_effect = side_effect

            with patch.object(contract_guard_module, "_get_medicalization_hash", return_value="actual_med"):
                with patch.object(contract_guard_module, "_get_normalization_hash", return_value="actual_norm"):
                    result = contract_guard_module.check_contracts()

                    assert result["medicalizationDrift"] is True
                    assert result["normalizationDrift"] is True
                    assert "medicalization_drift" in result["warnings"]
                    assert "normalization_drift" in result["warnings"]


class TestContractSnapshotFiles:
    """Tests for contract snapshot file existence and format."""

    def test_medicalization_contract_exists(self, contracts_dir):
        """medicalization_contract.json should exist."""
        path = contracts_dir / "medicalization_contract.json"
        assert path.exists(), f"Missing: {path}"

    def test_normalization_contract_exists(self, contracts_dir):
        """normalization_contract.json should exist."""
        path = contracts_dir / "normalization_contract.json"
        assert path.exists(), f"Missing: {path}"

    def test_medicalization_contract_valid_json(self, contracts_dir):
        """medicalization_contract.json should be valid JSON."""
        path = contracts_dir / "medicalization_contract.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_normalization_contract_valid_json(self, contracts_dir):
        """normalization_contract.json should be valid JSON."""
        path = contracts_dir / "normalization_contract.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_medicalization_contract_has_expected_hash(self, contracts_dir):
        """medicalization_contract.json should have expectedHash field."""
        path = contracts_dir / "medicalization_contract.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "expectedHash" in data
        assert isinstance(data["expectedHash"], str)
        assert len(data["expectedHash"]) == 64  # SHA256 hex

    def test_normalization_contract_has_expected_hash(self, contracts_dir):
        """normalization_contract.json should have expectedHash field."""
        path = contracts_dir / "normalization_contract.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "expectedHash" in data
        assert isinstance(data["expectedHash"], str)
        assert len(data["expectedHash"]) == 64  # SHA256 hex

    def test_medicalization_contract_has_version(self, contracts_dir):
        """medicalization_contract.json should have version field."""
        path = contracts_dir / "medicalization_contract.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "version" in data
        assert data["version"] == "v1"

    def test_normalization_contract_has_version(self, contracts_dir):
        """normalization_contract.json should have version field."""
        path = contracts_dir / "normalization_contract.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "version" in data
        assert data["version"] == "v1"
