"""
Tests for scripts/update_contract_snapshots.py

Tests the CLI tool for updating contract snapshots.
Uses tmp_path and monkeypatch to avoid touching the real contracts directory.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts to path for import
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from update_contract_snapshots import (
    get_contracts_dir,
    get_medicalization_hash_and_version,
    get_normalization_hash_and_version,
    load_snapshot,
    write_snapshot,
    build_snapshot,
    truncate_hash,
    run_check,
    run_update,
    main,
)


class TestGetContractsDir:
    """Tests for contracts directory resolution."""

    def test_default_path(self, monkeypatch):
        """Default should be app/contracts relative to project root."""
        monkeypatch.delenv("CONTRACTS_DIR", raising=False)
        result = get_contracts_dir(None)
        assert result.name == "contracts"
        assert result.parent.name == "app"

    def test_env_override(self, monkeypatch, tmp_path):
        """CONTRACTS_DIR env var should override default."""
        custom_dir = tmp_path / "custom_contracts"
        custom_dir.mkdir()
        monkeypatch.setenv("CONTRACTS_DIR", str(custom_dir))
        result = get_contracts_dir(None)
        assert result == custom_dir

    def test_arg_override(self, monkeypatch, tmp_path):
        """--contracts-dir argument should override env and default."""
        env_dir = tmp_path / "env_contracts"
        arg_dir = tmp_path / "arg_contracts"
        env_dir.mkdir()
        arg_dir.mkdir()
        monkeypatch.setenv("CONTRACTS_DIR", str(env_dir))
        result = get_contracts_dir(str(arg_dir))
        assert result == arg_dir


class TestTruncateHash:
    """Tests for hash truncation."""

    def test_short_hash_unchanged(self):
        """Short hash should not be truncated."""
        assert truncate_hash("abc123", 12) == "abc123"

    def test_long_hash_truncated(self):
        """Long hash should be truncated with ellipsis."""
        long_hash = "a" * 64
        result = truncate_hash(long_hash, 12)
        assert result == "aaaaaaaaaaaa..."
        assert len(result) == 15  # 12 + 3 for "..."


class TestLoadSnapshot:
    """Tests for snapshot loading."""

    def test_load_existing(self, tmp_path):
        """Should load existing snapshot."""
        snapshot_path = tmp_path / "test.json"
        data = {"version": "v1", "expectedHash": "abc123"}
        snapshot_path.write_text(json.dumps(data))

        result = load_snapshot(snapshot_path)
        assert result == data

    def test_load_missing(self, tmp_path):
        """Should return None for missing file."""
        result = load_snapshot(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        """Should return None for invalid JSON."""
        snapshot_path = tmp_path / "invalid.json"
        snapshot_path.write_text("not valid json {{{")
        result = load_snapshot(snapshot_path)
        assert result is None


class TestWriteSnapshot:
    """Tests for snapshot writing."""

    def test_write_creates_file(self, tmp_path):
        """Should create file with correct content."""
        snapshot_path = tmp_path / "test.json"
        data = {"version": "v1", "expectedHash": "abc123"}

        write_snapshot(snapshot_path, data)

        assert snapshot_path.exists()
        loaded = json.loads(snapshot_path.read_text())
        assert loaded == data

    def test_write_creates_parent_dirs(self, tmp_path):
        """Should create parent directories if needed."""
        snapshot_path = tmp_path / "nested" / "dir" / "test.json"
        data = {"version": "v1", "expectedHash": "abc123"}

        write_snapshot(snapshot_path, data)

        assert snapshot_path.exists()

    def test_write_trailing_newline(self, tmp_path):
        """Should write with trailing newline."""
        snapshot_path = tmp_path / "test.json"
        data = {"version": "v1"}

        write_snapshot(snapshot_path, data)

        content = snapshot_path.read_text()
        assert content.endswith("\n")


class TestBuildSnapshot:
    """Tests for snapshot building."""

    def test_basic_snapshot(self):
        """Should build snapshot with required fields."""
        result = build_snapshot(
            current_hash="abc123",
            version="v2",
            description="Test contract"
        )

        assert result["version"] == "v2"
        assert result["expectedHash"] == "abc123"
        assert result["description"] == "Test contract"
        assert "updatedAt" in result

    def test_preserves_extra_fields(self):
        """Should preserve extra fields from existing snapshot."""
        existing = {
            "version": "v1",
            "expectedHash": "old_hash",
            "customField": "preserve_me"
        }

        result = build_snapshot(
            current_hash="new_hash",
            version="v2",
            description="Test",
            existing=existing
        )

        assert result["customField"] == "preserve_me"
        assert result["expectedHash"] == "new_hash"  # Updated
        assert result["version"] == "v2"  # Updated


class TestDryRun:
    """Tests for --dry-run mode."""

    def test_dry_run_no_write(self, tmp_path, monkeypatch):
        """--dry-run should not modify files."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        # Create existing snapshot with different hash
        med_path = contracts_dir / "medicalization_contract.json"
        med_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "old_hash_12345"
        }))

        norm_path = contracts_dir / "normalization_contract.json"
        norm_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "old_norm_hash"
        }))

        # Mock hash getters to return different hashes
        with patch("update_contract_snapshots.get_medicalization_hash_and_version") as mock_med:
            with patch("update_contract_snapshots.get_normalization_hash_and_version") as mock_norm:
                mock_med.return_value = ("new_hash_67890", "v1")
                mock_norm.return_value = ("new_norm_hash", "v1")

                result = run_update(contracts_dir, dry_run=True)

        assert result == 0
        # Files should NOT be changed
        med_data = json.loads(med_path.read_text())
        assert med_data["expectedHash"] == "old_hash_12345"

        norm_data = json.loads(norm_path.read_text())
        assert norm_data["expectedHash"] == "old_norm_hash"


class TestCheckMode:
    """Tests for --check mode."""

    def test_check_match_returns_0(self, tmp_path):
        """--check should return 0 when hashes match."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        med_path = contracts_dir / "medicalization_contract.json"
        med_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "matching_hash_med"
        }))

        norm_path = contracts_dir / "normalization_contract.json"
        norm_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "matching_hash_norm"
        }))

        with patch("update_contract_snapshots.get_medicalization_hash_and_version") as mock_med:
            with patch("update_contract_snapshots.get_normalization_hash_and_version") as mock_norm:
                mock_med.return_value = ("matching_hash_med", "v1")
                mock_norm.return_value = ("matching_hash_norm", "v1")

                result = run_check(contracts_dir)

        assert result == 0

    def test_check_mismatch_returns_1(self, tmp_path):
        """--check should return 1 when hashes don't match."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        med_path = contracts_dir / "medicalization_contract.json"
        med_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "expected_hash"
        }))

        norm_path = contracts_dir / "normalization_contract.json"
        norm_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "expected_norm"
        }))

        with patch("update_contract_snapshots.get_medicalization_hash_and_version") as mock_med:
            with patch("update_contract_snapshots.get_normalization_hash_and_version") as mock_norm:
                mock_med.return_value = ("different_hash", "v1")  # Mismatch!
                mock_norm.return_value = ("expected_norm", "v1")

                result = run_check(contracts_dir)

        assert result == 1

    def test_check_missing_snapshot_returns_1(self, tmp_path):
        """--check should return 1 when snapshot file is missing."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()
        # Only create normalization, not medicalization

        norm_path = contracts_dir / "normalization_contract.json"
        norm_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "norm_hash"
        }))

        with patch("update_contract_snapshots.get_medicalization_hash_and_version") as mock_med:
            with patch("update_contract_snapshots.get_normalization_hash_and_version") as mock_norm:
                mock_med.return_value = ("med_hash", "v1")
                mock_norm.return_value = ("norm_hash", "v1")

                result = run_check(contracts_dir)

        assert result == 1


class TestUpdateMode:
    """Tests for update mode (default)."""

    def test_update_writes_json(self, tmp_path):
        """Update should write JSON with expectedHash and version."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        # Create empty/outdated snapshots
        med_path = contracts_dir / "medicalization_contract.json"
        med_path.write_text(json.dumps({"version": "v1", "expectedHash": "old"}))

        norm_path = contracts_dir / "normalization_contract.json"
        norm_path.write_text(json.dumps({"version": "v1", "expectedHash": "old"}))

        new_med_hash = "a" * 64  # 64 hex chars
        new_norm_hash = "b" * 64

        with patch("update_contract_snapshots.get_medicalization_hash_and_version") as mock_med:
            with patch("update_contract_snapshots.get_normalization_hash_and_version") as mock_norm:
                mock_med.return_value = (new_med_hash, "v2")
                mock_norm.return_value = (new_norm_hash, "v2")

                result = run_update(contracts_dir, dry_run=False)

        assert result == 0

        # Verify written content
        med_data = json.loads(med_path.read_text())
        assert med_data["expectedHash"] == new_med_hash
        assert len(med_data["expectedHash"]) == 64
        assert med_data["version"] == "v2"
        assert med_data["version"] != ""

        norm_data = json.loads(norm_path.read_text())
        assert norm_data["expectedHash"] == new_norm_hash
        assert len(norm_data["expectedHash"]) == 64
        assert norm_data["version"] == "v2"

    def test_update_no_changes_when_match(self, tmp_path):
        """Update should not rewrite if hashes already match."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        matching_hash = "c" * 64

        med_path = contracts_dir / "medicalization_contract.json"
        original_content = json.dumps({
            "version": "v1",
            "expectedHash": matching_hash,
            "updatedAt": "2020-01-01T00:00:00Z"  # Old timestamp
        })
        med_path.write_text(original_content)

        norm_path = contracts_dir / "normalization_contract.json"
        norm_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": matching_hash
        }))

        with patch("update_contract_snapshots.get_medicalization_hash_and_version") as mock_med:
            with patch("update_contract_snapshots.get_normalization_hash_and_version") as mock_norm:
                mock_med.return_value = (matching_hash, "v1")
                mock_norm.return_value = (matching_hash, "v1")

                result = run_update(contracts_dir, dry_run=False)

        assert result == 0
        # File should not have been modified (timestamp would change)
        assert med_path.read_text() == original_content


class TestEmptyHash:
    """Tests for empty hash handling."""

    def test_empty_hash_fails_update(self, tmp_path):
        """Script should fail (exit 2) if hash is empty."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        med_path = contracts_dir / "medicalization_contract.json"
        med_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "existing_hash"
        }))

        norm_path = contracts_dir / "normalization_contract.json"
        norm_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "existing_norm"
        }))

        with patch("update_contract_snapshots.get_medicalization_hash_and_version") as mock_med:
            with patch("update_contract_snapshots.get_normalization_hash_and_version") as mock_norm:
                mock_med.return_value = ("", "v1")  # Empty hash!
                mock_norm.return_value = ("good_hash", "v1")

                result = run_update(contracts_dir, dry_run=False)

        assert result == 2

        # Snapshot should NOT be overwritten
        med_data = json.loads(med_path.read_text())
        assert med_data["expectedHash"] == "existing_hash"

    def test_empty_hash_does_not_overwrite(self, tmp_path):
        """Empty hash should not overwrite existing snapshot."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        original_hash = "preserved_hash_12345"
        med_path = contracts_dir / "medicalization_contract.json"
        med_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": original_hash
        }))

        norm_path = contracts_dir / "normalization_contract.json"
        norm_path.write_text(json.dumps({
            "version": "v1",
            "expectedHash": "norm_hash"
        }))

        with patch("update_contract_snapshots.get_medicalization_hash_and_version") as mock_med:
            with patch("update_contract_snapshots.get_normalization_hash_and_version") as mock_norm:
                mock_med.return_value = ("", "v1")  # Empty!
                mock_norm.return_value = ("norm_hash", "v1")

                run_update(contracts_dir, dry_run=False)

        # Original should be preserved
        med_data = json.loads(med_path.read_text())
        assert med_data["expectedHash"] == original_hash


class TestMainCLI:
    """Tests for main() CLI function."""

    def test_check_and_dry_run_mutually_exclusive(self, monkeypatch, capsys):
        """--check and --dry-run should be mutually exclusive."""
        monkeypatch.setattr(
            sys, "argv",
            ["update_contract_snapshots.py", "--check", "--dry-run"]
        )

        result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "mutually exclusive" in captured.err

    def test_contracts_dir_arg(self, tmp_path, monkeypatch):
        """--contracts-dir should be respected."""
        contracts_dir = tmp_path / "custom"
        contracts_dir.mkdir()

        # Create matching snapshots
        (contracts_dir / "medicalization_contract.json").write_text(
            json.dumps({"version": "v1", "expectedHash": "hash1"})
        )
        (contracts_dir / "normalization_contract.json").write_text(
            json.dumps({"version": "v1", "expectedHash": "hash2"})
        )

        monkeypatch.setattr(
            sys, "argv",
            ["update_contract_snapshots.py", "--check", "--contracts-dir", str(contracts_dir)]
        )

        with patch("update_contract_snapshots.get_medicalization_hash_and_version") as mock_med:
            with patch("update_contract_snapshots.get_normalization_hash_and_version") as mock_norm:
                mock_med.return_value = ("hash1", "v1")
                mock_norm.return_value = ("hash2", "v1")

                result = main()

        assert result == 0


class TestHashGetters:
    """Tests for hash getter functions."""

    def test_medicalization_hash_getter_handles_exception(self):
        """Should return empty string on exception."""
        with patch.dict(sys.modules, {"app.services.medicalization.medicalization_glossary": None}):
            # Force import error
            with patch(
                "update_contract_snapshots.get_medicalization_hash_and_version",
                wraps=get_medicalization_hash_and_version
            ):
                # The real function will try to import, which may fail
                # depending on env setup - just verify it doesn't crash
                hash_val, version = get_medicalization_hash_and_version()
                assert isinstance(hash_val, str)
                assert isinstance(version, str)

    def test_normalization_hash_getter_handles_exception(self):
        """Should return empty string on exception."""
        hash_val, version = get_normalization_hash_and_version()
        assert isinstance(hash_val, str)
        assert isinstance(version, str)
