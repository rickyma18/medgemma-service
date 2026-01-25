"""
Tests for Drift Guard Safe Mode.

ÉPICA 13.9: Verifies that when DRIFT_GUARD_MODE="safe" and contract warnings exist,
the pipeline forces fallback_baseline instead of continuing with potentially drifted contracts.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.schemas.request import Transcript, TranscriptSegment


@pytest.fixture
def sample_transcript():
    """Create a minimal transcript for testing."""
    return Transcript(
        segments=[
            TranscriptSegment(
                speaker="doctor",
                text="El paciente presenta dolor de cabeza.",
                startMs=0,
                endMs=5000
            )
        ],
        durationMs=5000
    )


@pytest.fixture
def mock_settings_safe_mode():
    """Mock settings with drift_guard_mode='safe'."""
    mock = MagicMock()
    mock.drift_guard_mode = "safe"
    mock.drift_guard_cooldown_s = 3600
    mock.openai_compat_base_url = "http://localhost:1234/v1"
    mock.openai_compat_model = "test-model"
    mock.openai_compat_timeout_ms = 30000
    return mock


class TestDriftGuardSafeMode:
    """Tests for drift guard safe mode forcing fallback."""

    @pytest.mark.asyncio
    async def test_safe_mode_forces_fallback_on_drift(
        self, sample_transcript, mock_settings_safe_mode
    ):
        """
        When DRIFT_GUARD_MODE='safe' and warnings exist,
        pipeline should force fallback_baseline.
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_safe_mode):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                # Simulate drift detected
                mock_check.return_value = {
                    "medicalizationDrift": True,
                    "normalizationDrift": False,
                    "warnings": ["DRIFT:medicalization_drift"],
                    "details": {
                        "medicalization": {
                            "expected": "hash1",
                            "actual": "hash2",
                            "match": False,
                            "snapshotMissing": False,
                            "snapshotInvalid": False
                        }
                    }
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    # Mock the fallback extraction
                    mock_fields = MagicMock()
                    mock_extract.return_value = (mock_fields, 100, "test-model-v1")

                    with patch("app.services.telemetry.emit_event") as mock_emit:
                        from app.services.pipeline_orl import run_orl_pipeline

                        fields, metrics = await run_orl_pipeline(sample_transcript)

                        # Should have used fallback
                        assert metrics["pipelineUsed"] == "fallback_baseline"
                        assert metrics["fallbackReason"] == "contract_drift"

                        # Should have emitted telemetry event
                        mock_emit.assert_called_once()
                        call_args = mock_emit.call_args
                        assert call_args[1]["name"] == "contract_drift_detected"

    @pytest.mark.asyncio
    async def test_safe_mode_no_fallback_without_drift(
        self, sample_transcript, mock_settings_safe_mode
    ):
        """
        When DRIFT_GUARD_MODE='safe' but no warnings,
        pipeline should continue normally (not force fallback).
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_safe_mode):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                # No drift
                mock_check.return_value = {
                    "medicalizationDrift": False,
                    "normalizationDrift": False,
                    "warnings": [],
                    "details": {}
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_fields = MagicMock()
                    mock_extract.return_value = (mock_fields, 100, "test-model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_fields

                        with patch("app.services.telemetry.emit_event") as mock_emit:
                            from app.services.pipeline_orl import run_orl_pipeline

                            fields, metrics = await run_orl_pipeline(sample_transcript)

                            # Should NOT have forced fallback (fallbackReason should not be contract_drift)
                            assert metrics.get("fallbackReason") != "contract_drift"

                            # Should NOT have emitted telemetry (no warnings)
                            mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_safe_mode_fallback_reason_is_contract_drift(
        self, sample_transcript, mock_settings_safe_mode
    ):
        """
        Verify that fallbackReason is specifically 'contract_drift' when safe mode triggers.
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_safe_mode):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                mock_check.return_value = {
                    "medicalizationDrift": True,
                    "normalizationDrift": True,
                    "warnings": ["DRIFT:medicalization_drift", "DRIFT:normalization_drift"],
                    "details": {}
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_fields = MagicMock()
                    mock_extract.return_value = (mock_fields, 100, "v1")

                    with patch("app.services.telemetry.emit_event"):
                        from app.services.pipeline_orl import run_orl_pipeline

                        fields, metrics = await run_orl_pipeline(sample_transcript)

                        assert metrics["fallbackReason"] == "contract_drift"
                        assert metrics["pipelineUsed"] == "fallback_baseline"

    @pytest.mark.asyncio
    async def test_safe_mode_preserves_contract_warnings_in_metrics(
        self, sample_transcript, mock_settings_safe_mode
    ):
        """
        Even when forcing fallback, contract warnings should be in metrics.
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_safe_mode):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                expected_warnings = ["DRIFT:medicalization_drift", "DRIFT:normalization_drift"]
                mock_check.return_value = {
                    "medicalizationDrift": True,
                    "normalizationDrift": True,
                    "warnings": expected_warnings,
                    "details": {"test": "data"}
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_fields = MagicMock()
                    mock_extract.return_value = (mock_fields, 100, "v1")

                    with patch("app.services.telemetry.emit_event"):
                        from app.services.pipeline_orl import run_orl_pipeline

                        fields, metrics = await run_orl_pipeline(sample_transcript)

                        assert metrics["contractWarnings"] == expected_warnings
                        assert metrics["contractDetails"] == {"test": "data"}


class TestDriftGuardOffMode:
    """Tests for drift guard off mode."""

    @pytest.fixture
    def mock_settings_off_mode(self):
        """Mock settings with drift_guard_mode='off'."""
        mock = MagicMock()
        mock.drift_guard_mode = "off"
        mock.drift_guard_cooldown_s = 3600
        mock.openai_compat_base_url = "http://localhost:1234/v1"
        mock.openai_compat_model = "test-model"
        mock.openai_compat_timeout_ms = 30000
        return mock

    @pytest.mark.asyncio
    async def test_off_mode_no_fallback_on_drift(
        self, sample_transcript, mock_settings_off_mode
    ):
        """
        When DRIFT_GUARD_MODE='off', drift should not trigger fallback or telemetry.
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_off_mode):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                # Simulate drift
                mock_check.return_value = {
                    "medicalizationDrift": True,
                    "normalizationDrift": False,
                    "warnings": ["DRIFT:medicalization_drift"],
                    "details": {}
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_fields = MagicMock()
                    mock_extract.return_value = (mock_fields, 100, "v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_fields

                        with patch("app.services.telemetry.emit_event") as mock_emit:
                            from app.services.pipeline_orl import run_orl_pipeline

                            fields, metrics = await run_orl_pipeline(sample_transcript)

                            # Should NOT have forced fallback (off mode)
                            assert metrics.get("fallbackReason") != "contract_drift"

                            # Should NOT have emitted telemetry (off mode)
                            mock_emit.assert_not_called()

                            # But warnings should still be in metrics
                            assert metrics["contractWarnings"] == ["DRIFT:medicalization_drift"]
