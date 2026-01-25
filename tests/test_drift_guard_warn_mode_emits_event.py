"""
Tests for Drift Guard Warn Mode and Telemetry.

ÉPICA 13.9: Verifies that:
- "warn" mode emits telemetry event on drift
- Rate limiting (cooldown) prevents duplicate events
- PHI safety: payload must not contain transcript/patient data
"""
import json
import time
import pytest
from unittest.mock import patch, MagicMock

from app.schemas.request import Transcript, TranscriptSegment
from app.services.telemetry import (
    emit_event,
    reset_rate_limits,
    get_last_emit_time,
    _sanitize_payload,
    PHI_FORBIDDEN_KEYS,
)


@pytest.fixture(autouse=True)
def reset_telemetry_state():
    """Reset telemetry rate limit state before each test."""
    reset_rate_limits()
    yield
    reset_rate_limits()


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
def mock_settings_warn_mode():
    """Mock settings with drift_guard_mode='warn'."""
    mock = MagicMock()
    mock.drift_guard_mode = "warn"
    mock.drift_guard_cooldown_s = 3600
    mock.openai_compat_base_url = "http://localhost:1234/v1"
    mock.openai_compat_model = "test-model"
    mock.openai_compat_timeout_ms = 30000
    return mock


class TestWarnModeEmitsEvent:
    """Tests for warn mode telemetry emission."""

    @pytest.mark.asyncio
    async def test_warn_mode_emits_event_on_drift(
        self, sample_transcript, mock_settings_warn_mode
    ):
        """
        When DRIFT_GUARD_MODE='warn' and warnings exist,
        should emit telemetry event but NOT force fallback.
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_warn_mode):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                mock_check.return_value = {
                    "medicalizationDrift": True,
                    "normalizationDrift": False,
                    "warnings": ["medicalization_drift"],
                    "details": {
                        "medicalization": {
                            "expected": "expected_hash",
                            "actual": "actual_hash",
                            "match": False
                        }
                    }
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_fields = MagicMock()
                    mock_extract.return_value = (mock_fields, 100, "v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_fields

                        with patch("app.services.telemetry.emit_event") as mock_emit:
                            mock_emit.return_value = True

                            from app.services.pipeline_orl import run_orl_pipeline

                            fields, metrics = await run_orl_pipeline(sample_transcript)

                            # Should have emitted telemetry
                            mock_emit.assert_called_once()
                            call_kwargs = mock_emit.call_args[1]
                            assert call_kwargs["name"] == "contract_drift_detected"

                            # Should NOT have forced fallback (warn mode)
                            assert metrics.get("fallbackReason") != "contract_drift"

    @pytest.mark.asyncio
    async def test_warn_mode_continues_pipeline(
        self, sample_transcript, mock_settings_warn_mode
    ):
        """
        Warn mode should continue pipeline normally after emitting event.
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_warn_mode):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                mock_check.return_value = {
                    "warnings": ["medicalization_drift"],
                    "details": {}
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_fields = MagicMock()
                    mock_extract.return_value = (mock_fields, 100, "v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_fields

                        with patch("app.services.telemetry.emit_event"):
                            from app.services.pipeline_orl import run_orl_pipeline

                            fields, metrics = await run_orl_pipeline(sample_transcript)

                            # Pipeline should complete normally (not fallback_baseline due to drift)
                            # It might be fallback for other reasons, but not contract_drift
                            if metrics["pipelineUsed"] == "fallback_baseline":
                                assert metrics["fallbackReason"] != "contract_drift"


class TestTelemetryRateLimiting:
    """Tests for telemetry rate limiting / cooldown."""

    def test_emit_respects_cooldown(self):
        """Second emit within cooldown should be blocked."""
        # First emit should succeed
        result1 = emit_event(
            name="test_event",
            payload={"test": "data"},
            cooldown_s=3600
        )
        assert result1 is True

        # Second emit within cooldown should be blocked
        result2 = emit_event(
            name="test_event",
            payload={"test": "data2"},
            cooldown_s=3600
        )
        assert result2 is False

    def test_emit_different_events_not_blocked(self):
        """Different event names should have separate cooldowns."""
        result1 = emit_event(
            name="event_a",
            payload={"test": "data"},
            cooldown_s=3600
        )
        assert result1 is True

        result2 = emit_event(
            name="event_b",
            payload={"test": "data"},
            cooldown_s=3600
        )
        assert result2 is True

    def test_emit_no_cooldown_always_emits(self):
        """cooldown_s=0 should always emit."""
        for i in range(5):
            result = emit_event(
                name="no_cooldown_event",
                payload={"iteration": i},
                cooldown_s=0
            )
            assert result is True

    def test_emit_force_ignores_cooldown(self):
        """force=True should ignore cooldown."""
        # First emit
        emit_event(
            name="forced_event",
            payload={"test": 1},
            cooldown_s=3600
        )

        # Forced emit should succeed despite cooldown
        result = emit_event(
            name="forced_event",
            payload={"test": 2},
            cooldown_s=3600,
            force=True
        )
        assert result is True

    def test_last_emit_time_tracked(self):
        """get_last_emit_time should return correct timestamp."""
        event_name = "tracked_event"

        # Before emit
        assert get_last_emit_time(event_name) is None

        before = time.time()
        emit_event(name=event_name, payload={}, cooldown_s=3600)
        after = time.time()

        last_time = get_last_emit_time(event_name)
        assert last_time is not None
        assert before <= last_time <= after

    def test_reset_rate_limits_clears_state(self):
        """reset_rate_limits should clear all tracked events."""
        emit_event(name="event1", payload={}, cooldown_s=3600)
        emit_event(name="event2", payload={}, cooldown_s=3600)

        assert get_last_emit_time("event1") is not None
        assert get_last_emit_time("event2") is not None

        reset_rate_limits()

        assert get_last_emit_time("event1") is None
        assert get_last_emit_time("event2") is None


class TestPHISafety:
    """Tests to ensure telemetry payloads are PHI-safe."""

    def test_sanitize_removes_phi_keys(self):
        """_sanitize_payload should remove PHI-unsafe keys."""
        dirty_payload = {
            "warnings": ["drift"],
            "text": "SHOULD BE REMOVED",
            "transcript": "SHOULD BE REMOVED",
            "segments": ["SHOULD BE REMOVED"],
            "patient": "SHOULD BE REMOVED",
            "details": {
                "hash": "safe_to_keep",
                "content": "SHOULD BE REMOVED"
            }
        }

        clean = _sanitize_payload(dirty_payload)

        assert "text" not in clean
        assert "transcript" not in clean
        assert "segments" not in clean
        assert "patient" not in clean
        assert "warnings" in clean
        assert clean["details"]["hash"] == "safe_to_keep"
        assert "content" not in clean["details"]

    def test_sanitize_handles_nested_dicts(self):
        """Should recursively sanitize nested dicts."""
        payload = {
            "level1": {
                "level2": {
                    "text": "REMOVE",
                    "safe": "keep"
                }
            }
        }

        clean = _sanitize_payload(payload)

        assert "text" not in clean["level1"]["level2"]
        assert clean["level1"]["level2"]["safe"] == "keep"

    def test_sanitize_handles_lists(self):
        """Should sanitize dicts inside lists."""
        payload = {
            "items": [
                {"text": "REMOVE", "id": 1},
                {"transcript": "REMOVE", "id": 2}
            ]
        }

        clean = _sanitize_payload(payload)

        assert "text" not in clean["items"][0]
        assert clean["items"][0]["id"] == 1
        assert "transcript" not in clean["items"][1]
        assert clean["items"][1]["id"] == 2

    def test_all_forbidden_keys_defined(self):
        """PHI_FORBIDDEN_KEYS should contain expected dangerous keys."""
        expected_forbidden = {
            "text", "transcript", "segments", "patient",
            "content", "raw", "segment"
        }
        assert expected_forbidden.issubset(PHI_FORBIDDEN_KEYS)

    def test_emit_event_sanitizes_payload(self):
        """emit_event should sanitize payload before logging."""
        dirty_payload = {
            "warnings": ["drift"],
            "text": "PHI_TEXT_SHOULD_NOT_APPEAR",
            "patient": "PHI_PATIENT_SHOULD_NOT_APPEAR"
        }

        with patch("app.services.telemetry.logger") as mock_logger:
            emit_event(
                name="test_phi_safety",
                payload=dirty_payload,
                cooldown_s=0
            )

            # Check what was logged
            call_args = mock_logger.info.call_args
            logged_payload = call_args[1].get("telemetry_payload", "")

            # PHI keys should not appear in logged payload
            assert "PHI_TEXT_SHOULD_NOT_APPEAR" not in logged_payload
            assert "PHI_PATIENT_SHOULD_NOT_APPEAR" not in logged_payload
            # But safe data should be there
            assert "drift" in logged_payload

    def test_drift_payload_is_phi_safe(self):
        """
        Verify the actual drift payload structure is PHI-safe.
        """
        # This is the structure used in pipeline_orl.py
        drift_payload = {
            "warnings": ["medicalization_drift"],
            "details": {
                "medicalization": {
                    "expected": "hash1",
                    "actual": "hash2",
                    "match": False
                }
            },
            "pipelineUsed": "orl_pipeline_stub",
            "medicalizationVersion": "v1",
            "medicalizationGlossaryHash": "abc123",
            "normalizationVersion": "v1",
            "normalizationRulesHash": "def456",
            "driftGuardMode": "warn",
        }

        clean = _sanitize_payload(drift_payload)

        # All keys should be preserved (none are PHI)
        assert clean == drift_payload

        # Double-check no PHI keys
        payload_str = json.dumps(clean).lower()
        for forbidden in PHI_FORBIDDEN_KEYS:
            assert f'"{forbidden}"' not in payload_str

    def test_payload_with_accidental_phi_is_cleaned(self):
        """
        If someone accidentally adds PHI to drift payload, it should be removed.
        """
        bad_payload = {
            "warnings": ["drift"],
            "transcript": {"segments": [{"text": "patient data"}]},  # Accident!
            "details": {"hash": "safe"}
        }

        clean = _sanitize_payload(bad_payload)

        # PHI should be stripped
        assert "transcript" not in clean
        assert clean["warnings"] == ["drift"]
        assert clean["details"]["hash"] == "safe"
