
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.schemas.request import Transcript, TranscriptSegment
from app.services.pipeline_orl import run_orl_pipeline, PIPELINE_TIMEOUT_S


def _mock_settings_full_mode():
    """Create mock settings that use full extractor mode (legacy behavior)."""
    mock = MagicMock()
    mock.map_extractor_mode = "full"
    mock.include_evidence_in_response = False
    mock.lite_extractor_max_tokens = 512
    mock.evidence_max_snippets_per_chunk = 5
    mock.drift_guard_mode = "off"
    mock.drift_guard_cooldown_s = 3600
    mock.chunking_enabled = False
    mock.chunking_hard_token_limit = 2048
    mock.chunking_soft_duration_limit_ms = 180000
    mock.chunking_max_duration_ms = 300000
    mock.chunking_min_segments_per_chunk = 1
    mock.openai_compat_base_url = "http://localhost:1234/v1"
    mock.openai_compat_model = "test-model"
    mock.openai_compat_timeout_ms = 30000
    return mock


@pytest.mark.asyncio
async def test_pipeline_fallback_on_exception():
    """Verify pipeline falls back to baseline if map stage crashes."""

    # Mock extract_structured_v1 to raise an exception ONLY when called from loop
    # We can detect if it's loop or fallback by arguments or side effects,
    # but simpler to mock the whole extract function and side-effect it logic.

    transcript = Transcript(
        segments=[TranscriptSegment(speaker="doctor", text="Test", startMs=0, endMs=1000)],
        durationMs=1000
    )

    # We want the first call (Attempt 1 inside pipeline) to FAIL
    # And the second call (Fallback) to SUCCEED.

    success_result = (AsyncMock(), 100, "fallback-model")

    # Use full extractor mode to maintain legacy test behavior
    with patch("app.services.pipeline_orl.get_settings", return_value=_mock_settings_full_mode()):
        # side_effect can be an iterable
        with patch("app.services.pipeline_orl.extract_structured_v1", side_effect=[
            Exception("Boom inside loop"), # 1st call triggers fallback
            success_result # 2nd call (fallback) succeeds
        ]) as mock_extract:

            fields, metrics = await run_orl_pipeline(transcript)

            assert metrics["pipelineUsed"] == "fallback_baseline"
            assert "error_Exception" in metrics["fallbackReason"]
            assert mock_extract.call_count == 2
        

@pytest.mark.asyncio
async def test_pipeline_timeout_forced():
    """Verify pipeline timeout triggers fallback."""

    transcript = Transcript(
        segments=[TranscriptSegment(speaker="doctor", text="Test", startMs=0, endMs=1000)],
        durationMs=1000
    )

    # Mock extract to be very slow (slower than PIPELINE_TIMEOUT_S)
    # We need to temporarily shorten PIPELINE_TIMEOUT_S for the test,
    # or mock asyncio.wait_for.
    # Easiest is to monkeypatch the constant in the module import.

    # Use full extractor mode to maintain legacy test behavior
    with patch("app.services.pipeline_orl.get_settings", return_value=_mock_settings_full_mode()):
        with patch("app.services.pipeline_orl.PIPELINE_TIMEOUT_S", 0.05): # 50ms timeout
            async def slow_extract(*args, **kwargs):
                await asyncio.sleep(0.2) # 200ms
                return (AsyncMock(), 100, "slow-model")

            # We need the fallback to be fast though!
            # So we need separate behaviors.
            # But `extract_structured_v1` is utilized by BOTH pipeline and fallback.
            # This makes it tricky if we mock the function itself.

            # Helper to differentiate calls could be inspecting stack or args.
            # Or simpler:
            # run_orl_pipeline calls _run_pipeline_logic -> wait_for -> extract (SLOW)
            # on catch Timeout -> _fallback_to_baseline -> extract (FAST)

            call_counter = 0
            async def dynamic_extract(*args, **kwargs):
                nonlocal call_counter
                call_counter += 1
                if call_counter == 1:
                    # First call (Pipeline): Sleep
                    await asyncio.sleep(0.2)
                # Second call (Fallback): Return instantly
                return (AsyncMock(), 100, "fallback-model")

            with patch("app.services.pipeline_orl.extract_structured_v1", side_effect=dynamic_extract):
                fields, metrics = await run_orl_pipeline(transcript)

                assert metrics["pipelineUsed"] == "fallback_baseline"
                assert metrics["fallbackReason"] == "timeout_pipeline"
                assert call_counter == 2
