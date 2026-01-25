
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.pipeline_orl import run_orl_pipeline, _finalize_refine_fields
from app.schemas.request import Transcript, TranscriptSegment
from app.schemas.structured_fields_v1 import StructuredFieldsV1
import httpx


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
async def test_finalize_called_once():
    """Verify finalize stage is called and uses the aggregated input."""
    transcript = Transcript(
        segments=[TranscriptSegment(speaker="doctor", text="Test", startMs=0, endMs=1000)],
        durationMs=1000
    )

    # Mock extract to return valid fields so map/reduce works
    mock_map_result = (StructuredFieldsV1(motivoConsulta="Raw"), 10, "mock-v1")

    # Mock finalize internal call to httpx, or mock _finalize_refine_fields directly.
    # Mocking the helper is cleaner for unit test of orchestrator.

    refined_fields = StructuredFieldsV1(motivoConsulta="Refined")

    # Use full extractor mode to maintain legacy test behavior
    with patch("app.services.pipeline_orl.get_settings", return_value=_mock_settings_full_mode()):
        with patch("app.services.pipeline_orl.extract_structured_v1", return_value=mock_map_result):
            with patch("app.services.pipeline_orl._finalize_refine_fields", new_callable=AsyncMock) as mock_finalize:
                mock_finalize.return_value = refined_fields

                final_result, metrics = await run_orl_pipeline(transcript)

                assert final_result.motivo_consulta == "Refined"
                assert mock_finalize.call_count == 1
                assert "finalize" in metrics["stageMs"]

@pytest.mark.asyncio
async def test_finalize_fallback_on_invalid_json():
    """Verify fallback to aggregated fields if finalize fails."""
    transcript = Transcript(
        segments=[TranscriptSegment(speaker="doctor", text="Test", startMs=0, endMs=1000)],
        durationMs=1000
    )
    mock_map_result = (StructuredFieldsV1(motivoConsulta="Raw Aggregated"), 10, "mock-v1")

    # Use full extractor mode to maintain legacy test behavior
    with patch("app.services.pipeline_orl.get_settings", return_value=_mock_settings_full_mode()):
        with patch("app.services.pipeline_orl.extract_structured_v1", return_value=mock_map_result):
            # Mock finalize to raise exception (e.g. ModelError from parser)
            with patch("app.services.pipeline_orl._finalize_refine_fields", side_effect=ValueError("Bad JSON")):

                final_result, metrics = await run_orl_pipeline(transcript)

                # Should return the aggregated version, not crash
                assert final_result.motivo_consulta == "Raw Aggregated"
                # Metadata should reflect failure
                assert "finalize_failed" in metrics["fallbackReason"]
                assert metrics["stageMs"]["finalize"] >= 0


@pytest.mark.asyncio
async def test_finalize_timeout_fallback():
    """Verify fallback on timeout during finalize."""
    transcript = Transcript(
        segments=[TranscriptSegment(speaker="doctor", text="Test", startMs=0, endMs=1000)],
        durationMs=1000
    )
    mock_map_result = (StructuredFieldsV1(motivoConsulta="Raw"), 10, "mock-v1")

    # Use full extractor mode to maintain legacy test behavior
    with patch("app.services.pipeline_orl.get_settings", return_value=_mock_settings_full_mode()):
        with patch("app.services.pipeline_orl.extract_structured_v1", return_value=mock_map_result):
            # Mock finalize to sleep forever
            async def slow_finalize(*args):
                await asyncio.sleep(2)
                return StructuredFieldsV1(motivoConsulta="Never")

            # Patch FINALIZE_TIMEOUT_S to be tiny
            with patch("app.services.pipeline_orl.FINALIZE_TIMEOUT_S", 0.1):
                with patch("app.services.pipeline_orl._finalize_refine_fields", side_effect=slow_finalize):
                    final_result, metrics = await run_orl_pipeline(transcript)

                    assert final_result.motivo_consulta == "Raw"
                    assert "finalize_failed" in metrics["fallbackReason"]
