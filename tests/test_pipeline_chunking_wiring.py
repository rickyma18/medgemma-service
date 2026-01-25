"""
Tests for pipeline chunking wiring (ÉPICA 14 integration).

Validates:
1. CHUNKING_ENABLED=False -> chunk_transcript not called with intelligent params
2. CHUNKING_ENABLED=True + long transcript -> chunks processed in order, deterministic
3. Telemetry event emitted with PHI-safe metrics (no text)

PHI-safe: Uses fictitious text, no clinical content.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List

from app.schemas.request import Transcript, TranscriptSegment
from app.schemas.structured_fields_v1 import StructuredFieldsV1


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def short_transcript() -> Transcript:
    """Short transcript that fits in single chunk."""
    return Transcript(
        segments=[
            TranscriptSegment(speaker="doctor", text="Hola.", startMs=0, endMs=1000),
            TranscriptSegment(speaker="patient", text="Buenos dias.", startMs=1000, endMs=2000),
        ],
        durationMs=2000
    )


@pytest.fixture
def long_transcript() -> Transcript:
    """
    Long transcript (~10 min) that should trigger multiple chunks.
    10 segments, 1 minute each, ~50 chars each.
    """
    segments = []
    for i in range(10):
        start = i * 60_000
        end = start + 55_000
        speaker = "doctor" if i % 2 == 0 else "patient"
        text = f"Segmento {i:02d}: contenido de prueba para simular texto clinico largo."
        segments.append(
            TranscriptSegment(speaker=speaker, text=text, startMs=start, endMs=end)
        )
    return Transcript(segments=segments, durationMs=600_000)


@pytest.fixture
def mock_settings_disabled():
    """Settings with chunking disabled."""
    mock = MagicMock()
    mock.chunking_enabled = False
    mock.chunking_hard_token_limit = 2048
    mock.chunking_soft_duration_limit_ms = 180000
    mock.chunking_max_duration_ms = 300000
    mock.chunking_min_segments_per_chunk = 1
    mock.drift_guard_mode = "off"
    mock.drift_guard_cooldown_s = 3600
    mock.openai_compat_base_url = "http://localhost:1234/v1"
    mock.openai_compat_model = "test-model"
    mock.openai_compat_timeout_ms = 30000
    return mock


@pytest.fixture
def mock_settings_enabled():
    """Settings with chunking enabled and aggressive limits."""
    mock = MagicMock()
    mock.chunking_enabled = True
    mock.chunking_hard_token_limit = 100  # Low to force chunking
    mock.chunking_soft_duration_limit_ms = 120000  # 2 min soft
    mock.chunking_max_duration_ms = 180000  # 3 min hard
    mock.chunking_min_segments_per_chunk = 1
    mock.drift_guard_mode = "off"
    mock.drift_guard_cooldown_s = 3600
    mock.openai_compat_base_url = "http://localhost:1234/v1"
    mock.openai_compat_model = "test-model"
    mock.openai_compat_timeout_ms = 30000
    return mock


@pytest.fixture
def mock_structured_result():
    """Minimal valid extraction result."""
    return StructuredFieldsV1(
        motivo_consulta="Consulta de prueba",
        padecimiento_actual="Padecimiento ficticio."
    )


# =============================================================================
# Test: Chunking Disabled
# =============================================================================

class TestChunkingDisabled:
    """Tests when CHUNKING_ENABLED=False."""

    @pytest.mark.asyncio
    async def test_disabled_uses_legacy_chunking(
        self, short_transcript, mock_settings_disabled, mock_structured_result
    ):
        """
        When chunking disabled, chunk_transcript is called with legacy params
        (soft_duration_limit_ms=None).
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_disabled):
            with patch("app.services.chunking.chunk_transcript") as mock_chunk:
                # Return single chunk (the original)
                mock_chunk.return_value = [short_transcript]

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_result, 100, "model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_result

                        with patch("app.contracts.contract_guard.check_contracts") as mock_guard:
                            mock_guard.return_value = {"warnings": [], "details": {}}

                            from app.services.pipeline_orl import run_orl_pipeline
                            fields, metrics = await run_orl_pipeline(short_transcript)

                            # Verify chunk_transcript called with legacy params
                            mock_chunk.assert_called_once()
                            call_kwargs = mock_chunk.call_args.kwargs
                            assert call_kwargs.get("soft_duration_limit_ms") is None

                            # Verify metrics
                            assert metrics["chunkingEnabled"] is False

    @pytest.mark.asyncio
    async def test_disabled_no_telemetry_event(
        self, short_transcript, mock_settings_disabled, mock_structured_result
    ):
        """
        When chunking disabled, no 'chunking_applied' telemetry event is emitted.
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_disabled):
            with patch("app.services.chunking.chunk_transcript") as mock_chunk:
                mock_chunk.return_value = [short_transcript]

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_result, 100, "model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_result

                        with patch("app.contracts.contract_guard.check_contracts") as mock_guard:
                            mock_guard.return_value = {"warnings": [], "details": {}}

                            with patch("app.services.telemetry.emit_event") as mock_emit:
                                from app.services.pipeline_orl import run_orl_pipeline
                                await run_orl_pipeline(short_transcript)

                                # No chunking_applied event should be emitted
                                chunking_calls = [
                                    c for c in mock_emit.call_args_list
                                    if c.kwargs.get("name") == "chunking_applied"
                                    or (c.args and c.args[0] == "chunking_applied")
                                ]
                                assert len(chunking_calls) == 0


# =============================================================================
# Test: Chunking Enabled
# =============================================================================

class TestChunkingEnabled:
    """Tests when CHUNKING_ENABLED=True."""

    @pytest.mark.asyncio
    async def test_enabled_uses_intelligent_chunking(
        self, long_transcript, mock_settings_enabled, mock_structured_result
    ):
        """
        When chunking enabled, chunk_transcript is called with all config params.
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_enabled):
            with patch("app.services.chunking.chunk_transcript") as mock_chunk:
                # Simulate 3 chunks
                chunk1 = Transcript(
                    segments=long_transcript.segments[:3],
                    durationMs=180000
                )
                chunk2 = Transcript(
                    segments=long_transcript.segments[3:6],
                    durationMs=180000
                )
                chunk3 = Transcript(
                    segments=long_transcript.segments[6:],
                    durationMs=240000
                )
                mock_chunk.return_value = [chunk1, chunk2, chunk3]

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_result, 100, "model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_result

                        with patch("app.contracts.contract_guard.check_contracts") as mock_guard:
                            mock_guard.return_value = {"warnings": [], "details": {}}

                            with patch("app.services.telemetry.emit_event"):
                                from app.services.pipeline_orl import run_orl_pipeline
                                fields, metrics = await run_orl_pipeline(long_transcript)

                                # Verify chunk_transcript called with intelligent params
                                mock_chunk.assert_called_once()
                                call_kwargs = mock_chunk.call_args.kwargs

                                assert call_kwargs["hard_token_limit"] == 100
                                assert call_kwargs["soft_duration_limit_ms"] == 120000
                                assert call_kwargs["min_segments_per_chunk"] == 1

                                # Verify metrics
                                assert metrics["chunkingEnabled"] is True
                                assert metrics["chunksCount"] == 3

    @pytest.mark.asyncio
    async def test_enabled_extracts_all_chunks_in_order(
        self, long_transcript, mock_settings_enabled, mock_structured_result
    ):
        """
        When multiple chunks, extractor is called for each in order.
        """
        extraction_order = []

        async def track_extraction(transcript, context):
            # Track which chunk was processed (by segment count as proxy)
            extraction_order.append(len(transcript.segments))
            return (mock_structured_result, 100, "model-v1")

        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_enabled):
            with patch("app.services.chunking.chunk_transcript") as mock_chunk:
                # Create distinct chunks
                chunk1 = Transcript(segments=long_transcript.segments[:2], durationMs=120000)
                chunk2 = Transcript(segments=long_transcript.segments[2:5], durationMs=180000)
                chunk3 = Transcript(segments=long_transcript.segments[5:], durationMs=300000)
                mock_chunk.return_value = [chunk1, chunk2, chunk3]

                with patch("app.services.pipeline_orl.extract_structured_v1", side_effect=track_extraction):
                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_result

                        with patch("app.contracts.contract_guard.check_contracts") as mock_guard:
                            mock_guard.return_value = {"warnings": [], "details": {}}

                            with patch("app.services.telemetry.emit_event"):
                                from app.services.pipeline_orl import run_orl_pipeline
                                await run_orl_pipeline(long_transcript)

                                # Verify extraction order matches chunk order
                                assert extraction_order == [2, 3, 5]

    @pytest.mark.asyncio
    async def test_enabled_aggregates_results_deterministically(
        self, long_transcript, mock_settings_enabled
    ):
        """
        Results from multiple chunks are aggregated deterministically.
        """
        # Different results per chunk
        result1 = StructuredFieldsV1(motivo_consulta="Motivo A")
        result2 = StructuredFieldsV1(motivo_consulta="Motivo B")
        result3 = StructuredFieldsV1(motivo_consulta="Motivo C")

        call_count = [0]

        async def varying_extraction(transcript, context):
            call_count[0] += 1
            if call_count[0] == 1:
                return (result1, 100, "model-v1")
            elif call_count[0] == 2:
                return (result2, 100, "model-v1")
            else:
                return (result3, 100, "model-v1")

        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_enabled):
            with patch("app.services.chunking.chunk_transcript") as mock_chunk:
                chunk1 = Transcript(segments=long_transcript.segments[:3], durationMs=180000)
                chunk2 = Transcript(segments=long_transcript.segments[3:6], durationMs=180000)
                chunk3 = Transcript(segments=long_transcript.segments[6:], durationMs=240000)
                mock_chunk.return_value = [chunk1, chunk2, chunk3]

                with patch("app.services.pipeline_orl.extract_structured_v1", side_effect=varying_extraction):
                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        # Finalize returns the aggregated result unchanged
                        mock_finalize.side_effect = lambda x: x

                        with patch("app.contracts.contract_guard.check_contracts") as mock_guard:
                            mock_guard.return_value = {"warnings": [], "details": {}}

                            with patch("app.services.telemetry.emit_event"):
                                from app.services.pipeline_orl import run_orl_pipeline
                                fields, metrics = await run_orl_pipeline(long_transcript)

                                # Aggregator should combine: "Motivo A | Motivo B | Motivo C"
                                assert "Motivo A" in fields.motivo_consulta
                                assert "Motivo B" in fields.motivo_consulta
                                assert "Motivo C" in fields.motivo_consulta


# =============================================================================
# Test: Telemetry PHI-Safety
# =============================================================================

class TestChunkingTelemetry:
    """Tests for chunking telemetry events."""

    @pytest.mark.asyncio
    async def test_telemetry_emitted_on_multiple_chunks(
        self, long_transcript, mock_settings_enabled, mock_structured_result
    ):
        """
        Telemetry 'chunking_applied' event emitted when multiple chunks created.
        """
        emitted_events = []

        def capture_event(name, payload, cooldown_s=0):
            emitted_events.append({"name": name, "payload": payload})
            return True

        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_enabled):
            with patch("app.services.chunking.chunk_transcript") as mock_chunk:
                chunk1 = Transcript(segments=long_transcript.segments[:5], durationMs=300000)
                chunk2 = Transcript(segments=long_transcript.segments[5:], durationMs=300000)
                mock_chunk.return_value = [chunk1, chunk2]

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_result, 100, "model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_result

                        with patch("app.contracts.contract_guard.check_contracts") as mock_guard:
                            mock_guard.return_value = {"warnings": [], "details": {}}

                            with patch("app.services.telemetry.emit_event", side_effect=capture_event):
                                from app.services.pipeline_orl import run_orl_pipeline
                                await run_orl_pipeline(long_transcript)

                                # Find chunking_applied event
                                chunking_events = [
                                    e for e in emitted_events
                                    if e["name"] == "chunking_applied"
                                ]
                                assert len(chunking_events) == 1

                                payload = chunking_events[0]["payload"]
                                assert payload["numChunks"] == 2
                                assert "totalTokensEst" in payload
                                assert "hardTokenLimit" in payload

    @pytest.mark.asyncio
    async def test_telemetry_payload_phi_safe(
        self, long_transcript, mock_settings_enabled, mock_structured_result
    ):
        """
        Telemetry payload contains NO PHI (no text, segments, transcript).
        """
        captured_payload = [None]

        def capture_event(name, payload, cooldown_s=0):
            if name == "chunking_applied":
                captured_payload[0] = payload
            return True

        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_enabled):
            with patch("app.services.chunking.chunk_transcript") as mock_chunk:
                chunk1 = Transcript(segments=long_transcript.segments[:5], durationMs=300000)
                chunk2 = Transcript(segments=long_transcript.segments[5:], durationMs=300000)
                mock_chunk.return_value = [chunk1, chunk2]

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_result, 100, "model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_result

                        with patch("app.contracts.contract_guard.check_contracts") as mock_guard:
                            mock_guard.return_value = {"warnings": [], "details": {}}

                            with patch("app.services.telemetry.emit_event", side_effect=capture_event):
                                from app.services.pipeline_orl import run_orl_pipeline
                                await run_orl_pipeline(long_transcript)

                                payload = captured_payload[0]
                                assert payload is not None

                                # PHI-forbidden keys must NOT be present
                                phi_keys = {"text", "transcript", "segments", "segment", "patient", "content"}
                                payload_keys_lower = {k.lower() for k in payload.keys()}
                                assert phi_keys.isdisjoint(payload_keys_lower), "PHI key found in payload!"

                                # Only numeric/config values allowed
                                assert isinstance(payload["numChunks"], int)
                                assert isinstance(payload["totalTokensEst"], int)
                                assert isinstance(payload["totalSegments"], int)

    @pytest.mark.asyncio
    async def test_no_telemetry_on_single_chunk(
        self, short_transcript, mock_settings_enabled, mock_structured_result
    ):
        """
        No telemetry event when only 1 chunk (no actual chunking occurred).
        """
        emitted_events = []

        def capture_event(name, payload, cooldown_s=0):
            emitted_events.append(name)
            return True

        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_enabled):
            with patch("app.services.chunking.chunk_transcript") as mock_chunk:
                # Single chunk
                mock_chunk.return_value = [short_transcript]

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_result, 100, "model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_result

                        with patch("app.contracts.contract_guard.check_contracts") as mock_guard:
                            mock_guard.return_value = {"warnings": [], "details": {}}

                            with patch("app.services.telemetry.emit_event", side_effect=capture_event):
                                from app.services.pipeline_orl import run_orl_pipeline
                                await run_orl_pipeline(short_transcript)

                                # No chunking_applied event
                                assert "chunking_applied" not in emitted_events


# =============================================================================
# Test: Single Chunk Behavior (Determinism)
# =============================================================================

class TestSingleChunkDeterminism:
    """Tests that single chunk produces identical behavior to pre-chunking."""

    @pytest.mark.asyncio
    async def test_single_chunk_same_as_no_chunking(
        self, short_transcript, mock_settings_enabled, mock_structured_result
    ):
        """
        With 1 chunk, result is identical to direct extraction (no aggregation noise).
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings_enabled):
            with patch("app.services.chunking.chunk_transcript") as mock_chunk:
                # Return single chunk
                mock_chunk.return_value = [short_transcript]

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_result, 100, "model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_result

                        with patch("app.contracts.contract_guard.check_contracts") as mock_guard:
                            mock_guard.return_value = {"warnings": [], "details": {}}

                            with patch("app.services.telemetry.emit_event"):
                                from app.services.pipeline_orl import run_orl_pipeline
                                fields, metrics = await run_orl_pipeline(short_transcript)

                                # Extraction called exactly once
                                assert mock_extract.call_count == 1

                                # Result is the extraction result (not aggregated/modified)
                                assert fields.motivo_consulta == mock_structured_result.motivo_consulta
