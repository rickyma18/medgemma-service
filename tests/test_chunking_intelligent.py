"""
Tests for intelligent chunking module.

ÉPICA 14: Chunking Inteligente
- Token estimation (heuristic)
- Hard token limit + soft duration limit
- No segment splitting (boundary respect)
- Edge cases handling

PHI-safe: Uses fictitious text only.
"""
import pytest
from typing import List

from app.schemas.request import Transcript, TranscriptSegment
from app.services.chunking import (
    chunk_transcript,
    estimate_tokens,
    estimate_segment_tokens,
    validate_chunks_integrity,
    get_chunk_stats,
    DEFAULT_MAX_CHUNK_DURATION_MS,
    DEFAULT_HARD_TOKEN_LIMIT,
    CHARS_PER_TOKEN_ESTIMATE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def minimal_transcript() -> Transcript:
    """Single segment transcript."""
    return Transcript(
        segments=[
            TranscriptSegment(
                speaker="doctor",
                text="Hola, buenos dias.",
                startMs=0,
                endMs=2000
            )
        ],
        durationMs=2000
    )


@pytest.fixture
def short_transcript() -> Transcript:
    """Short transcript that should not be chunked."""
    return Transcript(
        segments=[
            TranscriptSegment(speaker="doctor", text="Hola", startMs=0, endMs=1000),
            TranscriptSegment(speaker="patient", text="Hola doctor", startMs=1000, endMs=2000),
            TranscriptSegment(speaker="doctor", text="Como se siente?", startMs=2000, endMs=3000),
        ],
        durationMs=3000
    )


@pytest.fixture
def fifteen_minute_transcript() -> Transcript:
    """
    ~15 minute transcript with segments every minute.
    Each segment has ~50 chars (~13 tokens).
    """
    segments = []
    for i in range(15):
        start = i * 60_000
        end = start + 50_000
        speaker = "doctor" if i % 2 == 0 else "patient"
        text = f"Segmento numero {i:02d} con contenido de prueba para simular texto clinico."
        segments.append(
            TranscriptSegment(speaker=speaker, text=text, startMs=start, endMs=end)
        )

    return Transcript(
        segments=segments,
        durationMs=15 * 60_000
    )


@pytest.fixture
def high_token_transcript() -> Transcript:
    """
    Transcript with many tokens (long text per segment).
    Each segment ~200 chars = ~50 tokens.
    10 segments = ~500 tokens (without speaker overhead).
    """
    segments = []
    base_text = "Este es un texto extenso para simular contenido clinico. " * 4  # ~200 chars

    for i in range(10):
        start = i * 30_000
        end = start + 25_000
        segments.append(
            TranscriptSegment(
                speaker="doctor" if i % 2 == 0 else "patient",
                text=f"{base_text} Segmento {i}.",
                startMs=start,
                endMs=end
            )
        )

    return Transcript(segments=segments, durationMs=300_000)


@pytest.fixture
def oversized_segment_transcript() -> Transcript:
    """
    Transcript with one segment that exceeds token limit by itself.
    ~5000 chars = ~1250 tokens (over 1024 limit if used).
    """
    giant_text = "Texto muy largo. " * 300  # ~5100 chars

    return Transcript(
        segments=[
            TranscriptSegment(speaker="doctor", text="Inicio.", startMs=0, endMs=1000),
            TranscriptSegment(speaker="doctor", text=giant_text, startMs=1000, endMs=60000),
            TranscriptSegment(speaker="patient", text="Entendido.", startMs=60000, endMs=62000),
        ],
        durationMs=62000
    )


# =============================================================================
# Token Estimation Tests
# =============================================================================

class TestTokenEstimation:
    """Tests for token estimation heuristics."""

    def test_empty_text_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_short_text_returns_minimum_one(self):
        # "Hi" = 2 chars -> ceil(2/4) = 1
        assert estimate_tokens("Hi") >= 1

    def test_known_length_estimation(self):
        # 100 chars -> ceil(100/4) = 25 tokens
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_spanish_medical_text(self):
        # Realistic Spanish medical text
        text = "El paciente presenta cefalea tensional de tres dias de evolucion."
        # ~67 chars -> ceil(67/4) = 17 tokens
        result = estimate_tokens(text)
        assert 15 <= result <= 20  # Reasonable range

    def test_segment_includes_speaker_overhead(self):
        segment = TranscriptSegment(
            speaker="doctor",
            text="Hola",  # 4 chars -> 1 token
            startMs=0,
            endMs=1000
        )
        # 1 text token + 2 speaker overhead = 3
        assert estimate_segment_tokens(segment) == 3


# =============================================================================
# Integrity Tests (No Loss/Duplication)
# =============================================================================

class TestChunkingIntegrity:
    """Tests ensuring no segment loss or duplication."""

    def test_all_segments_preserved_short(self, short_transcript):
        """Short transcript: all segments preserved."""
        chunks = chunk_transcript(short_transcript)
        assert validate_chunks_integrity(short_transcript, chunks)

    def test_all_segments_preserved_long(self, fifteen_minute_transcript):
        """Long transcript: all segments preserved after chunking."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,
            hard_token_limit=500,
            soft_duration_limit_ms=180_000
        )

        assert validate_chunks_integrity(fifteen_minute_transcript, chunks)

    def test_segment_order_preserved(self, fifteen_minute_transcript):
        """Segments maintain original order across chunks."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,
            soft_duration_limit_ms=180_000
        )

        all_texts = []
        for chunk in chunks:
            all_texts.extend(seg.text for seg in chunk.segments)

        original_texts = [seg.text for seg in fifteen_minute_transcript.segments]
        assert all_texts == original_texts

    def test_no_empty_chunks_produced(self, fifteen_minute_transcript):
        """No empty chunks in result."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            hard_token_limit=100  # Force many chunks
        )

        for chunk in chunks:
            assert len(chunk.segments) > 0


# =============================================================================
# Hard Token Limit Tests
# =============================================================================

class TestHardTokenLimit:
    """Tests for hard token limit enforcement."""

    def test_respects_hard_token_limit(self, high_token_transcript):
        """Each chunk should respect hard token limit."""
        hard_limit = 200

        chunks = chunk_transcript(
            high_token_transcript,
            hard_token_limit=hard_limit,
            soft_duration_limit_ms=None  # Disable soft limit
        )

        for chunk in chunks:
            chunk_tokens = sum(estimate_segment_tokens(s) for s in chunk.segments)
            # Allow single oversized segment (documented behavior)
            if len(chunk.segments) > 1:
                assert chunk_tokens <= hard_limit + 100  # Small tolerance for boundary

    def test_small_token_limit_creates_more_chunks(self, high_token_transcript):
        """Smaller token limit = more chunks."""
        chunks_large = chunk_transcript(
            high_token_transcript,
            hard_token_limit=2000,
            soft_duration_limit_ms=None
        )
        chunks_small = chunk_transcript(
            high_token_transcript,
            hard_token_limit=200,
            soft_duration_limit_ms=None
        )

        assert len(chunks_small) > len(chunks_large)

    def test_oversized_single_segment_not_split(self, oversized_segment_transcript):
        """Single segment exceeding limit goes in own chunk (no infinite loop)."""
        chunks = chunk_transcript(
            oversized_segment_transcript,
            hard_token_limit=100,  # Very low limit
            soft_duration_limit_ms=None
        )

        # Should have 3 chunks (or 2 if boundaries merge)
        assert len(chunks) >= 2

        # Verify no infinite loop / all segments present
        assert validate_chunks_integrity(oversized_segment_transcript, chunks)


# =============================================================================
# Soft Duration Limit Tests
# =============================================================================

class TestSoftDurationLimit:
    """Tests for soft duration limit behavior."""

    def test_soft_limit_creates_splits(self, fifteen_minute_transcript):
        """Soft duration limit triggers splits."""
        # 15 min transcript with 3 min soft limit
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=900_000,  # 15 min hard (no effect)
            hard_token_limit=10000,   # High (no effect)
            soft_duration_limit_ms=180_000  # 3 min soft
        )

        # Should have ~5 chunks (15min / 3min)
        assert len(chunks) >= 4

    def test_disabled_soft_limit(self, fifteen_minute_transcript):
        """soft_duration_limit_ms=None disables soft limit."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=900_000,
            hard_token_limit=10000,
            soft_duration_limit_ms=None  # Disabled
        )

        # With no limits exceeded, should be 1 chunk
        assert len(chunks) == 1

    def test_soft_limit_respects_min_segments(self, fifteen_minute_transcript):
        """Soft limit doesn't split below min_segments_per_chunk."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            soft_duration_limit_ms=10_000,  # Very short (10 sec)
            min_segments_per_chunk=3
        )

        # Each chunk should have at least 3 segments (except maybe last)
        for chunk in chunks[:-1]:
            assert len(chunk.segments) >= 3


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case handling tests."""

    def test_empty_segments_returns_original(self):
        """Empty transcript returns as-is."""
        # Need at least 1 segment for Transcript validation
        # So we test with minimal transcript
        t = Transcript(
            segments=[TranscriptSegment(speaker="doctor", text="x", startMs=0, endMs=1)],
            durationMs=1
        )
        chunks = chunk_transcript(t)
        assert len(chunks) == 1

    def test_single_segment_returns_original(self, minimal_transcript):
        """Single segment transcript not chunked."""
        chunks = chunk_transcript(minimal_transcript)

        assert len(chunks) == 1
        assert chunks[0].segments[0].text == minimal_transcript.segments[0].text

    def test_alternating_speakers_preserved(self):
        """Speaker alternation preserved across chunks."""
        segments = []
        for i in range(20):
            speaker = "doctor" if i % 2 == 0 else "patient"
            segments.append(
                TranscriptSegment(
                    speaker=speaker,
                    text=f"Turn {i}",
                    startMs=i * 10_000,
                    endMs=(i + 1) * 10_000 - 1000
                )
            )

        t = Transcript(segments=segments, durationMs=200_000)

        chunks = chunk_transcript(
            t,
            hard_token_limit=50,  # Force splits
            soft_duration_limit_ms=None
        )

        # Verify alternation preserved within each chunk
        all_speakers = []
        for chunk in chunks:
            for seg in chunk.segments:
                all_speakers.append(seg.speaker)

        original_speakers = [seg.speaker for seg in segments]
        assert all_speakers == original_speakers

    def test_language_preserved_in_chunks(self, short_transcript):
        """Language metadata preserved in all chunks."""
        # Modify language
        t = short_transcript.model_copy(deep=True)
        # Create new transcript with different language
        t = Transcript(
            segments=t.segments,
            language="en",
            durationMs=t.duration_ms
        )

        chunks = chunk_transcript(t, hard_token_limit=20)

        for chunk in chunks:
            assert chunk.language == "en"


# =============================================================================
# 15-Minute Transcript Test (Integration)
# =============================================================================

class TestFifteenMinuteTranscript:
    """Integration test with ~15 minute transcript."""

    def test_fifteen_min_chunked_correctly(self, fifteen_minute_transcript):
        """
        15 min transcript with 5 min max duration creates 3 chunks.
        Last chunk preserves final segments.
        """
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,  # 5 min
            soft_duration_limit_ms=None,
            hard_token_limit=10000  # High (no effect)
        )

        # Should have 3 chunks (0-5min, 5-10min, 10-15min)
        assert len(chunks) == 3

        # Last chunk contains final segments
        last_chunk = chunks[-1]
        last_original = fifteen_minute_transcript.segments[-1]
        assert last_chunk.segments[-1].text == last_original.text

    def test_fifteen_min_with_soft_limit(self, fifteen_minute_transcript):
        """
        15 min transcript with 3 min soft limit creates ~5 chunks.
        """
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=900_000,  # 15 min hard (no effect)
            soft_duration_limit_ms=180_000,  # 3 min soft
            hard_token_limit=10000
        )

        # Should have ~5 chunks
        assert 4 <= len(chunks) <= 6

        # Integrity preserved
        assert validate_chunks_integrity(fifteen_minute_transcript, chunks)

    def test_fifteen_min_stats(self, fifteen_minute_transcript):
        """Verify stats calculation for 15 min transcript."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000
        )

        stats = get_chunk_stats(chunks)

        assert stats["chunk_count"] == len(chunks)
        assert stats["total_segments"] == 15


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_validate_integrity_detects_loss(self, short_transcript):
        """validate_chunks_integrity detects segment loss."""
        chunks = chunk_transcript(short_transcript)

        # Remove a segment (simulate loss)
        bad_chunks = [
            Transcript(
                segments=chunks[0].segments[:-1],  # Missing last segment
                durationMs=chunks[0].duration_ms,
                language=chunks[0].language
            )
        ]

        assert not validate_chunks_integrity(short_transcript, bad_chunks)

    def test_validate_integrity_detects_duplication(self, short_transcript):
        """validate_chunks_integrity detects segment duplication."""
        # Duplicate a segment
        dup_segments = list(short_transcript.segments) + [short_transcript.segments[0]]
        bad_transcript = Transcript(
            segments=dup_segments,
            durationMs=short_transcript.duration_ms,
            language=short_transcript.language
        )

        chunks = [bad_transcript]

        assert not validate_chunks_integrity(short_transcript, chunks)

    def test_get_chunk_stats_empty(self):
        """get_chunk_stats handles empty list."""
        stats = get_chunk_stats([])

        assert stats["chunk_count"] == 0
        assert stats["total_segments"] == 0

    def test_get_chunk_stats_correct(self, short_transcript):
        """get_chunk_stats returns correct values."""
        chunks = [short_transcript]  # Single chunk

        stats = get_chunk_stats(chunks)

        assert stats["chunk_count"] == 1
        assert stats["total_segments"] == 3
        assert stats["total_duration_ms"] == short_transcript.duration_ms
        assert len(stats["chunks"]) == 1


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing callers."""

    def test_legacy_signature_works(self, fifteen_minute_transcript):
        """Old signature chunk_transcript(t, max_duration_ms) still works."""
        chunks = chunk_transcript(fifteen_minute_transcript, 300_000)

        assert len(chunks) >= 1
        assert validate_chunks_integrity(fifteen_minute_transcript, chunks)

    def test_default_values_unchanged(self):
        """Default values match expected constants."""
        assert DEFAULT_MAX_CHUNK_DURATION_MS == 300_000
        assert DEFAULT_HARD_TOKEN_LIMIT == 2048
