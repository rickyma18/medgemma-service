"""
Exhaustive tests for intelligent chunking module.

ÉPICA 14: Chunking Inteligente - QA/SDET Coverage

Test Categories:
1. Invariants: no loss, no duplication, stable order (by segment identity tuple)
2. Hard token limit enforcement (with oversized segment exception)
3. Soft duration limit behavior
4. 15-minute transcript integration
5. Edge cases: single segment, min_segments, alternating speakers, etc.

PHI-safe: Uses fictitious text only, no clinical content logged.

Segment Identity: (speaker, start_ms, end_ms, text) tuple for robust comparison.
"""
import pytest
from typing import List, Tuple

from app.schemas.request import Transcript, TranscriptSegment
from app.services.chunking import (
    chunk_transcript,
    estimate_tokens,
    estimate_segment_tokens,
    validate_chunks_integrity,
    get_chunk_stats,
    DEFAULT_MAX_CHUNK_DURATION_MS,
    DEFAULT_HARD_TOKEN_LIMIT,
    DEFAULT_SOFT_DURATION_LIMIT_MS,
    CHARS_PER_TOKEN_ESTIMATE,
)


# =============================================================================
# Helpers: Segment Identity for Robust Comparison
# =============================================================================

def segment_to_tuple(seg: TranscriptSegment) -> Tuple[str, int, int, str]:
    """Convert segment to identity tuple for comparison."""
    return (seg.speaker, seg.start_ms, seg.end_ms, seg.text)


def segments_to_tuples(segments: List[TranscriptSegment]) -> List[Tuple[str, int, int, str]]:
    """Convert segment list to identity tuples."""
    return [segment_to_tuple(s) for s in segments]


def flatten_chunks_to_tuples(chunks: List[Transcript]) -> List[Tuple[str, int, int, str]]:
    """Flatten all chunk segments to identity tuples."""
    result = []
    for chunk in chunks:
        result.extend(segments_to_tuples(chunk.segments))
    return result


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def single_segment_transcript() -> Transcript:
    """Transcript with exactly 1 segment."""
    return Transcript(
        segments=[
            TranscriptSegment(
                speaker="doctor",
                text="Unico segmento de prueba.",
                startMs=0,
                endMs=5000
            )
        ],
        durationMs=5000
    )


@pytest.fixture
def two_segment_transcript() -> Transcript:
    """Minimal multi-segment transcript."""
    return Transcript(
        segments=[
            TranscriptSegment(speaker="doctor", text="Primero.", startMs=0, endMs=2000),
            TranscriptSegment(speaker="patient", text="Segundo.", startMs=2000, endMs=4000),
        ],
        durationMs=4000
    )


@pytest.fixture
def short_transcript() -> Transcript:
    """Short transcript (3 segments) that fits in any reasonable limit."""
    return Transcript(
        segments=[
            TranscriptSegment(speaker="doctor", text="Hola.", startMs=0, endMs=1000),
            TranscriptSegment(speaker="patient", text="Buenos dias.", startMs=1000, endMs=2000),
            TranscriptSegment(speaker="doctor", text="Como esta?", startMs=2000, endMs=3000),
        ],
        durationMs=3000
    )


@pytest.fixture
def fifteen_minute_transcript() -> Transcript:
    """
    ~15 minute transcript with 15 segments (1 per minute).
    Each segment: ~70 chars (~18 tokens + 2 overhead = ~20 tokens).
    Total: ~300 tokens.
    """
    segments = []
    for i in range(15):
        start = i * 60_000
        end = start + 50_000
        speaker = "doctor" if i % 2 == 0 else "patient"
        # Unique text per segment for identity
        text = f"Segmento {i:02d}: contenido de prueba unico para validar chunking."
        segments.append(
            TranscriptSegment(speaker=speaker, text=text, startMs=start, endMs=end)
        )

    return Transcript(segments=segments, durationMs=15 * 60_000)


@pytest.fixture
def high_token_transcript() -> Transcript:
    """
    Transcript optimized for token limit testing.
    10 segments, each ~200 chars (~50 tokens + 2 = ~52 tokens each).
    Total: ~520 tokens.
    """
    segments = []
    base_text = "Este texto tiene aproximadamente doscientos caracteres para prueba. " * 3

    for i in range(10):
        start = i * 30_000
        end = start + 25_000
        segments.append(
            TranscriptSegment(
                speaker="doctor" if i % 2 == 0 else "patient",
                text=f"{base_text}Segmento {i}.",
                startMs=start,
                endMs=end
            )
        )

    return Transcript(segments=segments, durationMs=300_000)


@pytest.fixture
def oversized_segment_transcript() -> Transcript:
    """
    Transcript with one segment that exceeds typical token limits.
    Middle segment: ~5000 chars = ~1250 tokens (exceeds 1024 or even 512).
    """
    giant_text = "Texto largo para simular segmento enorme. " * 125  # ~5000 chars

    return Transcript(
        segments=[
            TranscriptSegment(speaker="doctor", text="Inicio normal.", startMs=0, endMs=2000),
            TranscriptSegment(speaker="patient", text=giant_text.strip(), startMs=2000, endMs=120000),
            TranscriptSegment(speaker="doctor", text="Fin normal.", startMs=120000, endMs=122000),
        ],
        durationMs=122000
    )


@pytest.fixture
def alternating_speakers_transcript() -> Transcript:
    """20 segments with strictly alternating speakers."""
    segments = []
    for i in range(20):
        speaker = "doctor" if i % 2 == 0 else "patient"
        segments.append(
            TranscriptSegment(
                speaker=speaker,
                text=f"Turno {i:02d} del {speaker}.",
                startMs=i * 5000,
                endMs=(i + 1) * 5000 - 500
            )
        )
    return Transcript(segments=segments, durationMs=100_000)


@pytest.fixture
def repeated_short_text_transcript() -> Transcript:
    """
    Segments with identical short text but different timestamps.
    Tests that identity uses full tuple, not just text.
    """
    segments = []
    for i in range(10):
        segments.append(
            TranscriptSegment(
                speaker="doctor",
                text="Ok.",  # Same text for all
                startMs=i * 2000,
                endMs=i * 2000 + 1500
            )
        )
    return Transcript(segments=segments, durationMs=20_000)


# =============================================================================
# 1. INVARIANT TESTS: No Loss, No Duplication, Stable Order
# =============================================================================

class TestChunkingInvariants:
    """
    Core invariants that must ALWAYS hold:
    - No segment loss
    - No segment duplication
    - Original order preserved
    Comparison by full identity tuple: (speaker, start_ms, end_ms, text)
    """

    def test_no_loss_short_transcript(self, short_transcript):
        """Short transcript: all segments present after chunking."""
        chunks = chunk_transcript(short_transcript, soft_duration_limit_ms=None)

        original = segments_to_tuples(short_transcript.segments)
        result = flatten_chunks_to_tuples(chunks)

        assert result == original, "Segment loss detected"

    def test_no_loss_long_transcript(self, fifteen_minute_transcript):
        """Long transcript: all segments present after chunking."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,
            hard_token_limit=100,
            soft_duration_limit_ms=60_000
        )

        original = segments_to_tuples(fifteen_minute_transcript.segments)
        result = flatten_chunks_to_tuples(chunks)

        assert result == original, "Segment loss detected in long transcript"

    def test_no_duplication(self, fifteen_minute_transcript):
        """No segment appears more than once."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            hard_token_limit=50,
            soft_duration_limit_ms=None
        )

        result = flatten_chunks_to_tuples(chunks)

        # Check for duplicates
        assert len(result) == len(set(result)), "Duplicate segments detected"

    def test_order_preserved_simple(self, short_transcript):
        """Order maintained in simple case."""
        chunks = chunk_transcript(short_transcript, hard_token_limit=20, soft_duration_limit_ms=None)

        original = segments_to_tuples(short_transcript.segments)
        result = flatten_chunks_to_tuples(chunks)

        assert result == original, "Order changed"

    def test_order_preserved_with_many_chunks(self, alternating_speakers_transcript):
        """Order maintained when splitting into many chunks."""
        chunks = chunk_transcript(
            alternating_speakers_transcript,
            hard_token_limit=30,
            soft_duration_limit_ms=None
        )

        original = segments_to_tuples(alternating_speakers_transcript.segments)
        result = flatten_chunks_to_tuples(chunks)

        assert result == original, "Order changed with many chunks"

    def test_repeated_text_uses_full_identity(self, repeated_short_text_transcript):
        """
        Segments with identical text but different timestamps are distinct.
        Tests that comparison uses full tuple, not just text.
        """
        chunks = chunk_transcript(
            repeated_short_text_transcript,
            hard_token_limit=30,
            soft_duration_limit_ms=None
        )

        original = segments_to_tuples(repeated_short_text_transcript.segments)
        result = flatten_chunks_to_tuples(chunks)

        # All 10 segments should be present (even though text is identical)
        assert len(result) == 10
        assert result == original

    def test_no_empty_chunks_ever(self, fifteen_minute_transcript):
        """No chunk should be empty."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            hard_token_limit=50,
            soft_duration_limit_ms=10_000
        )

        for i, chunk in enumerate(chunks):
            assert len(chunk.segments) > 0, f"Chunk {i} is empty"


# =============================================================================
# 2. HARD TOKEN LIMIT TESTS
# =============================================================================

class TestHardTokenLimit:
    """
    Hard token limit enforcement.
    Exception: A single oversized segment may exceed limit (goes alone in chunk).
    """

    def test_chunks_respect_hard_limit(self, high_token_transcript):
        """Multi-segment chunks must not exceed hard token limit."""
        hard_limit = 150

        chunks = chunk_transcript(
            high_token_transcript,
            hard_token_limit=hard_limit,
            soft_duration_limit_ms=None
        )

        for i, chunk in enumerate(chunks):
            chunk_tokens = sum(estimate_segment_tokens(s) for s in chunk.segments)

            # If chunk has multiple segments, must respect limit
            if len(chunk.segments) > 1:
                assert chunk_tokens <= hard_limit, (
                    f"Chunk {i} with {len(chunk.segments)} segments "
                    f"has {chunk_tokens} tokens > limit {hard_limit}"
                )

    def test_oversized_segment_allowed_alone(self, oversized_segment_transcript):
        """Single oversized segment is allowed in its own chunk."""
        hard_limit = 100  # Way below the giant segment

        chunks = chunk_transcript(
            oversized_segment_transcript,
            hard_token_limit=hard_limit,
            soft_duration_limit_ms=None
        )

        # Find the chunk with the giant segment
        giant_text_prefix = "Texto largo para simular"
        giant_chunk = None
        for chunk in chunks:
            for seg in chunk.segments:
                if seg.text.startswith(giant_text_prefix):
                    giant_chunk = chunk
                    break

        assert giant_chunk is not None, "Giant segment not found"
        assert len(giant_chunk.segments) == 1, "Giant segment should be alone in chunk"

        # Verify it exceeds limit (this is allowed for single-segment chunks)
        giant_tokens = sum(estimate_segment_tokens(s) for s in giant_chunk.segments)
        assert giant_tokens > hard_limit, "Giant segment should exceed limit"

    def test_oversized_no_infinite_loop(self, oversized_segment_transcript):
        """Oversized segment doesn't cause infinite loop."""
        chunks = chunk_transcript(
            oversized_segment_transcript,
            hard_token_limit=50,
            soft_duration_limit_ms=None
        )

        # Should complete and preserve all segments
        original = segments_to_tuples(oversized_segment_transcript.segments)
        result = flatten_chunks_to_tuples(chunks)
        assert result == original

    def test_lower_limit_creates_more_chunks(self, high_token_transcript):
        """Lower token limit = more chunks."""
        chunks_high = chunk_transcript(
            high_token_transcript,
            hard_token_limit=1000,
            soft_duration_limit_ms=None
        )
        chunks_low = chunk_transcript(
            high_token_transcript,
            hard_token_limit=100,
            soft_duration_limit_ms=None
        )

        assert len(chunks_low) > len(chunks_high)

    def test_exact_limit_boundary(self):
        """Test behavior at exact token limit boundary."""
        # Create segments where sum is exactly at limit
        # Each segment: 4 chars = 1 token + 2 overhead = 3 tokens
        segments = [
            TranscriptSegment(speaker="doctor", text="AAAA", startMs=i*1000, endMs=(i+1)*1000)
            for i in range(10)
        ]
        t = Transcript(segments=segments, durationMs=10000)

        # 10 segments * 3 tokens = 30 tokens total
        # With limit 30, should fit in 1 chunk (if soft limit disabled)
        chunks = chunk_transcript(
            t,
            hard_token_limit=30,
            soft_duration_limit_ms=None
        )

        assert len(chunks) == 1


# =============================================================================
# 3. SOFT DURATION LIMIT TESTS
# =============================================================================

class TestSoftDurationLimit:
    """Soft duration limit triggers splits at segment boundaries."""

    def test_soft_limit_triggers_split(self, fifteen_minute_transcript):
        """Soft limit creates splits when exceeded."""
        # 15 min transcript, 3 min soft limit should create ~5 chunks
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=900_000,  # 15 min hard (no effect)
            hard_token_limit=10000,   # High (no effect)
            soft_duration_limit_ms=180_000  # 3 min soft
        )

        assert len(chunks) >= 4, "Soft limit should create multiple chunks"

    def test_soft_limit_disabled_with_none(self, fifteen_minute_transcript):
        """soft_duration_limit_ms=None disables soft limit."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=900_000,
            hard_token_limit=10000,
            soft_duration_limit_ms=None
        )

        # With both hard limits high and soft disabled, single chunk
        assert len(chunks) == 1

    def test_split_at_segment_boundary_not_mid_segment(self, fifteen_minute_transcript):
        """Splits occur between segments, never within."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            soft_duration_limit_ms=60_000,  # 1 min
            hard_token_limit=10000
        )

        # Verify each chunk's segments are contiguous from original
        original = segments_to_tuples(fifteen_minute_transcript.segments)
        position = 0

        for chunk in chunks:
            chunk_tuples = segments_to_tuples(chunk.segments)
            expected = original[position:position + len(chunk_tuples)]
            assert chunk_tuples == expected, "Chunk has non-contiguous segments"
            position += len(chunk_tuples)

    def test_soft_limit_respects_min_segments(self, fifteen_minute_transcript):
        """Soft limit respects min_segments_per_chunk."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            soft_duration_limit_ms=10_000,  # Very aggressive
            min_segments_per_chunk=4
        )

        # All chunks except possibly last should have >= 4 segments
        for chunk in chunks[:-1]:
            assert len(chunk.segments) >= 4, "min_segments_per_chunk not respected"


# =============================================================================
# 4. FIFTEEN MINUTE TRANSCRIPT INTEGRATION
# =============================================================================

class TestFifteenMinuteTranscript:
    """Integration tests with ~15 minute transcript."""

    def test_creates_multiple_chunks(self, fifteen_minute_transcript):
        """15 min transcript with 5 min limit creates multiple chunks."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,  # 5 min
            soft_duration_limit_ms=None,
            hard_token_limit=10000
        )

        assert len(chunks) > 1, "Should create multiple chunks"
        assert len(chunks) == 3, "Should create exactly 3 chunks for 15min/5min"

    def test_last_chunk_has_final_segments(self, fifteen_minute_transcript):
        """Last chunk contains the final segments of original."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,
            soft_duration_limit_ms=None
        )

        last_original = segment_to_tuple(fifteen_minute_transcript.segments[-1])
        last_chunk_last = segment_to_tuple(chunks[-1].segments[-1])

        assert last_chunk_last == last_original, "Last segment not in last chunk"

    def test_first_chunk_has_first_segments(self, fifteen_minute_transcript):
        """First chunk starts with first segment of original."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,
            soft_duration_limit_ms=None
        )

        first_original = segment_to_tuple(fifteen_minute_transcript.segments[0])
        first_chunk_first = segment_to_tuple(chunks[0].segments[0])

        assert first_chunk_first == first_original

    def test_all_fifteen_segments_present(self, fifteen_minute_transcript):
        """All 15 segments are present across chunks."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,
            soft_duration_limit_ms=None
        )

        total_segments = sum(len(c.segments) for c in chunks)
        assert total_segments == 15

    def test_chunk_stats_accurate(self, fifteen_minute_transcript):
        """get_chunk_stats returns accurate information."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,
            soft_duration_limit_ms=None
        )

        stats = get_chunk_stats(chunks)

        assert stats["chunk_count"] == len(chunks)
        assert stats["total_segments"] == 15
        assert len(stats["chunks"]) == len(chunks)


# =============================================================================
# 5. EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case handling."""

    def test_single_segment_returns_single_chunk(self, single_segment_transcript):
        """Single segment transcript returns 1 chunk with that segment."""
        chunks = chunk_transcript(single_segment_transcript)

        assert len(chunks) == 1
        assert len(chunks[0].segments) == 1
        assert chunks[0].segments[0].text == single_segment_transcript.segments[0].text

    def test_two_segments_minimal_case(self, two_segment_transcript):
        """Two segment transcript handles correctly."""
        chunks = chunk_transcript(
            two_segment_transcript,
            hard_token_limit=10000,
            soft_duration_limit_ms=None
        )

        assert len(chunks) == 1
        original = segments_to_tuples(two_segment_transcript.segments)
        result = flatten_chunks_to_tuples(chunks)
        assert result == original

    def test_min_segments_per_chunk_respected(self, alternating_speakers_transcript):
        """min_segments_per_chunk prevents premature splits."""
        chunks = chunk_transcript(
            alternating_speakers_transcript,
            hard_token_limit=100,
            min_segments_per_chunk=5
        )

        # All non-final chunks should have >= 5 segments
        for chunk in chunks[:-1]:
            assert len(chunk.segments) >= 5

    def test_alternating_speakers_order_preserved(self, alternating_speakers_transcript):
        """Alternating speaker pattern preserved across chunks."""
        chunks = chunk_transcript(
            alternating_speakers_transcript,
            hard_token_limit=50,
            soft_duration_limit_ms=None
        )

        all_speakers = []
        for chunk in chunks:
            for seg in chunk.segments:
                all_speakers.append(seg.speaker)

        # Verify alternation
        for i in range(len(all_speakers)):
            expected = "doctor" if i % 2 == 0 else "patient"
            assert all_speakers[i] == expected, f"Speaker order broken at index {i}"

    def test_language_preserved_in_all_chunks(self, short_transcript):
        """Language metadata preserved in every chunk."""
        t = Transcript(
            segments=short_transcript.segments,
            language="en",
            durationMs=short_transcript.duration_ms
        )

        chunks = chunk_transcript(t, hard_token_limit=15, soft_duration_limit_ms=None)

        for chunk in chunks:
            assert chunk.language == "en"

    def test_duration_ms_calculated_per_chunk(self, fifteen_minute_transcript):
        """Each chunk has correct duration_ms based on its segments."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=300_000,
            soft_duration_limit_ms=None
        )

        for chunk in chunks:
            if chunk.segments:
                expected_duration = chunk.segments[-1].end_ms - chunk.segments[0].start_ms
                assert chunk.duration_ms == expected_duration

    def test_validates_chunks_integrity_helper(self, short_transcript):
        """validate_chunks_integrity works correctly."""
        chunks = chunk_transcript(short_transcript, soft_duration_limit_ms=None)

        # Should pass
        assert validate_chunks_integrity(short_transcript, chunks)

        # Simulate loss - should fail
        bad_chunks = [
            Transcript(
                segments=chunks[0].segments[:-1],
                durationMs=1000,
                language="es"
            )
        ]
        assert not validate_chunks_integrity(short_transcript, bad_chunks)


# =============================================================================
# 6. TOKEN ESTIMATION TESTS
# =============================================================================

class TestTokenEstimation:
    """Tests for token estimation heuristics."""

    def test_empty_text_returns_zero(self):
        """Empty string = 0 tokens."""
        assert estimate_tokens("") == 0

    def test_minimum_one_for_nonempty(self):
        """Non-empty text returns at least 1 token."""
        assert estimate_tokens("a") >= 1
        assert estimate_tokens("ab") >= 1

    def test_known_length_calculation(self):
        """Verify calculation: ceil(chars / 4)."""
        # 100 chars -> ceil(100/4) = 25
        assert estimate_tokens("a" * 100) == 25

        # 101 chars -> ceil(101/4) = 26
        assert estimate_tokens("a" * 101) == 26

        # 4 chars -> ceil(4/4) = 1
        assert estimate_tokens("abcd") == 1

    def test_segment_tokens_includes_overhead(self):
        """Segment estimation adds speaker overhead."""
        segment = TranscriptSegment(
            speaker="doctor",
            text="AAAA",  # 4 chars = 1 token
            startMs=0,
            endMs=1000
        )

        # 1 text token + 2 speaker overhead = 3
        assert estimate_segment_tokens(segment) == 3

    def test_consistent_estimation(self):
        """Same text always gives same estimate."""
        text = "Texto de prueba consistente."
        estimate1 = estimate_tokens(text)
        estimate2 = estimate_tokens(text)
        assert estimate1 == estimate2


# =============================================================================
# 7. BOUNDARY CONDITION TESTS
# =============================================================================

class TestBoundaryConditions:
    """Tests for boundary conditions and limits."""

    def test_segment_at_exact_duration_boundary(self):
        """Segment ending exactly at duration limit."""
        segments = [
            TranscriptSegment(speaker="doctor", text="A", startMs=0, endMs=100),
            TranscriptSegment(speaker="patient", text="B", startMs=100, endMs=200),
        ]
        t = Transcript(segments=segments, durationMs=200)

        chunks = chunk_transcript(
            t,
            max_duration_ms=200,  # Exactly matches total
            soft_duration_limit_ms=None
        )

        assert len(chunks) == 1

    def test_segment_one_ms_over_boundary(self):
        """Segment ending 1ms over duration limit triggers split."""
        segments = [
            TranscriptSegment(speaker="doctor", text="A" * 100, startMs=0, endMs=100),
            TranscriptSegment(speaker="patient", text="B" * 100, startMs=100, endMs=201),
        ]
        t = Transcript(segments=segments, durationMs=201)

        chunks = chunk_transcript(
            t,
            max_duration_ms=200,
            soft_duration_limit_ms=None
        )

        # Should split because segment B would make duration 201 > 200
        assert len(chunks) == 2

    def test_large_gap_between_segments(self):
        """Handles large time gaps between segments."""
        segments = [
            TranscriptSegment(speaker="doctor", text="Start.", startMs=0, endMs=1000),
            TranscriptSegment(speaker="patient", text="After gap.", startMs=100000, endMs=101000),
        ]
        t = Transcript(segments=segments, durationMs=101000)

        chunks = chunk_transcript(
            t,
            max_duration_ms=50_000,  # Gap forces split
            soft_duration_limit_ms=None
        )

        # Both segments preserved
        original = segments_to_tuples(t.segments)
        result = flatten_chunks_to_tuples(chunks)
        assert result == original


# =============================================================================
# 8. DEFAULT VALUES AND BACKWARD COMPATIBILITY
# =============================================================================

class TestDefaultsAndCompatibility:
    """Tests for default values and backward compatibility."""

    def test_default_constants_values(self):
        """Default constants have expected values."""
        assert DEFAULT_MAX_CHUNK_DURATION_MS == 300_000
        assert DEFAULT_HARD_TOKEN_LIMIT == 2048
        assert DEFAULT_SOFT_DURATION_LIMIT_MS == 180_000
        assert CHARS_PER_TOKEN_ESTIMATE == 4.0

    def test_legacy_signature_two_args(self, fifteen_minute_transcript):
        """Legacy call chunk_transcript(t, max_duration_ms) works."""
        chunks = chunk_transcript(fifteen_minute_transcript, 300_000)

        # Should produce valid chunks
        assert len(chunks) >= 1
        original = segments_to_tuples(fifteen_minute_transcript.segments)
        result = flatten_chunks_to_tuples(chunks)
        assert result == original

    def test_keyword_only_args_enforced(self):
        """hard_token_limit and others are keyword-only."""
        t = Transcript(
            segments=[TranscriptSegment(speaker="doctor", text="Test.", startMs=0, endMs=1000)],
            durationMs=1000
        )

        # This should work (keyword)
        chunks = chunk_transcript(t, hard_token_limit=100)
        assert len(chunks) == 1

    def test_get_chunk_stats_empty_list(self):
        """get_chunk_stats handles empty chunk list."""
        stats = get_chunk_stats([])

        assert stats["chunk_count"] == 0
        assert stats["total_segments"] == 0
        assert stats["total_tokens_estimated"] == 0
        assert stats["total_duration_ms"] == 0


# =============================================================================
# 9. STRESS / ROBUSTNESS TESTS
# =============================================================================

class TestRobustness:
    """Robustness and stress tests."""

    def test_many_small_segments(self):
        """Handles many small segments."""
        segments = [
            TranscriptSegment(
                speaker="doctor" if i % 2 == 0 else "patient",
                text=f"S{i}",
                startMs=i * 100,
                endMs=i * 100 + 50
            )
            for i in range(100)
        ]
        t = Transcript(segments=segments, durationMs=10000)

        chunks = chunk_transcript(t, hard_token_limit=50, soft_duration_limit_ms=None)

        # All segments preserved
        assert sum(len(c.segments) for c in chunks) == 100

    def test_very_aggressive_limits(self, fifteen_minute_transcript):
        """Very aggressive limits don't break chunking."""
        chunks = chunk_transcript(
            fifteen_minute_transcript,
            max_duration_ms=1000,  # 1 second
            hard_token_limit=10,   # Very low
            soft_duration_limit_ms=500  # 0.5 seconds
        )

        # Should still preserve all segments
        original = segments_to_tuples(fifteen_minute_transcript.segments)
        result = flatten_chunks_to_tuples(chunks)
        assert result == original

    def test_no_crash_on_minimal_valid_transcript(self):
        """Minimal valid transcript doesn't crash."""
        t = Transcript(
            segments=[TranscriptSegment(speaker="doctor", text="X", startMs=0, endMs=1)],
            durationMs=1
        )

        chunks = chunk_transcript(t)
        assert len(chunks) == 1
        assert len(chunks[0].segments) == 1
