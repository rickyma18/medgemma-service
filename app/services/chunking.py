"""
Intelligent chunking module for splitting long transcripts.

Features:
- Hard limit by estimated tokens (heuristic, no external tokenizer)
- Soft limit by duration (configurable)
- Never splits mid-segment (respects segment boundaries)
- Handles edge cases: empty, single segment, oversized segments

PHI-Safe: No content logging.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import math

from app.schemas.request import Transcript, TranscriptSegment


# =============================================================================
# Configuration Defaults
# =============================================================================

DEFAULT_MAX_CHUNK_DURATION_MS = 300_000  # 5 minutes
DEFAULT_HARD_TOKEN_LIMIT = 2048  # Conservative for most LLMs
DEFAULT_SOFT_DURATION_LIMIT_MS = 180_000  # 3 minutes soft target
DEFAULT_MIN_SEGMENTS_PER_CHUNK = 1

# Token estimation constants
CHARS_PER_TOKEN_ESTIMATE = 4.0  # ~4 chars per token (Spanish/English average)
MIN_TOKENS_PER_SEGMENT = 1  # Minimum tokens for any segment


# =============================================================================
# Token Estimation (Heuristic - No External Dependencies)
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using character-based heuristic.

    Heuristic: tokens ≈ ceil(chars / 4)
    This is conservative for Spanish medical text.

    Args:
        text: Input text string

    Returns:
        Estimated token count (minimum 1 for non-empty text)
    """
    if not text:
        return 0

    char_count = len(text)
    estimated = math.ceil(char_count / CHARS_PER_TOKEN_ESTIMATE)

    return max(estimated, MIN_TOKENS_PER_SEGMENT)


def estimate_segment_tokens(segment: TranscriptSegment) -> int:
    """
    Estimate token count for a transcript segment.

    Includes speaker label overhead (~2 tokens).

    Args:
        segment: TranscriptSegment to estimate

    Returns:
        Estimated token count
    """
    text_tokens = estimate_tokens(segment.text)
    speaker_overhead = 2  # "[doctor]: " or similar

    return text_tokens + speaker_overhead


# =============================================================================
# Chunking State (Internal)
# =============================================================================

@dataclass
class _ChunkState:
    """Internal state for chunk building."""
    segments: List[TranscriptSegment] = field(default_factory=list)
    token_count: int = 0
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None

    def add_segment(self, segment: TranscriptSegment, tokens: int) -> None:
        """Add a segment to this chunk."""
        self.segments.append(segment)
        self.token_count += tokens

        if self.start_ms is None:
            self.start_ms = segment.start_ms
        self.end_ms = segment.end_ms

    @property
    def duration_ms(self) -> int:
        """Calculate chunk duration in ms."""
        if self.start_ms is None or self.end_ms is None:
            return 0
        return self.end_ms - self.start_ms

    @property
    def segment_count(self) -> int:
        """Number of segments in chunk."""
        return len(self.segments)

    def is_empty(self) -> bool:
        """Check if chunk has no segments."""
        return len(self.segments) == 0

    def to_transcript(self, language: str) -> Transcript:
        """Convert chunk state to Transcript."""
        return Transcript(
            segments=self.segments,
            language=language,
            durationMs=self.duration_ms
        )

    def reset(self) -> None:
        """Reset state for new chunk."""
        self.segments = []
        self.token_count = 0
        self.start_ms = None
        self.end_ms = None


# =============================================================================
# Main Chunking Function
# =============================================================================

def chunk_transcript(
    transcript: Transcript,
    max_duration_ms: int = DEFAULT_MAX_CHUNK_DURATION_MS,
    *,
    hard_token_limit: int = DEFAULT_HARD_TOKEN_LIMIT,
    soft_duration_limit_ms: Optional[int] = DEFAULT_SOFT_DURATION_LIMIT_MS,
    min_segments_per_chunk: int = DEFAULT_MIN_SEGMENTS_PER_CHUNK,
) -> List[Transcript]:
    """
    Split a Transcript into smaller chunks respecting token and duration limits.

    Chunking Rules:
    1. NEVER split mid-segment (segment boundaries are sacred)
    2. Hard token limit: if adding segment exceeds limit and chunk has >= min_segments,
       start new chunk
    3. Soft duration limit: if duration exceeds soft limit and chunk has >= min_segments,
       prefer starting new chunk (but token limit takes precedence)
    4. Oversized segment: if a single segment exceeds hard_token_limit, it goes alone
       in its own chunk (documented behavior, no infinite loop)

    Args:
        transcript: Full transcript to chunk
        max_duration_ms: Maximum duration per chunk (legacy param, used as hard duration cap)
        hard_token_limit: Maximum estimated tokens per chunk
        soft_duration_limit_ms: Soft target duration (triggers split if exceeded and has min segments)
        min_segments_per_chunk: Minimum segments before allowing split (default 1)

    Returns:
        List of Transcript chunks (never empty if input has segments)

    Guarantees:
        - No segment loss or duplication
        - Each chunk is a valid Transcript
        - Original language preserved
    """
    # Edge case: no segments
    if not transcript.segments:
        return [transcript]

    # Edge case: single segment
    if len(transcript.segments) == 1:
        return [transcript]

    # Pre-compute token estimates for all segments
    segment_tokens = [
        estimate_segment_tokens(seg) for seg in transcript.segments
    ]

    # Check if entire transcript fits (all limits)
    total_tokens = sum(segment_tokens)
    total_duration = transcript.segments[-1].end_ms - transcript.segments[0].start_ms

    fits_in_tokens = total_tokens <= hard_token_limit
    fits_in_hard_duration = total_duration <= max_duration_ms
    fits_in_soft_duration = (
        soft_duration_limit_ms is None
        or total_duration <= soft_duration_limit_ms
    )

    # Only skip chunking if ALL limits are satisfied
    if fits_in_tokens and fits_in_hard_duration and fits_in_soft_duration:
        return [transcript]

    # Build chunks
    chunks: List[Transcript] = []
    current = _ChunkState()

    for i, segment in enumerate(transcript.segments):
        seg_tokens = segment_tokens[i]

        # Calculate what would happen if we add this segment
        would_exceed_tokens = (current.token_count + seg_tokens) > hard_token_limit

        # Duration check (from chunk start to this segment's end)
        # Use current.start_ms if set, otherwise this is first segment so use its start
        chunk_start = current.start_ms if current.start_ms is not None else segment.start_ms
        projected_duration = segment.end_ms - chunk_start
        would_exceed_hard_duration = projected_duration > max_duration_ms
        would_exceed_soft_duration = (
            soft_duration_limit_ms is not None
            and projected_duration > soft_duration_limit_ms
        )

        # Determine if we should start a new chunk
        should_split = False

        # Rule 1: Hard token limit exceeded
        if would_exceed_tokens and current.segment_count >= min_segments_per_chunk:
            should_split = True

        # Rule 2: Hard duration limit exceeded
        if would_exceed_hard_duration and current.segment_count >= min_segments_per_chunk:
            should_split = True

        # Rule 3: Soft duration exceeded (only if we have min segments)
        if (
            would_exceed_soft_duration
            and current.segment_count >= min_segments_per_chunk
            and not current.is_empty()
        ):
            should_split = True

        # Execute split if needed
        if should_split and not current.is_empty():
            chunks.append(current.to_transcript(transcript.language))
            current.reset()

        # Add segment to current chunk
        current.add_segment(segment, seg_tokens)

    # Don't forget the last chunk
    if not current.is_empty():
        chunks.append(current.to_transcript(transcript.language))

    return chunks


# =============================================================================
# Utility Functions
# =============================================================================

def validate_chunks_integrity(
    original: Transcript,
    chunks: List[Transcript]
) -> bool:
    """
    Validate that chunks contain all original segments without loss or duplication.

    Args:
        original: Original transcript
        chunks: List of chunk transcripts

    Returns:
        True if integrity check passes
    """
    original_texts = [seg.text for seg in original.segments]
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.extend(seg.text for seg in chunk.segments)

    return original_texts == chunk_texts


def get_chunk_stats(chunks: List[Transcript]) -> dict:
    """
    Get statistics about chunking results.
    PHI-safe: only returns counts and sizes, no content.

    Args:
        chunks: List of transcript chunks

    Returns:
        Dict with chunk statistics
    """
    if not chunks:
        return {
            "chunk_count": 0,
            "total_segments": 0,
            "total_tokens_estimated": 0,
            "total_duration_ms": 0,
            "chunks": []
        }

    chunk_stats = []
    for i, chunk in enumerate(chunks):
        tokens = sum(estimate_segment_tokens(seg) for seg in chunk.segments)
        chunk_stats.append({
            "index": i,
            "segment_count": len(chunk.segments),
            "token_estimate": tokens,
            "duration_ms": chunk.duration_ms
        })

    return {
        "chunk_count": len(chunks),
        "total_segments": sum(c["segment_count"] for c in chunk_stats),
        "total_tokens_estimated": sum(c["token_estimate"] for c in chunk_stats),
        "total_duration_ms": sum(c["duration_ms"] for c in chunk_stats),
        "chunks": chunk_stats
    }
