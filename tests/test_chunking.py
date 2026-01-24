
import pytest
from app.schemas.request import Transcript, TranscriptSegment
from app.services.chunking import chunk_transcript, DEFAULT_MAX_CHUNK_DURATION_MS

def test_chunking_short_transcript():
    """Verify short transcripts are not chunked."""
    segments = [
        TranscriptSegment(speaker="doctor", text="Hola", startMs=0, endMs=1000),
        TranscriptSegment(speaker="patient", text="Hola", startMs=1000, endMs=2000),
    ]
    t = Transcript(segments=segments, durationMs=2000)
    
    chunks = chunk_transcript(t, max_duration_ms=5000)
    
    assert len(chunks) == 1
    assert chunks[0] == t
    assert chunks[0].duration_ms == 2000

def test_chunking_long_transcript():
    """Verify long transcripts are split correctly."""
    # Create segments that span 15 mins (3 chunks of 5 mins)
    # Segments every 1 minute (60,000 ms)
    segments = []
    for i in range(15):
        start = i * 60000
        end = start + 50000 # 50 sec speech
        segments.append(
            TranscriptSegment(speaker="doctor", text=f"Minuto {i}", startMs=start, endMs=end)
        )
    
    # Total duration 15 mins
    t = Transcript(segments=segments, durationMs=15 * 60000)
    
    # Run chunking with 5 min limit (300,000 ms)
    chunks = chunk_transcript(t, max_duration_ms=300000)
    
    # Expect 3 chunks: 0-4min, 5-9min, 10-14min
    assert len(chunks) == 3
    
    # Verify chunk contents
    # Chunk 1: min 0, 1, 2, 3, 4 (5 segments)
    # Last segment end: 4*60000 + 50000 = 290,000 < 300,000
    # Next segment starts at 300,000
    assert len(chunks[0].segments) == 5
    assert chunks[0].segments[0].text == "Minuto 0"
    assert chunks[0].segments[-1].text == "Minuto 4"
    
    # Chunk 2: min 5, 6, 7, 8, 9
    assert len(chunks[1].segments) == 5
    assert chunks[1].segments[0].text == "Minuto 5"
    assert chunks[1].segments[-1].text == "Minuto 9"
    
    # Check NO segments lost
    total_segments = sum(len(c.segments) for c in chunks)
    assert total_segments == 15

def test_chunking_boundary_edge_case():
    """Verify exact boundary behavior."""
    # 2 segments: one ends at 1000, next ends at 3000
    # max_duration = 2000
    segments = [
        TranscriptSegment(speaker="doctor", text="A", startMs=0, endMs=1000),
        TranscriptSegment(speaker="doctor", text="B", startMs=1500, endMs=3000), 
    ]
    t = Transcript(segments=segments, durationMs=3000)
    
    # Seg A ends at 1000 (<= 2000) -> Chunk 1
    # Seg B ends at 3000. Start of chunk is 0. 3000-0 > 2000? Yes.
    # Should force split BEFORE Seg B.
    chunks = chunk_transcript(t, max_duration_ms=2000)
    
    assert len(chunks) == 2
    assert len(chunks[0].segments) == 1
    assert len(chunks[1].segments) == 1
    assert chunks[0].segments[0].text == "A"
    assert chunks[1].segments[0].text == "B"
