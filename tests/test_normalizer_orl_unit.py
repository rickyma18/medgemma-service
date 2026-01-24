
import pytest
from app.services.text_normalizer_orl import normalize_transcript_orl
from app.schemas.request import Transcript, TranscriptSegment

def test_normalization_basics():
    # Construct a transcript with known typos from whitelist
    seg1 = TranscriptSegment(speaker="doctor", text="Tiene las migdalas inflamadas.", startMs=0, endMs=1000)
    seg2 = TranscriptSegment(speaker="patient", text="Siento un faringeo raro.", startMs=1000, endMs=2000)
    transcript = Transcript(segments=[seg1, seg2], durationMs=2000)
    
    norm_t, replacements = normalize_transcript_orl(transcript)
    
    assert replacements == 2
    # Verify text changed
    assert "amígdalas" in norm_t.segments[0].text
    assert "migdalas" not in norm_t.segments[0].text
    
    assert "faríngeo" in norm_t.segments[1].text
