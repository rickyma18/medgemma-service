"""
Transcript Cleaner
Removes non-clinical artifacts from transcript segments.
"""
import re
from app.schemas.request import Transcript

# Regex for common filler words (Spanish)
# \b word boundary
RE_FILLERS = re.compile(
    r"\b(eh+|em+|este|bueno|o sea|pues|mm+|ah+)\b", 
    re.IGNORECASE
)

# Regex for multiple spaces
RE_SPACES = re.compile(r"\s+")

# Regex for simple stuttering "si si" -> "si"
# Matches word repeated with optional space
RE_STUTTER = re.compile(r"\b(\w+)( \1\b)+", re.IGNORECASE)

def clean_transcript_text(text: str) -> str:
    """
    Applies basic cleaning to text.
    - Remove fillers
    - Collapse stutters
    - Collapse spaces
    """
    # 1. Remove fillers
    text = RE_FILLERS.sub(" ", text)
    
    # 2. Collapse stutters (run twice to catch multi)
    text = RE_STUTTER.sub(r"\1", text)
    
    # 3. Collapse spaces
    text = RE_SPACES.sub(" ", text).strip()
    
    return text

def clean_transcript(transcript: Transcript) -> Transcript:
    """
    Applies cleaning to all segments in transcript.
    Returns new Transcript object.
    """
    new_t = transcript.model_copy(deep=True)
    for seg in new_t.segments:
        seg.text = clean_transcript_text(seg.text)
    return new_t
