"""
Transcript Cleaner
Removes non-clinical artifacts from transcript segments.
Port from previous cleaner logic to new namespace.
"""
import re

# Regex for common filler words (Spanish)
# \b word boundary
# Regex for common filler words (Spanish)
# Matches filler word + optional trailing dots (e.g. "eh...")
# \b ensures full word match. \s* handles potential space before dots if any? No.
# text is "Este..." -> "Este" matches. "..." is next.
RE_FILLERS = re.compile(
    r"\b(eh+|em+|este|bueno|o sea|pues|mm+|ah+)\b(\.{2,})?", 
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
