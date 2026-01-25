"""
PHI-safe evidence sanitizer for Epic 15.
Sanitizes evidence snippets before storage/response.

Rules:
- Max 160 chars (hard truncation)
- Remove potential PII patterns (names, IDs, dates of birth)
- Never log the original or sanitized content
"""
import re
from typing import List, Optional

# Max length for evidence snippets
MAX_EVIDENCE_LENGTH = 160

# PII detection patterns (conservative - only detect high-confidence PII)
# Intentionally narrow to avoid false positives on clinical terms
PII_PATTERNS = [
    # Mexican CURP (18 chars alphanumeric pattern) - very specific format
    (r"\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z\d]\d\b", "[CURP]"),
    # Mexican RFC (12-13 chars) - very specific format
    (r"\b[A-ZÑ&]{3,4}\d{6}[A-Z\d]{3}\b", "[RFC]"),
    # Phone numbers (10 digits, various formats)
    (r"\b\d{2}[\s\-]?\d{4}[\s\-]?\d{4}\b", "[TEL]"),
    (r"\b\d{3}[\s\-]?\d{3}[\s\-]?\d{4}\b", "[TEL]"),
    # Email addresses - standard format
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b", "[EMAIL]"),
    # Dates that could be DOB (dd/mm/yyyy or yyyy-mm-dd)
    (r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", "[FECHA]"),
    (r"\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b", "[FECHA]"),
    # Patient/medical record numbers (with explicit keywords)
    (r"(?:expediente|folio|registro)\s*(?:no\.?|num\.?|#|:)\s*\d{4,}", "[ID]"),
    # NSS (IMSS number - exactly 11 digits, standalone)
    (r"(?<!\d)\d{11}(?!\d)", "[NSS]"),
    # Full names only when preceded by explicit markers
    (r"(?:paciente|sr\.?|sra\.?|don|doña)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})", "[NOMBRE]"),
]

# Compile patterns for efficiency
_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in PII_PATTERNS]


def sanitize_evidence(text: str) -> str:
    """
    Sanitize a single evidence snippet.

    - Applies PII pattern replacement
    - Truncates to MAX_EVIDENCE_LENGTH
    - Strips whitespace

    Args:
        text: Raw evidence text (may contain PHI)

    Returns:
        Sanitized text, max 160 chars

    Note: NEVER log the input or output of this function.
    """
    if not text:
        return ""

    # Normalize whitespace
    result = " ".join(text.split())

    # Apply PII patterns
    for pattern, replacement in _COMPILED_PATTERNS:
        result = pattern.sub(replacement, result)

    # Truncate if needed
    if len(result) > MAX_EVIDENCE_LENGTH:
        result = result[:MAX_EVIDENCE_LENGTH - 3] + "..."

    return result.strip()


def sanitize_evidence_list(texts: List[str]) -> List[str]:
    """
    Sanitize a list of evidence snippets.

    Args:
        texts: List of raw evidence texts

    Returns:
        List of sanitized texts (empty strings filtered out)
    """
    sanitized = [sanitize_evidence(t) for t in texts if t]
    return [s for s in sanitized if s]  # Filter empty results


def extract_evidence_from_text(
    text: str,
    field_path: str,
    max_snippets: int = 3
) -> List[tuple[str, str]]:
    """
    Extract evidence snippets from source text for a given field.

    Simple heuristic: Split by sentence boundaries, take first N non-empty.
    This is a baseline implementation - can be enhanced with NLP.

    Args:
        text: Source text to extract from
        field_path: Field path this evidence supports
        max_snippets: Maximum snippets to extract

    Returns:
        List of (sanitized_text, field_path) tuples
    """
    if not text:
        return []

    # Simple sentence splitting (Spanish punctuation)
    sentences = re.split(r'[.!?]+', text)

    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= 10:  # Min meaningful length
            sanitized = sanitize_evidence(sentence)
            if sanitized and len(sanitized) >= 10:
                result.append((sanitized, field_path))
                if len(result) >= max_snippets:
                    break

    return result


def is_potentially_phi(text: str) -> bool:
    """
    Quick check if text potentially contains PHI.
    Used for deciding whether to include evidence in response.

    Returns True if any PII pattern matches.
    """
    if not text:
        return False

    for pattern, _ in _COMPILED_PATTERNS:
        if pattern.search(text):
            return True

    return False
