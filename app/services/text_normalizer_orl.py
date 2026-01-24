"""
Módulo de normalización de texto para transcripciones ORL (Otorrinolaringología).
Objetivo: Corregir errores comunes de STT (typos, fonética) antes de la extracción LLM.

PHI-safe:
- No loguea contenido de transcripciones.
- Solo retorna métricas numéricas.
"""
import re
from app.schemas.request import Transcript

# Lista de tuplas (Pattern, Replacement).
# Se compilan una sola vez al importar el módulo.
# Se separan variantes singulares/plurales y de género para reemplazo exacto.

ORL_STT_WHITELIST: list[tuple[re.Pattern, str]] = [
    # Amígdalas (singular/plural)
    (re.compile(r"\bmigda\b", re.IGNORECASE), "amígdala"),
    (re.compile(r"\bmigdalas\b", re.IGNORECASE), "amígdalas"),
    (re.compile(r"\bamigdala\b", re.IGNORECASE), "amígdala"),
    (re.compile(r"\bamigdalas\b", re.IGNORECASE), "amígdalas"),

    # Faringe / Orofaringe / Nasofaringe
    (re.compile(r"\bfaringeo\b", re.IGNORECASE), "faríngeo"),
    (re.compile(r"\bfaringea\b", re.IGNORECASE), "faríngea"),
    (re.compile(r"\borofaringeo\b", re.IGNORECASE), "orofaríngeo"),
    (re.compile(r"\borofaringea\b", re.IGNORECASE), "orofaríngea"),

    # separaciones STT (usar \s+)
    (re.compile(r"\boro\s+faringe\b", re.IGNORECASE), "orofaringe"),
    (re.compile(r"\bnaso\s+faringe\b", re.IGNORECASE), "nasofaringe"),

    # nasofaringeo/a (acentos)
    (re.compile(r"\bnasofaringeo\b", re.IGNORECASE), "nasofaríngeo"),
    (re.compile(r"\bnasofaringea\b", re.IGNORECASE), "nasofaríngea"),

    # Otología
    (re.compile(r"\botalguia\b", re.IGNORECASE), "otalgia"),
    (re.compile(r"\botalgía\b", re.IGNORECASE), "otalgia"),
    (re.compile(r"\bhipocusia\b", re.IGNORECASE), "hipoacusia"),
    (re.compile(r"\bhipo\s+acusia\b", re.IGNORECASE), "hipoacusia"),
    (re.compile(r"\boto\s+rrea\b", re.IGNORECASE), "otorrea"),
    (re.compile(r"\botorréa\b", re.IGNORECASE), "otorrea"),
    (re.compile(r"\bacufeno\b", re.IGNORECASE), "acúfeno"),
    (re.compile(r"\bacufenos\b", re.IGNORECASE), "acúfenos"),
    (re.compile(r"\btinitus\b", re.IGNORECASE), "tinnitus"),
    (re.compile(r"\btimpanica\b", re.IGNORECASE), "timpánica"),
    (re.compile(r"\btimpanico\b", re.IGNORECASE), "timpánico"),

    # Laringe / Voz
    (re.compile(r"\bdisfonia\b", re.IGNORECASE), "disfonía"),
    (re.compile(r"\bafonia\b", re.IGNORECASE), "afonía"),

    # Vestibular / General ORL
    (re.compile(r"\bvertigo\b", re.IGNORECASE), "vértigo"),

    # Rinología (separación STT)
    (re.compile(r"\brinoria\b", re.IGNORECASE), "rinorrea"),
    (re.compile(r"\brino\s+rrea\b", re.IGNORECASE), "rinorrea"),

    # Exudado
    (re.compile(r"\bexsudado\b", re.IGNORECASE), "exudado"),
]


def normalize_transcript_orl(transcript: Transcript) -> tuple[Transcript, int]:
    """
    Aplica una whitelist de correcciones ortográficas/fonéticas ORL al transcript.

    Args:
        transcript: Objeto Transcript original (no se modifica in-place).

    Returns:
        tuple[Transcript, int]:
            - Nuevo objeto Transcript normalizado.
            - Número total de reemplazos realizados.

    PHI-Safe: No loguea nada.
    """
    normalized_transcript = transcript.model_copy(deep=True)
    total_replacements = 0

    for segment in normalized_transcript.segments:
        current_text = segment.text
        replacements_in_segment = 0

        for pattern, replacement in ORL_STT_WHITELIST:
            current_text, count = pattern.subn(replacement, current_text)
            replacements_in_segment += count

        if replacements_in_segment > 0:
            segment.text = current_text
            total_replacements += replacements_in_segment

    return normalized_transcript, total_replacements
