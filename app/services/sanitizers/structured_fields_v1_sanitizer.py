"""
Epic 16.2: Sanitizer for StructuredFieldsV1.

Deterministic sanitization rules:
- Trim whitespace, collapse multiple spaces
- Convert garbage values to None (case-insensitive)
- Empty strings become None
- Objects with all-None fields remain as empty objects (schema compatibility)
- Diagnostico with garbage texto becomes None

Compatible with structured_fields_schema_v1.dart (Flutter).
PHI-safe: No logging of field values.
"""
import re
from typing import Optional, Set

from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1,
    Antecedentes,
    ExploracionFisica,
    Diagnostico,
)


# Garbage values to convert to None (case-insensitive, after trim)
GARBAGE_VALUES: Set[str] = {
    "no sé",
    "no se",
    "nose",
    "n/a",
    "na",
    "-",
    "--",
    "---",
    "sin datos",
    "ninguno",
    "ninguna",
    "no refiere",
    "no ref",
    "s/d",
    "sd",
    "sin información",
    "sin informacion",
    "no especificado",
    "no especifica",
    "no aplica",
    "no hay",
    "nada",
    "null",
    "none",
    "undefined",
    ".",
    "..",
    "...",
    "x",
    "xx",
    "xxx",
}

# Regex to collapse multiple whitespace into single space
_WHITESPACE_REGEX = re.compile(r"\s+")


def sanitize_string_field(value: Optional[str]) -> Optional[str]:
    """
    Sanitize a single string field.

    Rules:
    1. If None or empty, return None
    2. Trim whitespace
    3. Collapse multiple spaces to single space
    4. If matches garbage value (case-insensitive), return None
    5. If empty after processing, return None

    Args:
        value: Raw string value (may contain garbage)

    Returns:
        Sanitized string or None
    """
    if value is None:
        return None

    # Trim and collapse whitespace
    cleaned = _WHITESPACE_REGEX.sub(" ", value.strip())

    # Check if empty
    if not cleaned:
        return None

    # Check garbage values (case-insensitive)
    if cleaned.casefold() in {g.casefold() for g in GARBAGE_VALUES}:
        return None

    return cleaned


def _sanitize_antecedentes(antecedentes: Antecedentes) -> Antecedentes:
    """
    Sanitize Antecedentes object.

    Returns Antecedentes with sanitized fields.
    If all fields are None after sanitization, returns empty Antecedentes.
    """
    return Antecedentes(
        heredofamiliares=sanitize_string_field(antecedentes.heredofamiliares),
        personalesNoPatologicos=sanitize_string_field(antecedentes.personales_no_patologicos),
        personalesPatologicos=sanitize_string_field(antecedentes.personales_patologicos),
    )


def _sanitize_exploracion_fisica(exploracion: ExploracionFisica) -> ExploracionFisica:
    """
    Sanitize ExploracionFisica object.

    Returns ExploracionFisica with sanitized fields.
    If all fields are None after sanitization, returns empty ExploracionFisica.
    """
    return ExploracionFisica(
        signosVitales=sanitize_string_field(exploracion.signos_vitales),
        rinoscopia=sanitize_string_field(exploracion.rinoscopia),
        orofaringe=sanitize_string_field(exploracion.orofaringe),
        cuello=sanitize_string_field(exploracion.cuello),
        laringoscopia=sanitize_string_field(exploracion.laringoscopia),
        otoscopia=sanitize_string_field(exploracion.otoscopia),
        otomicroscopia=sanitize_string_field(exploracion.otomicroscopia),
        endoscopiaNasal=sanitize_string_field(exploracion.endoscopia_nasal),
    )


def _sanitize_diagnostico(diagnostico: Optional[Diagnostico]) -> Optional[Diagnostico]:
    """
    Sanitize Diagnostico object.

    If texto becomes garbage/None after sanitization, returns None.
    texto is required for Diagnostico to be valid.
    """
    if diagnostico is None:
        return None

    texto_sanitized = sanitize_string_field(diagnostico.texto)

    # If texto is garbage, entire diagnostico is invalid
    if texto_sanitized is None:
        return None

    return Diagnostico(
        texto=texto_sanitized,
        tipo=diagnostico.tipo,
        cie10=sanitize_string_field(diagnostico.cie10),
    )


def sanitize_structured_fields_v1(fields: StructuredFieldsV1) -> StructuredFieldsV1:
    """
    Sanitize all fields in StructuredFieldsV1.

    Applies deterministic sanitization rules:
    - Trim whitespace, collapse spaces
    - Convert garbage values to None
    - Empty strings become None
    - Nested objects are recursively sanitized

    Args:
        fields: Raw StructuredFieldsV1 (may contain garbage)

    Returns:
        Sanitized StructuredFieldsV1 (same shape, compatible with Flutter)

    Note:
        Does NOT modify the input object. Returns a new instance.
        Does NOT change the public contract - only cleans values.
    """
    return StructuredFieldsV1(
        motivoConsulta=sanitize_string_field(fields.motivo_consulta),
        padecimientoActual=sanitize_string_field(fields.padecimiento_actual),
        antecedentes=_sanitize_antecedentes(fields.antecedentes),
        exploracionFisica=_sanitize_exploracion_fisica(fields.exploracion_fisica),
        diagnostico=_sanitize_diagnostico(fields.diagnostico),
        planTratamiento=sanitize_string_field(fields.plan_tratamiento),
        pronostico=sanitize_string_field(fields.pronostico),
        estudiosIndicados=sanitize_string_field(fields.estudios_indicados),
        notasAdicionales=sanitize_string_field(fields.notas_adicionales),
    )
