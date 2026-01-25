"""
Epic 16: Deterministic Reducer V2 for ORL Pipeline.

Improved reduction with:
- Field-specific merge strategies (concat, prefer_first, prefer_last)
- Stable deduplication (trim + casefold normalization)
- Conflict detection for finalize stage
- Intermediate representation with internal flags (not exposed to client)

PHI-safe: Never log field values, only conflict counts.
"""
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field

from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1,
    ExploracionFisica,
    Antecedentes,
    Diagnostico,
)


class MergeStrategy(str, Enum):
    """Merge strategies for string fields."""
    CONCAT_DEDUPE = "concat_dedupe"  # Join with " | ", dedupe by casefold
    PREFER_FIRST = "prefer_first"    # Take first non-null value
    PREFER_LAST = "prefer_last"      # Take last non-null value


# Field-specific merge strategies
FIELD_STRATEGIES: Dict[str, MergeStrategy] = {
    # Top-level fields
    "motivo_consulta": MergeStrategy.PREFER_FIRST,
    "padecimiento_actual": MergeStrategy.CONCAT_DEDUPE,
    "plan_tratamiento": MergeStrategy.PREFER_LAST,
    "pronostico": MergeStrategy.PREFER_LAST,
    "estudios_indicados": MergeStrategy.CONCAT_DEDUPE,
    "notas_adicionales": MergeStrategy.CONCAT_DEDUPE,
    # Antecedentes sub-fields (all concat)
    "heredofamiliares": MergeStrategy.CONCAT_DEDUPE,
    "personales_no_patologicos": MergeStrategy.CONCAT_DEDUPE,
    "personales_patologicos": MergeStrategy.CONCAT_DEDUPE,
    # ExploracionFisica sub-fields (all concat)
    "signos_vitales": MergeStrategy.CONCAT_DEDUPE,
    "rinoscopia": MergeStrategy.CONCAT_DEDUPE,
    "orofaringe": MergeStrategy.CONCAT_DEDUPE,
    "cuello": MergeStrategy.CONCAT_DEDUPE,
    "laringoscopia": MergeStrategy.CONCAT_DEDUPE,
    "otoscopia": MergeStrategy.CONCAT_DEDUPE,
    "otomicroscopia": MergeStrategy.CONCAT_DEDUPE,
    "endoscopia_nasal": MergeStrategy.CONCAT_DEDUPE,
    # Diagnostico sub-fields
    "diagnostico_texto": MergeStrategy.CONCAT_DEDUPE,
    "diagnostico_cie10": MergeStrategy.PREFER_LAST,
}

SEPARATOR = " | "


@dataclass
class ConflictMarker:
    """Internal marker for field conflicts (not exposed to client)."""
    field_path: str
    values: List[str]
    resolved_value: str


@dataclass
class IntermediateResult:
    """
    Internal intermediate representation with conflict markers.

    This is used between reduce and finalize stages.
    NOT exposed to client - only the final StructuredFieldsV1 is returned.
    """
    fields: StructuredFieldsV1
    conflicts: List[ConflictMarker] = field(default_factory=list)
    chunk_count: int = 0

    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    def conflict_count(self) -> int:
        """PHI-safe: returns count only, not values."""
        return len(self.conflicts)

    def conflict_paths(self) -> List[str]:
        """PHI-safe: returns field paths only."""
        return [c.field_path for c in self.conflicts]


def _normalize_for_dedupe(s: str) -> str:
    """Normalize string for deduplication: trim + casefold."""
    return s.strip().casefold() if s else ""


def _merge_strings_concat_dedupe(values: List[Optional[str]]) -> Optional[str]:
    """
    Merge strings using CONCAT_DEDUPE strategy.
    - Filter None/empty
    - Dedupe using casefold normalization
    - Preserve original case of first occurrence
    - Stable order (input order preserved)
    """
    valid = [v.strip() for v in values if v and v.strip()]
    if not valid:
        return None

    seen: Set[str] = set()
    unique: List[str] = []

    for val in valid:
        norm = _normalize_for_dedupe(val)
        if norm not in seen:
            seen.add(norm)
            unique.append(val)

    return SEPARATOR.join(unique) if unique else None


def _merge_strings_prefer_first(values: List[Optional[str]]) -> Optional[str]:
    """Take first non-null, non-empty value."""
    for v in values:
        if v and v.strip():
            return v.strip()
    return None


def _merge_strings_prefer_last(values: List[Optional[str]]) -> Optional[str]:
    """Take last non-null, non-empty value."""
    for v in reversed(values):
        if v and v.strip():
            return v.strip()
    return None


def _merge_string_field(
    values: List[Optional[str]],
    strategy: MergeStrategy
) -> Optional[str]:
    """Merge string values using specified strategy."""
    if strategy == MergeStrategy.CONCAT_DEDUPE:
        return _merge_strings_concat_dedupe(values)
    elif strategy == MergeStrategy.PREFER_FIRST:
        return _merge_strings_prefer_first(values)
    elif strategy == MergeStrategy.PREFER_LAST:
        return _merge_strings_prefer_last(values)
    else:
        # Default fallback
        return _merge_strings_concat_dedupe(values)


def _detect_conflict(values: List[Optional[str]], merged: Optional[str]) -> bool:
    """
    Detect if there's a meaningful conflict in values.
    A conflict exists when:
    - Multiple distinct non-null values exist
    - They don't normalize to the same string
    """
    valid = [v.strip() for v in values if v and v.strip()]
    if len(valid) <= 1:
        return False

    # Check if all values normalize to the same thing
    normalized = set(_normalize_for_dedupe(v) for v in valid)
    return len(normalized) > 1


def _merge_antecedentes_v2(
    items: List[Antecedentes],
    conflicts: List[ConflictMarker]
) -> Antecedentes:
    """Merge Antecedentes with field-specific strategies."""
    if not items:
        return Antecedentes()

    field_map = {
        "heredofamiliares": "heredofamiliares",
        "personales_no_patologicos": "personales_no_patologicos",
        "personales_patologicos": "personales_patologicos",
    }

    merged = Antecedentes()

    for attr, field_name in field_map.items():
        values = [getattr(item, attr) for item in items]
        strategy = FIELD_STRATEGIES.get(field_name, MergeStrategy.CONCAT_DEDUPE)
        result = _merge_string_field(values, strategy)
        setattr(merged, attr, result)

        # Check for conflict
        if _detect_conflict(values, result):
            conflicts.append(ConflictMarker(
                field_path=f"antecedentes.{field_name}",
                values=[v for v in values if v],
                resolved_value=result or ""
            ))

    return merged


def _merge_exploracion_v2(
    items: List[ExploracionFisica],
    conflicts: List[ConflictMarker]
) -> ExploracionFisica:
    """Merge ExploracionFisica with field-specific strategies."""
    if not items:
        return ExploracionFisica()

    field_names = [
        "signos_vitales", "rinoscopia", "orofaringe", "cuello",
        "laringoscopia", "otoscopia", "otomicroscopia", "endoscopia_nasal"
    ]

    merged = ExploracionFisica()

    for field_name in field_names:
        values = [getattr(item, field_name) for item in items]
        strategy = FIELD_STRATEGIES.get(field_name, MergeStrategy.CONCAT_DEDUPE)
        result = _merge_string_field(values, strategy)
        setattr(merged, field_name, result)

        # Check for conflict
        if _detect_conflict(values, result):
            conflicts.append(ConflictMarker(
                field_path=f"exploracionFisica.{field_name}",
                values=[v for v in values if v],
                resolved_value=result or ""
            ))

    return merged


def _merge_diagnostico_v2(
    items: List[Optional[Diagnostico]],
    conflicts: List[ConflictMarker]
) -> Optional[Diagnostico]:
    """
    Merge Diagnostico with special handling.
    - texto: CONCAT_DEDUPE + conflict detection
    - tipo: HIGHEST_CERTAINTY (definitivo > presuntivo > sindromico)
    - cie10: PREFER_LAST
    """
    valid_items = [x for x in items if x]
    if not valid_items:
        return None

    # Merge texto
    textos = [x.texto for x in valid_items]
    final_texto = _merge_string_field(textos, MergeStrategy.CONCAT_DEDUPE)

    if not final_texto:
        return None

    # Detect texto conflict
    if _detect_conflict(textos, final_texto):
        conflicts.append(ConflictMarker(
            field_path="diagnostico.texto",
            values=textos,
            resolved_value=final_texto
        ))

    # Merge tipo (highest certainty)
    priorities = {"definitivo": 3, "presuntivo": 2, "sindromico": 1}
    best_tipo = "sindromico"
    max_prio = 0

    for x in valid_items:
        p = priorities.get(x.tipo, 0)
        if p > max_prio:
            max_prio = p
            best_tipo = x.tipo

    # Merge cie10 (prefer last)
    cies = [x.cie10 for x in valid_items]
    final_cie = _merge_string_field(cies, MergeStrategy.PREFER_LAST)

    return Diagnostico(
        texto=final_texto,
        tipo=best_tipo,  # type: ignore
        cie10=final_cie
    )


def reduce_chunk_fields_v2(
    results: List[StructuredFieldsV1]
) -> IntermediateResult:
    """
    Reduce multiple StructuredFieldsV1 into one with conflict tracking.

    This is the Epic 16 improved reducer that:
    - Uses field-specific merge strategies
    - Detects and marks conflicts for finalize
    - Returns intermediate result with conflict metadata

    Args:
        results: List of StructuredFieldsV1 from chunks, ordered by chunk index

    Returns:
        IntermediateResult with merged fields and conflict markers

    Note: Conflicts are internal - not exposed to client.
    """
    if not results:
        return IntermediateResult(
            fields=StructuredFieldsV1(),
            conflicts=[],
            chunk_count=0
        )

    if len(results) == 1:
        return IntermediateResult(
            fields=results[0],
            conflicts=[],
            chunk_count=1
        )

    conflicts: List[ConflictMarker] = []

    # Merge top-level string fields
    def merge_top_field(field_name: str) -> Optional[str]:
        values = [getattr(r, field_name) for r in results]
        strategy = FIELD_STRATEGIES.get(field_name, MergeStrategy.CONCAT_DEDUPE)
        result = _merge_string_field(values, strategy)

        if _detect_conflict(values, result):
            conflicts.append(ConflictMarker(
                field_path=field_name,
                values=[v for v in values if v],
                resolved_value=result or ""
            ))

        return result

    merged = StructuredFieldsV1(
        motivoConsulta=merge_top_field("motivo_consulta"),
        padecimientoActual=merge_top_field("padecimiento_actual"),
        antecedentes=_merge_antecedentes_v2(
            [r.antecedentes for r in results],
            conflicts
        ),
        exploracionFisica=_merge_exploracion_v2(
            [r.exploracion_fisica for r in results],
            conflicts
        ),
        diagnostico=_merge_diagnostico_v2(
            [r.diagnostico for r in results],
            conflicts
        ),
        planTratamiento=merge_top_field("plan_tratamiento"),
        pronostico=merge_top_field("pronostico"),
        estudiosIndicados=merge_top_field("estudios_indicados"),
        notasAdicionales=merge_top_field("notas_adicionales"),
    )

    return IntermediateResult(
        fields=merged,
        conflicts=conflicts,
        chunk_count=len(results)
    )


def reduce_to_final(results: List[StructuredFieldsV1]) -> StructuredFieldsV1:
    """
    Convenience function: reduce and return only final fields.
    Discards conflict info (for backward compatibility).

    Epic 16.2: Applies sanitization before returning.
    """
    from app.services.sanitizers.structured_fields_v1_sanitizer import (
        sanitize_structured_fields_v1,
    )

    intermediate = reduce_chunk_fields_v2(results)
    return sanitize_structured_fields_v1(intermediate.fields)
