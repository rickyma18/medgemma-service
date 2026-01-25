"""
Unit tests for Epic 16 Reducer V2.
Tests deterministic reduction, deduplication, conflict detection, and field strategies.
"""
import pytest
from app.services.reducer_v2 import (
    reduce_chunk_fields_v2,
    reduce_to_final,
    MergeStrategy,
    FIELD_STRATEGIES,
    _merge_strings_concat_dedupe,
    _merge_strings_prefer_first,
    _merge_strings_prefer_last,
    _normalize_for_dedupe,
    _detect_conflict,
    IntermediateResult,
)
from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1,
    Diagnostico,
    Antecedentes,
    ExploracionFisica,
)


class TestNormalization:
    """Tests for string normalization."""

    def test_normalize_strips_whitespace(self):
        assert _normalize_for_dedupe("  hello  ") == "hello"

    def test_normalize_casefold(self):
        assert _normalize_for_dedupe("HELLO") == "hello"
        assert _normalize_for_dedupe("Hola MUNDO") == "hola mundo"

    def test_normalize_empty(self):
        assert _normalize_for_dedupe("") == ""
        assert _normalize_for_dedupe(None) == ""


class TestMergeStrategies:
    """Tests for individual merge strategies."""

    def test_concat_dedupe_basic(self):
        result = _merge_strings_concat_dedupe(["A", "B", "C"])
        assert result == "A | B | C"

    def test_concat_dedupe_removes_duplicates(self):
        result = _merge_strings_concat_dedupe(["dolor", "Dolor", "DOLOR"])
        assert result == "dolor"  # First occurrence preserved

    def test_concat_dedupe_preserves_order(self):
        result = _merge_strings_concat_dedupe(["primero", "segundo", "tercero"])
        assert result == "primero | segundo | tercero"

    def test_concat_dedupe_filters_none_and_empty(self):
        result = _merge_strings_concat_dedupe([None, "A", "", "B", None])
        assert result == "A | B"

    def test_concat_dedupe_all_empty_returns_none(self):
        result = _merge_strings_concat_dedupe([None, "", "  "])
        assert result is None

    def test_prefer_first_takes_first_valid(self):
        result = _merge_strings_prefer_first([None, "", "first", "second"])
        assert result == "first"

    def test_prefer_first_all_none(self):
        result = _merge_strings_prefer_first([None, None, ""])
        assert result is None

    def test_prefer_last_takes_last_valid(self):
        result = _merge_strings_prefer_last(["first", "second", None, ""])
        assert result == "second"

    def test_prefer_last_all_none(self):
        result = _merge_strings_prefer_last([None, "", None])
        assert result is None


class TestConflictDetection:
    """Tests for conflict detection."""

    def test_no_conflict_single_value(self):
        assert not _detect_conflict(["only one"], "only one")

    def test_no_conflict_all_same_normalized(self):
        assert not _detect_conflict(["Same", "SAME", "same"], "Same")

    def test_conflict_different_values(self):
        assert _detect_conflict(["Faringitis", "Amigdalitis"], "Faringitis | Amigdalitis")

    def test_no_conflict_with_none(self):
        assert not _detect_conflict([None, "value", None], "value")

    def test_no_conflict_empty_list(self):
        assert not _detect_conflict([], None)


class TestReduceChunkFieldsV2:
    """Tests for the main reducer function."""

    def test_empty_input_returns_empty(self):
        result = reduce_chunk_fields_v2([])
        assert isinstance(result, IntermediateResult)
        assert result.chunk_count == 0
        assert result.conflicts == []

    def test_single_input_passthrough(self):
        fields = StructuredFieldsV1(
            motivoConsulta="dolor de garganta",
            diagnostico=Diagnostico(texto="Faringitis", tipo="definitivo")
        )
        result = reduce_chunk_fields_v2([fields])
        assert result.chunk_count == 1
        assert result.fields.motivo_consulta == "dolor de garganta"
        assert not result.has_conflicts()

    def test_motivo_consulta_prefer_first(self):
        """motivoConsulta should use PREFER_FIRST strategy."""
        chunk1 = StructuredFieldsV1(motivoConsulta="primer motivo")
        chunk2 = StructuredFieldsV1(motivoConsulta="segundo motivo")

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        assert result.fields.motivo_consulta == "primer motivo"

    def test_plan_tratamiento_prefer_last(self):
        """planTratamiento should use PREFER_LAST strategy."""
        chunk1 = StructuredFieldsV1(planTratamiento="plan inicial")
        chunk2 = StructuredFieldsV1(planTratamiento="plan final")

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        assert result.fields.plan_tratamiento == "plan final"

    def test_padecimiento_actual_concat_dedupe(self):
        """padecimientoActual should use CONCAT_DEDUPE strategy."""
        chunk1 = StructuredFieldsV1(padecimientoActual="dolor desde hace 3 dias")
        chunk2 = StructuredFieldsV1(padecimientoActual="con fiebre")

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        assert result.fields.padecimiento_actual == "dolor desde hace 3 dias | con fiebre"

    def test_dedupe_removes_exact_duplicates(self):
        chunk1 = StructuredFieldsV1(padecimientoActual="Fiebre de 38.5")
        chunk2 = StructuredFieldsV1(padecimientoActual="fiebre de 38.5")  # Same, different case

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        # Should dedupe (casefold match)
        assert result.fields.padecimiento_actual == "Fiebre de 38.5"

    def test_diagnostico_texto_concat_with_conflict(self):
        """Different diagnostico texts should be merged and marked as conflict."""
        chunk1 = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="Faringitis aguda", tipo="presuntivo")
        )
        chunk2 = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="Amigdalitis", tipo="definitivo")
        )

        result = reduce_chunk_fields_v2([chunk1, chunk2])

        # Texts should be merged
        assert "Faringitis aguda" in result.fields.diagnostico.texto
        assert "Amigdalitis" in result.fields.diagnostico.texto

        # Should detect conflict
        assert result.has_conflicts()
        assert "diagnostico.texto" in result.conflict_paths()

    def test_diagnostico_tipo_highest_certainty(self):
        """Diagnostico tipo should prefer highest certainty."""
        chunk1 = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="dx1", tipo="sindromico")
        )
        chunk2 = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="dx2", tipo="definitivo")
        )
        chunk3 = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="dx3", tipo="presuntivo")
        )

        result = reduce_chunk_fields_v2([chunk1, chunk2, chunk3])
        assert result.fields.diagnostico.tipo == "definitivo"

    def test_diagnostico_cie10_prefer_last(self):
        """CIE10 should use PREFER_LAST strategy."""
        chunk1 = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="dx", tipo="definitivo", cie10="J02.9")
        )
        chunk2 = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="dx", tipo="definitivo", cie10="J03.9")
        )

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        assert result.fields.diagnostico.cie10 == "J03.9"

    def test_exploracion_fisica_concat(self):
        """ExploracionFisica fields should concat."""
        chunk1 = StructuredFieldsV1(
            exploracionFisica=ExploracionFisica(orofaringe="eritema")
        )
        chunk2 = StructuredFieldsV1(
            exploracionFisica=ExploracionFisica(orofaringe="exudado amigdalino")
        )

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        assert "eritema" in result.fields.exploracion_fisica.orofaringe
        assert "exudado amigdalino" in result.fields.exploracion_fisica.orofaringe

    def test_antecedentes_concat(self):
        """Antecedentes fields should concat."""
        chunk1 = StructuredFieldsV1(
            antecedentes=Antecedentes(heredofamiliares="madre con HTA")
        )
        chunk2 = StructuredFieldsV1(
            antecedentes=Antecedentes(heredofamiliares="padre con DM2")
        )

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        assert "madre con HTA" in result.fields.antecedentes.heredofamiliares
        assert "padre con DM2" in result.fields.antecedentes.heredofamiliares


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self):
        """Same input should always produce same output."""
        chunk1 = StructuredFieldsV1(motivoConsulta="A", padecimientoActual="X")
        chunk2 = StructuredFieldsV1(motivoConsulta="B", padecimientoActual="Y")

        result1 = reduce_chunk_fields_v2([chunk1, chunk2])
        result2 = reduce_chunk_fields_v2([chunk1, chunk2])

        assert result1.fields.motivo_consulta == result2.fields.motivo_consulta
        assert result1.fields.padecimiento_actual == result2.fields.padecimiento_actual
        assert result1.conflict_count() == result2.conflict_count()

    def test_order_matters_for_prefer_strategies(self):
        """Order should matter for PREFER_FIRST and PREFER_LAST."""
        chunk1 = StructuredFieldsV1(planTratamiento="plan A")
        chunk2 = StructuredFieldsV1(planTratamiento="plan B")

        # Order 1
        result1 = reduce_chunk_fields_v2([chunk1, chunk2])
        # Order 2
        result2 = reduce_chunk_fields_v2([chunk2, chunk1])

        # PREFER_LAST for planTratamiento
        assert result1.fields.plan_tratamiento == "plan B"
        assert result2.fields.plan_tratamiento == "plan A"

    def test_stable_dedupe_order(self):
        """Dedupe should preserve order of first occurrence."""
        chunk1 = StructuredFieldsV1(padecimientoActual="primero")
        chunk2 = StructuredFieldsV1(padecimientoActual="segundo")
        chunk3 = StructuredFieldsV1(padecimientoActual="PRIMERO")  # Duplicate

        result = reduce_chunk_fields_v2([chunk1, chunk2, chunk3])
        assert result.fields.padecimiento_actual == "primero | segundo"


class TestEdgeCases:
    """Edge case tests."""

    def test_all_none_fields(self):
        chunk1 = StructuredFieldsV1()
        chunk2 = StructuredFieldsV1()

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        assert result.fields.motivo_consulta is None
        assert result.fields.diagnostico is None

    def test_mixed_none_and_values(self):
        chunk1 = StructuredFieldsV1(motivoConsulta=None)
        chunk2 = StructuredFieldsV1(motivoConsulta="motivo real")

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        # PREFER_FIRST should skip None
        assert result.fields.motivo_consulta == "motivo real"

    def test_whitespace_only_treated_as_none(self):
        chunk1 = StructuredFieldsV1(motivoConsulta="   ")
        chunk2 = StructuredFieldsV1(motivoConsulta="motivo real")

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        assert result.fields.motivo_consulta == "motivo real"

    def test_no_diagnostico_returns_none(self):
        chunk1 = StructuredFieldsV1()
        chunk2 = StructuredFieldsV1()

        result = reduce_chunk_fields_v2([chunk1, chunk2])
        assert result.fields.diagnostico is None

    def test_conflict_count_phi_safe(self):
        """Conflict count should be PHI-safe (no content)."""
        chunk1 = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="Secreto 1", tipo="definitivo")
        )
        chunk2 = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="Secreto 2", tipo="definitivo")
        )

        result = reduce_chunk_fields_v2([chunk1, chunk2])

        # Should be able to get count without exposing values
        count = result.conflict_count()
        paths = result.conflict_paths()

        assert isinstance(count, int)
        assert isinstance(paths, list)
        # Paths don't contain PHI, just field names
        assert all(isinstance(p, str) for p in paths)


class TestReduceToFinal:
    """Tests for convenience function."""

    def test_reduce_to_final_returns_only_fields(self):
        chunk1 = StructuredFieldsV1(motivoConsulta="test")
        chunk2 = StructuredFieldsV1(motivoConsulta="test 2")

        result = reduce_to_final([chunk1, chunk2])

        assert isinstance(result, StructuredFieldsV1)
        # No intermediate info exposed
        assert not hasattr(result, "conflicts")


class TestFieldStrategiesConfig:
    """Tests for field strategy configuration."""

    def test_all_top_level_fields_have_strategy(self):
        """All top-level string fields should have defined strategies."""
        top_level_fields = [
            "motivo_consulta", "padecimiento_actual", "plan_tratamiento",
            "pronostico", "estudios_indicados", "notas_adicionales"
        ]
        for field in top_level_fields:
            assert field in FIELD_STRATEGIES, f"Missing strategy for {field}"

    def test_all_antecedentes_fields_have_strategy(self):
        """All antecedentes fields should have defined strategies."""
        antecedentes_fields = [
            "heredofamiliares", "personales_no_patologicos", "personales_patologicos"
        ]
        for field in antecedentes_fields:
            assert field in FIELD_STRATEGIES, f"Missing strategy for {field}"

    def test_all_exploracion_fields_have_strategy(self):
        """All exploracion fields should have defined strategies."""
        exploracion_fields = [
            "signos_vitales", "rinoscopia", "orofaringe", "cuello",
            "laringoscopia", "otoscopia", "otomicroscopia", "endoscopia_nasal"
        ]
        for field in exploracion_fields:
            assert field in FIELD_STRATEGIES, f"Missing strategy for {field}"
