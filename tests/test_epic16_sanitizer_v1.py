"""
Unit tests for Epic 16.2 Sanitizer for StructuredFieldsV1.
Tests garbage value removal, whitespace handling, and schema compatibility.
"""
import pytest
from app.services.sanitizers.structured_fields_v1_sanitizer import (
    sanitize_string_field,
    sanitize_structured_fields_v1,
    GARBAGE_VALUES,
    _sanitize_antecedentes,
    _sanitize_exploracion_fisica,
    _sanitize_diagnostico,
)
from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1,
    Diagnostico,
    Antecedentes,
    ExploracionFisica,
)


class TestSanitizeStringField:
    """Tests for individual string field sanitization."""

    def test_none_returns_none(self):
        assert sanitize_string_field(None) is None

    def test_empty_string_returns_none(self):
        assert sanitize_string_field("") is None

    def test_whitespace_only_returns_none(self):
        assert sanitize_string_field("   ") is None
        assert sanitize_string_field("\t\n") is None

    def test_trims_whitespace(self):
        assert sanitize_string_field("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert sanitize_string_field("hello    world") == "hello world"
        assert sanitize_string_field("a   b   c") == "a b c"

    def test_valid_value_unchanged(self):
        assert sanitize_string_field("dolor de garganta") == "dolor de garganta"

    def test_preserves_case(self):
        assert sanitize_string_field("Faringitis Aguda") == "Faringitis Aguda"


class TestGarbageValues:
    """Tests for garbage value detection and removal."""

    def test_no_se_variants(self):
        assert sanitize_string_field("no sé") is None
        assert sanitize_string_field("no se") is None
        assert sanitize_string_field("nose") is None
        assert sanitize_string_field("NO SÉ") is None  # Case insensitive

    def test_na_variants(self):
        assert sanitize_string_field("n/a") is None
        assert sanitize_string_field("N/A") is None
        assert sanitize_string_field("na") is None

    def test_dashes(self):
        assert sanitize_string_field("-") is None
        assert sanitize_string_field("--") is None
        assert sanitize_string_field("---") is None

    def test_sin_datos_variants(self):
        assert sanitize_string_field("sin datos") is None
        assert sanitize_string_field("Sin Datos") is None
        assert sanitize_string_field("sin información") is None
        assert sanitize_string_field("sin informacion") is None

    def test_ninguno_variants(self):
        assert sanitize_string_field("ninguno") is None
        assert sanitize_string_field("ninguna") is None

    def test_no_refiere_variants(self):
        assert sanitize_string_field("no refiere") is None
        assert sanitize_string_field("no ref") is None
        assert sanitize_string_field("No Refiere") is None

    def test_sd_variants(self):
        assert sanitize_string_field("s/d") is None
        assert sanitize_string_field("sd") is None
        assert sanitize_string_field("S/D") is None

    def test_no_especificado_variants(self):
        assert sanitize_string_field("no especificado") is None
        assert sanitize_string_field("no especifica") is None

    def test_no_aplica(self):
        assert sanitize_string_field("no aplica") is None
        assert sanitize_string_field("No Aplica") is None

    def test_no_hay_nada(self):
        assert sanitize_string_field("no hay") is None
        assert sanitize_string_field("nada") is None

    def test_null_none_undefined(self):
        assert sanitize_string_field("null") is None
        assert sanitize_string_field("none") is None
        assert sanitize_string_field("undefined") is None
        assert sanitize_string_field("NULL") is None

    def test_dots(self):
        assert sanitize_string_field(".") is None
        assert sanitize_string_field("..") is None
        assert sanitize_string_field("...") is None

    def test_x_variants(self):
        assert sanitize_string_field("x") is None
        assert sanitize_string_field("xx") is None
        assert sanitize_string_field("xxx") is None
        assert sanitize_string_field("X") is None

    def test_garbage_with_whitespace(self):
        # Should trim then detect garbage
        assert sanitize_string_field("  n/a  ") is None
        assert sanitize_string_field("  -  ") is None

    def test_valid_similar_to_garbage_preserved(self):
        # Make sure we don't falsely match valid clinical text
        assert sanitize_string_field("nada relevante") == "nada relevante"
        assert sanitize_string_field("sin dolor") == "sin dolor"
        assert sanitize_string_field("no refiere dolor") == "no refiere dolor"


class TestSanitizeAntecedentes:
    """Tests for Antecedentes sanitization."""

    def test_empty_antecedentes(self):
        result = _sanitize_antecedentes(Antecedentes())
        assert result.heredofamiliares is None
        assert result.personales_no_patologicos is None
        assert result.personales_patologicos is None

    def test_all_garbage_returns_all_none(self):
        ant = Antecedentes(
            heredofamiliares="n/a",
            personalesNoPatologicos="sin datos",
            personalesPatologicos="-"
        )
        result = _sanitize_antecedentes(ant)
        assert result.heredofamiliares is None
        assert result.personales_no_patologicos is None
        assert result.personales_patologicos is None

    def test_mixed_values(self):
        ant = Antecedentes(
            heredofamiliares="madre con HTA",
            personalesNoPatologicos="n/a",
            personalesPatologicos="DM2 hace 5 años"
        )
        result = _sanitize_antecedentes(ant)
        assert result.heredofamiliares == "madre con HTA"
        assert result.personales_no_patologicos is None
        assert result.personales_patologicos == "DM2 hace 5 años"


class TestSanitizeExploracionFisica:
    """Tests for ExploracionFisica sanitization."""

    def test_empty_exploracion(self):
        result = _sanitize_exploracion_fisica(ExploracionFisica())
        assert result.signos_vitales is None
        assert result.rinoscopia is None
        assert result.orofaringe is None

    def test_all_garbage_returns_all_none(self):
        exp = ExploracionFisica(
            signosVitales="sin datos",
            rinoscopia="---",
            orofaringe="x",
            cuello="n/a"
        )
        result = _sanitize_exploracion_fisica(exp)
        assert result.signos_vitales is None
        assert result.rinoscopia is None
        assert result.orofaringe is None
        assert result.cuello is None

    def test_mixed_values(self):
        exp = ExploracionFisica(
            signosVitales="TA 120/80, FC 72",
            rinoscopia="n/a",
            orofaringe="eritema en faringe posterior",
            cuello="sin adenopatías"
        )
        result = _sanitize_exploracion_fisica(exp)
        assert result.signos_vitales == "TA 120/80, FC 72"
        assert result.rinoscopia is None
        assert result.orofaringe == "eritema en faringe posterior"
        assert result.cuello == "sin adenopatías"


class TestSanitizeDiagnostico:
    """Tests for Diagnostico sanitization."""

    def test_none_returns_none(self):
        assert _sanitize_diagnostico(None) is None

    def test_garbage_texto_returns_none(self):
        """If texto is garbage, entire diagnostico is invalid."""
        dx = Diagnostico(texto="n/a", tipo="definitivo", cie10="J02.9")
        assert _sanitize_diagnostico(dx) is None

    def test_valid_texto_preserved(self):
        dx = Diagnostico(texto="Faringitis aguda", tipo="definitivo", cie10="J02.9")
        result = _sanitize_diagnostico(dx)
        assert result is not None
        assert result.texto == "Faringitis aguda"
        assert result.tipo == "definitivo"
        assert result.cie10 == "J02.9"

    def test_garbage_cie10_becomes_none(self):
        dx = Diagnostico(texto="Faringitis", tipo="definitivo", cie10="n/a")
        result = _sanitize_diagnostico(dx)
        assert result is not None
        assert result.texto == "Faringitis"
        assert result.cie10 is None

    def test_tipo_preserved(self):
        """Tipo should not be sanitized (enum value)."""
        dx = Diagnostico(texto="Amigdalitis", tipo="presuntivo")
        result = _sanitize_diagnostico(dx)
        assert result.tipo == "presuntivo"


class TestSanitizeStructuredFieldsV1:
    """Tests for full StructuredFieldsV1 sanitization."""

    def test_empty_fields(self):
        result = sanitize_structured_fields_v1(StructuredFieldsV1())
        assert result.motivo_consulta is None
        assert result.diagnostico is None
        # Should return valid empty objects (schema compatibility)
        assert isinstance(result.antecedentes, Antecedentes)
        assert isinstance(result.exploracion_fisica, ExploracionFisica)

    def test_all_garbage_fields(self):
        fields = StructuredFieldsV1(
            motivoConsulta="n/a",
            padecimientoActual="sin datos",
            planTratamiento="-",
            pronostico="...",
            estudiosIndicados="x",
            notasAdicionales="null"
        )
        result = sanitize_structured_fields_v1(fields)
        assert result.motivo_consulta is None
        assert result.padecimiento_actual is None
        assert result.plan_tratamiento is None
        assert result.pronostico is None
        assert result.estudios_indicados is None
        assert result.notas_adicionales is None

    def test_mixed_values(self):
        fields = StructuredFieldsV1(
            motivoConsulta="dolor de garganta",
            padecimientoActual="n/a",
            diagnostico=Diagnostico(texto="Faringitis", tipo="definitivo"),
            planTratamiento="ibuprofeno 400mg c/8h"
        )
        result = sanitize_structured_fields_v1(fields)
        assert result.motivo_consulta == "dolor de garganta"
        assert result.padecimiento_actual is None
        assert result.diagnostico.texto == "Faringitis"
        assert result.plan_tratamiento == "ibuprofeno 400mg c/8h"

    def test_nested_objects_sanitized(self):
        fields = StructuredFieldsV1(
            antecedentes=Antecedentes(
                heredofamiliares="madre con HTA",
                personalesNoPatologicos="n/a"
            ),
            exploracionFisica=ExploracionFisica(
                orofaringe="eritema",
                rinoscopia="---"
            )
        )
        result = sanitize_structured_fields_v1(fields)
        assert result.antecedentes.heredofamiliares == "madre con HTA"
        assert result.antecedentes.personales_no_patologicos is None
        assert result.exploracion_fisica.orofaringe == "eritema"
        assert result.exploracion_fisica.rinoscopia is None

    def test_whitespace_collapsed(self):
        fields = StructuredFieldsV1(
            motivoConsulta="  dolor    de   garganta  ",
            padecimientoActual="  inició   hace   3   días  "
        )
        result = sanitize_structured_fields_v1(fields)
        assert result.motivo_consulta == "dolor de garganta"
        assert result.padecimiento_actual == "inició hace 3 días"


class TestSchemaCompatibility:
    """Tests to ensure sanitization doesn't break schema shape."""

    def test_returns_structuredfieldsv1(self):
        result = sanitize_structured_fields_v1(StructuredFieldsV1())
        assert isinstance(result, StructuredFieldsV1)

    def test_empty_objects_not_none(self):
        """Empty nested objects should remain as objects, not None."""
        fields = StructuredFieldsV1(
            antecedentes=Antecedentes(),
            exploracionFisica=ExploracionFisica()
        )
        result = sanitize_structured_fields_v1(fields)
        # Should be empty objects, not None
        assert result.antecedentes is not None
        assert result.exploracion_fisica is not None
        assert isinstance(result.antecedentes, Antecedentes)
        assert isinstance(result.exploracion_fisica, ExploracionFisica)

    def test_can_serialize_to_json(self):
        """Sanitized result should be JSON serializable."""
        fields = StructuredFieldsV1(
            motivoConsulta="dolor",
            diagnostico=Diagnostico(texto="Faringitis", tipo="definitivo")
        )
        result = sanitize_structured_fields_v1(fields)
        # Should not raise
        json_data = result.model_dump()
        assert isinstance(json_data, dict)

    def test_does_not_modify_input(self):
        """Sanitization should return new object, not modify input."""
        original = StructuredFieldsV1(
            motivoConsulta="  spaces  ",
            padecimientoActual="n/a"
        )
        original_motivo = original.motivo_consulta
        original_padecimiento = original.padecimiento_actual

        result = sanitize_structured_fields_v1(original)

        # Original should be unchanged
        assert original.motivo_consulta == original_motivo
        assert original.padecimiento_actual == original_padecimiento
        # Result should be different
        assert result.motivo_consulta == "spaces"
        assert result.padecimiento_actual is None


class TestGarbageValuesSet:
    """Tests for GARBAGE_VALUES constant."""

    def test_garbage_set_not_empty(self):
        assert len(GARBAGE_VALUES) > 0

    def test_common_garbage_included(self):
        # Verify key garbage values are in the set
        expected = ["n/a", "na", "-", "sin datos", "ninguno", "no refiere"]
        for val in expected:
            assert val in GARBAGE_VALUES, f"Missing garbage value: {val}"

    def test_all_lowercase(self):
        """All garbage values should be lowercase for case-insensitive matching."""
        for val in GARBAGE_VALUES:
            assert val == val.lower(), f"Garbage value not lowercase: {val}"
