"""
Tests for scope mask (FIX #8).

Validates:
- _is_effectively_empty: detects truly empty values
- _apply_scope_mask: non-destructive; preserves cross-scope data
- All four scopes: interview, exam, studies, assessment
- Integration via _parse_v1_output with scope
"""
import pytest

from app.services.structured_v1_extractor import (
    _is_effectively_empty,
    _apply_scope_mask,
    _parse_v1_output,
    SCOPE_ALLOWED_FIELDS,
)


# ---------------------------------------------------------------------------
# Unit: _is_effectively_empty
# ---------------------------------------------------------------------------

class TestIsEffectivelyEmpty:

    def test_none_is_empty(self):
        assert _is_effectively_empty(None) is True

    def test_empty_string_is_empty(self):
        assert _is_effectively_empty("") is True

    def test_whitespace_string_is_empty(self):
        assert _is_effectively_empty("   ") is True

    def test_non_empty_string(self):
        assert _is_effectively_empty("Dolor") is False

    def test_empty_dict_is_empty(self):
        assert _is_effectively_empty({}) is True

    def test_dict_all_none_is_empty(self):
        assert _is_effectively_empty({"a": None, "b": None}) is True

    def test_dict_with_value(self):
        assert _is_effectively_empty({"a": None, "b": "data"}) is False

    def test_empty_list_is_empty(self):
        assert _is_effectively_empty([]) is True

    def test_non_empty_list(self):
        assert _is_effectively_empty(["item"]) is False

    def test_number_not_empty(self):
        assert _is_effectively_empty(0) is False

    def test_nested_empty_dicts(self):
        assert _is_effectively_empty({"a": {}, "b": None}) is True

    def test_nested_dict_with_value(self):
        assert _is_effectively_empty({"a": {"x": "data"}}) is False


# ---------------------------------------------------------------------------
# Unit: _apply_scope_mask — interview scope
# ---------------------------------------------------------------------------

class TestScopeMaskInterview:
    """Interview scope: motivoConsulta, padecimientoActual, antecedentes."""

    def test_interview_preserves_core_fields(self):
        data = {
            "motivoConsulta": "Dolor de garganta",
            "padecimientoActual": "Inicio hace 3 dias",
            "antecedentes": {
                "heredofamiliares": "Padre con DM2",
                "personalesNoPatologicos": "Niega tabaquismo",
                "personalesPatologicos": "Alergia a sulfas",
            },
            "exploracionFisica": {},
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "interview")

        assert result["motivoConsulta"] == "Dolor de garganta"
        assert result["padecimientoActual"] == "Inicio hace 3 dias"
        assert result["antecedentes"]["heredofamiliares"] == "Padre con DM2"
        assert result["antecedentes"]["personalesNoPatologicos"] == "Niega tabaquismo"
        assert result["antecedentes"]["personalesPatologicos"] == "Alergia a sulfas"

    def test_interview_preserves_cross_scope_bonus_data(self):
        """If LLM extracted exam findings during interview, keep them."""
        data = {
            "motivoConsulta": "Dolor de oido",
            "padecimientoActual": "Otalgia 2 dias",
            "antecedentes": {},
            "exploracionFisica": {
                "otoscopia": "CAE eritematoso",
            },
            "diagnostico": {
                "texto": "Otitis externa",
                "tipo": "presuntivo",
                "cie10": None,
            },
            "planTratamiento": "Gotas oticas ciprofloxacino",
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "interview")

        # In-scope preserved
        assert result["motivoConsulta"] == "Dolor de oido"
        assert result["padecimientoActual"] == "Otalgia 2 dias"

        # Out-of-scope with data: preserved (bonus data)
        assert result["exploracionFisica"]["otoscopia"] == "CAE eritematoso"
        assert result["diagnostico"]["texto"] == "Otitis externa"
        assert result["planTratamiento"] == "Gotas oticas ciprofloxacino"

    def test_interview_normalizes_empty_out_of_scope(self):
        """Empty out-of-scope fields are normalized to null/{}."""
        data = {
            "motivoConsulta": "Dolor",
            "padecimientoActual": None,
            "antecedentes": {},
            "exploracionFisica": {},
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "interview")

        # Dict-typed out-of-scope stays {}
        assert result["exploracionFisica"] == {}
        # Scalar out-of-scope stays None
        assert result["diagnostico"] is None
        assert result["planTratamiento"] is None

    def test_interview_antecedentes_subfields_preserved(self):
        """All antecedentes subfields are preserved in interview scope."""
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {
                "heredofamiliares": "Madre con HTA",
                "personalesNoPatologicos": "Niega tabaquismo. Niega alcoholismo",
                "personalesPatologicos": "Circuncision a los 5 anos",
            },
            "exploracionFisica": {},
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "interview")

        ant = result["antecedentes"]
        assert ant["heredofamiliares"] == "Madre con HTA"
        assert ant["personalesNoPatologicos"] == "Niega tabaquismo. Niega alcoholismo"
        assert ant["personalesPatologicos"] == "Circuncision a los 5 anos"


# ---------------------------------------------------------------------------
# Unit: _apply_scope_mask — exam scope
# ---------------------------------------------------------------------------

class TestScopeMaskExam:
    """Exam scope: exploracionFisica."""

    def test_exam_preserves_exploracion(self):
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {},
            "exploracionFisica": {
                "orofaringe": "Amigdalas grado II",
                "cuello": "Adenopatia submandibular",
                "rinoscopia": "Cornetes hipertroficos",
            },
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "exam")

        assert result["exploracionFisica"]["orofaringe"] == "Amigdalas grado II"
        assert result["exploracionFisica"]["cuello"] == "Adenopatia submandibular"
        assert result["exploracionFisica"]["rinoscopia"] == "Cornetes hipertroficos"

    def test_exam_preserves_cross_scope_motivo(self):
        """If LLM extracted motivo during exam step, keep it."""
        data = {
            "motivoConsulta": "Dolor de garganta",
            "padecimientoActual": None,
            "antecedentes": {},
            "exploracionFisica": {
                "orofaringe": "Amigdalas hiperhemicas",
            },
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "exam")

        # Out-of-scope with value: preserved
        assert result["motivoConsulta"] == "Dolor de garganta"
        # Out-of-scope empty: normalized
        assert result["antecedentes"] == {}

    def test_exam_empty_antecedentes_normalized(self):
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {"heredofamiliares": None, "personalesPatologicos": None},
            "exploracionFisica": {"rinoscopia": "Normal"},
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "exam")

        # antecedentes with all-null subfields is effectively empty
        assert result["antecedentes"] == {}


# ---------------------------------------------------------------------------
# Unit: _apply_scope_mask — assessment scope
# ---------------------------------------------------------------------------

class TestScopeMaskAssessment:
    """Assessment scope: diagnostico, planTratamiento, pronostico."""

    def test_assessment_preserves_dx_plan(self):
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {},
            "exploracionFisica": {},
            "diagnostico": {"texto": "Faringitis aguda", "tipo": "definitivo", "cie10": None},
            "planTratamiento": "Amoxicilina 500mg c/8h x7d",
            "pronostico": "Bueno",
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "assessment")

        assert result["diagnostico"]["texto"] == "Faringitis aguda"
        assert result["planTratamiento"] == "Amoxicilina 500mg c/8h x7d"
        assert result["pronostico"] == "Bueno"

    def test_assessment_preserves_cross_scope_antecedentes(self):
        """If LLM mentions allergies during assessment, keep them."""
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {"personalesPatologicos": "Alergia a penicilina"},
            "exploracionFisica": {},
            "diagnostico": {"texto": "Faringitis", "tipo": "definitivo", "cie10": None},
            "planTratamiento": "Azitromicina 500mg",
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "assessment")

        # Cross-scope allergy data preserved
        assert result["antecedentes"]["personalesPatologicos"] == "Alergia a penicilina"


# ---------------------------------------------------------------------------
# Unit: _apply_scope_mask — studies scope
# ---------------------------------------------------------------------------

class TestScopeMaskStudies:
    """Studies scope: estudiosIndicados."""

    def test_studies_preserves_estudios(self):
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {},
            "exploracionFisica": {},
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": "TAC de senos paranasales",
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "studies")

        assert result["estudiosIndicados"] == "TAC de senos paranasales"

    def test_studies_normalizes_empty(self):
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {},
            "exploracionFisica": {},
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "studies")

        assert result["exploracionFisica"] == {}
        assert result["antecedentes"] == {}
        assert result["motivoConsulta"] is None


# ---------------------------------------------------------------------------
# Unit: _apply_scope_mask — unknown scope
# ---------------------------------------------------------------------------

class TestScopeMaskUnknown:

    def test_unknown_scope_preserves_populated_fields(self):
        """Unknown scope has no allowed set; populated data still preserved."""
        data = {
            "motivoConsulta": "Dolor",
            "padecimientoActual": None,
            "antecedentes": {},
            "exploracionFisica": {},
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "unknown_scope")

        # Populated out-of-scope data preserved
        assert result["motivoConsulta"] == "Dolor"
        # Empty out-of-scope normalized
        assert result["exploracionFisica"] == {}

    def test_unknown_scope_all_empty(self):
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {},
            "exploracionFisica": {},
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
        }

        result = _apply_scope_mask(data, "unknown_scope")

        assert result["antecedentes"] == {}
        assert result["exploracionFisica"] == {}
        assert result["motivoConsulta"] is None


# ---------------------------------------------------------------------------
# Integration: _parse_v1_output with scope
# ---------------------------------------------------------------------------

class TestParseV1OutputWithScope:

    def _make_json_output(self, data: dict) -> str:
        import json
        return json.dumps(data, ensure_ascii=False)

    def test_interview_scope_keeps_antecedentes(self):
        """Full pipeline: JSON output parsed with interview scope keeps antecedentes."""
        raw = self._make_json_output({
            "motivoConsulta": "Dolor de oido",
            "padecimientoActual": "Otalgia de 2 dias",
            "antecedentes": {
                "heredofamiliares": "Padre con DM2",
                "personalesNoPatologicos": "Niega tabaquismo",
                "personalesPatologicos": "Niega alergias",
            },
            "exploracionFisica": {},
            "diagnostico": {"texto": "Otalgia en estudio", "tipo": "sindromico"},
        })

        result = _parse_v1_output(raw, scope="interview")

        assert result.motivo_consulta == "Dolor de oido"
        assert result.padecimiento_actual == "Otalgia de 2 dias"
        assert result.antecedentes.heredofamiliares == "Padre con DM2"
        assert result.antecedentes.personales_no_patologicos == "Niega tabaquismo"
        assert result.antecedentes.personales_patologicos == "Niega alergias"

    def test_exam_scope_keeps_bonus_motivo(self):
        """Full pipeline: exam scope preserves cross-scope motivoConsulta."""
        raw = self._make_json_output({
            "motivoConsulta": "Dolor de garganta",
            "padecimientoActual": None,
            "antecedentes": {},
            "exploracionFisica": {
                "orofaringe": "Amigdalas grado II",
            },
            "diagnostico": {"texto": "Faringitis", "tipo": "presuntivo"},
        })

        result = _parse_v1_output(raw, scope="exam")

        assert result.exploracion_fisica.orofaringe == "Amigdalas grado II"
        # Cross-scope bonus preserved
        assert result.motivo_consulta == "Dolor de garganta"

    def test_no_scope_returns_all_fields(self):
        """Without scope, all fields are preserved as-is."""
        raw = self._make_json_output({
            "motivoConsulta": "Dolor",
            "padecimientoActual": "Inicio ayer",
            "antecedentes": {"heredofamiliares": "Padre DM2"},
            "exploracionFisica": {"orofaringe": "Normal"},
            "diagnostico": {"texto": "Odinofagia", "tipo": "sindromico"},
            "planTratamiento": "Ibuprofeno PRN",
        })

        result = _parse_v1_output(raw, scope=None)

        assert result.motivo_consulta == "Dolor"
        assert result.padecimiento_actual == "Inicio ayer"
        assert result.antecedentes.heredofamiliares == "Padre DM2"
        assert result.exploracion_fisica.orofaringe == "Normal"
        assert result.plan_tratamiento == "Ibuprofeno PRN"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestScopeMaskEdgeCases:

    def test_missing_field_in_data(self):
        """Fields not present in data are treated as None/{}."""
        data = {
            "motivoConsulta": "Dolor",
            # Missing all other fields
        }

        result = _apply_scope_mask(data, "interview")

        assert result["motivoConsulta"] == "Dolor"
        assert result["padecimientoActual"] is None
        assert result["antecedentes"] is None  # not in data, in-scope → None from .get()
        assert result["exploracionFisica"] == {}  # out-of-scope empty → {}

    def test_all_scopes_have_entries(self):
        """Verify SCOPE_ALLOWED_FIELDS covers all four scopes."""
        expected = {"interview", "exam", "studies", "assessment"}
        assert set(SCOPE_ALLOWED_FIELDS.keys()) == expected

    def test_scope_fields_no_overlap(self):
        """No field appears in multiple scopes."""
        all_fields = []
        for fields in SCOPE_ALLOWED_FIELDS.values():
            all_fields.extend(fields)
        assert len(all_fields) == len(set(all_fields)), "Fields overlap between scopes"
