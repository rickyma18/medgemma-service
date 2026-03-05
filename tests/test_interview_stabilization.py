"""
Tests for Interview scope stabilization (Epic: Interview V1 Stabilization).

Validates:
1. motivoConsulta — no mid-sentence truncation (prompt constraint)
2. cirugías → always in personalesPatologicos
3. Negation format: diseases = "Niega ...", habits = "No ..."
4. Guard doesn't discard when only antecedentes are present
5. Scope mask strictly nulls out-of-scope fields
6. Few-shot habit negation format consistency
7. Telemetry: compute_extraction_meta is scope-aware
8. Interview postprocess: padecimientoActual → antecedentes rerouting
9. Negation normalization: incomplete token cleanup and fragment merge
"""
import pytest

from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1,
    Antecedentes,
    ExploracionFisica,
)
from app.services.structured_v1_extractor import (
    _build_v1_system_prompt,
    _apply_scope_mask,
    _merge_negations_into_antecedentes,
    _postprocess_interview_fields,
    _normalize_negations,
    _repair_v1_dict,
    compute_extraction_meta,
    _build_short_transcript_fewshot,
    SHORT_TRANSCRIPT_THRESHOLD,
    SHORT_TRANSCRIPT_THRESHOLD_INTERVIEW,
)


# ---------------------------------------------------------------------------
# 1. motivoConsulta — prompt forbids mid-sentence truncation
# ---------------------------------------------------------------------------

class TestMotivoNotTruncated:
    """Base prompt and interview scope must forbid word-count truncation."""

    def test_no_word_count_in_base_prompt(self):
        prompt = _build_v1_system_prompt()
        assert "3-15 palabras" not in prompt

    def test_complete_sentence_rule_in_base(self):
        prompt = _build_v1_system_prompt()
        assert "nunca cortar" in prompt.lower()

    def test_anti_dangling_rule_in_interview_scope(self):
        prompt = _build_v1_system_prompt(scope="interview")
        assert "de/del/con/y" in prompt

    def test_schema_shows_complete_sentences(self):
        prompt = _build_v1_system_prompt()
        assert "oraciones cortas completas" in prompt.lower()


# ---------------------------------------------------------------------------
# 2. cirugías → personalesPatologicos
# ---------------------------------------------------------------------------

class TestCirugiaRouting:
    """Cirugía keywords must route to personalesPatologicos via negation merge."""

    def test_negation_merge_routes_cirugia_to_app(self):
        data = _repair_v1_dict({
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {"heredofamiliares": None, "personalesNoPatologicos": None, "personalesPatologicos": None},
            "negations": ["Apendicectomía hace 5 años"],
        })
        result = _merge_negations_into_antecedentes(data, "interview")
        assert result["antecedentes"]["personalesPatologicos"] is not None
        assert "pendicectom" in result["antecedentes"]["personalesPatologicos"].lower()

    def test_cirugia_keywords_in_interview_prompt(self):
        prompt = _build_v1_system_prompt(scope="interview")
        lower = prompt.lower()
        assert "cirugía" in lower or "cirugia" in lower
        assert "operaron" in lower

    def test_cirugia_targets_patologicos_in_prompt(self):
        prompt = _build_v1_system_prompt(scope="interview")
        assert "personalesPatologicos" in prompt


# ---------------------------------------------------------------------------
# 3. Negation format: diseases = "Niega", habits = "No"
# ---------------------------------------------------------------------------

class TestNegationFormat:
    """Interview scope must specify Niega for diseases, No for habits."""

    def test_niega_for_diseases_in_prompt(self):
        prompt = _build_v1_system_prompt(scope="interview")
        assert "Niega diabetes" in prompt or "Niega hipertensión" in prompt

    def test_no_for_habits_in_prompt(self):
        prompt = _build_v1_system_prompt(scope="interview")
        assert "No fuma" in prompt or "No toma" in prompt

    def test_fewshot_habits_use_no_format(self):
        """Few-shot examples must model 'No fuma' not 'Niega tabaquismo'."""
        fewshot = _build_short_transcript_fewshot("interview")
        assert "No fuma" in fewshot or "No toma alcohol" in fewshot

    def test_fewshot_diseases_use_niega_format(self):
        """Few-shot examples must model 'Niega diabetes' not 'No diabetes'."""
        fewshot = _build_short_transcript_fewshot("interview")
        assert "Niega diabetes" in fewshot or "Niega hipertensión" in fewshot

    def test_sentence_separation_rule(self):
        prompt = _build_v1_system_prompt(scope="interview")
        assert "ORACIÓN SEPARADA" in prompt


# ---------------------------------------------------------------------------
# 4. Guard doesn't discard when only antecedentes present
# ---------------------------------------------------------------------------

class TestGuardScopeAware:
    """compute_extraction_meta with scope=interview must not flag antecedentes-only as sparse."""

    def test_antecedentes_only_is_useful(self):
        fields = StructuredFieldsV1(
            antecedentes=Antecedentes(
                personales_patologicos="Niega diabetes. Niega hipertensión"
            )
        )
        meta = compute_extraction_meta(fields, scope="interview")
        assert meta["hasContent"] is True

    def test_apnp_only_is_useful(self):
        fields = StructuredFieldsV1(
            antecedentes=Antecedentes(
                personales_no_patologicos="No fuma. No toma alcohol"
            )
        )
        meta = compute_extraction_meta(fields, scope="interview")
        assert meta["hasContent"] is True

    def test_heredofam_only_is_useful(self):
        fields = StructuredFieldsV1(
            antecedentes=Antecedentes(
                heredofamiliares="Padre con diabetes mellitus"
            )
        )
        meta = compute_extraction_meta(fields, scope="interview")
        assert meta["hasContent"] is True

    def test_negations_count_as_useful(self):
        fields = StructuredFieldsV1(negations=["No fuma"])
        meta = compute_extraction_meta(fields, scope="interview")
        assert meta["hasContent"] is True
        assert meta["negatedFindingsCount"] == 1

    def test_empty_interview_is_sparse(self):
        fields = StructuredFieldsV1()
        meta = compute_extraction_meta(fields, scope="interview")
        assert meta["hasContent"] is False

    def test_exam_fields_dont_count_for_interview(self):
        """If only exam fields are filled, interview scope should show sparse."""
        fields = StructuredFieldsV1(
            exploracion_fisica=ExploracionFisica(otoscopia="CAE sin alteraciones")
        )
        meta = compute_extraction_meta(fields, scope="interview")
        assert meta["hasContent"] is False

    def test_full_scope_backward_compat(self):
        """Without scope, compute_extraction_meta behaves as before."""
        fields = StructuredFieldsV1(
            antecedentes=Antecedentes(personales_patologicos="Niega diabetes")
        )
        meta = compute_extraction_meta(fields)
        assert meta["hasContent"] is True

    def test_full_scope_empty(self):
        fields = StructuredFieldsV1()
        meta = compute_extraction_meta(fields)
        assert meta["hasContent"] is False


# ---------------------------------------------------------------------------
# 5. Scope mask strictly nulls out-of-scope fields
# ---------------------------------------------------------------------------

class TestScopeMaskStrict:
    """_apply_scope_mask must strictly null out-of-scope fields even if they have data."""

    def test_interview_nulls_diagnostico(self):
        data = _repair_v1_dict({
            "motivoConsulta": "Dolor de garganta",
            "diagnostico": {"texto": "Faringitis", "tipo": "presuntivo"},
        })
        masked = _apply_scope_mask(data, "interview")
        assert masked["motivoConsulta"] == "Dolor de garganta"
        assert masked["diagnostico"] is None, "Out-of-scope diagnostico should be strictly nulled"

    def test_interview_nulls_plan(self):
        data = _repair_v1_dict({
            "motivoConsulta": "Dolor",
            "planTratamiento": "Amoxicilina 500mg",
        })
        masked = _apply_scope_mask(data, "interview")
        assert masked["planTratamiento"] is None

    def test_interview_nulls_exploracion(self):
        data = _repair_v1_dict({
            "motivoConsulta": "Dolor",
            "exploracionFisica": {"otoscopia": "Normal"},
        })
        masked = _apply_scope_mask(data, "interview")
        assert masked["exploracionFisica"] is None

    def test_interview_preserves_antecedentes(self):
        data = _repair_v1_dict({
            "antecedentes": {
                "heredofamiliares": "Madre con HTA",
                "personalesPatologicos": "Niega diabetes",
            },
        })
        masked = _apply_scope_mask(data, "interview")
        assert masked["antecedentes"]["heredofamiliares"] == "Madre con HTA"
        assert masked["antecedentes"]["personalesPatologicos"] == "Niega diabetes"

    def test_interview_preserves_negations(self):
        data = _repair_v1_dict({"negations": ["No fuma"]})
        masked = _apply_scope_mask(data, "interview")
        assert masked["negations"] == ["No fuma"]

    def test_exam_nulls_motivo(self):
        data = _repair_v1_dict({
            "motivoConsulta": "Dolor",
            "exploracionFisica": {"otoscopia": "Normal"},
        })
        masked = _apply_scope_mask(data, "exam")
        assert masked["motivoConsulta"] is None
        assert masked["exploracionFisica"]["otoscopia"] == "Normal"

    def test_interview_structuredfields_only_has_scope_keys(self):
        """End-to-end: after mask + validate, out-of-scope fields are null/empty."""
        data = _repair_v1_dict({
            "motivoConsulta": "Dolor abdominal de tres días de evolución",
            "padecimientoActual": "Inicio hace tres días, progresivo.",
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": "No fuma. No toma alcohol",
                "personalesPatologicos": "Apendicectomía hace 10 años. Niega diabetes. Niega hipertensión",
            },
            "exploracionFisica": {"otoscopia": "Normal"},
            "diagnostico": {"texto": "Gastritis", "tipo": "sindromico"},
            "planTratamiento": "Omeprazol",
        })
        masked = _apply_scope_mask(data, "interview")
        fields = StructuredFieldsV1.model_validate(masked)

        # In-scope should be preserved
        assert fields.motivo_consulta == "Dolor abdominal de tres días de evolución"
        assert fields.padecimiento_actual is not None
        assert fields.antecedentes.personales_no_patologicos is not None
        assert fields.antecedentes.personales_patologicos is not None

        # Out-of-scope should be strictly null
        assert fields.diagnostico is None
        assert fields.plan_tratamiento is None
        assert fields.pronostico is None
        assert fields.estudios_indicados is None
        assert fields.exploracion_fisica is None


# ---------------------------------------------------------------------------
# 6. Few-shot threshold and content
# ---------------------------------------------------------------------------

class TestFewShotInterview:
    """Interview few-shot uses higher threshold and correct format."""

    def test_threshold_values(self):
        assert SHORT_TRANSCRIPT_THRESHOLD == 150
        assert SHORT_TRANSCRIPT_THRESHOLD_INTERVIEW == 400

    def test_interview_200_chars_gets_fewshot(self):
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=200)
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" in prompt

    def test_exam_200_chars_no_fewshot(self):
        prompt = _build_v1_system_prompt(scope="exam", transcript_len=200)
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" not in prompt

    def test_fewshot_has_cirugia_example(self):
        fewshot = _build_short_transcript_fewshot("interview")
        assert "Apendicectomía" in fewshot

    def test_fewshot_has_mixed_negation_example(self):
        """The new few-shot example has both diseases (Niega) and habits (No)."""
        fewshot = _build_short_transcript_fewshot("interview")
        assert "No fuma" in fewshot
        assert "Niega diabetes" in fewshot


# ---------------------------------------------------------------------------
# 7. Example of final structuredFields for scope=interview
# ---------------------------------------------------------------------------

class TestInterviewFinalShape:
    """Validate the exact shape of structuredFields after full interview pipeline."""

    def test_expected_interview_shape(self):
        """Simulate the full parse + mask + validate pipeline for interview scope."""
        # Simulated LLM output after repair
        llm_output = {
            "motivoConsulta": "Dolor abdominal de tres días de evolución",
            "padecimientoActual": "Inicio hace tres días con dolor epigástrico progresivo.",
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": "No fuma. No toma alcohol",
                "personalesPatologicos": "Apendicectomía hace 10 años. Niega diabetes. Niega hipertensión. Niega alergias",
            },
            "exploracionFisica": {"rinoscopia": "Mucosa eritematosa"},
            "diagnostico": {"texto": "Gastritis", "tipo": "sindromico"},
            "planTratamiento": "Omeprazol 20mg VO c/24h",
            "negations": [],
        }

        repaired = _repair_v1_dict(llm_output)
        repaired = _merge_negations_into_antecedentes(repaired, "interview")
        masked = _apply_scope_mask(repaired, "interview")
        fields = StructuredFieldsV1.model_validate(masked)
        output = fields.model_dump(by_alias=True, exclude_none=False)

        # Verify interview keys present and complete
        assert output["motivoConsulta"] == "Dolor abdominal de tres días de evolución"
        assert output["padecimientoActual"] is not None
        assert output["antecedentes"]["personalesNoPatologicos"] == "No fuma. No toma alcohol"
        assert "Apendicectomía" in output["antecedentes"]["personalesPatologicos"]
        assert "Niega diabetes" in output["antecedentes"]["personalesPatologicos"]

        # Verify out-of-scope keys are strictly null
        assert output["diagnostico"] is None
        assert output["planTratamiento"] is None
        assert output["pronostico"] is None
        assert output["estudiosIndicados"] is None
        assert output["exploracionFisica"] is None


# ---------------------------------------------------------------------------
# 8. Interview postprocess: reroute misplaced antecedentes from padecimientoActual
# ---------------------------------------------------------------------------

class TestInterviewPostprocess:
    """_postprocess_interview_fields moves antecedentes out of padecimientoActual."""

    def test_habits_moved_to_apnp(self):
        data = _repair_v1_dict({
            "padecimientoActual": "No fuma, no toma alcohol",
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
        })
        result = _postprocess_interview_fields(data, "interview")
        assert result["padecimientoActual"] is None
        apnp = result["antecedentes"]["personalesNoPatologicos"]
        assert apnp is not None
        assert "No fuma" in apnp or "no fuma" in apnp
        assert "alcohol" in apnp.lower()

    def test_disease_negations_moved_to_app(self):
        data = _repair_v1_dict({
            "padecimientoActual": "Niega diabetes, niega hipertensión",
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
        })
        result = _postprocess_interview_fields(data, "interview")
        assert result["padecimientoActual"] is None
        app = result["antecedentes"]["personalesPatologicos"]
        assert app is not None
        assert "diabetes" in app.lower()
        assert "hipertensión" in app.lower()

    def test_surgeries_moved_to_app(self):
        data = _repair_v1_dict({
            "padecimientoActual": "Apendicectomía hace 5 años",
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
        })
        result = _postprocess_interview_fields(data, "interview")
        assert result["padecimientoActual"] is None
        assert "Apendicectomía" in result["antecedentes"]["personalesPatologicos"]

    def test_hf_moved_to_heredofamiliares(self):
        data = _repair_v1_dict({
            "padecimientoActual": "En familiares padre con diabetes",
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
        })
        result = _postprocess_interview_fields(data, "interview")
        hf = result["antecedentes"]["heredofamiliares"]
        assert hf is not None
        assert "familiares" in hf.lower()

    def test_mixed_content_keeps_padecimiento(self):
        data = _repair_v1_dict({
            "padecimientoActual": "Dolor de garganta de 5 días. No fuma, no toma alcohol. Niega diabetes",
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
        })
        result = _postprocess_interview_fields(data, "interview")
        pa = result["padecimientoActual"]
        assert pa is not None
        assert "dolor de garganta" in pa.lower()
        assert "no fuma" not in pa.lower()
        assert "niega" not in pa.lower()
        assert result["antecedentes"]["personalesNoPatologicos"] is not None
        assert result["antecedentes"]["personalesPatologicos"] is not None

    def test_concatenates_with_existing_antecedentes(self):
        data = _repair_v1_dict({
            "padecimientoActual": "No fuma",
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": "No consume drogas",
                "personalesPatologicos": None,
            },
        })
        result = _postprocess_interview_fields(data, "interview")
        apnp = result["antecedentes"]["personalesNoPatologicos"]
        assert "No consume drogas" in apnp
        assert "fuma" in apnp.lower()

    def test_noop_for_non_interview_scope(self):
        data = _repair_v1_dict({
            "padecimientoActual": "No fuma",
            "antecedentes": {},
        })
        result = _postprocess_interview_fields(data, "exam")
        assert result["padecimientoActual"] == "No fuma"

    def test_noop_when_no_padecimiento(self):
        data = _repair_v1_dict({
            "padecimientoActual": None,
            "antecedentes": {},
        })
        result = _postprocess_interview_fields(data, "interview")
        assert result["padecimientoActual"] is None

    def test_full_real_world_scenario(self):
        """Real-world case: all antecedentes dumped into padecimientoActual."""
        data = _repair_v1_dict({
            "padecimientoActual": (
                "no fuma, no toma alcohol, apendicectomía hace 5 años, "
                "niega diabetes, niega alergias"
            ),
            "antecedentes": {
                "heredofamiliares": "Padre con HTA",
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
        })
        result = _postprocess_interview_fields(data, "interview")

        # padecimientoActual should be empty now
        assert result["padecimientoActual"] is None

        # antecedentes should be populated
        assert result["antecedentes"]["personalesNoPatologicos"] is not None
        assert "fuma" in result["antecedentes"]["personalesNoPatologicos"].lower()
        assert "alcohol" in result["antecedentes"]["personalesNoPatologicos"].lower()

        assert result["antecedentes"]["personalesPatologicos"] is not None
        assert "apendicectomía" in result["antecedentes"]["personalesPatologicos"].lower()
        assert "diabetes" in result["antecedentes"]["personalesPatologicos"].lower()

        # HF should be preserved
        assert "Padre con HTA" in result["antecedentes"]["heredofamiliares"]

    def test_operaron_keyword(self):
        data = _repair_v1_dict({
            "padecimientoActual": "me operaron de apéndice hace 3 años",
            "antecedentes": {},
        })
        result = _postprocess_interview_fields(data, "interview")
        assert result["padecimientoActual"] is None
        assert result["antecedentes"]["personalesPatologicos"] is not None

    def test_circuncision_keyword(self):
        data = _repair_v1_dict({
            "padecimientoActual": "Circuncisión a los 5 años",
            "antecedentes": {},
        })
        result = _postprocess_interview_fields(data, "interview")
        assert result["padecimientoActual"] is None
        assert "ircuncis" in result["antecedentes"]["personalesPatologicos"].lower()


# ---------------------------------------------------------------------------
# 9. Negation normalization
# ---------------------------------------------------------------------------

class TestNegationNormalization:
    """_normalize_negations cleans incomplete tokens and merges fragments."""

    def test_removes_trailing_conjunction(self):
        result = _normalize_negations(["diabetes e", "hipertensión"])
        assert "diabetes" in result
        assert "hipertensión" in result
        assert "diabetes e" not in result

    def test_removes_standalone_incomplete(self):
        result = _normalize_negations(["ha presentado", "alergias"])
        assert "ha presentado" not in result
        assert "alergias" in result

    def test_removes_short_tokens(self):
        result = _normalize_negations(["a", "de", "y", "diabetes"])
        assert len(result) == 1
        assert "diabetes" in result

    def test_merges_alergias_medicamentos(self):
        result = _normalize_negations(["alergias", "medicamentos"])
        assert len(result) == 1
        assert "alergias a medicamentos" in result

    def test_preserves_clean_entries(self):
        result = _normalize_negations(["diabetes", "hipertensión", "asma"])
        assert result == ["diabetes", "hipertensión", "asma"]

    def test_empty_input(self):
        assert _normalize_negations([]) == []
        assert _normalize_negations(None) == []

    def test_trims_trailing_preposition(self):
        result = _normalize_negations(["alergias a"])
        assert result == ["alergias"]

    def test_strips_punctuation(self):
        result = _normalize_negations(["diabetes.", "hipertensión;"])
        assert "diabetes" in result
        assert "hipertensión" in result

    def test_mixed_real_world_negations(self):
        """Real-world negation list from client with noise."""
        raw = ["diabetes e", "ha presentado", "alergias", "medicamentos", "a", "hipertensión"]
        result = _normalize_negations(raw)
        assert "ha presentado" not in result
        assert "a" not in result
        assert any("alergias a medicamentos" in r for r in result)
        assert "hipertensión" in result
        assert "diabetes" in result

    def test_deduplicates_preserving_order(self):
        result = _normalize_negations(["diabetes", "hipertensión", "diabetes", "asma"])
        assert result == ["diabetes", "hipertensión", "asma"]

    def test_deduplicates_case_insensitive(self):
        result = _normalize_negations(["Diabetes", "diabetes", "DIABETES"])
        assert result == ["Diabetes"]


# ---------------------------------------------------------------------------
# 10. Postprocess detects patterns WITHIN sentences (not just at start)
# ---------------------------------------------------------------------------

class TestPostprocessMidSentenceDetection:
    """Verify patterns are detected anywhere in the phrase, not just at the start."""

    def test_refiere_que_no_fuma(self):
        data = _repair_v1_dict({
            "padecimientoActual": "refiere que no fuma",
            "antecedentes": {},
        })
        result = _postprocess_interview_fields(data, "interview")
        assert result["padecimientoActual"] is None
        apnp = result["antecedentes"]["personalesNoPatologicos"]
        assert apnp is not None
        assert "fuma" in apnp.lower()

    def test_menciona_que_no_toma_alcohol(self):
        data = _repair_v1_dict({
            "padecimientoActual": "menciona que no toma alcohol",
            "antecedentes": {},
        })
        result = _postprocess_interview_fields(data, "interview")
        assert result["padecimientoActual"] is None
        assert result["antecedentes"]["personalesNoPatologicos"] is not None

    def test_paciente_niega_diabetes(self):
        data = _repair_v1_dict({
            "padecimientoActual": "el paciente niega diabetes e hipertensión",
            "antecedentes": {},
        })
        result = _postprocess_interview_fields(data, "interview")
        assert result["padecimientoActual"] is None
        assert result["antecedentes"]["personalesPatologicos"] is not None

    def test_mixed_mid_sentence_keeps_padecimiento(self):
        """Real padecimiento stays; mid-sentence antecedentes move."""
        data = _repair_v1_dict({
            "padecimientoActual": (
                "Dolor de oído de 3 días. "
                "Refiere que no fuma ni toma alcohol. "
                "El paciente niega alergias"
            ),
            "antecedentes": {},
        })
        result = _postprocess_interview_fields(data, "interview")
        pa = result["padecimientoActual"]
        assert pa is not None
        assert "dolor" in pa.lower()
        assert "fuma" not in pa.lower()
        assert "niega" not in pa.lower()
        assert result["antecedentes"]["personalesNoPatologicos"] is not None
        assert result["antecedentes"]["personalesPatologicos"] is not None


# ---------------------------------------------------------------------------
# 11. Finalize + merge: reproduce production JSON where negations are populated
#     but personalesNoPatologicos / personalesPatologicos are null
# ---------------------------------------------------------------------------

class TestFinalizeInterviewMerge:
    """
    Reproduce the exact production case:
      - heredofamiliares has text
      - personalesNoPatologicos = null, personalesPatologicos = null
      - negations = ["fuma", "toma alcohol", "asma", "alergias a medicamentos", ...]
    After the pipeline, personalesNoPatologicos and personalesPatologicos must be filled.
    """

    def test_production_json_negations_fill_antecedentes(self):
        """End-to-end: simulate what finalize does for scope=interview."""
        # This is the raw JSON the client sends to /v1/finalize
        raw_fields = StructuredFieldsV1.model_validate({
            "motivoConsulta": "Dolor de garganta",
            "padecimientoActual": "Odinofagia de 3 días de evolución",
            "antecedentes": {
                "heredofamiliares": "Padre con diabetes mellitus tipo 2",
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
            "exploracionFisica": None,
            "diagnostico": None,
            "planTratamiento": None,
            "pronostico": None,
            "estudiosIndicados": None,
            "notasAdicionales": None,
            "negations": [
                "fuma",
                "toma alcohol",
                "drogas",
                "asma",
                "diabetes",
                "hipertensión",
                "alergias a medicamentos",
                "diabetes",        # duplicate
            ],
        })

        # Simulate the finalize pipeline for scope=interview
        data_dict = raw_fields.model_dump(by_alias=True)
        raw_negations = list(data_dict.get("negations") or [])
        clean_negations = _normalize_negations(raw_negations)
        data_dict["negations"] = clean_negations
        data_dict = _postprocess_interview_fields(data_dict, "interview")
        data_dict = _merge_negations_into_antecedentes(data_dict, "interview")
        data_dict["negations"] = clean_negations
        final = StructuredFieldsV1.model_validate(data_dict)

        # ── Assertions ──

        # HF preserved
        assert final.antecedentes.heredofamiliares is not None
        assert "diabetes" in final.antecedentes.heredofamiliares.lower()

        # personalesNoPatologicos NOW populated with habits
        apnp = final.antecedentes.personales_no_patologicos
        assert apnp is not None, "personalesNoPatologicos must not be null"
        assert "fuma" in apnp.lower()
        assert "alcohol" in apnp.lower()
        assert "droga" in apnp.lower()

        # personalesPatologicos NOW populated with disease negations
        app = final.antecedentes.personales_patologicos
        assert app is not None, "personalesPatologicos must not be null"
        assert "asma" in app.lower()
        assert "hipertensi" in app.lower()
        assert "alergias a medicamentos" in app.lower()

        # padecimientoActual untouched (no misplaced content)
        assert final.padecimiento_actual is not None
        assert "odinofagia" in final.padecimiento_actual.lower()

        # negations deduplicated (diabetes appeared twice)
        assert clean_negations.count("diabetes") == 1

        # negations restored in output
        assert final.negations == clean_negations
