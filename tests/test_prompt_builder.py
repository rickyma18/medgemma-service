"""
Tests for prompt builder (FIX #4 + FIX #5).

Validates:
- FIX #4: prompt includes MISION, negation rules, MAPEO, schema
- FIX #5: short transcripts (<150 chars) get few-shot examples; long ones don't
"""
import pytest

from app.services.structured_v1_extractor import (
    _build_v1_system_prompt,
    SHORT_TRANSCRIPT_THRESHOLD,
    SHORT_TRANSCRIPT_THRESHOLD_INTERVIEW,
)


# ---------------------------------------------------------------------------
# FIX #4: Prompt structure and content
# ---------------------------------------------------------------------------

class TestPromptMisionDirective:
    """MISION directive must be present at the top."""

    def test_contains_extraer_no_redactar(self):
        prompt = _build_v1_system_prompt()
        assert "EXTRAER" in prompt
        assert "NO REDACTAR" in prompt

    def test_contains_no_inferir(self):
        prompt = _build_v1_system_prompt()
        assert "NO INFERIR" in prompt

    def test_contains_no_inventar(self):
        prompt = _build_v1_system_prompt()
        assert "NO INVENTAR" in prompt


class TestPromptNegationRules:
    """Negation handling rules must be explicit."""

    def test_preserva_negaciones(self):
        prompt = _build_v1_system_prompt()
        assert "niega" in prompt.lower()
        # Must mention that negatives ARE valid data
        assert "datos validos" in prompt.lower() or "DEBEN registrarse" in prompt

    def test_negativos_pertinentes_listed(self):
        prompt = _build_v1_system_prompt()
        for neg in ["no fuma", "niega alergias", "sin cirugias"]:
            assert neg in prompt.lower(), f"Missing negation example: {neg}"

    def test_null_only_when_not_discussed(self):
        prompt = _build_v1_system_prompt()
        assert "no se toco en la conversacion" in prompt.lower()


class TestPromptMapeo:
    """MAPEO DE CAMPOS section must be present."""

    def test_contains_mapeo_section(self):
        prompt = _build_v1_system_prompt()
        assert "## MAPEO DE CAMPOS" in prompt

    def test_mapeo_has_critical_fields(self):
        prompt = _build_v1_system_prompt()
        for field in [
            "motivoConsulta",
            "padecimientoActual",
            "heredofamiliares",
            "personalesNoPatologicos",
            "personalesPatologicos",
        ]:
            assert field in prompt, f"MAPEO missing field: {field}"


class TestPromptSchema:
    """JSON schema reference must be included."""

    def test_contains_schema_section(self):
        prompt = _build_v1_system_prompt()
        assert "## SCHEMA" in prompt

    def test_schema_has_required_fields(self):
        prompt = _build_v1_system_prompt()
        for field in [
            "motivoConsulta",
            "padecimientoActual",
            "antecedentes",
            "exploracionFisica",
            "diagnostico",
            "planTratamiento",
        ]:
            assert field in prompt


class TestPromptAntecedentesRouting:
    """Antecedentes routing rules (rule 1B) must be present."""

    def test_routing_alergias_to_patologicos(self):
        prompt = _build_v1_system_prompt()
        assert "alergi" in prompt.lower()
        assert "personalesPatologicos" in prompt

    def test_routing_habitos_to_no_patologicos(self):
        prompt = _build_v1_system_prompt()
        assert "personalesNoPatologicos" in prompt

    def test_routing_cirugias_to_patologicos(self):
        prompt = _build_v1_system_prompt()
        assert "cirugi" in prompt.lower() or "circuncis" in prompt.lower()


class TestPromptImpresion:
    """Rule 4: impresion diagnostica = diagnostico."""

    def test_impresion_rule(self):
        prompt = _build_v1_system_prompt()
        assert "impresion" in prompt.lower()
        assert "diagnostico" in prompt.lower()
        # Must warn against confusing with depression
        assert "depresion" in prompt.lower()


# ---------------------------------------------------------------------------
# FIX #5: Few-shot injection for short transcripts
# ---------------------------------------------------------------------------

class TestFewShotShortTranscript:
    """Few-shot examples injected when short transcript AND scope is provided.
    
    NOTE: Few-shot examples are now scope-aware. They are only injected when:
    1. transcript_len > 0 and < SHORT_TRANSCRIPT_THRESHOLD
    2. scope is explicitly provided (e.g., "interview")
    
    Non-scoped extractions (full extraction) do not inject few-shot examples
    because the main prompt already has comprehensive examples.
    """

    def test_short_transcript_interview_has_fewshot(self):
        """Short transcript with interview scope gets few-shot examples."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=100)
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" in prompt

    def test_short_transcript_no_scope_no_fewshot(self):
        """Short transcript WITHOUT scope does NOT get scope-specific few-shot."""
        prompt = _build_v1_system_prompt(transcript_len=100)
        # Non-scoped extractions rely on the comprehensive base prompt
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS – INTERVIEW" not in prompt

    def test_long_transcript_no_fewshot(self):
        """Transcript above interview threshold should NOT get few-shot."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=500)
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" not in prompt

    def test_exact_threshold_no_fewshot(self):
        """Exact interview threshold boundary should NOT get few-shot."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=SHORT_TRANSCRIPT_THRESHOLD_INTERVIEW)
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" not in prompt

    def test_zero_len_no_fewshot(self):
        """Default (len=0) should not inject few-shot."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=0)
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" not in prompt

    def test_fewshot_interview_contains_negation_examples(self):
        """Interview few-shot must contain negation routing examples."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=50)
        lower = prompt.lower()
        assert "niega alergias" in lower
        assert "niega tabaquismo" in lower

    def test_fewshot_interview_contains_heredofamiliares_example(self):
        """Interview few-shot must contain family history examples."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=50)
        # Updated to match new expanded medical terminology
        assert "diabetes mellitus" in prompt.lower() or "hipertensión arterial" in prompt.lower()

    def test_fewshot_interview_nulls_for_missing_fields(self):
        """Interview few-shot examples must show null for fields not in transcript."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=50)
        # motivoConsulta should be null in short examples
        assert '"motivoConsulta": null' in prompt
        assert '"padecimientoActual": null' in prompt

    def test_fewshot_interview_no_out_of_scope_fields(self):
        """Interview few-shot must NOT contain out-of-scope fields.
        
        Test the few-shot function directly since the full prompt contains
        all field names in schema documentation.
        """
        from app.services.structured_v1_extractor import _build_short_transcript_fewshot
        fewshot = _build_short_transcript_fewshot("interview")
        # These fields should NOT appear in interview few-shot examples
        assert '"exploracionFisica"' not in fewshot
        assert '"diagnostico"' not in fewshot
        assert '"planTratamiento"' not in fewshot
        assert '"pronostico"' not in fewshot
        assert '"estudiosIndicados"' not in fewshot
        assert '"notasAdicionales"' not in fewshot

    def test_fewshot_no_phi(self):
        """Few-shot examples must not contain real PHI."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=50)
        # Check that no real names/dates/IDs appear
        # (Synthetic examples use generic terms)
        assert "Juan" not in prompt
        assert "Maria" not in prompt
        assert "CURP" not in prompt


class TestPromptWithScope:
    """Scope instructions work with and without few-shot."""

    def test_scope_interview_added(self):
        prompt = _build_v1_system_prompt(scope="interview")
        assert "SCOPE:" in prompt
        assert "motivoConsulta" in prompt

    def test_scope_with_fewshot(self):
        """Scope and few-shot can coexist."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=80)
        assert "SCOPE:" in prompt
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" in prompt

    def test_scope_without_fewshot(self):
        prompt = _build_v1_system_prompt(scope="exam", transcript_len=500)
        assert "SCOPE:" in prompt
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" not in prompt


# ---------------------------------------------------------------------------
# Interview scope: cirugía routing, negation format, anti-dangling motivo
# ---------------------------------------------------------------------------

class TestInterviewScopeCirugiaRouting:
    """Interview scope must explicitly route cirugías to personalesPatologicos."""

    def test_interview_scope_mentions_cirugia(self):
        prompt = _build_v1_system_prompt(scope="interview")
        lower = prompt.lower()
        assert "cirugía" in lower or "cirugia" in lower

    def test_interview_scope_routes_cirugia_to_patologicos(self):
        prompt = _build_v1_system_prompt(scope="interview")
        # The SCOPE block must mention both cirugía and personalesPatologicos
        assert "personalesPatologicos" in prompt
        lower = prompt.lower()
        assert "apendicectom" in lower or "operaron" in lower


class TestInterviewScopeNegationFormat:
    """Interview scope must enforce Niega for diseases, No for habits."""

    def test_interview_scope_niega_for_diseases(self):
        prompt = _build_v1_system_prompt(scope="interview")
        assert "Niega diabetes" in prompt or "Niega hipertensión" in prompt

    def test_interview_scope_no_for_habits(self):
        prompt = _build_v1_system_prompt(scope="interview")
        assert "No fuma" in prompt or "No toma" in prompt

    def test_interview_scope_separate_sentences(self):
        prompt = _build_v1_system_prompt(scope="interview")
        assert "ORACIÓN SEPARADA" in prompt


class TestMotivoConsultaAntiDangling:
    """motivoConsulta must not allow mid-sentence truncation."""

    def test_base_prompt_no_word_count_limit(self):
        prompt = _build_v1_system_prompt()
        assert "3-15 palabras" not in prompt

    def test_base_prompt_complete_sentences(self):
        prompt = _build_v1_system_prompt()
        lower = prompt.lower()
        assert "nunca cortar a mitad de frase" in lower or "nunca cortar" in lower

    def test_interview_scope_anti_dangling_rule(self):
        prompt = _build_v1_system_prompt(scope="interview")
        # Must warn against dangling prepositions
        assert "de/del/con/y" in prompt


class TestInterviewFewShotThreshold:
    """Interview scope uses higher few-shot threshold than other scopes."""

    def test_interview_threshold_is_higher(self):
        assert SHORT_TRANSCRIPT_THRESHOLD_INTERVIEW > SHORT_TRANSCRIPT_THRESHOLD

    def test_interview_200_chars_gets_fewshot(self):
        """200 chars is above base threshold (150) but below interview (400)."""
        prompt = _build_v1_system_prompt(scope="interview", transcript_len=200)
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" in prompt

    def test_exam_200_chars_no_fewshot(self):
        """200 chars is above base threshold (150) — no few-shot for exam scope."""
        prompt = _build_v1_system_prompt(scope="exam", transcript_len=200)
        assert "EJEMPLOS PARA TRANSCRIPTS CORTOS" not in prompt

    def test_fewshot_contains_cirugia_example(self):
        """Interview few-shot must include circuncisión/apendicectomía example."""
        from app.services.structured_v1_extractor import _build_short_transcript_fewshot
        fewshot = _build_short_transcript_fewshot("interview")
        assert "Apendicectomía" in fewshot or "operaron" in fewshot.lower()


# ---------------------------------------------------------------------------
# Interview scope: symptom exclusion from antecedentes + negations[] quality
# ---------------------------------------------------------------------------

class TestInterviewScopeSymptomExclusion:
    """Interview prompt must forbid placing symptoms in antecedentes fields."""

    def test_symptoms_must_not_go_into_antecedentes(self):
        prompt = _build_v1_system_prompt(scope="interview")
        lower = prompt.lower()
        assert "síntomas" in lower or "sintomas" in lower
        assert "antecedentes" in lower
        # Must explicitly forbid symptoms in antecedentes
        assert "prohibido colocar síntomas en antecedentes" in lower or \
               "no van en ningún campo de antecedentes" in lower

    def test_symptom_examples_listed(self):
        """Prompt must list common symptoms that should NOT go into antecedentes."""
        prompt = _build_v1_system_prompt(scope="interview")
        lower = prompt.lower()
        for symptom in ["fiebre", "tos", "disnea", "dolor", "rinorrea", "nausea", "diarrea"]:
            assert symptom in lower, f"Missing symptom exclusion example: {symptom}"

    def test_symptom_negations_go_to_padecimiento_or_negations(self):
        """Prompt must route symptom negations to padecimientoActual or negations[], not antecedentes."""
        prompt = _build_v1_system_prompt(scope="interview")
        lower = prompt.lower()
        assert "padecimientoactual" in lower
        assert "negations" in lower


class TestInterviewScopeNegationsQuality:
    """Interview prompt must enforce clean, atomic negation items."""

    def test_no_dangling_conjunctions_rule(self):
        """Prompt must explicitly forbid dangling conjunctions like 'ni', 'y', 'e', 'o'."""
        prompt = _build_v1_system_prompt(scope="interview")
        lower = prompt.lower()
        assert "conjunciones colgantes" in lower or "nunca terminar un item con" in lower
        # Must mention the specific conjunctions
        assert "'ni'" in lower

    def test_fuma_ni_example(self):
        """Prompt must show the 'fuma ni' anti-pattern explicitly."""
        prompt = _build_v1_system_prompt(scope="interview")
        # Must show that "fuma ni" is wrong
        assert "fuma ni" in prompt.lower()

    def test_correct_splitting_example(self):
        """Prompt must show the correct split: ["fuma", "toma alcohol"]."""
        prompt = _build_v1_system_prompt(scope="interview")
        assert '"fuma"' in prompt
        assert '"toma alcohol"' in prompt

    def test_no_duplication_with_antecedentes(self):
        """Prompt must state negations[] should not duplicate antecedentes content."""
        prompt = _build_v1_system_prompt(scope="interview")
        lower = prompt.lower()
        assert "duplica" in lower or "ya" in lower

    def test_antecedentes_field_definitions(self):
        """Prompt must define what each antecedentes subfield contains."""
        prompt = _build_v1_system_prompt(scope="interview")
        lower = prompt.lower()
        # personalesNoPatologicos = habits
        assert "hábitos" in lower or "habitos" in lower
        # personalesPatologicos = chronic diseases + allergies + surgeries
        assert "alergias" in lower or "alergia" in lower
        assert "cirugías" in lower or "cirugias" in lower
