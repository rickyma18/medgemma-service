"""
Tests for ORL interview extraction quality improvements (P0).

Covers:
1. _filter_negations_against_positive_fields — chief-complaint terms removed
   from negations[]; truly negated symptoms (fiebre, tos) preserved.
2. rescue_surgeries_from_transcript — ORL procedures (rinoplastia, septoplastia,
   timpanoplastia, adenoidectomía) rescued from raw transcript.
3. _SURGERY_KEYWORDS_PA — ORL keywords present for postprocess rerouting.
4. _build_v1_user_prompt — interview scope always yields CONSULTA mode even
   when all segments have speaker="unknown".
5. Full post-processing pipeline with the real ORL sample transcript.
"""
import pytest

from app.schemas.request import Context, Transcript, TranscriptSegment
from app.services.structured_v1_extractor import (
    _filter_negations_against_positive_fields,
    _normalize_negations,
    _postprocess_interview_fields,
    _repair_v1_dict,
    _build_v1_user_prompt,
    _SURGERY_KEYWORDS_PA,
    rescue_surgeries_from_transcript,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(text: str, speaker: str = "unknown") -> TranscriptSegment:
    return TranscriptSegment(
        speaker=speaker,
        text=text,
        startMs=0,
        endMs=1000,
    )


def _make_transcript(text: str, speaker: str = "unknown") -> Transcript:
    return Transcript(
        segments=[_make_segment(text, speaker)],
        language="es",
        durationMs=1000,
    )


# Full real ORL sample text (verbatim from the bug report)
_ORL_TRANSCRIPT = (
    "que lo trae el día de hoy vengo porque desde hace tres días tengo dolor en el oído "
    "derecho como presión y punzadas no he tenido fiebre ni tos desde cuándo exactamente "
    "inició el dolor empezó el dolor en la tarde después de bañarme no he tenido secreción "
    "ni mareo tiene alguna enfermedad crónica no tengo diabetes ni hipertensión y no tomo "
    "medicamentos actualmente cirugías previas si me hicieron una rinoplastia en 2021 y una "
    "colesistectomía hace como ocho años no he tenido otras cirugías alergias no tengo "
    "alergias a medicamentos"
)


# ---------------------------------------------------------------------------
# 1. _filter_negations_against_positive_fields
# ---------------------------------------------------------------------------

class TestFilterNegationsAgainstPositiveFields:
    """Chief-complaint terms must be removed; truly negated symptoms kept."""

    def test_dolor_removed_when_chief_complaint(self):
        """'dolor' in negations[] must be dropped if motivoConsulta mentions dolor."""
        data = {
            "motivoConsulta": "Dolor en el oído derecho de 3 días",
            "padecimientoActual": "Dolor con presión y punzadas desde hace tres días.",
            "negations": ["dolor", "fiebre", "tos"],
        }
        result = _filter_negations_against_positive_fields(data)
        assert "dolor" not in result["negations"], (
            "'dolor' should be removed — it is the chief complaint, not a negation"
        )

    def test_fiebre_preserved_when_not_chief_complaint(self):
        """'fiebre' must stay in negations[] — it is truly negated and not in motivo."""
        data = {
            "motivoConsulta": "Dolor en el oído derecho de 3 días",
            "padecimientoActual": "Dolor con presión y punzadas.",
            "negations": ["fiebre", "tos"],
        }
        result = _filter_negations_against_positive_fields(data)
        assert "fiebre" in result["negations"]

    def test_tos_preserved_when_not_chief_complaint(self):
        """'tos' must stay in negations[] — truly negated in the ORL sample."""
        data = {
            "motivoConsulta": "Dolor en el oído derecho de 3 días",
            "padecimientoActual": "Dolor con presión y punzadas.",
            "negations": ["fiebre", "tos"],
        }
        result = _filter_negations_against_positive_fields(data)
        assert "tos" in result["negations"]

    def test_orl_sample_negations_filtered(self):
        """Real observed negations list: 'dolor' out, fiebre/tos in."""
        observed_negations = [
            "fiebre", "tos", "dolor", "mareo", "medicamentos",
            "alergias", "secreción", "diabetes", "tomo medicamentos", "otras cirugías",
        ]
        data = {
            "motivoConsulta": "Dolor en el oído derecho de 3 días, presión y punzadas",
            "padecimientoActual": (
                "Dolor en el oído derecho desde hace tres días, presión y punzadas. "
                "Inicio del dolor en la tarde después de bañarse."
            ),
            "negations": observed_negations,
        }
        result = _filter_negations_against_positive_fields(data)
        negations_out = result["negations"]
        assert "dolor" not in negations_out
        assert "fiebre" in negations_out
        assert "tos" in negations_out

    def test_empty_positive_fields_keeps_all_negations(self):
        """When motivo + padecimiento are null, negations[] is untouched."""
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "negations": ["fiebre", "tos", "dolor"],
        }
        result = _filter_negations_against_positive_fields(data)
        # "dolor" is in _CHIEF_COMPLAINT_EXCLUSIONS so still removed
        assert "fiebre" in result["negations"]
        assert "tos" in result["negations"]

    def test_known_exclusion_always_removed(self):
        """Terms in _CHIEF_COMPLAINT_EXCLUSIONS are always removed."""
        for term in ["dolor", "molestia", "malestar"]:
            data = {
                "motivoConsulta": None,
                "padecimientoActual": None,
                "negations": [term, "fiebre"],
            }
            result = _filter_negations_against_positive_fields(data)
            assert term not in result["negations"], (
                f"'{term}' should be removed as chief-complaint exclusion"
            )
            assert "fiebre" in result["negations"]

    def test_empty_negations_returns_unchanged(self):
        data = {"motivoConsulta": "Dolor", "padecimientoActual": None, "negations": []}
        result = _filter_negations_against_positive_fields(data)
        assert result["negations"] == []

    def test_no_negations_key_returns_unchanged(self):
        data = {"motivoConsulta": "Dolor", "padecimientoActual": None}
        result = _filter_negations_against_positive_fields(data)
        assert "negations" not in result or result.get("negations") == []

    def test_non_string_items_dropped(self):
        data = {
            "motivoConsulta": None,
            "padecimientoActual": None,
            "negations": [42, None, "fiebre"],
        }
        result = _filter_negations_against_positive_fields(data)
        assert result["negations"] == ["fiebre"]

    def test_combined_normalize_then_filter(self):
        """Integration: _normalize_negations + filter reproduce pipeline behaviour."""
        raw = ["fiebre ni", "tos", "dolor", "mareo", "medicamentos", "alergias"]
        cleaned = _normalize_negations(raw)
        data = {
            "motivoConsulta": "Dolor en el oído",
            "padecimientoActual": "Dolor de oído de 3 días.",
            "negations": cleaned,
        }
        result = _filter_negations_against_positive_fields(data)
        assert "dolor" not in result["negations"]
        assert "fiebre" in result["negations"]
        assert "tos" in result["negations"]
        assert "mareo" in result["negations"]


# ---------------------------------------------------------------------------
# 2. rescue_surgeries_from_transcript — ORL procedures
# ---------------------------------------------------------------------------

class TestRescueSurgeriesORL:
    """ORL-specific surgeries must be rescued from the raw transcript text."""

    def _base_data(self) -> dict:
        return _repair_v1_dict({
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
            "negations": [],
        })

    def test_rinoplastia_rescued(self):
        transcript = "si me hicieron una rinoplastia en 2021"
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "rinoplast" in app.lower(), (
            "rinoplastia should be rescued into personalesPatologicos"
        )

    def test_rinoplastia_with_year_rescued(self):
        transcript = _ORL_TRANSCRIPT
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "rinoplast" in app.lower()

    def test_septoplastia_rescued(self):
        transcript = "me realizaron una septoplastia hace dos años por desviación de septum"
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "septoplast" in app.lower()

    def test_timpanoplastia_rescued(self):
        transcript = "tuve una timpanoplastia izquierda hace tres años"
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "timpanoplast" in app.lower()

    def test_adenoidectomia_rescued(self):
        transcript = "de niño me hicieron una adenoidectomía"
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "adenoidectom" in app.lower()

    def test_rescue_skips_if_already_present(self):
        """If rinoplastia already in personalesPatologicos, no duplicate added."""
        data = _repair_v1_dict({
            "antecedentes": {
                "personalesPatologicos": "Rinoplastia en 2021.",
            },
        })
        transcript = "me hicieron una rinoplastia en 2021"
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        # Should appear exactly once
        assert app.lower().count("rinoplast") == 1

    def test_rescue_noop_for_non_interview_scope(self):
        """rescue_surgeries_from_transcript is a no-op for scope != interview."""
        data = self._base_data()
        transcript = "rinoplastia en 2021"
        result = rescue_surgeries_from_transcript(data, transcript, "exam")
        assert result["antecedentes"]["personalesPatologicos"] is None

    def test_orl_full_transcript_rescues_rinoplastia_and_colecistectomia(self):
        """Real ORL sample must rescue both rinoplastia and colecistectomía."""
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, _ORL_TRANSCRIPT, "interview")
        app = (result["antecedentes"]["personalesPatologicos"] or "").lower()
        assert "rinoplast" in app
        assert "colesistectom" in app or "colecistectom" in app

    def test_rescued_phrase_ends_with_period(self):
        """Each rescued phrase must be a complete sentence ending with '.'."""
        data = self._base_data()
        transcript = "rinoplastia en 2021"
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert app.strip().endswith(".")


# ---------------------------------------------------------------------------
# 3. _SURGERY_KEYWORDS_PA — ORL keywords present
# ---------------------------------------------------------------------------

class TestSurgeryKeywordsORL:
    """ORL surgery keywords must be in _SURGERY_KEYWORDS_PA for rerouting."""

    @pytest.mark.parametrize("keyword", [
        "rinoplast",
        "septoplast",
        "turbinoplast",
        "timpanoplast",
        "adenoidectom",
        "traqueotom",
    ])
    def test_orl_keyword_present(self, keyword: str):
        assert any(keyword in kw for kw in _SURGERY_KEYWORDS_PA), (
            f"'{keyword}' not found in _SURGERY_KEYWORDS_PA"
        )

    def test_existing_keywords_preserved(self):
        """Existing keywords must still be present after the edit."""
        for kw in ["apendicectom", "colecistectom", "circuncis", "tiroidectom"]:
            assert any(kw in k for k in _SURGERY_KEYWORDS_PA), (
                f"Existing keyword '{kw}' was accidentally removed"
            )


# ---------------------------------------------------------------------------
# 4. _build_v1_user_prompt — interview scope always CONSULTA
# ---------------------------------------------------------------------------

class TestUserPromptInterviewMode:
    """For scope=interview, user prompt must never say DICTADO."""

    def _ctx(self, scope: str) -> Context:
        return Context(scope=scope)

    def test_interview_unknown_speaker_is_consulta(self):
        """Single segment with speaker='unknown' + interview scope → CONSULTA."""
        transcript = _make_transcript("vengo por dolor de oído", speaker="unknown")
        ctx = self._ctx("interview")
        prompt = _build_v1_user_prompt(transcript, ctx)
        assert "CONSULTA" in prompt
        assert "DICTADO" not in prompt

    def test_interview_doctor_only_speaker_is_consulta(self):
        """Single segment with speaker='doctor' + interview scope → CONSULTA."""
        transcript = _make_transcript("dolor de garganta de tres días", speaker="doctor")
        ctx = self._ctx("interview")
        prompt = _build_v1_user_prompt(transcript, ctx)
        assert "CONSULTA" in prompt
        assert "DICTADO" not in prompt

    def test_exam_unknown_speaker_is_dictado(self):
        """For exam scope (dictation), unknown speaker → DICTADO as before."""
        transcript = _make_transcript("amígdalas grado II sin exudado", speaker="unknown")
        ctx = self._ctx("exam")
        prompt = _build_v1_user_prompt(transcript, ctx)
        assert "DICTADO" in prompt

    def test_no_scope_unknown_speaker_is_dictado(self):
        """Without scope, unknown speaker → DICTADO (existing behaviour)."""
        transcript = _make_transcript("texto clínico", speaker="unknown")
        prompt = _build_v1_user_prompt(transcript, None)
        assert "DICTADO" in prompt

    def test_interview_unsegmented_note_in_mode(self):
        """Interview without diarisation gets descriptive mode label."""
        transcript = _make_transcript(_ORL_TRANSCRIPT, speaker="unknown")
        ctx = self._ctx("interview")
        prompt = _build_v1_user_prompt(transcript, ctx)
        assert "sin segmentación de turnos" in prompt.lower() or "CONSULTA" in prompt

    def test_interview_patient_speaker_is_consulta(self):
        """If at least one patient segment exists, mode is still CONSULTA."""
        from app.schemas.request import TranscriptSegment, Transcript
        segments = [
            TranscriptSegment(speaker="doctor", text="¿Qué lo trae?", startMs=0, endMs=500),
            TranscriptSegment(speaker="patient", text="Dolor de oído.", startMs=500, endMs=1000),
        ]
        transcript = Transcript(segments=segments, language="es", durationMs=1000)
        ctx = self._ctx("interview")
        prompt = _build_v1_user_prompt(transcript, ctx)
        assert "CONSULTA" in prompt
        assert "DICTADO" not in prompt


# ---------------------------------------------------------------------------
# 5. Full pipeline integration — real ORL transcript
# ---------------------------------------------------------------------------

class TestFullPipelineORLTranscript:
    """
    Simulate the complete deterministic post-processing pipeline for the real
    ORL sample, without invoking the LLM.

    This tests that even if the LLM produces the exact bad output described in
    the bug report, the post-processing layer fixes it.
    """

    # Exact bad output the LLM produced (from the bug report)
    _LLM_OUTPUT = {
        "motivoConsulta": "Dolor en el oído derecho de 3 días, presión y punzadas",
        "padecimientoActual": (
            "Dolor en el oído derecho desde hace tres días, presión y punzadas. "
            "Inicio del dolor en la tarde después de bañarse."
        ),
        "antecedentes": {
            "heredofamiliares": None,
            "personalesNoPatologicos": None,
            "personalesPatologicos": (
                "fiebre. tos. dolor. mareo. medicamentos. alergias. secreción. "
                "diabetes. tomo medicamentos. otras cirugías. "
                "Colesistectomía hace como ocho años no he tenido otras cirugías "
                "alergias no tengo alergias a medicamentos."
            ),
        },
        "negations": [
            "fiebre", "tos", "dolor", "mareo", "medicamentos",
            "alergias", "secreción", "diabetes", "tomo medicamentos", "otras cirugías",
        ],
        "exploracionFisica": {},
        "diagnostico": None,
    }

    def _run_pipeline(self, llm_output: dict, transcript_text: str) -> dict:
        """Run only the deterministic post-processing steps (no LLM call)."""
        from app.services.structured_v1_extractor import (
            _normalize_negations,
            _filter_negations_against_positive_fields,
            _postprocess_interview_fields,
            _merge_negations_into_antecedentes,
            rescue_surgeries_from_transcript,
        )
        data = _repair_v1_dict(llm_output)
        data["negations"] = _normalize_negations(data.get("negations", []))
        data = _filter_negations_against_positive_fields(data)
        data = _postprocess_interview_fields(data, "interview")
        data = rescue_surgeries_from_transcript(data, transcript_text, "interview")
        data = _merge_negations_into_antecedentes(data, "interview")
        return data

    def test_dolor_not_in_negations_after_pipeline(self):
        """After pipeline, 'dolor' (chief complaint) must not be in negations[]."""
        data = self._run_pipeline(dict(self._LLM_OUTPUT), _ORL_TRANSCRIPT)
        assert "dolor" not in data.get("negations", [])

    def test_fiebre_and_tos_in_negations_after_pipeline(self):
        """After pipeline, truly negated 'fiebre' and 'tos' must be in negations[]
        OR merged into personalesPatologicos."""
        data = self._run_pipeline(dict(self._LLM_OUTPUT), _ORL_TRANSCRIPT)
        app = (data.get("antecedentes") or {}).get("personalesPatologicos") or ""
        negations = data.get("negations", [])
        # fiebre and tos must appear somewhere: either in negations[] or merged into app
        fiebre_present = "fiebre" in negations or "fiebre" in app.lower()
        tos_present = "tos" in negations or "tos" in app.lower()
        assert fiebre_present, "'fiebre' must survive pipeline (negations[] or antecedentes)"
        assert tos_present, "'tos' must survive pipeline (negations[] or antecedentes)"

    def test_rinoplastia_in_personales_patologicos(self):
        """Rinoplastia (2021) must appear in personalesPatologicos after rescue."""
        data = self._run_pipeline(dict(self._LLM_OUTPUT), _ORL_TRANSCRIPT)
        app = (data.get("antecedentes") or {}).get("personalesPatologicos") or ""
        assert "rinoplast" in app.lower(), (
            f"'rinoplastia' not found in personalesPatologicos: {app!r}"
        )

    def test_colecistectomia_in_personales_patologicos(self):
        """Colecistectomía must appear in personalesPatologicos."""
        data = self._run_pipeline(dict(self._LLM_OUTPUT), _ORL_TRANSCRIPT)
        app = (data.get("antecedentes") or {}).get("personalesPatologicos") or ""
        assert "colecistectom" in app.lower() or "colesistectom" in app.lower(), (
            f"Colecistectomía not found in personalesPatologicos: {app!r}"
        )

    def test_motivoConsulta_preserved(self):
        """motivoConsulta must be unchanged through the pipeline."""
        data = self._run_pipeline(dict(self._LLM_OUTPUT), _ORL_TRANSCRIPT)
        mc = data.get("motivoConsulta") or ""
        assert "oído" in mc.lower() or "dolor" in mc.lower()

    def test_padecimiento_preserved(self):
        """padecimientoActual must still describe the ear pain episode."""
        data = self._run_pipeline(dict(self._LLM_OUTPUT), _ORL_TRANSCRIPT)
        pa = data.get("padecimientoActual") or ""
        assert "oído" in pa.lower() or "dolor" in pa.lower() or "punzadas" in pa.lower()

    def test_no_negation_duplicates_after_normalize(self):
        """After normalization, 'medicamentos' and 'tomo medicamentos' should
        not both appear (verb-prefix stripping + exact dedup handles this)."""
        negations = [
            "fiebre", "tos", "dolor", "mareo", "medicamentos",
            "alergias", "secreción", "diabetes", "tomo medicamentos", "otras cirugías",
        ]
        from app.services.structured_v1_extractor import _normalize_negations
        cleaned = _normalize_negations(negations)
        seen = set()
        for item in cleaned:
            key = item.lower()
            assert key not in seen, f"Duplicate negation: {item!r}"
            seen.add(key)

    def test_tomo_medicamentos_deduped_with_medicamentos(self):
        """'tomo medicamentos' is stripped to 'medicamentos' → deduped with bare form."""
        from app.services.structured_v1_extractor import _normalize_negations
        cleaned = _normalize_negations(["medicamentos", "tomo medicamentos"])
        med_count = sum(1 for r in cleaned if r.lower() == "medicamentos")
        assert med_count == 1, (
            f"Expected exactly one 'medicamentos', got {med_count}: {cleaned}"
        )


# ---------------------------------------------------------------------------
# 6. Production ASR noise cases
# ---------------------------------------------------------------------------

class TestProductionASRNoise:
    """
    Reproduce production garbage observed in negations[]:
    'secreción por', 'se me' must be removed; clinical items survive.
    """

    def test_secrecion_por_removed(self):
        from app.services.structured_v1_extractor import _normalize_negations
        result = _normalize_negations(["secreción por", "fiebre", "tos"])
        assert all("secreción por" not in r for r in result), (
            "'secreción por' must not appear in normalized negations"
        )

    def test_se_me_removed(self):
        from app.services.structured_v1_extractor import _normalize_negations
        result = _normalize_negations(["se me", "fiebre", "mareo"])
        assert "se me" not in result, (
            "'se me' is pure ASR noise and must be removed"
        )

    def test_clinical_items_survive_noise(self):
        """After noise filtering, fiebre and tos must still be present."""
        from app.services.structured_v1_extractor import _normalize_negations
        result = _normalize_negations([
            "se me", "secreción por", "fiebre", "tos", "mareo",
        ])
        assert "fiebre" in result
        assert "tos" in result
        assert "mareo" in result

    def test_full_production_noise_list(self):
        """Real noisy negations[] from ORL log — all garbage gone, real items kept."""
        from app.services.structured_v1_extractor import _normalize_negations
        noisy = [
            "fiebre", "tos", "dolor", "mareo",
            "medicamentos", "tomo medicamentos",   # near-duplicate
            "alergias", "secreción", "diabetes",
            "se me",                               # pure ASR noise
            "secreción por",                       # trailing preposition noise
            "otras cirugías",
        ]
        result = _normalize_negations(noisy)
        lower = [r.lower() for r in result]

        # Noise items must be gone
        assert "se me" not in lower
        assert "secreción por" not in lower
        # Near-duplicate: only one form of medicamentos
        med_hits = [r for r in lower if "medicamentos" in r]
        assert len(med_hits) == 1, f"Expected 1 medicamentos form, got {med_hits}"

        # Clinical items must survive
        assert "fiebre" in lower
        assert "tos" in lower
        assert "mareo" in lower
        assert "diabetes" in lower


# ---------------------------------------------------------------------------
# 7. Septoplastia rescue with ASR typo
# ---------------------------------------------------------------------------

class TestSeptoplastiaASRTypo:
    """'esceptoplastía' must be rescued and normalised to 'Septoplastia …'."""

    def _base_data(self) -> dict:
        from app.services.structured_v1_extractor import _repair_v1_dict
        return _repair_v1_dict({
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
            "negations": [],
        })

    def test_esceptoplastia_typo_rescued(self):
        """ASR typo 'esceptoplastía' is matched by the tolerant regex."""
        transcript = "me hicieron una esceptoplastía en 2021"
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = (result["antecedentes"]["personalesPatologicos"] or "").lower()
        assert "septoplast" in app, (
            f"Expected 'septoplast' in personalesPatologicos after rescue, got: {app!r}"
        )

    def test_esceptoplastia_normalized_to_canonical(self):
        """Rescued phrase must contain 'Septoplastia' (canonical, not the typo)."""
        transcript = "esceptoplastía en 2021"
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "Septoplastia" in app, (
            f"Expected canonical 'Septoplastia', got: {app!r}"
        )

    def test_esceptoplastia_not_duplicated_if_already_in_field(self):
        """If 'esceptoplastía' is already in the field, rescue must skip."""
        from app.services.structured_v1_extractor import _repair_v1_dict
        data = _repair_v1_dict({
            "antecedentes": {
                "personalesPatologicos": "Esceptoplastía en 2021.",
            },
        })
        transcript = "me hicieron una esceptoplastía en 2021"
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        # Should appear exactly once (no duplication)
        assert app.lower().count("plast") == 1, (
            f"Duplicate rescue detected: {app!r}"
        )

    def test_canonical_septoplastia_still_rescued(self):
        """Correctly spelled 'septoplastia' also matches (existing behaviour)."""
        transcript = "me realizaron una septoplastia hace dos años"
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = (result["antecedentes"]["personalesPatologicos"] or "").lower()
        assert "septoplast" in app

    def test_siptoplastia_variant_rescued(self):
        """Another ASR variant 'siptoplastía' (vowel confusion) is also matched."""
        transcript = "tiene una siptoplastía previa"
        data = self._base_data()
        result = rescue_surgeries_from_transcript(data, transcript, "interview")
        app = (result["antecedentes"]["personalesPatologicos"] or "").lower()
        assert "septoplast" in app or "siptoplast" in app


# ---------------------------------------------------------------------------
# 8. "Niega" prefix in merge
# ---------------------------------------------------------------------------

class TestNiegaPrefixInMerge:
    """
    _merge_negations_into_antecedentes must produce 'Niega <item>.' for disease/
    surgery negations and 'No <item>.' for habit negations.
    """

    def _merge(self, negations: list, existing_app: str = "") -> dict:
        from app.services.structured_v1_extractor import (
            _merge_negations_into_antecedentes,
            _repair_v1_dict,
        )
        data = _repair_v1_dict({
            "motivoConsulta": None,
            "padecimientoActual": None,
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": existing_app or None,
            },
            "negations": negations,
        })
        return _merge_negations_into_antecedentes(data, "interview")

    def test_otras_cirugia_gets_niega_prefix(self):
        """'otras cirugías' → 'Niega otras cirugías.' in personalesPatologicos."""
        result = self._merge(["otras cirugías"])
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "Niega otras cirugías" in app, (
            f"Expected 'Niega otras cirugías' in personalesPatologicos, got: {app!r}"
        )

    def test_diabetes_gets_niega_prefix(self):
        result = self._merge(["diabetes"])
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "Niega diabetes" in app

    def test_hipertension_gets_niega_prefix(self):
        result = self._merge(["hipertensión"])
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "Niega hipertensión" in app

    def test_habit_fuma_gets_no_prefix(self):
        """Habit items go to personalesNoPatologicos with 'No' prefix."""
        result = self._merge(["fuma"])
        apnp = result["antecedentes"]["personalesNoPatologicos"] or ""
        assert "No fuma" in apnp, (
            f"Expected 'No fuma' in personalesNoPatologicos, got: {apnp!r}"
        )

    def test_already_prefixed_niega_not_doubled(self):
        """Item starting with 'Niega' must not get a second prefix."""
        result = self._merge(["Niega diabetes"])
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "Niega Niega" not in app
        assert "Niega diabetes" in app

    def test_already_prefixed_no_not_doubled(self):
        """Item starting with 'No' must not get a second prefix."""
        result = self._merge(["No fuma"])
        apnp = result["antecedentes"]["personalesNoPatologicos"] or ""
        assert "No No fuma" not in apnp

    def test_each_phrase_ends_with_period(self):
        """Every merged phrase must end with '.'."""
        result = self._merge(["diabetes", "hipertensión", "alergias"])
        app = result["antecedentes"]["personalesPatologicos"] or ""
        # All individual sentences end in period — the joined string ends in period
        assert app.rstrip().endswith(".")

    def test_appends_to_existing_field(self):
        """New clinical phrases append correctly to pre-existing field text."""
        result = self._merge(["otras cirugías"], existing_app="Rinoplastia en 2021.")
        app = result["antecedentes"]["personalesPatologicos"] or ""
        assert "Rinoplastia en 2021" in app
        assert "Niega otras cirugías" in app
        # No double period between existing and new text
        assert ".." not in app

    def test_full_orl_pipeline_has_niega_otras_cirugia(self):
        """End-to-end pipeline produces 'Niega otras cirugías.' in the field."""
        from app.services.structured_v1_extractor import (
            _normalize_negations,
            _filter_negations_against_positive_fields,
            _postprocess_interview_fields,
            _merge_negations_into_antecedentes,
            rescue_surgeries_from_transcript,
        )
        llm_output = {
            "motivoConsulta": "Dolor en el oído derecho de 3 días, presión y punzadas",
            "padecimientoActual": (
                "Dolor en el oído derecho desde hace tres días. "
                "Inicio del dolor en la tarde después de bañarse."
            ),
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
            "negations": [
                "fiebre", "tos", "dolor", "mareo", "medicamentos",
                "alergias", "secreción", "diabetes", "tomo medicamentos",
                "otras cirugías", "se me", "secreción por",
            ],
        }
        from app.services.structured_v1_extractor import _repair_v1_dict
        data = _repair_v1_dict(llm_output)
        data["negations"] = _normalize_negations(data.get("negations", []))
        data = _filter_negations_against_positive_fields(data)
        data = _postprocess_interview_fields(data, "interview")
        data = rescue_surgeries_from_transcript(data, _ORL_TRANSCRIPT, "interview")
        data = _merge_negations_into_antecedentes(data, "interview")

        app = (data.get("antecedentes") or {}).get("personalesPatologicos") or ""
        app_lower = app.lower()

        # Core requirements from the bug report
        assert "niega otras cirugías" in app_lower, (
            f"'Niega otras cirugías' not found in personalesPatologicos: {app!r}"
        )
        assert "se me" not in app_lower
        assert "secreción por" not in app_lower
        assert "rinoplast" in app_lower
