"""
Tests for interview extraction quality fixes.

Covers three focused scenarios:
1. heredofamiliares rescued from transcript when LLM missed family history.
2. Vague objectless phrases ("No estoy tomando.") removed from no_patologicos;
   alcohol/tabaco handled correctly; meds redirected to patologicos.
3. Negated pathological history captured: diabetes/HTA/cirugías/transfusiones
   appear in personalesPatologicos.
"""
import pytest

from app.services.structured_v1_extractor import (
    rescue_family_history_from_transcript,
    rescue_negated_history_from_transcript,
    _sanitize_vague_no_patologicos,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_interview_data(**overrides) -> dict:
    """Minimal interview-scope data dict with optional overrides."""
    data = {
        "motivoConsulta": None,
        "padecimientoActual": None,
        "antecedentes": {
            "heredofamiliares": None,
            "personalesNoPatologicos": None,
            "personalesPatologicos": None,
        },
        "negations": [],
    }
    if "heredofamiliares" in overrides:
        data["antecedentes"]["heredofamiliares"] = overrides["heredofamiliares"]
    if "personalesNoPatologicos" in overrides:
        data["antecedentes"]["personalesNoPatologicos"] = overrides["personalesNoPatologicos"]
    if "personalesPatologicos" in overrides:
        data["antecedentes"]["personalesPatologicos"] = overrides["personalesPatologicos"]
    return data


# ===========================================================================
# 1. heredofamiliares rescued from transcript lines
# ===========================================================================

class TestRescueFamilyHistory:
    """Family history mentions in the transcript must populate heredofamiliares."""

    def test_padre_con_hta_madre_con_dm(self):
        """Classic case: 'padre con HTA, madre con diabetes' must be rescued."""
        data = _base_interview_data()
        transcript = "mi padre tiene hipertensión y mi madre es diabética"
        result = rescue_family_history_from_transcript(data, transcript, "interview")

        hf = result["antecedentes"]["heredofamiliares"]
        assert hf is not None, "heredofamiliares should not be None"
        assert "Padre con hipertensión arterial" in hf
        assert "Madre con diabetes mellitus" in hf

    def test_hermano_con_asma(self):
        """'hermano con asma' should be captured."""
        data = _base_interview_data()
        transcript = "mi hermano tiene asma desde chiquito"
        result = rescue_family_history_from_transcript(data, transcript, "interview")

        hf = result["antecedentes"]["heredofamiliares"]
        assert hf is not None
        assert "Hermano con asma" in hf

    def test_abuelo_con_cancer(self):
        """'abuelo con cáncer' should be captured."""
        data = _base_interview_data()
        transcript = "mi abuelo tuvo cáncer de pulmón"
        result = rescue_family_history_from_transcript(data, transcript, "interview")

        hf = result["antecedentes"]["heredofamiliares"]
        assert hf is not None
        assert "cáncer" in hf.lower()

    def test_no_family_mention_no_change(self):
        """If transcript has no family terms, data is unchanged."""
        data = _base_interview_data()
        transcript = "me duele la garganta desde hace tres días"
        result = rescue_family_history_from_transcript(data, transcript, "interview")

        assert result["antecedentes"]["heredofamiliares"] is None

    def test_already_present_no_duplicate(self):
        """If heredofamiliares already has the relation+disease, don't duplicate."""
        data = _base_interview_data(heredofamiliares="Padre con hipertensión arterial.")
        transcript = "mi padre tiene hipertensión"
        result = rescue_family_history_from_transcript(data, transcript, "interview")

        hf = result["antecedentes"]["heredofamiliares"]
        # Should not add a second "Padre con hipertensión arterial"
        assert hf.count("Padre") == 1

    def test_non_interview_scope_noop(self):
        """Non-interview scope should not trigger rescue."""
        data = _base_interview_data()
        transcript = "padre con diabetes"
        result = rescue_family_history_from_transcript(data, transcript, "exam")
        assert result["antecedentes"]["heredofamiliares"] is None

    def test_mama_hipertensa_stt_variant(self):
        """Common STT variant: 'mama hipertensa' (without accent)."""
        data = _base_interview_data()
        transcript = "mi mama es hipertensa"
        result = rescue_family_history_from_transcript(data, transcript, "interview")

        hf = result["antecedentes"]["heredofamiliares"]
        assert hf is not None
        assert "hipertensión" in hf.lower()


# ===========================================================================
# 2. Vague no_patologicos handling: "No estoy tomando." removed;
#    alcohol/tabaco stay; meds go to patologicos
# ===========================================================================

class TestSanitizeVagueNoPatologicos:
    """Objectless phrases must be stripped; meds redirected to patologicos."""

    def test_no_estoy_tomando_removed(self):
        """'No estoy tomando.' (vague, no object) must be removed."""
        data = _base_interview_data(
            personalesNoPatologicos="No fuma. No estoy tomando."
        )
        result = _sanitize_vague_no_patologicos(data, "interview")

        apnp = result["antecedentes"]["personalesNoPatologicos"]
        assert apnp is not None, "Should keep 'No fuma.'"
        assert "No fuma" in apnp
        assert "No estoy tomando" not in apnp

    def test_no_tomo_removed(self):
        """'No tomo.' (vague) must be removed."""
        data = _base_interview_data(
            personalesNoPatologicos="No tomo."
        )
        result = _sanitize_vague_no_patologicos(data, "interview")

        apnp = result["antecedentes"]["personalesNoPatologicos"]
        assert apnp is None, "Only content was vague; field should become None"

    def test_no_consumo_removed(self):
        """'No consumo.' (vague) must be removed."""
        data = _base_interview_data(
            personalesNoPatologicos="No consumo."
        )
        result = _sanitize_vague_no_patologicos(data, "interview")

        apnp = result["antecedentes"]["personalesNoPatologicos"]
        assert apnp is None

    def test_alcohol_tabaco_preserved(self):
        """'No toma alcohol. No fuma.' should be preserved (object is explicit)."""
        data = _base_interview_data(
            personalesNoPatologicos="No toma alcohol. No fuma."
        )
        result = _sanitize_vague_no_patologicos(data, "interview")

        apnp = result["antecedentes"]["personalesNoPatologicos"]
        assert apnp is not None
        assert "No toma alcohol" in apnp
        assert "No fuma" in apnp

    def test_meds_redirected_to_patologicos(self):
        """'No toma medicamentos.' redirected to personalesPatologicos."""
        data = _base_interview_data(
            personalesNoPatologicos="No fuma. No toma medicamentos."
        )
        result = _sanitize_vague_no_patologicos(data, "interview")

        apnp = result["antecedentes"]["personalesNoPatologicos"]
        app = result["antecedentes"]["personalesPatologicos"]

        assert apnp is not None
        assert "No fuma" in apnp
        assert "medicamentos" not in (apnp or "")

        assert app is not None
        assert "Niega medicamentos" in app

    def test_meds_not_duplicated_in_patologicos(self):
        """If patologicos already has 'Niega medicamentos.', don't add again."""
        data = _base_interview_data(
            personalesNoPatologicos="No toma medicamentos.",
            personalesPatologicos="Niega medicamentos. Niega alergias."
        )
        result = _sanitize_vague_no_patologicos(data, "interview")

        app = result["antecedentes"]["personalesPatologicos"]
        assert app.count("Niega medicamentos") == 1

    def test_non_interview_scope_noop(self):
        """Non-interview scope should not sanitize."""
        data = _base_interview_data(
            personalesNoPatologicos="No estoy tomando."
        )
        result = _sanitize_vague_no_patologicos(data, "exam")
        assert result["antecedentes"]["personalesNoPatologicos"] == "No estoy tomando."


# ===========================================================================
# 3. Negated pathological history captured: diabetes/HTA/cirugías/transfusiones
# ===========================================================================

class TestRescueNegatedHistory:
    """Negated pat history in transcript must appear in personalesPatologicos."""

    def test_niega_diabetes_captured(self):
        """'niega diabetes' should be rescued to personalesPatologicos."""
        data = _base_interview_data()
        transcript = "el paciente niega diabetes"
        result = rescue_negated_history_from_transcript(data, transcript, "interview")

        app = result["antecedentes"]["personalesPatologicos"]
        assert app is not None
        assert "Niega diabetes" in app

    def test_niega_hipertension_captured(self):
        """'niega hipertensión' should be rescued."""
        data = _base_interview_data()
        transcript = "niega hipertensión"
        result = rescue_negated_history_from_transcript(data, transcript, "interview")

        app = result["antecedentes"]["personalesPatologicos"]
        assert "Niega hipertensión" in app

    def test_niega_cirugias_captured(self):
        """'niega cirugías previas' should be rescued."""
        data = _base_interview_data()
        transcript = "niega cirugías previas"
        result = rescue_negated_history_from_transcript(data, transcript, "interview")

        app = result["antecedentes"]["personalesPatologicos"]
        assert "Niega cirugías previas" in app

    def test_niega_transfusiones_captured(self):
        """'niega transfusiones' should be rescued."""
        data = _base_interview_data()
        transcript = "niega transfusiones"
        result = rescue_negated_history_from_transcript(data, transcript, "interview")

        app = result["antecedentes"]["personalesPatologicos"]
        assert "Niega transfusiones" in app

    def test_multiple_negated_items(self):
        """Multiple negated items should all appear."""
        data = _base_interview_data()
        transcript = (
            "niega diabetes niega hipertensión niega cirugías niega transfusiones"
        )
        result = rescue_negated_history_from_transcript(data, transcript, "interview")

        app = result["antecedentes"]["personalesPatologicos"]
        assert "Niega diabetes" in app
        assert "Niega hipertensión" in app
        assert "Niega cirugías previas" in app
        assert "Niega transfusiones" in app

    def test_already_present_no_duplicate(self):
        """If 'Niega diabetes' already in patologicos, don't add again."""
        data = _base_interview_data(
            personalesPatologicos="Niega diabetes."
        )
        transcript = "niega diabetes"
        result = rescue_negated_history_from_transcript(data, transcript, "interview")

        app = result["antecedentes"]["personalesPatologicos"]
        assert app.count("diabetes") == 1

    def test_appends_to_existing_content(self):
        """Rescued items append to existing personalesPatologicos."""
        data = _base_interview_data(
            personalesPatologicos="Alergia a sulfas."
        )
        transcript = "niega diabetes niega cirugías"
        result = rescue_negated_history_from_transcript(data, transcript, "interview")

        app = result["antecedentes"]["personalesPatologicos"]
        assert "Alergia a sulfas" in app
        assert "Niega diabetes" in app
        assert "Niega cirugías previas" in app

    def test_no_tengo_diabetes_variant(self):
        """'no tengo diabetes' should also be captured."""
        data = _base_interview_data()
        transcript = "no tengo diabetes ni hipertensión"
        result = rescue_negated_history_from_transcript(data, transcript, "interview")

        app = result["antecedentes"]["personalesPatologicos"]
        assert app is not None
        assert "diabetes" in app.lower()

    def test_non_interview_scope_noop(self):
        """Non-interview scope should not trigger rescue."""
        data = _base_interview_data()
        transcript = "niega diabetes"
        result = rescue_negated_history_from_transcript(data, transcript, "exam")
        assert result["antecedentes"]["personalesPatologicos"] is None

    def test_symptom_negations_not_captured(self):
        """Symptom negations (fiebre, tos) should NOT go into patologicos."""
        data = _base_interview_data()
        transcript = "niega fiebre niega tos niega mareo"
        result = rescue_negated_history_from_transcript(data, transcript, "interview")

        # These symptom negations should NOT be rescued into patologicos
        app = result["antecedentes"]["personalesPatologicos"]
        assert app is None, "Symptom negations should not appear in patologicos"
