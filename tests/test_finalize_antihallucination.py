"""
Tests for anti-hallucination evidence policy in /v1/finalize (FIX #6).

Validates:
- _has_evidence: keyword detection per field
- _enforce_evidence_policy: null fields stay null when no evidence
- _enforce_evidence_policy: null fields CAN be filled when evidence exists
- Full endpoint integration: finalize doesn't invent content
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi import Request
from fastapi.testclient import TestClient

from app.main import create_app
from app.core.auth import verify_auth_header
from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1,
    Antecedentes,
    ExploracionFisica,
    Diagnostico,
)
from app.api.finalize import _has_evidence, _enforce_evidence_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def mock_verify_auth_header(request: Request) -> None:
    request.state.uid = "test_user"


@pytest.fixture
def test_app():
    app = create_app()
    app.dependency_overrides[verify_auth_header] = mock_verify_auth_header
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


@pytest.fixture
def mock_contracts():
    with patch("app.api.finalize.check_contracts") as mock:
        mock.return_value = {"warnings": [], "details": None}
        yield mock


def _make_transcript_payload(*texts):
    segments = []
    offset = 0
    for t in texts:
        dur = len(t) * 50
        segments.append({
            "speaker": "doctor",
            "text": t,
            "startMs": offset,
            "endMs": offset + dur,
        })
        offset += dur
    return {"segments": segments, "language": "es", "durationMs": offset}


# ---------------------------------------------------------------------------
# Unit: _has_evidence
# ---------------------------------------------------------------------------

class TestHasEvidence:

    def test_empty_text_returns_false(self):
        assert _has_evidence("", "motivoConsulta") is False

    def test_matching_keyword(self):
        assert _has_evidence("paciente refiere dolor intenso", "motivoConsulta") is True

    def test_no_matching_keyword(self):
        assert _has_evidence("el cielo esta nublado", "motivoConsulta") is False

    def test_heredofamiliares_evidence(self):
        assert _has_evidence("padre diabetico", "heredofamiliares") is True

    def test_personalesNoPatologicos_evidence(self):
        assert _has_evidence("no fuma ni toma alcohol", "personalesNoPatologicos") is True

    def test_personalesPatologicos_evidence(self):
        assert _has_evidence("alergico a sulfas", "personalesPatologicos") is True

    def test_exploracion_evidence(self):
        assert _has_evidence("cornetes hipertroficos", "exploracionFisica") is True

    def test_plan_evidence(self):
        assert _has_evidence("amoxicilina 500mg cada 8h", "planTratamiento") is True

    def test_unknown_field_returns_false(self):
        assert _has_evidence("algo de texto", "campoInventado") is False

    def test_case_insensitive(self):
        assert _has_evidence("PADRE DIABETICO", "heredofamiliares") is True


# ---------------------------------------------------------------------------
# Unit: _enforce_evidence_policy
# ---------------------------------------------------------------------------

class TestEnforceEvidencePolicy:

    def test_null_stays_null_without_evidence(self):
        """Fields null in original stay null if transcript has no evidence."""
        original = StructuredFieldsV1()  # all nulls
        refined = StructuredFieldsV1(
            motivoConsulta="Dolor de garganta inventado",
            padecimientoActual="Inicio hace 3 dias inventado",
            planTratamiento="Amoxicilina inventada",
        )
        result = _enforce_evidence_policy(
            original, refined, "el cielo esta nublado hoy"
        )

        assert result.motivo_consulta is None
        assert result.padecimiento_actual is None
        assert result.plan_tratamiento is None

    def test_field_kept_when_evidence_exists(self):
        """Fields can be filled if transcript contains evidence."""
        original = StructuredFieldsV1()  # all nulls
        refined = StructuredFieldsV1(
            motivoConsulta="Dolor de garganta",
        )
        result = _enforce_evidence_policy(
            original, refined, "paciente refiere dolor de garganta"
        )

        # "dolor" is a keyword for motivoConsulta
        assert result.motivo_consulta is not None

    def test_existing_fields_not_reverted(self):
        """Fields already non-null in original are never touched."""
        original = StructuredFieldsV1(
            motivoConsulta="Dolor original",
        )
        refined = StructuredFieldsV1(
            motivoConsulta="Dolor reescrito",
        )
        result = _enforce_evidence_policy(
            original, refined, "texto sin evidencia alguna"
        )

        # Original was non-null, so refinement is kept regardless of evidence
        assert result.motivo_consulta == "Dolor reescrito"

    def test_antecedentes_heredofamiliares_no_evidence(self):
        """Heredofamiliares reverted if no family keywords in transcript."""
        original = StructuredFieldsV1(
            antecedentes=Antecedentes(heredofamiliares=None)
        )
        refined = StructuredFieldsV1(
            antecedentes=Antecedentes(heredofamiliares="Padre con DM2")
        )
        result = _enforce_evidence_policy(
            original, refined, "paciente con tos cronica"
        )

        assert result.antecedentes.heredofamiliares is None

    def test_antecedentes_heredofamiliares_with_evidence(self):
        """Heredofamiliares kept when transcript mentions 'padre'."""
        original = StructuredFieldsV1(
            antecedentes=Antecedentes(heredofamiliares=None)
        )
        refined = StructuredFieldsV1(
            antecedentes=Antecedentes(heredofamiliares="Padre con DM2")
        )
        result = _enforce_evidence_policy(
            original, refined, "padre diabetico, madre sana"
        )

        assert result.antecedentes.heredofamiliares == "Padre con DM2"

    def test_antecedentes_patologicos_alergias(self):
        """Alergias preserved when transcript mentions 'alergico'."""
        original = StructuredFieldsV1(
            antecedentes=Antecedentes(personales_patologicos=None)
        )
        refined = StructuredFieldsV1(
            antecedentes=Antecedentes(personales_patologicos="Alergia a sulfas")
        )
        result = _enforce_evidence_policy(
            original, refined, "paciente alergico a sulfas"
        )

        assert result.antecedentes.personales_patologicos == "Alergia a sulfas"

    def test_exploracion_no_evidence(self):
        """Exploracion fields reverted without exam keywords."""
        original = StructuredFieldsV1(
            exploracion_fisica=ExploracionFisica(orofaringe=None)
        )
        refined = StructuredFieldsV1(
            exploracion_fisica=ExploracionFisica(orofaringe="Amigdalas eritematosas")
        )
        result = _enforce_evidence_policy(
            original, refined, "el paciente dice que le duele"
        )

        assert result.exploracion_fisica.orofaringe is None

    def test_exploracion_with_evidence(self):
        """Exploracion kept when transcript has exam findings."""
        original = StructuredFieldsV1(
            exploracion_fisica=ExploracionFisica(orofaringe=None)
        )
        refined = StructuredFieldsV1(
            exploracion_fisica=ExploracionFisica(orofaringe="Amigdalas eritematosas")
        )
        result = _enforce_evidence_policy(
            original, refined, "a la exploracion amigdalas eritematosas"
        )

        assert result.exploracion_fisica.orofaringe == "Amigdalas eritematosas"

    def test_mixed_fields_partial_evidence(self):
        """Only fields WITH evidence survive; others revert."""
        original = StructuredFieldsV1()
        refined = StructuredFieldsV1(
            motivoConsulta="Dolor inventado",
            antecedentes=Antecedentes(
                heredofamiliares="Padre con DM2",
                personales_no_patologicos="Niega tabaquismo",
            ),
        )
        # Transcript has evidence for heredofamiliares but NOT for motivo
        result = _enforce_evidence_policy(
            original, refined, "padre diabetico, madre sana"
        )

        assert result.motivo_consulta is None  # no evidence
        assert result.antecedentes.heredofamiliares == "Padre con DM2"  # has evidence
        assert result.antecedentes.personales_no_patologicos is None  # no "fuma/alcohol"


# ---------------------------------------------------------------------------
# Integration: full /v1/finalize endpoint
# ---------------------------------------------------------------------------

class TestFinalizeEndpointAntiHallucination:

    def test_empty_draft_no_transcript_stays_null(self, client, mock_contracts):
        """Empty reduce draft + no transcript = all fields stay null."""
        empty_fields = StructuredFieldsV1()

        payload = {
            "structuredFields": empty_fields.model_dump(by_alias=True),
        }

        response = client.post("/v1/finalize", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]

        assert data["motivoConsulta"] is None
        assert data["padecimientoActual"] is None
        assert data["planTratamiento"] is None

    def test_empty_draft_with_transcript_evidence_preserves(
        self, client, mock_contracts
    ):
        """With evidence in transcript, fields populated by refinement survive."""
        empty_fields = StructuredFieldsV1()

        payload = {
            "structuredFields": empty_fields.model_dump(by_alias=True),
            "transcript": _make_transcript_payload(
                "Padre diabetico.",
                "Niega alergias.",
            ),
            # refine=False, so no LLM call; fields stay null naturally
        }

        response = client.post("/v1/finalize", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]

        # Without refinement, fields stay null (policy only checks refined vs original)
        assert data["motivoConsulta"] is None
        assert data["padecimientoActual"] is None

    def test_refine_does_not_invent_motivo(self, client, mock_contracts):
        """Refinement must not invent motivoConsulta from thin air."""
        fields = StructuredFieldsV1(
            padecimientoActual="Odinofagia de 3 dias",
        )

        payload = {
            "structuredFields": fields.model_dump(by_alias=True),
            "transcript": _make_transcript_payload("Odinofagia de 3 dias"),
            "refine": True,
        }

        # Mock refinement to NOT invent new fields
        with patch(
            "app.services.pipeline_orl._finalize_refine_fields",
            new_callable=MagicMock,
        ) as mock_refine:
            async def passthrough(x):
                return x

            mock_refine.side_effect = passthrough

            response = client.post("/v1/finalize", json=payload)

        assert response.status_code == 200
        data = response.json()["data"]
        # motivoConsulta was null in input and should stay null
        assert data["motivoConsulta"] is None

    def test_refine_hallucinated_field_reverted(self, client, mock_contracts):
        """If refinement invents a field, evidence policy reverts it."""
        original = StructuredFieldsV1()

        payload = {
            "structuredFields": original.model_dump(by_alias=True),
            "transcript": _make_transcript_payload("el cielo esta nublado"),
            "refine": True,
        }

        # Mock refinement that hallucinated motivoConsulta
        with patch(
            "app.services.pipeline_orl._finalize_refine_fields",
            new_callable=MagicMock,
        ) as mock_refine:
            async def hallucinate(fields):
                fields.motivo_consulta = "Dolor de garganta inventado"
                return fields

            mock_refine.side_effect = hallucinate

            response = client.post("/v1/finalize", json=payload)

        assert response.status_code == 200
        data = response.json()["data"]
        # Evidence policy should revert the hallucination
        assert data["motivoConsulta"] is None
