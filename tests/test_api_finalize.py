import pytest
from unittest.mock import patch, MagicMock
from fastapi import Request
from fastapi.testclient import TestClient
from app.main import create_app
from app.core.auth import verify_auth_header
from app.schemas.structured_fields_v1 import StructuredFieldsV1
from app.schemas.request import Transcript, TranscriptSegment


# Helper to bypass auth - MUST have Request type annotation!
async def mock_verify_auth_header(request: Request) -> None:
    request.state.uid = "test_user"


@pytest.fixture
def test_app():
    """Create a fresh app instance using create_app() for realistic testing."""
    app = create_app()
    app.dependency_overrides[verify_auth_header] = mock_verify_auth_header
    return app


@pytest.fixture
def client(test_app):
    """Create TestClient with properly configured app."""
    return TestClient(test_app)


@pytest.fixture
def mock_contracts():
    with patch("app.api.finalize.check_contracts") as mock:
        yield mock


@pytest.fixture
def sample_fields():
    return StructuredFieldsV1(
        motivoConsulta="Dolor de garganta",
        padecimientoActual="Inicia hace 3 dias",
        diagnostico={"texto": "Faringitis", "tipo": "presuntivo"}
    )


def test_finalize_success_ok(client, mock_contracts, sample_fields):
    """Test standard success case with clean contracts."""
    mock_contracts.return_value = {
        "warnings": [],
        "details": {"some": "detail"},
        "medicalizationDrift": False,
        "normalizationDrift": False
    }

    payload = {
        "structuredFields": sample_fields.model_dump(by_alias=True)
    }

    response = client.post("/v1/finalize", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["motivoConsulta"] == "Dolor de garganta"

    metadata = data["metadata"]
    assert metadata["contractStatus"] == "ok"
    assert metadata["contractWarnings"] == []
    assert metadata["confidence"] == 1.0
    assert "requestId" in metadata

    # Flutter compat fields
    assert "warnings" in metadata
    assert metadata["warnings"] == metadata["contractWarnings"]
    assert "confidenceLabel" in metadata
    assert metadata["confidenceLabel"] in ("baja", "media", "alta")
    assert "usedEvidence" in metadata
    assert isinstance(metadata["usedEvidence"], bool)


def test_finalize_success_warning(client, mock_contracts, sample_fields):
    """Test success case but with contract warnings."""
    mock_contracts.return_value = {
        "warnings": ["medicalization_snapshot_missing"],
        "details": {},
        "medicalizationDrift": False,
        "normalizationDrift": False
    }

    payload = {
        "structuredFields": sample_fields.model_dump(by_alias=True)
    }

    response = client.post("/v1/finalize", json=payload)

    assert response.status_code == 200
    data = response.json()
    metadata = data["metadata"]
    assert metadata["contractStatus"] == "warning"
    assert "medicalization_snapshot_missing" in metadata["contractWarnings"]

    # Flutter compat: warnings must match contractWarnings
    assert metadata["warnings"] == metadata["contractWarnings"]
    assert "medicalization_snapshot_missing" in metadata["warnings"]


def test_finalize_missing_fields(client, mock_contracts):
    """Test validation error when structuredFields is missing.

    Note: create_app() includes PHI-safe error handler that returns 400
    with generic message instead of 422 with field details.
    """
    payload = {
        "transcript": "some text"
    }
    response = client.post("/v1/finalize", json=payload)
    # PHI-safe handler converts validation errors to 400 BAD_REQUEST
    assert response.status_code == 400
    assert response.json()["success"] is False
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_finalize_with_refine_flag(client, mock_contracts, sample_fields):
    """Test that refine flag constructs request but logic is mocked/handled."""
    mock_contracts.return_value = {"warnings": [], "details": None}

    payload = {
        "structuredFields": sample_fields.model_dump(by_alias=True),
        "refine": True
    }

    # We need to mock _finalize_refine_fields since it would try to call LLM
    with patch("app.services.pipeline_orl._finalize_refine_fields", new_callable=MagicMock) as mock_refine:
        # Mock it returning the SAME fields
        async def async_refine(x):
            return x
        mock_refine.side_effect = async_refine

        response = client.post("/v1/finalize", json=payload)

        assert response.status_code == 200
        assert mock_refine.called


# --- Flutter backwards compatibility tests ---

def test_finalize_flutter_compat_warnings_alias(client, mock_contracts, sample_fields):
    """Test that warnings field is an alias of contractWarnings."""
    test_warnings = ["warn1", "warn2", "normalization_drift"]
    mock_contracts.return_value = {
        "warnings": test_warnings,
        "details": None,
    }

    payload = {"structuredFields": sample_fields.model_dump(by_alias=True)}
    response = client.post("/v1/finalize", json=payload)

    assert response.status_code == 200
    metadata = response.json()["metadata"]

    # warnings must exist and be identical to contractWarnings
    assert "warnings" in metadata
    assert "contractWarnings" in metadata
    assert metadata["warnings"] == metadata["contractWarnings"]
    assert metadata["warnings"] == test_warnings


def test_finalize_flutter_compat_confidence_label(client, mock_contracts, sample_fields):
    """Test that confidenceLabel exists and is valid."""
    mock_contracts.return_value = {"warnings": [], "details": None}

    payload = {"structuredFields": sample_fields.model_dump(by_alias=True)}
    response = client.post("/v1/finalize", json=payload)

    assert response.status_code == 200
    metadata = response.json()["metadata"]

    # confidenceLabel must exist and be one of the valid labels
    assert "confidenceLabel" in metadata
    assert metadata["confidenceLabel"] in ("baja", "media", "alta")

    # With default confidence=1.0, label should be "alta"
    assert metadata["confidenceLabel"] == "alta"


def test_finalize_flutter_compat_used_evidence_bool(client, mock_contracts, sample_fields):
    """Test that usedEvidence is a boolean (not null/list)."""
    mock_contracts.return_value = {"warnings": [], "details": None}

    payload = {"structuredFields": sample_fields.model_dump(by_alias=True)}
    response = client.post("/v1/finalize", json=payload)

    assert response.status_code == 200
    metadata = response.json()["metadata"]

    # usedEvidence must be bool, not None or list
    assert "usedEvidence" in metadata
    assert isinstance(metadata["usedEvidence"], bool)
    # Default should be False when no evidence
    assert metadata["usedEvidence"] is False


def test_finalize_legacy_structured_v1_field(client, mock_contracts, sample_fields):
    """Test that legacy structuredV1 field is accepted and mapped to structuredFields."""
    mock_contracts.return_value = {"warnings": [], "details": None}

    # Use legacy field name
    payload = {
        "structuredV1": sample_fields.model_dump(by_alias=True)
    }

    response = client.post("/v1/finalize", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # Data should be populated from structuredV1
    assert data["data"]["motivoConsulta"] == "Dolor de garganta"


def test_finalize_both_structured_fields_prefers_new(client, mock_contracts, sample_fields):
    """Test that structuredFields takes precedence over structuredV1 if both provided."""
    mock_contracts.return_value = {"warnings": [], "details": None}

    # Create different sample for legacy
    legacy_fields = StructuredFieldsV1(
        motivoConsulta="Legacy motivo",
        padecimientoActual="Legacy padecimiento"
    )

    payload = {
        "structuredFields": sample_fields.model_dump(by_alias=True),
        "structuredV1": legacy_fields.model_dump(by_alias=True)
    }

    response = client.post("/v1/finalize", json=payload)

    assert response.status_code == 200
    data = response.json()
    # Should use structuredFields, not structuredV1
    assert data["data"]["motivoConsulta"] == "Dolor de garganta"
    assert data["data"]["motivoConsulta"] != "Legacy motivo"


# --- Consistency check tests (Epic 6) ---

def _make_transcript_payload(*texts: str) -> dict:
    """Build a minimal Transcript dict from plain text strings."""
    segments = []
    offset = 0
    for t in texts:
        dur = len(t) * 50  # arbitrary ms per char
        segments.append({
            "speaker": "doctor",
            "text": t,
            "startMs": offset,
            "endMs": offset + dur,
        })
        offset += dur
    return {
        "segments": segments,
        "language": "es",
        "durationMs": offset,
    }


def test_consistency_false_no_extra_warnings(client, mock_contracts):
    """check_consistency=false should NOT add consistency warnings."""
    mock_contracts.return_value = {"warnings": [], "details": None}

    fields = StructuredFieldsV1(
        motivoConsulta="Dolor de garganta",
        antecedentes={"personalesNoPatologicos": "Tabaquismo positivo, 10 cigarros/día"},
        diagnostico={"texto": "Faringitis", "tipo": "presuntivo"},
    )

    payload = {
        "structuredFields": fields.model_dump(by_alias=True),
        "transcript": _make_transcript_payload(
            "Paciente niega tabaquismo."
        ),
        "checkConsistency": False,
    }

    response = client.post("/v1/finalize", json=payload)
    assert response.status_code == 200
    metadata = response.json()["metadata"]

    # No consistency warnings should appear
    assert metadata["contractWarnings"] == []
    assert metadata["warnings"] == metadata["contractWarnings"]
    assert metadata["contractStatus"] == "ok"


def test_consistency_true_detects_negation_contradiction(client, mock_contracts):
    """check_consistency=true + 'niega tabaquismo' + field with tabaquismo => warning."""
    mock_contracts.return_value = {"warnings": [], "details": None}

    fields = StructuredFieldsV1(
        motivoConsulta="Dolor de garganta",
        antecedentes={"personalesNoPatologicos": "Tabaquismo positivo, 10 cigarros/día"},
        diagnostico={"texto": "Faringitis", "tipo": "presuntivo"},
    )

    payload = {
        "structuredFields": fields.model_dump(by_alias=True),
        "transcript": _make_transcript_payload(
            "Paciente niega tabaquismo."
        ),
        "checkConsistency": True,
    }

    response = client.post("/v1/finalize", json=payload)
    assert response.status_code == 200
    metadata = response.json()["metadata"]

    # At least one consistency warning must be present
    assert len(metadata["contractWarnings"]) >= 1
    assert metadata["contractStatus"] == "warning"

    # Find the consistency dict warning
    consistency_warns = [
        w for w in metadata["contractWarnings"]
        if isinstance(w, dict) and w.get("type") == "consistency"
    ]
    assert len(consistency_warns) >= 1

    w = consistency_warns[0]
    assert w["severity"] == "warning"
    assert w["field"] == "antecedentes.personalesNoPatologicos"
    assert "tabaquismo" in w["message"].lower()
    assert "evidence" in w
    assert len(w["evidence"]) <= 160

    # Flutter compat: warnings mirrors contractWarnings
    assert metadata["warnings"] == metadata["contractWarnings"]


def test_consistency_true_no_transcript_ok(client, mock_contracts):
    """check_consistency=true but no transcript => 200 OK, no warnings."""
    mock_contracts.return_value = {"warnings": [], "details": None}

    fields = StructuredFieldsV1(
        motivoConsulta="Dolor de garganta",
        diagnostico={"texto": "Faringitis", "tipo": "presuntivo"},
    )

    payload = {
        "structuredFields": fields.model_dump(by_alias=True),
        "checkConsistency": True,
        # No transcript provided
    }

    response = client.post("/v1/finalize", json=payload)
    assert response.status_code == 200
    metadata = response.json()["metadata"]

    assert metadata["contractWarnings"] == []
    assert metadata["warnings"] == []
    assert metadata["contractStatus"] == "ok"


def test_consistency_true_accepts_string_transcript(client, mock_contracts):
    """check_consistency=true with plain string transcript detects contradiction."""
    mock_contracts.return_value = {"warnings": [], "details": None}

    fields = StructuredFieldsV1(
        motivoConsulta="Dolor de garganta",
        antecedentes={"personalesPatologicos": "Alergia a penicilina"},
        diagnostico={"texto": "Faringitis", "tipo": "presuntivo"},
    )

    # Plain string transcript instead of Transcript object
    payload = {
        "structuredFields": fields.model_dump(by_alias=True),
        "transcript": "Paciente niega alergias conocidas. Refiere dolor de garganta hace 3 días.",
        "checkConsistency": True,
    }

    response = client.post("/v1/finalize", json=payload)
    assert response.status_code == 200
    metadata = response.json()["metadata"]

    # Should detect contradiction: "niega alergias" but field has allergy
    assert len(metadata["contractWarnings"]) >= 1
    assert metadata["contractStatus"] == "warning"

    # Find the consistency warning
    consistency_warns = [
        w for w in metadata["contractWarnings"]
        if isinstance(w, dict) and w.get("type") == "consistency"
    ]
    assert len(consistency_warns) >= 1

    w = consistency_warns[0]
    assert w["severity"] == "warning"
    assert w["field"] == "antecedentes.personalesPatologicos"
    assert "evidence" in w
    assert len(w["evidence"]) <= 160

    # Flutter compat
    assert metadata["warnings"] == metadata["contractWarnings"]
