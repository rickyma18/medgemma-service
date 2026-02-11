"""
Tests for interview scope anti-sparse behavior.

Ensures that:
1. Context negations are merged into antecedentes for scope=interview.
2. Job status response includes extractionMeta with hasContent/negatedFindingsCount.
3. Finalize response includes hasContent/negatedFindingsCount metadata.
4. V1 antecedentes keys use correct camelCase aliases.
5. Backward compatibility is preserved.

No PHI - all test data is synthetic.
"""
import pytest
from unittest.mock import patch, MagicMock

from fastapi import Request
from fastapi.testclient import TestClient

from app.main import create_app
from app.core.auth import verify_auth_header
from app.schemas.structured_fields_v1 import StructuredFieldsV1, Antecedentes
from app.schemas.request import ExtractRequest, Transcript, TranscriptSegment
from app.services.structured_v1_extractor import (
    _merge_negations_into_antecedentes,
    compute_extraction_meta,
)
from app.services.job_manager import Job, JobManager


# ── Helpers ──────────────────────────────────────────────────────────────────


async def mock_verify_auth_header(request: Request) -> None:
    request.state.uid = "test_user"


def _make_transcript_payload(*texts: str) -> dict:
    """Build a minimal Transcript dict from plain text strings."""
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
    return {
        "segments": segments,
        "language": "es",
        "durationMs": offset,
    }


dummy_transcript = Transcript(
    segments=[
        TranscriptSegment(speaker="doctor", text="Test", startMs=0, endMs=1000)
    ],
    durationMs=1000,
    language="es",
)
dummy_request = ExtractRequest(transcript=dummy_transcript)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def reset_job_manager():
    JobManager._instance = None
    yield
    JobManager._instance = None


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


# ── Unit tests: _merge_negations_into_antecedentes ───────────────────────────


class TestMergeNegationsIntoAntecedentes:
    """Unit tests for the negation -> antecedentes routing logic."""

    def test_allergy_negation_routes_to_patologicos(self):
        """'niega alergias' should route to personalesPatologicos."""
        data = {
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
            "negations": ["niega alergias"],
        }
        result = _merge_negations_into_antecedentes(data, "interview")

        assert result["antecedentes"]["personalesPatologicos"] is not None
        assert "niega alergias" in result["antecedentes"]["personalesPatologicos"].lower()

    def test_family_diabetes_routes_to_heredofamiliares(self):
        """'Padre diabetico' should route to heredofamiliares."""
        data = {
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
            "negations": ["Padre diabetico"],
        }
        result = _merge_negations_into_antecedentes(data, "interview")

        assert result["antecedentes"]["heredofamiliares"] is not None
        assert "padre" in result["antecedentes"]["heredofamiliares"].lower()

    def test_habit_negation_routes_to_no_patologicos(self):
        """'no fuma' should route to personalesNoPatologicos."""
        data = {
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
            "negations": ["no fuma"],
        }
        result = _merge_negations_into_antecedentes(data, "interview")

        assert result["antecedentes"]["personalesNoPatologicos"] is not None
        assert "fuma" in result["antecedentes"]["personalesNoPatologicos"].lower()

    def test_non_interview_scope_is_noop(self):
        """Non-interview scopes should NOT merge negations."""
        data = {
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
            "negations": ["niega alergias"],
        }
        result = _merge_negations_into_antecedentes(data, "exam")

        assert result["antecedentes"]["personalesPatologicos"] is None
        # negations remain untouched
        assert result["negations"] == ["niega alergias"]

    def test_appends_to_existing_antecedentes(self):
        """Merged negations should append to existing antecedentes content."""
        data = {
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": "DM2 con metformina",
            },
            "negations": ["niega alergias"],
        }
        result = _merge_negations_into_antecedentes(data, "interview")

        pp = result["antecedentes"]["personalesPatologicos"]
        assert "DM2 con metformina" in pp
        assert "niega alergias" in pp.lower()

    def test_clears_negations_after_merge(self):
        """After merging for interview scope, negations array is cleared."""
        data = {
            "antecedentes": {},
            "negations": ["niega alergias"],
        }
        result = _merge_negations_into_antecedentes(data, "interview")

        assert result["negations"] == []

    def test_empty_negations_is_noop(self):
        """Empty negations should not modify antecedentes."""
        data = {
            "antecedentes": {
                "heredofamiliares": None,
                "personalesNoPatologicos": None,
                "personalesPatologicos": None,
            },
            "negations": [],
        }
        result = _merge_negations_into_antecedentes(data, "interview")

        assert result["antecedentes"]["heredofamiliares"] is None
        assert result["antecedentes"]["personalesNoPatologicos"] is None
        assert result["antecedentes"]["personalesPatologicos"] is None


# ── Unit tests: compute_extraction_meta ──────────────────────────────────────


class TestComputeExtractionMeta:
    """Unit tests for the PHI-safe extraction metadata helper."""

    def test_all_null_no_negations_is_not_content(self):
        fields = StructuredFieldsV1()
        meta = compute_extraction_meta(fields)
        assert meta["hasContent"] is False
        assert meta["negatedFindingsCount"] == 0

    def test_negations_present_is_content(self):
        fields = StructuredFieldsV1(negations=["niega alergias"])
        meta = compute_extraction_meta(fields)
        assert meta["hasContent"] is True
        assert meta["negatedFindingsCount"] == 1

    def test_antecedentes_present_is_content(self):
        fields = StructuredFieldsV1(
            antecedentes=Antecedentes(heredofamiliares="Padre con DM2")
        )
        meta = compute_extraction_meta(fields)
        assert meta["hasContent"] is True
        assert meta["negatedFindingsCount"] == 0

    def test_motivo_consulta_present_is_content(self):
        fields = StructuredFieldsV1(motivoConsulta="Dolor de garganta")
        meta = compute_extraction_meta(fields)
        assert meta["hasContent"] is True

    def test_only_diagnostico_is_not_content(self):
        """Diagnostico alone (auto-generated fallback) should not count as content."""
        fields = StructuredFieldsV1(
            diagnostico={"texto": "Consulta ORL en estudio", "tipo": "sindromico"}
        )
        meta = compute_extraction_meta(fields)
        # diagnostico is not counted in hasContent (it's always auto-generated)
        assert meta["hasContent"] is False


# ── Integration: Job status response includes extractionMeta ─────────────────


class TestJobStatusExtractionMeta:
    """Job status response should include extractionMeta for sparse detection."""

    @pytest.fixture(autouse=True)
    def setup(self, reset_job_manager):
        pass

    @pytest.fixture
    def jobs_client(self):
        app = create_app()
        app.dependency_overrides[verify_auth_header] = mock_verify_auth_header
        client = TestClient(app)
        yield client
        app.dependency_overrides.clear()

    def test_done_job_has_extraction_meta(self, jobs_client):
        manager = JobManager.get_instance()
        job_id = "job-interview-meta"
        manager._jobs[job_id] = Job(
            id=job_id,
            user_id="test_user",
            request=dummy_request,
            status="done",
            result=StructuredFieldsV1(
                antecedentes=Antecedentes(
                    personalesPatologicos="Niega alergias"
                ),
                negations=["niega alergias"],
            ),
        )

        response = jobs_client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "done"
        assert "extractionMeta" in data["result"]

        meta = data["result"]["extractionMeta"]
        assert meta["hasContent"] is True
        assert meta["negatedFindingsCount"] == 1

    def test_done_job_empty_extraction_meta(self, jobs_client):
        manager = JobManager.get_instance()
        job_id = "job-interview-empty"
        manager._jobs[job_id] = Job(
            id=job_id,
            user_id="test_user",
            request=dummy_request,
            status="done",
            result=StructuredFieldsV1(),
        )

        response = jobs_client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200

        meta = response.json()["result"]["extractionMeta"]
        assert meta["hasContent"] is False
        assert meta["negatedFindingsCount"] == 0

    def test_backward_compat_structured_fields_still_present(self, jobs_client):
        """extractionMeta is additive; structuredFields key must still exist."""
        manager = JobManager.get_instance()
        job_id = "job-compat-check"
        manager._jobs[job_id] = Job(
            id=job_id,
            user_id="test_user",
            request=dummy_request,
            status="done",
            result=StructuredFieldsV1(
                motivoConsulta="Dolor de garganta",
                negations=["niega fiebre"],
            ),
        )

        response = jobs_client.get(f"/v1/jobs/{job_id}")
        result = response.json()["result"]

        assert "structuredFields" in result
        assert result["structuredFields"]["motivoConsulta"] == "Dolor de garganta"
        assert result["structuredFields"]["negations"] == ["niega fiebre"]


# ── Integration: Finalize response includes hasContent/negatedFindingsCount ──


class TestFinalizeExtractionMeta:
    """Finalize response metadata should include anti-sparse indicators."""

    def test_finalize_with_negations_has_content(self, client, mock_contracts):
        fields = StructuredFieldsV1(
            antecedentes=Antecedentes(
                personalesPatologicos="Niega alergias"
            ),
            negations=["niega alergias"],
            diagnostico={"texto": "Consulta ORL en estudio", "tipo": "sindromico"},
        )

        payload = {
            "structuredFields": fields.model_dump(by_alias=True),
        }

        response = client.post("/v1/finalize", json=payload)
        assert response.status_code == 200

        metadata = response.json()["metadata"]
        assert metadata["hasContent"] is True
        assert metadata["negatedFindingsCount"] == 1

    def test_finalize_empty_fields_no_content(self, client, mock_contracts):
        fields = StructuredFieldsV1()

        payload = {
            "structuredFields": fields.model_dump(by_alias=True),
        }

        response = client.post("/v1/finalize", json=payload)
        assert response.status_code == 200

        metadata = response.json()["metadata"]
        assert metadata["hasContent"] is False
        assert metadata["negatedFindingsCount"] == 0

    def test_finalize_preserves_negations_and_has_content(self, client, mock_contracts):
        """negations field passes through and metadata reflects them."""
        fields = StructuredFieldsV1(
            negations=["niega fiebre", "sin alergias"],
            diagnostico={"texto": "Consulta ORL en estudio", "tipo": "sindromico"},
        )

        payload = {
            "structuredFields": fields.model_dump(by_alias=True),
        }

        response = client.post("/v1/finalize", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["negations"] == ["niega fiebre", "sin alergias"]
        assert data["metadata"]["hasContent"] is True
        assert data["metadata"]["negatedFindingsCount"] == 2


# ── V1 key correctness: antecedentes use camelCase ───────────────────────────


class TestV1AntecedentesKeys:
    """Ensure antecedentes sub-fields use correct camelCase keys in API responses."""

    def test_antecedentes_camelcase_in_job_response(self, reset_job_manager):
        app = create_app()
        app.dependency_overrides[verify_auth_header] = mock_verify_auth_header
        jobs_client = TestClient(app)

        manager = JobManager.get_instance()
        job_id = "job-camelcase"
        manager._jobs[job_id] = Job(
            id=job_id,
            user_id="test_user",
            request=dummy_request,
            status="done",
            result=StructuredFieldsV1(
                antecedentes=Antecedentes(
                    heredofamiliares="Padre con DM2",
                    personalesNoPatologicos="Niega tabaquismo",
                    personalesPatologicos="Niega alergias",
                ),
            ),
        )

        response = jobs_client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200

        ant = response.json()["result"]["structuredFields"]["antecedentes"]

        # Must use camelCase keys
        assert "heredofamiliares" in ant
        assert "personalesNoPatologicos" in ant
        assert "personalesPatologicos" in ant

        # Must NOT use snake_case
        assert "personales_no_patologicos" not in ant
        assert "personales_patologicos" not in ant

        # Values should be correct
        assert ant["heredofamiliares"] == "Padre con DM2"
        assert ant["personalesNoPatologicos"] == "Niega tabaquismo"
        assert ant["personalesPatologicos"] == "Niega alergias"

        app.dependency_overrides.clear()

    def test_antecedentes_camelcase_in_finalize_response(self, client, mock_contracts):
        fields = StructuredFieldsV1(
            antecedentes=Antecedentes(
                heredofamiliares="Madre con HTA",
                personalesNoPatologicos="Niega alcoholismo",
                personalesPatologicos="Alergia a sulfas",
            ),
            diagnostico={"texto": "Faringitis", "tipo": "presuntivo"},
        )

        payload = {
            "structuredFields": fields.model_dump(by_alias=True),
        }

        response = client.post("/v1/finalize", json=payload)
        assert response.status_code == 200

        ant = response.json()["data"]["antecedentes"]

        assert "heredofamiliares" in ant
        assert "personalesNoPatologicos" in ant
        assert "personalesPatologicos" in ant
        assert "personales_no_patologicos" not in ant
        assert "personales_patologicos" not in ant
