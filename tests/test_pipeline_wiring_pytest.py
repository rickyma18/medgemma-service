
import os
import pytest
from fastapi.testclient import TestClient

# --- SETUP ENV before imports ---
os.environ["AUTH_MODE"] = "dev"
os.environ["DEV_BEARER_TOKEN"] = "dev-token"

from app.main import app

client = TestClient(app)

def test_pipeline_endpoint_integration():
    """
    Test the new pipeline endpoint using TestClient.
    Validates:
    - Status 200
    - Proper stub metadata presence
    """
    payload = {
        "transcript": {
            "segments": [
                {"speaker": "doctor", "text": "Hola paciente.", "startMs": 0, "endMs": 1000},
                {"speaker": "patient", "text": "Hola doctor, me duele la garganta.", "startMs": 1000, "endMs": 2000}
            ],
            "durationMs": 2000
        },
        "context": {},
        # This config ensures we don't actually hit external APIs if "mock" or similar is default, 
        # but in this repo, structured_v1_extractor uses OpenAI compat.
        # Assuming the environment or default settings allow this to run or fail gracefully.
        # If it fails with connection error, we catch 503/500, but structured_v1 might need a mock backend.
        # For this wiring test, we hope it either hits a mock or we are just testing the structure.
        # Ideally we should mock the inner extractor, but user asked to test integration.
        "config": {"modelVersion": "test-model"}
    }

    # We can mock the `run_orl_pipeline` for a pure unit test, 
    # but let's try to run it to see if the wiring works.
    # If the backend is not running, extract_structured_v1 might raise BackendUnavailable.
    # We'll check for 200 OR typical error codes if backend is missing, 
    # but the requirement says "valida status 200".
    # Assuming the user has a dev environment where this works or mocks are in place.
    # Alternatively, we can mock `extract_structured_v1` inside the pipeline if needed.
    # Let's rely on the previous successful run of `test_pipeline_wiring.py` suggesting it works.

    response = client.post(
        "/v1/extract-structured-pipeline",
        json=payload,
        headers={"Authorization": "Bearer dev-token"}
    )
    
    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    
    data = response.json()
    assert data["success"] is True
    
    metadata = data["metadata"]
    # Check required fields
    assert metadata["pipelineUsed"] == "orl_pipeline_stub"
    assert metadata["chunksCount"] == 1
    assert metadata["source"] == "pipeline"
    assert "totalMs" in metadata
    assert isinstance(metadata["totalMs"], int)
    assert metadata["normalizationReplacements"] == 0

