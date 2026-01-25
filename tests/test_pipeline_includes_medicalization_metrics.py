
import pytest
import os
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.structured_fields_v1 import StructuredFieldsV1
from app.services.pipeline_orl import PIPELINE_TIMEOUT_S

client = TestClient(app)

# Use dummy auth headers if needed by your AuthMiddleware
AUTH_HEADERS = {"Authorization": "Bearer dev-token"} # Check your auth implementation if needed

def test_pipeline_includes_medicalization_metrics():
    """
    Integration test for Medicalization in Pipeline.
    Verifies that metrics are correctly populated in the response metadata.
    """
    
    # Setup environment
    # We expect the test runner to provide correct MEDICALIZATION_GLOSSARY_PATH
    # or rely on the default if it resolves.
    
    # Input transcript that triggers:
    # 1. Medicalization: "Me duele la cabeza" -> "Refiere cefalea" (1 replacement)
    # 2. Negation: "No tengo fiebre" -> Negation span (1 span)
    
    payload = {
        "transcript": {
            "segments": [
                {
                    "speaker": "patient",
                    "text": "Me duele la cabeza y no tengo fiebre.",
                    "startMs": 0,
                    "endMs": 1000
                }
            ],
            "durationMs": 1000,
            "language": "es"
        }
    }
    
    # Mock the LLM extractor calls to avoid external dependencies
    # We mock `extract_structured_v1` which is called by the pipeline map phase
    mock_files = StructuredFieldsV1(motivoConsulta="Cefalea")
    
    with patch("app.services.pipeline_orl.extract_structured_v1", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = (mock_files, 100, "mock-v1")
        
        # We also mock finalize to avoid that stage call
        with patch("app.services.pipeline_orl._finalize_refine_fields", new_callable=AsyncMock) as mock_finalize:
            mock_finalize.return_value = mock_files

            response = client.post(
                "/v1/extract-structured-pipeline",
                json=payload,
                headers=AUTH_HEADERS # Assuming DevAuthMiddleware accepts anything or valid token in dev mode
            )
            
            assert response.status_code == 200, f"Response: {response.text}"
            
            data = response.json()
            assert data["success"] is True
            metadata = data["metadata"]
            
            # Verify Medicalization Metrics
            # Note: "Me duele la cabeza" counts as 1 replacement in our glossary
            # "No tengo fiebre" counts as 1 negation span
            
            assert "medicalizationReplacements" in metadata
            assert metadata["medicalizationReplacements"] >= 1
            
            assert "negationSpans" in metadata
            assert metadata["negationSpans"] >= 1
            
            # Verify Stage Timings
            assert "stageMs" in metadata
            assert "medicalization" in metadata["stageMs"]
            assert metadata["stageMs"]["medicalization"] >= 0
            
            # Verify Flow
            assert metadata["source"] == "pipeline"
            assert metadata["pipelineUsed"] == "orl_pipeline_stub" # or similar
