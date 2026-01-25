"""
Unit tests for Epic 15 backward compatibility.
Ensures response shape is identical for existing clients.
"""
import pytest
from pydantic import ValidationError

from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1,
    V1ResponseMetadata,
    V1SuccessResponse,
    Diagnostico,
)
from app.schemas.chunk_extraction_result import ChunkEvidenceSummary


class TestV1ResponseMetadataBackwardCompat:
    """Tests that V1ResponseMetadata maintains backward compatibility."""

    def test_existing_fields_unchanged(self):
        """All existing fields should work without changes."""
        metadata = V1ResponseMetadata(
            modelVersion="test-v1",
            inferenceMs=100,
            requestId="req-123",
            schemaVersion="v1",
            pipelineUsed="orl_pipeline_stub",
            chunksCount=2,
            normalizationReplacements=5,
            medicalizationReplacements=3,
            negationSpans=1,
            totalMs=500,
            stageMs={"map": 100, "reduce": 50},
            source="pipeline",
            fallbackReason=None,
            medicalizationVersion="v1",
            medicalizationGlossaryHash="abc123",
            normalizationVersion="v1",
            normalizationRulesHash="def456",
            contractWarnings=["test_warning"],
        )

        assert metadata.model_version == "test-v1"
        assert metadata.chunks_count == 2
        assert metadata.contract_warnings == ["test_warning"]

    def test_chunk_evidence_default_none(self):
        """chunkEvidence should default to None."""
        metadata = V1ResponseMetadata(
            modelVersion="test",
            inferenceMs=0,
            requestId="req"
        )
        assert metadata.chunk_evidence is None

    def test_chunk_evidence_omitted_when_none(self):
        """chunkEvidence should be omitted from JSON when None."""
        metadata = V1ResponseMetadata(
            modelVersion="test",
            inferenceMs=0,
            requestId="req"
        )

        # Serialize with exclude_none
        data = metadata.model_dump(by_alias=True, exclude_none=True)

        assert "chunkEvidence" not in data

    def test_chunk_evidence_included_when_set(self):
        """chunkEvidence should be included when explicitly set."""
        evidence = [
            ChunkEvidenceSummary(chunkIndex=0, snippets=["snippet1"]),
            ChunkEvidenceSummary(chunkIndex=1, snippets=["snippet2"]),
        ]
        metadata = V1ResponseMetadata(
            modelVersion="test",
            inferenceMs=0,
            requestId="req",
            chunkEvidence=evidence
        )

        data = metadata.model_dump(by_alias=True)

        assert "chunkEvidence" in data
        assert len(data["chunkEvidence"]) == 2


class TestV1SuccessResponseBackwardCompat:
    """Tests that V1SuccessResponse maintains backward compatibility."""

    def test_response_shape_without_evidence(self):
        """Response without evidence should match original shape."""
        fields = StructuredFieldsV1(
            motivoConsulta="Dolor de garganta",
            diagnostico=Diagnostico(texto="Faringitis", tipo="definitivo")
        )
        metadata = V1ResponseMetadata(
            modelVersion="test-v1",
            inferenceMs=100,
            requestId="req-123",
            pipelineUsed="orl_pipeline_stub",
            chunksCount=1,
            contractWarnings=[]
        )

        response = V1SuccessResponse(
            success=True,
            data=fields,
            metadata=metadata
        )

        # Serialize
        data = response.model_dump(by_alias=True, exclude_none=True)

        # Check structure
        assert data["success"] is True
        assert "data" in data
        assert "metadata" in data
        assert data["data"]["motivoConsulta"] == "Dolor de garganta"
        assert data["metadata"]["modelVersion"] == "test-v1"

        # chunkEvidence should NOT be present
        assert "chunkEvidence" not in data["metadata"]

    def test_response_shape_with_evidence(self):
        """Response with evidence should include chunkEvidence."""
        fields = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="Test", tipo="sindromico")
        )
        evidence = [ChunkEvidenceSummary(chunkIndex=0, snippets=["test"])]
        metadata = V1ResponseMetadata(
            modelVersion="test",
            inferenceMs=0,
            requestId="req",
            chunkEvidence=evidence
        )

        response = V1SuccessResponse(
            success=True,
            data=fields,
            metadata=metadata
        )

        data = response.model_dump(by_alias=True)

        assert "chunkEvidence" in data["metadata"]
        assert data["metadata"]["chunkEvidence"][0]["chunkIndex"] == 0


class TestContractWarningsNeverNone:
    """Ensure contractWarnings is always a list, never None."""

    def test_default_empty_list(self):
        metadata = V1ResponseMetadata(
            modelVersion="test",
            inferenceMs=0,
            requestId="req"
        )
        assert metadata.contract_warnings == []
        assert metadata.contract_warnings is not None

    def test_explicit_empty_list(self):
        metadata = V1ResponseMetadata(
            modelVersion="test",
            inferenceMs=0,
            requestId="req",
            contractWarnings=[]
        )
        assert metadata.contract_warnings == []

    def test_with_warnings(self):
        metadata = V1ResponseMetadata(
            modelVersion="test",
            inferenceMs=0,
            requestId="req",
            contractWarnings=["medicalization_drift", "normalization_drift"]
        )
        assert len(metadata.contract_warnings) == 2


class TestAliasesPreserved:
    """Ensure all camelCase aliases are preserved for API compatibility."""

    def test_metadata_aliases(self):
        metadata = V1ResponseMetadata(
            modelVersion="test",
            inferenceMs=100,
            requestId="req",
            schemaVersion="v1",
            pipelineUsed="pipeline",
            chunksCount=2,
            normalizationReplacements=1,
            medicalizationReplacements=2,
            negationSpans=0,
            totalMs=500,
            stageMs={"map": 100},
            fallbackReason="test",
            medicalizationVersion="v1",
            medicalizationGlossaryHash="hash1",
            normalizationVersion="v1",
            normalizationRulesHash="hash2",
            contractWarnings=["warn"],
            chunkEvidence=[ChunkEvidenceSummary(chunkIndex=0, snippets=["s"])]
        )

        data = metadata.model_dump(by_alias=True)

        # All expected camelCase keys
        expected_keys = [
            "modelVersion", "inferenceMs", "requestId", "schemaVersion",
            "pipelineUsed", "chunksCount", "normalizationReplacements",
            "medicalizationReplacements", "negationSpans", "totalMs",
            "stageMs", "fallbackReason", "medicalizationVersion",
            "medicalizationGlossaryHash", "normalizationVersion",
            "normalizationRulesHash", "contractWarnings", "chunkEvidence"
        ]

        for key in expected_keys:
            assert key in data, f"Missing alias: {key}"
