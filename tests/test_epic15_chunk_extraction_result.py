"""
Unit tests for Epic 15 ChunkExtractionResult schema.
Tests internal schemas and validation.
"""
import pytest
from pydantic import ValidationError

from app.schemas.chunk_extraction_result import (
    EvidenceSnippet,
    ChunkExtractionResult,
    ChunkEvidenceSummary,
)
from app.schemas.structured_fields_v1 import StructuredFieldsV1, Diagnostico


class TestEvidenceSnippet:
    """Tests for EvidenceSnippet model."""

    def test_valid_snippet(self):
        snippet = EvidenceSnippet(
            text="Amigdalas hiperhemicas",
            fieldPath="exploracionFisica.orofaringe"
        )
        assert snippet.text == "Amigdalas hiperhemicas"
        assert snippet.field_path == "exploracionFisica.orofaringe"

    def test_truncates_long_text(self):
        long_text = "A" * 200
        snippet = EvidenceSnippet(text=long_text, fieldPath="test")
        assert len(snippet.text) <= 160
        assert snippet.text.endswith("...")

    def test_exactly_160_chars_no_truncation(self):
        text = "A" * 160
        snippet = EvidenceSnippet(text=text, fieldPath="test")
        assert snippet.text == text
        assert not snippet.text.endswith("...")

    def test_alias_serialization(self):
        snippet = EvidenceSnippet(text="test", fieldPath="diagnostico.texto")
        data = snippet.model_dump(by_alias=True)
        assert "fieldPath" in data
        assert data["fieldPath"] == "diagnostico.texto"


class TestChunkExtractionResult:
    """Tests for ChunkExtractionResult model."""

    def test_minimal_valid_result(self):
        fields = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="Test", tipo="sindromico")
        )
        result = ChunkExtractionResult(
            chunkIndex=0,
            fields=fields
        )
        assert result.chunk_index == 0
        assert result.extractor_used == "lite"
        assert result.evidence == []

    def test_with_evidence(self):
        fields = StructuredFieldsV1(
            diagnostico=Diagnostico(texto="Faringitis", tipo="definitivo")
        )
        evidence = [
            EvidenceSnippet(text="Dolor de garganta", fieldPath="motivoConsulta")
        ]
        result = ChunkExtractionResult(
            chunkIndex=1,
            fields=fields,
            evidence=evidence,
            extractorUsed="full"
        )
        assert result.chunk_index == 1
        assert result.extractor_used == "full"
        assert len(result.evidence) == 1

    def test_chunk_index_must_be_non_negative(self):
        fields = StructuredFieldsV1()
        with pytest.raises(ValidationError):
            ChunkExtractionResult(chunkIndex=-1, fields=fields)

    def test_alias_serialization(self):
        fields = StructuredFieldsV1()
        result = ChunkExtractionResult(chunkIndex=2, fields=fields)
        data = result.model_dump(by_alias=True)
        assert "chunkIndex" in data
        assert "extractorUsed" in data


class TestChunkEvidenceSummary:
    """Tests for ChunkEvidenceSummary model (public opt-in response)."""

    def test_valid_summary(self):
        summary = ChunkEvidenceSummary(
            chunkIndex=0,
            snippets=["Evidence 1", "Evidence 2"]
        )
        assert summary.chunk_index == 0
        assert len(summary.snippets) == 2

    def test_empty_snippets(self):
        summary = ChunkEvidenceSummary(chunkIndex=0, snippets=[])
        assert summary.snippets == []

    def test_chunk_index_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            ChunkEvidenceSummary(chunkIndex=-1, snippets=[])

    def test_alias_serialization(self):
        summary = ChunkEvidenceSummary(chunkIndex=3, snippets=["test"])
        data = summary.model_dump(by_alias=True)
        assert "chunkIndex" in data
        assert data["chunkIndex"] == 3
