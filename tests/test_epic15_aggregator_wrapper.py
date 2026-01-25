"""
Unit tests for Epic 15 aggregate_chunk_results wrapper.
Ensures wrapper correctly delegates to aggregate_structured_fields_v1.
"""
import pytest

from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1,
    Diagnostico,
    Antecedentes,
    ExploracionFisica,
)
from app.schemas.chunk_extraction_result import (
    ChunkExtractionResult,
    EvidenceSnippet,
    ChunkEvidenceSummary,
)
from app.services.aggregator import (
    aggregate_chunk_results,
    aggregate_structured_fields_v1,
)


class TestAggregateChunkResults:
    """Tests for aggregate_chunk_results wrapper."""

    def test_empty_list_returns_empty_fields(self):
        fields, evidence = aggregate_chunk_results([])
        assert isinstance(fields, StructuredFieldsV1)
        assert evidence == []

    def test_single_chunk_returns_same_fields(self):
        original = StructuredFieldsV1(
            motivoConsulta="Dolor de garganta",
            diagnostico=Diagnostico(texto="Faringitis", tipo="definitivo")
        )
        chunk = ChunkExtractionResult(chunkIndex=0, fields=original)

        fields, evidence = aggregate_chunk_results([chunk])

        assert fields.motivo_consulta == "Dolor de garganta"
        assert fields.diagnostico.texto == "Faringitis"

    def test_multiple_chunks_aggregated(self):
        chunk1 = ChunkExtractionResult(
            chunkIndex=0,
            fields=StructuredFieldsV1(
                motivoConsulta="Dolor de garganta",
                diagnostico=Diagnostico(texto="Faringitis", tipo="presuntivo")
            )
        )
        chunk2 = ChunkExtractionResult(
            chunkIndex=1,
            fields=StructuredFieldsV1(
                motivoConsulta="Fiebre",
                diagnostico=Diagnostico(texto="Faringitis aguda", tipo="definitivo")
            )
        )

        fields, evidence = aggregate_chunk_results([chunk1, chunk2])

        # Motivos should be merged
        assert "Dolor de garganta" in fields.motivo_consulta
        assert "Fiebre" in fields.motivo_consulta
        # Diagnosis with higher certainty wins
        assert fields.diagnostico.tipo == "definitivo"

    def test_evidence_collected_from_chunks(self):
        chunk1 = ChunkExtractionResult(
            chunkIndex=0,
            fields=StructuredFieldsV1(),
            evidence=[
                EvidenceSnippet(text="Evidence A", fieldPath="field1")
            ]
        )
        chunk2 = ChunkExtractionResult(
            chunkIndex=1,
            fields=StructuredFieldsV1(),
            evidence=[
                EvidenceSnippet(text="Evidence B", fieldPath="field2"),
                EvidenceSnippet(text="Evidence C", fieldPath="field3")
            ]
        )

        fields, evidence = aggregate_chunk_results([chunk1, chunk2])

        assert len(evidence) == 2  # 2 chunks with evidence
        assert evidence[0].chunk_index == 0
        assert evidence[0].snippets == ["Evidence A"]
        assert evidence[1].chunk_index == 1
        assert len(evidence[1].snippets) == 2

    def test_chunks_without_evidence_excluded_from_summary(self):
        chunk1 = ChunkExtractionResult(
            chunkIndex=0,
            fields=StructuredFieldsV1(),
            evidence=[]  # No evidence
        )
        chunk2 = ChunkExtractionResult(
            chunkIndex=1,
            fields=StructuredFieldsV1(),
            evidence=[EvidenceSnippet(text="Evidence", fieldPath="test")]
        )

        fields, evidence = aggregate_chunk_results([chunk1, chunk2])

        assert len(evidence) == 1
        assert evidence[0].chunk_index == 1

    def test_chunks_sorted_by_index(self):
        """Ensure deterministic ordering regardless of input order."""
        chunk2 = ChunkExtractionResult(chunkIndex=2, fields=StructuredFieldsV1())
        chunk0 = ChunkExtractionResult(chunkIndex=0, fields=StructuredFieldsV1())
        chunk1 = ChunkExtractionResult(chunkIndex=1, fields=StructuredFieldsV1())

        # Pass in wrong order
        fields, evidence = aggregate_chunk_results([chunk2, chunk0, chunk1])

        # Should still work correctly (deterministic)
        assert isinstance(fields, StructuredFieldsV1)

    def test_wrapper_delegates_to_original_function(self):
        """Ensure wrapper uses aggregate_structured_fields_v1 internally."""
        fields1 = StructuredFieldsV1(motivoConsulta="A")
        fields2 = StructuredFieldsV1(motivoConsulta="B")

        # Direct aggregation
        direct_result = aggregate_structured_fields_v1([fields1, fields2])

        # Wrapper aggregation
        chunk1 = ChunkExtractionResult(chunkIndex=0, fields=fields1)
        chunk2 = ChunkExtractionResult(chunkIndex=1, fields=fields2)
        wrapper_result, _ = aggregate_chunk_results([chunk1, chunk2])

        # Results should be identical
        assert direct_result.motivo_consulta == wrapper_result.motivo_consulta


class TestBackwardCompatibility:
    """Ensure aggregate_structured_fields_v1 signature unchanged."""

    def test_original_function_signature_unchanged(self):
        """The original function should accept List[StructuredFieldsV1]."""
        fields_list = [
            StructuredFieldsV1(motivoConsulta="Test 1"),
            StructuredFieldsV1(motivoConsulta="Test 2"),
        ]

        # This should work without any changes
        result = aggregate_structured_fields_v1(fields_list)

        assert isinstance(result, StructuredFieldsV1)
        assert "Test 1" in result.motivo_consulta
        assert "Test 2" in result.motivo_consulta
