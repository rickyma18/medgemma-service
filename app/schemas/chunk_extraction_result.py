"""
Internal schemas for chunked extraction pipeline (Epic 15).
These are INTERNAL types - NOT part of the public API contract.

PHI note: EvidenceSnippet.text may contain sanitized clinical content.
         Never log ChunkExtractionResult instances.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class EvidenceSnippet(BaseModel):
    """
    A short PHI-sanitized text snippet supporting an extracted field.
    Max 160 chars, sanitized before storage.

    INTERNAL: Not exposed in API response directly.
    """
    text: str = Field(
        ...,
        max_length=160,
        description="Sanitized evidence text (max 160 chars)"
    )
    field_path: str = Field(
        ...,
        alias="fieldPath",
        description="Dot-notation path to the field this evidence supports (e.g. 'diagnostico.texto')"
    )

    @field_validator("text", mode="before")
    @classmethod
    def truncate_text(cls, v: str) -> str:
        """Ensure text never exceeds 160 chars."""
        if v and len(v) > 160:
            return v[:157] + "..."
        return v or ""

    class Config:
        populate_by_name = True


class ChunkExtractionResult(BaseModel):
    """
    Internal result from extracting a single chunk.
    Carries the partial fields + evidence for traceability.

    INTERNAL: Used only within pipeline, never serialized to client directly.
    PHI: Contains clinical data - NEVER log.
    """
    chunk_index: int = Field(
        ...,
        alias="chunkIndex",
        ge=0,
        description="0-based index of the chunk in the transcript"
    )
    fields: "StructuredFieldsV1" = Field(
        ...,
        description="Partial extracted fields from this chunk"
    )
    evidence: List[EvidenceSnippet] = Field(
        default_factory=list,
        description="Evidence snippets supporting extracted fields"
    )
    extractor_used: Literal["lite", "full"] = Field(
        default="lite",
        alias="extractorUsed",
        description="Which extractor was used for this chunk"
    )

    class Config:
        populate_by_name = True


class ChunkEvidenceSummary(BaseModel):
    """
    Summary of evidence for a single chunk - for opt-in API response.
    Only included if client requests evidence via config/header.

    PUBLIC (opt-in): Can be included in API response if requested.
    """
    chunk_index: int = Field(
        ...,
        alias="chunkIndex",
        ge=0,
        description="0-based index of the chunk"
    )
    snippets: List[str] = Field(
        default_factory=list,
        description="List of sanitized evidence strings (text only, no field paths)"
    )

    class Config:
        populate_by_name = True


# Avoid circular import - import at runtime
from app.schemas.structured_fields_v1 import StructuredFieldsV1

# Update forward reference
ChunkExtractionResult.model_rebuild()
