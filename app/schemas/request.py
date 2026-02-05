"""
Request schemas with strict Pydantic validation.
PHI note: These schemas handle PHI data - NEVER log instances.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TranscriptSegment(BaseModel):
    """A single segment of the transcript."""

    speaker: Literal["doctor", "patient", "unknown"] = Field(
        ...,
        description="Who is speaking in this segment"
    )
    text: str = Field(
        ...,
        min_length=1,
        description="The transcribed text content"
    )
    start_ms: int = Field(
        ...,
        ge=0,
        alias="startMs",
        description="Start time in milliseconds"
    )
    end_ms: int = Field(
        ...,
        ge=0,
        alias="endMs",
        description="End time in milliseconds"
    )

    @field_validator("end_ms", mode="after")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        """Validate that end_ms >= start_ms."""
        start = info.data.get("start_ms")
        if start is not None and v < start:
            raise ValueError("endMs must be >= startMs")
        return v

    class Config:
        populate_by_name = True


class Transcript(BaseModel):
    """The full transcript with segments."""

    segments: list[TranscriptSegment] = Field(
        ...,
        min_length=1,
        description="List of transcript segments"
    )
    language: str = Field(
        default="es",
        min_length=2,
        max_length=10,
        description="Language code (e.g., 'es', 'en')"
    )
    duration_ms: int = Field(
        ...,
        ge=0,
        alias="durationMs",
        description="Total duration in milliseconds"
    )

    class Config:
        populate_by_name = True


class Context(BaseModel):
    """Clinical context for the extraction."""

    specialty: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Medical specialty"
    )
    encounter_type: Optional[str] = Field(
        default=None,
        alias="encounterType",
        max_length=100,
        description="Type of clinical encounter"
    )
    patient_age: Optional[int] = Field(
        default=None,
        ge=0,
        le=150,
        alias="patientAge",
        description="Patient age in years"
    )
    patient_gender: Optional[Literal["male", "female", "unknown"]] = Field(
        default=None,
        alias="patientGender",
        description="Patient gender"
    )
    scope: Optional[Literal["interview", "exam", "studies", "assessment"]] = Field(
        default=None,
        description="Extraction scope: interview, exam, studies, or assessment. If null, full extraction."
    )

    class Config:
        populate_by_name = True


class ExtractConfig(BaseModel):
    """Configuration for the extraction."""

    model_version: Optional[str] = Field(
        default=None,
        alias="modelVersion",
        max_length=50,
        description="Requested model version"
    )

    class Config:
        populate_by_name = True


class ExtractRequest(BaseModel):
    """
    Request body for POST /v1/extract.
    CRITICAL: Never log this object - contains PHI.
    """

    transcript: Transcript = Field(
        ...,
        description="The clinical transcript to process"
    )
    context: Optional[Context] = Field(
        default=None,
        description="Optional clinical context"
    )
    config: Optional[ExtractConfig] = Field(
        default=None,
        description="Optional extraction configuration"
    )

    class Config:
        populate_by_name = True
