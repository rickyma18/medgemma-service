"""
Response schemas for the extraction API.
PHI note: ClinicalFacts contains PHI - NEVER log.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# === Clinical Facts DTOs (PHI - never log) ===

class ChiefComplaint(BaseModel):
    """Chief complaint section."""
    text: Optional[str] = Field(default=None, description="Chief complaint text")


class HPI(BaseModel):
    """History of Present Illness."""
    narrative: Optional[str] = Field(default=None, description="HPI narrative")


class ROS(BaseModel):
    """Review of Systems."""
    positives: list[str] = Field(default_factory=list, description="Positive findings")
    negatives: list[str] = Field(default_factory=list, description="Negative findings")


class VitalSign(BaseModel):
    """A single vital sign measurement."""
    name: str = Field(..., description="Vital sign name")
    value: str = Field(..., description="Measured value")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")


class PhysicalExam(BaseModel):
    """Physical examination findings."""
    findings: list[str] = Field(default_factory=list, description="Exam findings")
    vitals: list[VitalSign] = Field(default_factory=list, description="Vital signs")


class Diagnosis(BaseModel):
    """A diagnosis entry."""
    description: str = Field(..., description="Diagnosis description")
    icd10: Optional[str] = Field(default=None, description="ICD-10 code if applicable")


class Assessment(BaseModel):
    """Clinical assessment."""
    primary: Optional[Diagnosis] = Field(default=None, description="Primary diagnosis")
    differential: list[Diagnosis] = Field(default_factory=list, description="Differential diagnoses")


class DiagnosticOrder(BaseModel):
    """A diagnostic test order."""
    name: str = Field(..., description="Test name")
    reason: Optional[str] = Field(default=None, description="Reason for ordering")


class Treatment(BaseModel):
    """A treatment or prescription."""
    name: str = Field(..., description="Treatment name")
    dosage: Optional[str] = Field(default=None, description="Dosage if applicable")
    instructions: Optional[str] = Field(default=None, description="Instructions")


class Plan(BaseModel):
    """Treatment plan."""
    diagnostics: list[DiagnosticOrder] = Field(default_factory=list, description="Ordered diagnostics")
    treatments: list[Treatment] = Field(default_factory=list, description="Treatments")
    follow_up: Optional[str] = Field(default=None, alias="followUp", description="Follow-up instructions")

    class Config:
        populate_by_name = True


class ClinicalFacts(BaseModel):
    """
    Complete clinical facts extracted from transcript.
    CRITICAL: This contains PHI - NEVER log this object.
    """
    chief_complaint: ChiefComplaint = Field(
        default_factory=ChiefComplaint,
        alias="chiefComplaint",
        description="Chief complaint"
    )
    hpi: HPI = Field(default_factory=HPI, description="History of Present Illness")
    ros: ROS = Field(default_factory=ROS, description="Review of Systems")
    physical_exam: PhysicalExam = Field(
        default_factory=PhysicalExam,
        alias="physicalExam",
        description="Physical examination"
    )
    assessment: Assessment = Field(default_factory=Assessment, description="Assessment")
    plan: Plan = Field(default_factory=Plan, description="Treatment plan")

    class Config:
        populate_by_name = True
        json_schema_serialization_defaults_required = True


# === API Response Wrappers ===

class ResponseMetadata(BaseModel):
    """Metadata included in all responses."""
    model_version: str = Field(..., alias="modelVersion", description="Model version used")
    inference_ms: int = Field(..., alias="inferenceMs", description="Inference time in ms")
    request_id: str = Field(..., alias="requestId", description="Request ID for tracing")

    class Config:
        populate_by_name = True


class ErrorDetail(BaseModel):
    """Error details for failed requests."""
    code: Literal[
        "UNAUTHORIZED",
        "BAD_REQUEST",
        "MODEL_ERROR",
        "BACKEND_UNAVAILABLE",
        "TIMEOUT",
        "RATE_LIMITED"
    ] = Field(
        ...,
        description="Error code"
    )
    message: str = Field(..., description="Human-readable error message")
    retryable: bool = Field(default=False, description="Whether the request can be retried")


class SuccessResponse(BaseModel):
    """Successful extraction response."""
    success: Literal[True] = True
    data: ClinicalFacts = Field(..., description="Extracted clinical facts")
    metadata: ResponseMetadata = Field(..., description="Response metadata")

    class Config:
        populate_by_name = True


class ErrorResponse(BaseModel):
    """Error response."""
    success: Literal[False] = False
    error: ErrorDetail = Field(..., description="Error details")
    metadata: ResponseMetadata = Field(..., description="Response metadata")

    class Config:
        populate_by_name = True
