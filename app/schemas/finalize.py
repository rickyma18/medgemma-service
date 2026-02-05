"""
Finalize endpoint schemas.
"""
from typing import List, Literal, Optional, Any, Dict, Union

from pydantic import BaseModel, Field, model_validator

from app.schemas.request import Transcript, Context
from app.schemas.structured_fields_v1 import StructuredFieldsV1


class FinalizeRequest(BaseModel):
    """
    Request model for /v1/finalize.
    Receives already extracted structured fields and optional transcript/context.

    Backwards Compatibility:
        - structuredFields: preferred field name (use this)
        - structuredV1: DEPRECATED, mapped internally to structuredFields
    """
    structured_fields: Optional[StructuredFieldsV1] = Field(
        default=None,
        alias="structuredFields",
        title="Structured Fields",
        description="The structured fields extracted by the job queue/pipeline. This is the preferred field.",
        examples=[{
            "motivoConsulta": "Dolor de garganta",
            "padecimientoActual": "Inicio hace 3 días",
            "diagnostico": {"texto": "Faringitis", "tipo": "presuntivo"}
        }]
    )
    # Legacy field support (Flutter backwards compatibility)
    structured_v1: Optional[StructuredFieldsV1] = Field(
        default=None,
        alias="structuredV1",
        title="Structured Fields (DEPRECATED)",
        description="DEPRECATED: Use 'structuredFields' instead. This field is kept for backwards compatibility with older Flutter clients and will be removed in a future version.",
        deprecated=True
    )
    transcript: Optional[Union[Transcript, str]] = Field(
        default=None,
        description="Original transcript (optional, for reference/logging context). Can be Transcript object or plain string."
    )
    context: Optional[Context] = Field(
        default=None,
        description="Encounter context (optional)"
    )
    # Optional flags for future expansion
    refine: bool = Field(
        default=False,
        description="Whether to run LLM refinement (expensive). Default False (just contract checks)."
    )
    check_consistency: bool = Field(
        default=False,
        alias="checkConsistency",
        description="Run deterministic consistency checks between transcript and structured fields (no LLM)."
    )

    @model_validator(mode="after")
    def map_legacy_structured_v1(self) -> "FinalizeRequest":
        """Map legacy structuredV1 to structured_fields if present."""
        if self.structured_fields is None and self.structured_v1 is not None:
            self.structured_fields = self.structured_v1
        if self.structured_fields is None:
            raise ValueError("Either structuredFields or structuredV1 must be provided")
        return self

    class Config:
        populate_by_name = True


class FinalizeMetadata(BaseModel):
    """
    Metadata for finalize response.
    Aligned with V1ResponseMetadata but focused on contract/quality.

    Flutter compatibility fields:
    - warnings: alias for contractWarnings
    - confidenceLabel: "baja" | "media" | "alta" derived from confidence float
    - usedEvidence: bool (false if no evidence)
    """
    # Standard fields
    model_version: str = Field(..., alias="modelVersion")
    request_id: str = Field(..., alias="requestId")
    timestamp_ms: int = Field(..., alias="timestampMs")

    # Contract/Quality fields
    contract_status: Literal["ok", "warning", "drift"] = Field(..., alias="contractStatus")
    contract_warnings: List[Any] = Field(default_factory=list, alias="contractWarnings")
    contract_details: Optional[Dict[str, Any]] = Field(default=None, alias="contractDetails")

    classification_confidence: float = Field(
        default=1.0,
        alias="confidence",
        description="Overall confidence score (0.0-1.0). Placeholder for now."
    )

    # --- Flutter backwards compatibility fields ---
    # warnings: mirrors contractWarnings for Flutter
    warnings: List[Any] = Field(
        default_factory=list,
        description="[Flutter compat] Alias for contractWarnings"
    )

    # confidenceLabel: human-readable label derived from confidence float
    confidence_label: Literal["baja", "media", "alta"] = Field(
        default="alta",
        alias="confidenceLabel",
        description="[Flutter compat] Confidence as label: baja (<0.5), media (0.5-0.8), alta (>=0.8)"
    )

    # usedEvidence: bool instead of nullable list for Flutter
    used_evidence: bool = Field(
        default=False,
        alias="usedEvidence",
        description="[Flutter compat] Whether evidence was used (bool). True if evidence list is non-empty."
    )

    # Original evidence list (optional, for advanced clients)
    evidence_list: Optional[List[Any]] = Field(
        default=None,
        alias="evidenceList",
        description="Raw evidence snippets (if available). usedEvidence is derived from this."
    )

    class Config:
        populate_by_name = True


class FinalizeResponse(BaseModel):
    """
    Response model for /v1/finalize.
    Returns the (possibly refined) fields and validation metadata.
    """
    success: Literal[True] = True
    data: StructuredFieldsV1 = Field(..., description="Finalized structured fields")
    metadata: FinalizeMetadata = Field(..., description="Finalization metadata")

    class Config:
        populate_by_name = True
