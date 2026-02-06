"""
Request/response schemas for the suggest_plan endpoint.
PHI note: motivo_consulta and diagnostico contain PHI — NEVER log.
"""
from typing import Literal

from pydantic import BaseModel, Field


class SuggestPlanRequest(BaseModel):
    """Request body for POST /v1/suggest_plan."""

    motivo_consulta: str = Field(
        ...,
        min_length=1,
        description="Chief complaint / reason for visit",
    )
    diagnostico: str = Field(
        ...,
        min_length=1,
        description="Working diagnosis",
    )
    language: str = Field(
        default="es",
        description="Output language (ISO 639-1)",
    )
    style: Literal["bullets", "paragraph"] = Field(
        default="bullets",
        description="Output style: bullet list or paragraph",
    )


class SuggestPlanResponse(BaseModel):
    """Response body for POST /v1/suggest_plan."""

    plan_tratamiento: str = Field(
        ...,
        description="Generated treatment plan text",
    )
