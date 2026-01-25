"""
StructuredFieldsV1 schema for ORL clinical documentation.
Maps directly to Flutter wizard fields.

PHI note: Contains patient data - NEVER log instances.
"""
from typing import List, Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.schemas.chunk_extraction_result import ChunkEvidenceSummary


class Antecedentes(BaseModel):
    """Antecedentes del paciente - estructura simplificada."""
    heredofamiliares: Optional[str] = Field(
        default=None,
        description="Antecedentes familiares: DM, HTA, cancer, etc."
    )
    personales_no_patologicos: Optional[str] = Field(
        default=None,
        alias="personalesNoPatologicos",
        description="Tabaquismo, alcoholismo, ocupacion, vivienda"
    )
    personales_patologicos: Optional[str] = Field(
        default=None,
        alias="personalesPatologicos",
        description="Enfermedades cronicas, cirugias, alergias, medicamentos actuales"
    )

    class Config:
        populate_by_name = True


class ExploracionFisica(BaseModel):
    """
    Exploracion fisica ORL estructurada.
    Cada campo es null si no se exploro o no se menciona.
    """
    signos_vitales: Optional[str] = Field(
        default=None,
        alias="signosVitales",
        description="TA, FC, FR, Temp, SpO2, peso, talla"
    )
    rinoscopia: Optional[str] = Field(
        default=None,
        description="Mucosa, cornetes, septum"
    )
    orofaringe: Optional[str] = Field(
        default=None,
        description="Amigdalas, faringe, uvula"
    )
    cuello: Optional[str] = Field(
        default=None,
        description="Adenopatias, tiroides"
    )
    laringoscopia: Optional[str] = Field(
        default=None,
        description="Cuerdas vocales, glotis - SOLO si se realizo"
    )
    # Campos adicionales ORL opcionales:
    otoscopia: Optional[str] = Field(
        default=None,
        description="CAE, membrana timpanica"
    )
    otomicroscopia: Optional[str] = Field(
        default=None,
        description="Hallazgos de otomicroscopia"
    )
    endoscopia_nasal: Optional[str] = Field(
        default=None,
        alias="endoscopiaNasal",
        description="Hallazgos de endoscopia nasal"
    )

    class Config:
        populate_by_name = True


class Diagnostico(BaseModel):
    """
    Diagnostico clinico.
    tipo indica el nivel de certeza.
    """
    texto: str = Field(..., description="Diagnostico - OBLIGATORIO")
    tipo: Literal["definitivo", "presuntivo", "sindromico"] = Field(
        default="sindromico",
        description="Nivel de certeza del diagnostico"
    )
    cie10: Optional[str] = Field(
        default=None,
        description="Codigo CIE-10 si se menciona"
    )


class StructuredFieldsV1(BaseModel):
    """
    Schema V1 para documentacion clinica ORL.
    Mapea directamente a los campos del wizard de Flutter.

    CRITICAL: Contains PHI - NEVER log this object.
    """
    # === Motivo y padecimiento ===
    motivo_consulta: Optional[str] = Field(
        default=None,
        alias="motivoConsulta",
        description="3-15 palabras"
    )
    padecimiento_actual: Optional[str] = Field(
        default=None,
        alias="padecimientoActual",
        description="1-4 oraciones: inicio, evolucion, caracteristicas"
    )

    # === Antecedentes ===
    antecedentes: Antecedentes = Field(
        default_factory=Antecedentes,
        description="Antecedentes del paciente"
    )

    # === Exploracion fisica ===
    exploracion_fisica: ExploracionFisica = Field(
        default_factory=ExploracionFisica,
        alias="exploracionFisica",
        description="Exploracion fisica ORL"
    )

    # === Diagnostico ===
    diagnostico: Optional[Diagnostico] = Field(
        default=None,
        description="Diagnostico clinico"
    )

    # === Plan ===
    plan_tratamiento: Optional[str] = Field(
        default=None,
        alias="planTratamiento",
        description="Medicamentos con dosis, via, frecuencia, duracion"
    )
    pronostico: Optional[str] = Field(
        default=None,
        description="Solo si el medico lo menciona explicitamente"
    )
    estudios_indicados: Optional[str] = Field(
        default=None,
        alias="estudiosIndicados",
        description="Estudios de laboratorio o gabinete solicitados"
    )
    notas_adicionales: Optional[str] = Field(
        default=None,
        alias="notasAdicionales",
        description="Citas de seguimiento, referencias, indicaciones especiales"
    )

    class Config:
        populate_by_name = True
        json_schema_serialization_defaults_required = True


# === Response wrapper for V1 ===

class V1ResponseMetadata(BaseModel):
    """Metadata for V1 responses."""
    model_version: str = Field(..., alias="modelVersion")
    inference_ms: int = Field(..., alias="inferenceMs")
    request_id: str = Field(..., alias="requestId")
    schema_version: str = Field(default="v1", alias="schemaVersion")

    pipeline_used: Optional[str] = Field(default=None, alias="pipelineUsed")
    chunks_count: Optional[int] = Field(default=None, alias="chunksCount")
    normalization_replacements: Optional[int] = Field(default=None, alias="normalizationReplacements")
    medicalization_replacements: Optional[int] = Field(default=None, alias="medicalizationReplacements")
    negation_spans: Optional[int] = Field(default=None, alias="negationSpans")
    total_ms: Optional[int] = Field(default=None, alias="totalMs")
    stage_ms: Optional[dict] = Field(default=None, alias="stageMs")
    source: Optional[str] = Field(default=None, description="Extraction source (legacy/pipeline)")
    fallback_reason: Optional[str] = Field(default=None, alias="fallbackReason")

    # Contract freeze / anti-drift fields (PHI-safe)
    medicalization_version: Optional[str] = Field(
        default=None,
        alias="medicalizationVersion",
        description="Medicalization glossary version (e.g. 'v1')"
    )
    medicalization_glossary_hash: Optional[str] = Field(
        default=None,
        alias="medicalizationGlossaryHash",
        description="SHA256 hex of glossary JSON for drift detection"
    )
    normalization_version: Optional[str] = Field(
        default=None,
        alias="normalizationVersion",
        description="Normalization rules version (e.g. 'v1')"
    )
    normalization_rules_hash: Optional[str] = Field(
        default=None,
        alias="normalizationRulesHash",
        description="SHA256 hex of canonical rules for drift detection"
    )

    # Contract guard warnings (drift detection)
    contract_warnings: List[str] = Field(
        default_factory=list,
        alias="contractWarnings",
        description="List of contract drift warnings (e.g. 'medicalization_drift')"
    )
    
    # Epic 16.3: Contract Status Summary (UX)
    contract_status: Literal["ok", "warning", "drift"] = Field(
        default="ok",
        alias="contractStatus",
        description="Simple drift status for UX: ok|warning|drift"
    )

    # Epic 15: Chunk evidence (opt-in, backward compatible)
    chunk_evidence: Optional[List["ChunkEvidenceSummary"]] = Field(
        default=None,
        alias="chunkEvidence",
        description=(
            "Evidence snippets per chunk (opt-in). "
            "Only included if include_evidence_in_response=True. "
            "Omitted (null) by default for backward compatibility."
        )
    )

    class Config:
        populate_by_name = True


class V1SuccessResponse(BaseModel):
    """Successful V1 extraction response."""
    success: Literal[True] = True
    data: StructuredFieldsV1 = Field(..., description="Extracted structured fields")
    metadata: V1ResponseMetadata = Field(..., description="Response metadata")

    class Config:
        populate_by_name = True


# Resolve forward references for Epic 15
# Import here to avoid circular dependency at module load time
from app.schemas.chunk_extraction_result import ChunkEvidenceSummary  # noqa: E402

V1ResponseMetadata.model_rebuild()
