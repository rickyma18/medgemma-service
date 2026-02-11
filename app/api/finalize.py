"""
Finalize API endpoint.
Handles post-processing, contract verification, and quality checks for extracted fields.

(c) Anti-hallucination policy (FIX #6): _enforce_evidence_policy() ensures that
    fields which were null in the input are NOT filled by the LLM unless there is
    keyword evidence in the transcript.  Applied after any refinement step.
"""
import re
import time
import uuid
from typing import Annotated, Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.responses import JSONResponse

from app.core.auth import verify_auth_header
from app.core.config import get_settings
from app.core.logging import get_safe_logger
from app.core.metrics import get_metrics_collector
from app.core.rate_limiter import get_rate_limiter
from app.schemas.finalize import FinalizeRequest, FinalizeResponse, FinalizeMetadata
from app.schemas.response import ErrorResponse, ErrorDetail, ResponseMetadata
from app.schemas.structured_fields_v1 import StructuredFieldsV1
from app.services.extractor import get_model_version
from app.services.structured_v1_extractor import compute_extraction_meta

# Reusing contract logic (No new logic invented)
from app.contracts.contract_guard import check_contracts, get_contract_warnings

# Shared dependencies
from app.api.extract import get_request_id, check_rate_limit

router = APIRouter(prefix="/v1", tags=["finalize"], dependencies=[Depends(verify_auth_header)])
logger = get_safe_logger(__name__)


def _compute_confidence_label(confidence: float) -> str:
    """
    Convert numeric confidence to Flutter-friendly label.
    - baja: < 0.5
    - media: 0.5 - 0.8
    - alta: >= 0.8
    """
    if confidence < 0.5:
        return "baja"
    elif confidence < 0.8:
        return "media"
    else:
        return "alta"


# ---------------------------------------------------------------------------
# Deterministic consistency check helpers (no LLM)
# ---------------------------------------------------------------------------

_NEGATION_RX = re.compile(
    r'(?:niega|sin|no\s+refiere|no\s+tiene|no\s+presenta|no\s+consume|no\s+usa)\s+'
    r'(.+?)(?:\.|,|;|\s+y\s|$)',
    re.IGNORECASE,
)

_ALLERGY_ASSERT_RX = re.compile(
    r'al[eé]rgic[oa]\s+a\s+(.+?)(?:\.|,|;|$)',
    re.IGNORECASE,
)

_ALLERGY_NEGATION_RX = re.compile(
    r'(?:niega|sin)\s+alergias(?:\s+conocidas)?',
    re.IGNORECASE,
)

# (field_path used in warning, accessor returning Optional[str])
_CONSISTENCY_FIELDS = [
    ("antecedentes.personalesNoPatologicos",
     lambda sf: sf.antecedentes.personales_no_patologicos if sf.antecedentes else None),
    ("antecedentes.personalesPatologicos",
     lambda sf: sf.antecedentes.personales_patologicos if sf.antecedentes else None),
    ("antecedentes.heredofamiliares",
     lambda sf: sf.antecedentes.heredofamiliares if sf.antecedentes else None),
    ("diagnostico.texto",
     lambda sf: sf.diagnostico.texto if sf.diagnostico else None),
    ("planTratamiento",
     lambda sf: sf.plan_tratamiento),
]


def _transcript_full_text(transcript) -> str:
    """Flatten transcript into a single string.

    Handles:
    - None -> ""
    - str -> returns as-is
    - Transcript object -> joins seg.text
    - dict -> extracts segments[].text defensively
    """
    if transcript is None:
        return ""
    # String: use directly
    if isinstance(transcript, str):
        return transcript
    # Transcript object: has .segments attribute
    if hasattr(transcript, "segments"):
        return " ".join(seg.text for seg in transcript.segments if seg.text)
    # Dict fallback (defensive)
    if isinstance(transcript, dict):
        segments = transcript.get("segments", [])
        texts = []
        for seg in segments:
            if isinstance(seg, dict):
                t = seg.get("text", "")
                if t:
                    texts.append(t)
        return " ".join(texts)
    return ""


def _field_affirms_item(field_value: str, item: str) -> bool:
    """Return True if *field_value* mentions *item* without a preceding negation."""
    field_lower = field_value.lower()
    item_lower = item.lower()
    if item_lower not in field_lower:
        return False
    # Check a 40-char window before the item for negation words
    idx = field_lower.index(item_lower)
    prefix = field_lower[max(0, idx - 40):idx]
    if re.search(r'(?:niega|sin\b|no\b|negativo)', prefix):
        return False
    return True


def _make_warning(
    field: str, message: str, evidence: str,
) -> Dict[str, Any]:
    return {
        "type": "consistency",
        "severity": "warning",
        "field": field,
        "message": message,
        "evidence": evidence[:160],
    }


def _check_consistency(structured_fields, transcript) -> List[Dict[str, Any]]:
    """
    Deterministic consistency check: transcript negations vs structured fields.
    Returns a list of warning dicts.  No LLM calls.
    """
    text = _transcript_full_text(transcript)
    if not text.strip():
        return []

    warnings: List[Dict[str, Any]] = []
    text_lower = text.lower()

    # (a) General negations vs all fields
    for match in _NEGATION_RX.finditer(text_lower):
        negated_item = match.group(1).strip()
        if len(negated_item) < 3:
            continue
        evidence = match.group(0).strip()[:160]
        for field_path, accessor in _CONSISTENCY_FIELDS:
            field_value = accessor(structured_fields)
            if field_value and _field_affirms_item(field_value, negated_item):
                warnings.append(_make_warning(
                    field=field_path,
                    message=(
                        f"Posible contradicción: el transcript dice "
                        f"'{evidence}' pero en {field_path} se registra "
                        f"'{negated_item}'."
                    ),
                    evidence=evidence,
                ))

    # (b) Allergy-specific checks
    patologicos = ""
    if structured_fields.antecedentes:
        patologicos = (structured_fields.antecedentes.personales_patologicos or "").lower()

    # "niega alergias" but field records allergies
    allergy_neg = _ALLERGY_NEGATION_RX.search(text_lower)
    if allergy_neg and re.search(r'al[eé]rgi', patologicos):
        ev = allergy_neg.group(0).strip()[:160]
        warnings.append(_make_warning(
            field="antecedentes.personalesPatologicos",
            message=(
                f"Posible contradicción: el transcript dice '{ev}' "
                f"pero en antecedentes se registran alergias."
            ),
            evidence=ev,
        ))

    # "alérgico a X" asserted in transcript but missing from field
    for match in _ALLERGY_ASSERT_RX.finditer(text_lower):
        allergen = match.group(1).strip()
        if allergen and allergen not in patologicos:
            ev = match.group(0).strip()[:160]
            warnings.append(_make_warning(
                field="antecedentes.personalesPatologicos",
                message=(
                    f"Posible omisión: el transcript dice '{ev}' "
                    f"pero no se registra en antecedentes."
                ),
                evidence=ev,
            ))

    return warnings


# ---------------------------------------------------------------------------
# FIX #6: Anti-hallucination evidence policy
# ---------------------------------------------------------------------------

# Keywords per field that constitute transcript evidence.
# If a field was null in the input and becomes non-null after refinement,
# we require at least one keyword hit in the transcript to keep the value.
_EVIDENCE_KEYWORDS: Dict[str, List[str]] = {
    "motivoConsulta": [
        "duele", "dolor", "molestia", "trae", "motivo", "consulta",
        "problema", "queja", "viene por", "refiere",
    ],
    "padecimientoActual": [
        "dia", "semana", "inicio", "empezo", "desde", "presenta",
        "evolucion", "fiebre", "tos", "hora", "mes",
    ],
    "heredofamiliares": [
        "padre", "madre", "hermano", "familia", "abuelo", "abuela",
        "heredo",
    ],
    "personalesNoPatologicos": [
        "fuma", "alcohol", "toma", "droga", "tabaco", "mascotas",
        "mascota", "ocupacion", "vivienda", "trabaja", "cigarro",
    ],
    "personalesPatologicos": [
        "alergi", "cirugi", "opera", "diabetes", "hipertension",
        "medicament", "circuncis", "sulfa", "penicilina", "asma",
    ],
    "exploracionFisica": [
        "explora", "observa", "cornete", "amigdala", "faringe",
        "timpan", "cuello", "rinoscop", "otoscop", "mucosa", "septum",
        "adenopat", "signos vitales", "tension",
    ],
    "planTratamiento": [
        "tratamiento", "medicament", "dosis", "recet", "indic",
        "mg", "cada", "tableta", "gotas",
    ],
}


def _has_evidence(text: str, field_key: str) -> bool:
    """Return True if *text* contains at least one keyword for *field_key*."""
    if not text:
        return False
    text_lower = text.lower()
    keywords = _EVIDENCE_KEYWORDS.get(field_key, [])
    return any(kw in text_lower for kw in keywords)


def _enforce_evidence_policy(
    original_fields: StructuredFieldsV1,
    refined_fields: StructuredFieldsV1,
    transcript_text: str,
) -> StructuredFieldsV1:
    """
    Anti-hallucination guard (FIX #6).

    Rule: if a field was null/empty in *original_fields* and became non-null in
    *refined_fields*, revert to null UNLESS the transcript contains keyword
    evidence for that field.  Fields that were already non-null in the original
    are left untouched (they come from extraction, not refinement).
    """
    result = refined_fields.model_copy(deep=True)

    # --- top-level string fields ---
    _top_fields = [
        ("motivoConsulta", "motivo_consulta"),
        ("padecimientoActual", "padecimiento_actual"),
        ("planTratamiento", "plan_tratamiento"),
    ]
    for evidence_key, attr_name in _top_fields:
        orig = getattr(original_fields, attr_name, None)
        ref = getattr(result, attr_name, None)
        if orig is None and ref is not None:
            if not _has_evidence(transcript_text, evidence_key):
                setattr(result, attr_name, None)

    # --- antecedentes sub-fields ---
    if original_fields.antecedentes and result.antecedentes:
        _ant_fields = [
            ("heredofamiliares", "heredofamiliares"),
            ("personalesNoPatologicos", "personales_no_patologicos"),
            ("personalesPatologicos", "personales_patologicos"),
        ]
        for evidence_key, attr_name in _ant_fields:
            orig = getattr(original_fields.antecedentes, attr_name, None)
            ref = getattr(result.antecedentes, attr_name, None)
            if orig is None and ref is not None:
                if not _has_evidence(transcript_text, evidence_key):
                    setattr(result.antecedentes, attr_name, None)

    # --- exploracionFisica sub-fields (single evidence key) ---
    if original_fields.exploracion_fisica and result.exploracion_fisica:
        for attr_name in (
            "signos_vitales", "rinoscopia", "orofaringe", "cuello",
            "laringoscopia", "otoscopia", "otomicroscopia", "endoscopia_nasal",
        ):
            orig = getattr(original_fields.exploracion_fisica, attr_name, None)
            ref = getattr(result.exploracion_fisica, attr_name, None)
            if orig is None and ref is not None:
                if not _has_evidence(transcript_text, "exploracionFisica"):
                    setattr(result.exploracion_fisica, attr_name, None)

    return result


@router.post(
    "/finalize",
    response_model=Union[FinalizeResponse, ErrorResponse],
    status_code=status.HTTP_200_OK,
    summary="Finalize and validate structured fields",
    description="""
    Performs post-processing checks on extracted clinical fields.
    - Validates against current backend contracts (medicalization/normalization drift).
    - Returns quality metrics and contract status.
    - Optional: Refinement (future use).
    """,
    responses={
        200: {"model": FinalizeResponse, "description": "Successful finalization"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limited"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    dependencies=[Depends(check_rate_limit)]
)
async def finalize_extraction(
    request_body: FinalizeRequest,
    request_id: Annotated[str, Depends(get_request_id)],
) -> Union[FinalizeResponse, JSONResponse]:
    """
    Finalize extraction result.
    """
    start_time = time.perf_counter()
    metrics_collector = get_metrics_collector()
    settings = get_settings()

    # Log request (PHI-safe)
    logger.info(
        "Finalize request started",
        request_id=request_id,
        method="POST",
        path="/v1/finalize",
        has_transcript=bool(request_body.transcript),
        refine_requested=request_body.refine,
        check_consistency=request_body.check_consistency
    )

    try:
        # 1. Contract Guard Check
        # Re-using existing logic to detect drift
        contract_result = check_contracts()
        contract_warnings = contract_result.get("warnings", []) or []
        contract_details = contract_result.get("details")

        # Determine status (aligned with pipeline_orl.py logic)
        if not contract_warnings:
            contract_status = "ok"
        else:
            contract_status = "warning"
            # distinct "drift" status reserved for fallback scenarios or specific policy
            # For now, following pipeline logic: warnings = warning status.

        # 2. Refinement (Optional)
        # If refinement logic is needed in future or requested via flag
        # Deep copy original so in-place mutations during refine don't affect it.
        original_fields = request_body.structured_fields.model_copy(deep=True)
        final_fields = request_body.structured_fields
        if request_body.refine:
            # Reusing existing refinement logic
            from app.services.pipeline_orl import _finalize_refine_fields
            try:
                final_fields = await _finalize_refine_fields(final_fields)
            except Exception as e:
                logger.warning("Refinement failed during finalize", error=str(e))
                # Fallback to original fields, but add warning
                contract_warnings.append(f"refinement_failed:{type(e).__name__}")
                contract_status = "warning"

        # 2.1 FIX #6: Anti-hallucination evidence policy (no LLM, deterministic).
        # Prevents refinement from filling null fields without transcript evidence.
        transcript_text = _transcript_full_text(request_body.transcript)
        if transcript_text:
            final_fields = _enforce_evidence_policy(
                original_fields, final_fields, transcript_text
            )

        # Preserve upstream negations as first-class field in finalized output.
        if original_fields.negations and not final_fields.negations:
            final_fields.negations = list(original_fields.negations)

        # 2.5 Deterministic consistency check (no LLM)
        consistency_warnings: list = []
        if request_body.check_consistency:
            consistency_warnings = _check_consistency(
                final_fields, request_body.transcript
            )
            if consistency_warnings:
                contract_warnings = contract_warnings + consistency_warnings
                contract_status = "warning"

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # 3. Construct Response
        # Confidence placeholder (can be computed from model output in future)
        confidence_value = 1.0

        # Evidence handling - convert to bool for Flutter compat
        evidence_list = None  # Future: extract from request or inference result
        used_evidence_bool = bool(evidence_list)

        # Anti-sparse metadata (PHI-safe)
        extraction_meta = compute_extraction_meta(final_fields)

        response = FinalizeResponse(
            success=True,
            data=final_fields,
            metadata=FinalizeMetadata(
                model_version=get_model_version(),  # Or explicit version if refinement used
                request_id=request_id,
                timestamp_ms=int(time.time() * 1000),
                contract_status=contract_status,
                contract_warnings=contract_warnings,
                contract_details=contract_details,
                classification_confidence=confidence_value,
                # Flutter compat fields
                warnings=contract_warnings,  # Alias for contractWarnings
                confidence_label=_compute_confidence_label(confidence_value),
                used_evidence=used_evidence_bool,
                evidence_list=evidence_list,
                # Anti-sparse fields
                has_content=extraction_meta["hasContent"],
                negated_findings_count=extraction_meta["negatedFindingsCount"],
            )
        )

        logger.info(
            "Finalize request completed",
            request_id=request_id,
            status="success",
            status_code=200,
            latency_ms=latency_ms,
            contract_status=contract_status
        )

        # Record generic success metric
        metrics_collector.record_request(
            latency_ms=latency_ms,
            inference_ms=0,
            success=True,
            cache_hit=False
        )

        return response

    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        logger.error(
            "Finalize request failed",
            error_code="MODEL_ERROR",
            request_id=request_id,
            status="error",
            status_code=500,
            latency_ms=latency_ms
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_ERROR",
                message="Internal processing error during finalization",
                retryable=True
            ),
            metadata=ResponseMetadata(
                modelVersion=get_model_version(),
                inferenceMs=0,
                requestId=request_id
            )
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(by_alias=True)
        )
