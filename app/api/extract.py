"""
Extraction API endpoint.
PHI-safe: No logging of request body, response data, or user identifiers.
"""
import time
import uuid
from typing import Annotated, Union

from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.responses import JSONResponse

from app.core.auth import verify_auth_header
from app.core.cache import get_extraction_cache
from app.core.logging import get_safe_logger
from app.core.metrics import get_metrics_collector
from app.core.rate_limiter import get_rate_limiter
from app.schemas.request import ExtractRequest
from app.schemas.response import (
    ErrorDetail,
    ErrorResponse,
    ResponseMetadata,
    SuccessResponse,
)
from app.schemas.structured_fields_v1 import (
    V1ResponseMetadata,
    V1SuccessResponse,
)
from app.services.extractor import extract, get_model_version
from app.services.structured_v1_extractor import (
    extract_structured_v1,
    get_v1_model_version,
)
from app.services.exceptions import ExtractorError

# Auth dependency at router level - executes BEFORE body parsing
router = APIRouter(prefix="/v1", tags=["extraction"], dependencies=[Depends(verify_auth_header)])
logger = get_safe_logger(__name__)


def get_request_id(
    x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None
) -> str:
    """
    Get or generate request ID from header.
    """
    if x_request_id and len(x_request_id) <= 100:
        return x_request_id
    return str(uuid.uuid4())


async def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency to check rate limit.
    
    Raises HTTPException 429 if rate limit exceeded.
    PHI-safe: Does not log UID.
    """
    # Get uid from request.state (set by verify_auth_header)
    uid = getattr(request.state, "uid", None)
    if uid is None:
        # Should not happen if auth is properly configured
        return
    
    rate_limiter = get_rate_limiter()
    allowed, remaining = rate_limiter.check_and_record(uid)
    
    if not allowed:
        # Record rate limit in metrics
        metrics = get_metrics_collector()
        metrics.record_rate_limited()
        
        reset_seconds = rate_limiter.get_reset_time(uid)
        
        logger.warning(
            "Rate limit exceeded",
            error_code="RATE_LIMITED",
            reset_seconds=reset_seconds
        )
        
        # Return 429 with structured error
        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="RATE_LIMITED",
                message=f"Rate limit exceeded. Try again in {reset_seconds} seconds.",
                retryable=True
            ),
            metadata=ResponseMetadata(
                modelVersion=get_model_version(),
                inferenceMs=0,
                requestId=request.headers.get("X-Request-ID", str(uuid.uuid4()))
            )
        )
        
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error_response.model_dump(by_alias=True)
        )
    
    # Store remaining in request.state for response headers (optional)
    request.state.rate_limit_remaining = remaining


@router.post(
    "/extract",
    response_model=Union[SuccessResponse, ErrorResponse],
    status_code=status.HTTP_200_OK,
    summary="Extract clinical facts from transcript",
    description="Processes a clinical transcript and extracts structured clinical facts",
    responses={
        200: {"model": SuccessResponse, "description": "Successful extraction"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limited"},
        500: {"model": ErrorResponse, "description": "Internal error"},
        503: {"model": ErrorResponse, "description": "Backend unavailable"},
    },
    dependencies=[Depends(check_rate_limit)]
)
async def extract_clinical_facts(
    request_body: ExtractRequest,
    request_id: Annotated[str, Depends(get_request_id)],
) -> Union[SuccessResponse, JSONResponse]:
    """
    Extract clinical facts from a transcript.

    PHI Safety:
    - Auth verified at router level (verify_auth_header) BEFORE body parsing
    - request_body contains PHI and is NEVER logged
    - Only requestId, latencyMs, status, errorCode are logged
    """
    start_time = time.perf_counter()
    metrics = get_metrics_collector()
    cache = get_extraction_cache()
    cache_hit = False

    # Log request start with safe fields only
    logger.info(
        "Extract request started",
        request_id=request_id,
        method="POST",
        path="/v1/extract"
    )

    try:
        # Check cache BEFORE calling LLM
        cached_result = cache.get(
            request_body.transcript,
            request_body.context,
            request_body.config
        )
        
        if cached_result is not None:
            cache_hit = True
            facts, cached_inference_ms, model_version = cached_result
            
            # Calculate total latency
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Log success with cache hit
            logger.info(
                "Extract request completed",
                request_id=request_id,
                status="success",
                status_code=200,
                latency_ms=latency_ms,
                inference_ms=0,  # No inference on cache hit
                model_version=model_version,
                cache_hit=True
            )
            
            # Record metrics
            metrics.record_request(
                latency_ms=latency_ms,
                inference_ms=0,
                success=True,
                cache_hit=True
            )
            
            return SuccessResponse(
                success=True,
                data=facts,
                metadata=ResponseMetadata(
                    modelVersion=model_version,
                    inferenceMs=cached_inference_ms,  # Report original inference time
                    requestId=request_id
                )
            )

        # Perform extraction using configured backend
        facts, inference_ms, model_version = await extract(
            transcript=request_body.transcript,
            context=request_body.context,
            config=request_body.config,
        )
        
        # Cache the result
        cache.set(
            request_body.transcript,
            request_body.context,
            request_body.config,
            facts,
            inference_ms,
            model_version
        )

        # Calculate total latency
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Log success with safe fields only
        logger.info(
            "Extract request completed",
            request_id=request_id,
            status="success",
            status_code=200,
            latency_ms=latency_ms,
            inference_ms=inference_ms,
            model_version=model_version,
            cache_hit=False
        )
        
        # Record metrics
        metrics.record_request(
            latency_ms=latency_ms,
            inference_ms=inference_ms,
            success=True,
            cache_hit=False
        )

        return SuccessResponse(
            success=True,
            data=facts,
            metadata=ResponseMetadata(
                modelVersion=model_version,
                inferenceMs=inference_ms,
                requestId=request_id
            )
        )

    except ExtractorError as e:
        # Handle known extractor errors with proper status codes
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.error(
            "Extract request failed",
            error_code=e.error_code.value,
            request_id=request_id,
            status="error",
            status_code=e.status_code,
            latency_ms=latency_ms
        )
        
        # Record metrics
        metrics.record_request(
            latency_ms=latency_ms,
            inference_ms=0,
            success=False,
            error_code=e.error_code.value,
            cache_hit=cache_hit
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code=e.error_code.value,
                message=e.message,
                retryable=e.retryable
            ),
            metadata=ResponseMetadata(
                modelVersion=get_model_version(),
                inferenceMs=0,
                requestId=request_id
            )
        )

        return JSONResponse(
            status_code=e.status_code,
            content=error_response.model_dump(by_alias=True)
        )

    except Exception:
        # Handle unexpected errors
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Log error with safe fields only - NO exception details (may contain PHI)
        logger.error(
            "Extract request failed",
            error_code="MODEL_ERROR",
            request_id=request_id,
            status="error",
            status_code=500,
            latency_ms=latency_ms
        )
        
        # Record metrics
        metrics.record_request(
            latency_ms=latency_ms,
            inference_ms=0,
            success=False,
            error_code="MODEL_ERROR",
            cache_hit=cache_hit
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_ERROR",
                message="Internal processing error",
                retryable=True
            ),
            metadata=ResponseMetadata(
                modelVersion=get_model_version(),
                inferenceMs=0,
                requestId=request_id
            )
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(by_alias=True)
        )


# =============================================================================
# V1 STRUCTURED ENDPOINT - ORL-specific schema
# =============================================================================

@router.post(
    "/extract-structured",
    response_model=Union[V1SuccessResponse, ErrorResponse],
    status_code=status.HTTP_200_OK,
    summary="Extract structured ORL fields from transcript (V1)",
    description="""
    Processes a clinical transcript and extracts structured fields
    optimized for ORL (ENT) documentation.

    Returns StructuredFieldsV1 schema with:
    - motivoConsulta, padecimientoActual
    - antecedentes (heredofamiliares, personalesNoPatologicos, personalesPatologicos)
    - exploracionFisica (signosVitales, rinoscopia, orofaringe, cuello, laringoscopia, otoscopia)
    - diagnostico (texto, tipo, cie10)
    - planTratamiento, pronostico, estudiosIndicados, notasAdicionales
    """,
    responses={
        200: {"model": V1SuccessResponse, "description": "Successful extraction"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limited"},
        500: {"model": ErrorResponse, "description": "Internal error"},
        503: {"model": ErrorResponse, "description": "Backend unavailable"},
    },
    dependencies=[Depends(check_rate_limit)]
)
async def extract_structured_fields_v1(
    request_body: ExtractRequest,
    request_id: Annotated[str, Depends(get_request_id)],
) -> Union[V1SuccessResponse, JSONResponse]:
    """
    Extract structured ORL fields from a transcript.

    PHI Safety:
    - Auth verified at router level BEFORE body parsing
    - request_body contains PHI and is NEVER logged
    - Only requestId, latencyMs, status, errorCode are logged
    """
    start_time = time.perf_counter()
    metrics = get_metrics_collector()

    logger.info(
        "V1 Extract request started",
        request_id=request_id,
        method="POST",
        path="/v1/extract-structured"
    )

    try:
        # Perform V1 extraction
        fields, inference_ms, model_version = await extract_structured_v1(
            transcript=request_body.transcript,
            context=request_body.context,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "V1 Extract request completed",
            request_id=request_id,
            status="success",
            status_code=200,
            latency_ms=latency_ms,
            inference_ms=inference_ms,
            model_version=model_version,
        )

        metrics.record_request(
            latency_ms=latency_ms,
            inference_ms=inference_ms,
            success=True,
            cache_hit=False
        )

        return V1SuccessResponse(
            success=True,
            data=fields,
            metadata=V1ResponseMetadata(
                modelVersion=model_version,
                inferenceMs=inference_ms,
                requestId=request_id,
                schemaVersion="v1"
            )
        )

    except ExtractorError as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.error(
            "V1 Extract request failed",
            error_code=e.error_code.value,
            request_id=request_id,
            status="error",
            status_code=e.status_code,
            latency_ms=latency_ms
        )

        metrics.record_request(
            latency_ms=latency_ms,
            inference_ms=0,
            success=False,
            error_code=e.error_code.value,
            cache_hit=False
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code=e.error_code.value,
                message=e.message,
                retryable=e.retryable
            ),
            metadata=ResponseMetadata(
                modelVersion=get_v1_model_version(),
                inferenceMs=0,
                requestId=request_id
            )
        )

        return JSONResponse(
            status_code=e.status_code,
            content=error_response.model_dump(by_alias=True)
        )

    except Exception:
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.error(
            "V1 Extract request failed",
            error_code="MODEL_ERROR",
            request_id=request_id,
            status="error",
            status_code=500,
            latency_ms=latency_ms
        )

        metrics.record_request(
            latency_ms=latency_ms,
            inference_ms=0,
            success=False,
            error_code="MODEL_ERROR",
            cache_hit=False
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_ERROR",
                message="Internal processing error",
                retryable=True
            ),
            metadata=ResponseMetadata(
                modelVersion=get_v1_model_version(),
                inferenceMs=0,
                requestId=request_id
            )
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(by_alias=True)
        )


@router.post(
    "/extract-structured-pipeline",
    response_model=Union[V1SuccessResponse, ErrorResponse],
    status_code=status.HTTP_200_OK,
    summary="Extract structured ORL fields (Pipeline Map-Reduce)",
    description="Processes a clinical transcript using the new Map-Reduce pipeline (Stub Mode).",
    dependencies=[Depends(check_rate_limit)]
)
async def extract_structured_pipeline(
    request_body: ExtractRequest,
    request_id: Annotated[str, Depends(get_request_id)],
) -> Union[V1SuccessResponse, JSONResponse]:
    """
    Pipeline-based extraction.
    Currently stubbed to use legacy extractor via pipeline orchestrator.
    """
    from app.services.pipeline_orl import run_orl_pipeline
    
    start_time = time.perf_counter()
    metrics_collector = get_metrics_collector()
    
    logger.info(
        "Pipeline extract request started",
        request_id=request_id,
        method="POST",
        path="/v1/extract-structured-pipeline"
    )

    try:
        # Exec pipeline
        fields, pipeline_metrics = await run_orl_pipeline(
            transcript=request_body.transcript,
            context=request_body.context
        )
        
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Extract specific metrics for metadata
        model_version = pipeline_metrics.pop("modelVersion", "unknown")
        # In this stub, inference_ms is functionally similar to map time or total pipeline time minus overhead
        # We can approximate inference_ms as the "map" stage time if present
        inference_ms = pipeline_metrics.get("stageMs", {}).get("map", 0)
        
        logger.info(
            "Pipeline extract request completed",
            request_id=request_id,
            status="success",
            status_code=200,
            latency_ms=latency_ms,
            model_version=model_version,
            pipeline_metrics=pipeline_metrics
        )
        
        metrics_collector.record_request(
            latency_ms=latency_ms,
            inference_ms=inference_ms,
            success=True,
            cache_hit=False
        )
        
        return V1SuccessResponse(
            success=True,
            data=fields,
            metadata=V1ResponseMetadata(
                modelVersion=model_version,
                inferenceMs=inference_ms,
                requestId=request_id,
                schemaVersion="v1",
                # Pipeline metadata
                pipelineUsed=pipeline_metrics.get("pipelineUsed"),
                chunksCount=pipeline_metrics.get("chunksCount"),
                normalizationReplacements=pipeline_metrics.get("normalizationReplacements"),
                medicalizationReplacements=pipeline_metrics.get("medicalizationReplacements"),
                negationSpans=pipeline_metrics.get("negationSpans"),
                totalMs=latency_ms,
                stageMs=pipeline_metrics.get("stageMs"),
                source="pipeline",
                fallbackReason=pipeline_metrics.get("fallbackReason")
            )
        )

    except ExtractorError as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.error(
            "Pipeline extract request failed",
            error_code=e.error_code.value,
            request_id=request_id,
            status="error",
            status_code=e.status_code,
            latency_ms=latency_ms
        )

        metrics_collector.record_request(
            latency_ms=latency_ms,
            inference_ms=0,
            success=False,
            error_code=e.error_code.value,
            cache_hit=False
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code=e.error_code.value,
                message=e.message,
                retryable=e.retryable
            ),
            metadata=ResponseMetadata(
                modelVersion=get_v1_model_version(),
                inferenceMs=0,
                requestId=request_id
            )
        )

        return JSONResponse(
            status_code=e.status_code,
            content=error_response.model_dump(by_alias=True)
        )

    except Exception:
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.error(
            "Pipeline extract request failed",
            error_code="MODEL_ERROR",
            request_id=request_id,
            status="error",
            status_code=500,
            latency_ms=latency_ms
        )

        metrics_collector.record_request(
            latency_ms=latency_ms,
            inference_ms=0,
            success=False,
            error_code="MODEL_ERROR",
            cache_hit=False
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_ERROR",
                message="Internal processing error",
                retryable=True
            ),
            metadata=ResponseMetadata(
                modelVersion=get_v1_model_version(),
                inferenceMs=0,
                requestId=request_id
            )
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(by_alias=True)
        )
