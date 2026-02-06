"""
Suggest Plan API endpoint.
Lightweight LLM call to generate a treatment plan from motivo + diagnóstico.

PHI-safe: NEVER log request body or plan text — only requestId, status, latency, plan_length.
"""
import time
from typing import Annotated, Union

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.core.auth import verify_auth_header
from app.core.logging import get_safe_logger
from app.core.metrics import get_metrics_collector
from app.schemas.suggest_plan import SuggestPlanRequest, SuggestPlanResponse
from app.schemas.response import ErrorDetail, ErrorResponse, ResponseMetadata
from app.services.extractor import get_model_version
from app.services.suggest_plan_service import suggest_plan
from app.services.exceptions import ExtractorError

# Reuse shared dependencies
from app.api.extract import get_request_id, check_rate_limit

router = APIRouter(
    prefix="/v1",
    tags=["suggest_plan"],
    dependencies=[Depends(verify_auth_header)],
)
logger = get_safe_logger(__name__)


@router.post(
    "/suggest_plan",
    response_model=Union[SuggestPlanResponse, ErrorResponse],
    status_code=status.HTTP_200_OK,
    summary="Suggest a treatment plan",
    description=(
        "Generates a treatment plan suggestion from motivo_consulta and diagnostico. "
        "Lightweight single LLM call — no extraction pipeline, no contracts."
    ),
    responses={
        200: {"model": SuggestPlanResponse, "description": "Plan generated"},
        400: {"model": ErrorResponse, "description": "Missing or empty fields"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limited"},
        500: {"model": ErrorResponse, "description": "Internal error"},
        503: {"model": ErrorResponse, "description": "Backend unavailable"},
    },
    dependencies=[Depends(check_rate_limit)],
)
async def suggest_plan_endpoint(
    request_body: SuggestPlanRequest,
    request_id: Annotated[str, Depends(get_request_id)],
) -> Union[SuggestPlanResponse, JSONResponse]:
    """Generate a treatment plan suggestion."""
    start_time = time.perf_counter()
    metrics = get_metrics_collector()

    logger.info(
        "suggest_plan request started",
        request_id=request_id,
        method="POST",
        path="/v1/suggest_plan",
    )

    try:
        plan_text, inference_ms = await suggest_plan(
            motivo_consulta=request_body.motivo_consulta,
            diagnostico=request_body.diagnostico,
            language=request_body.language,
            style=request_body.style,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # PHI-safe log: length only, never content
        logger.info(
            "suggest_plan request completed",
            request_id=request_id,
            status="success",
            status_code=200,
            latency_ms=latency_ms,
            inference_ms=inference_ms,
        )

        metrics.record_request(
            latency_ms=latency_ms,
            inference_ms=inference_ms,
            success=True,
            cache_hit=False,
        )

        return SuggestPlanResponse(plan_tratamiento=plan_text)

    except ExtractorError as exc:
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.error(
            "suggest_plan request failed",
            error_code=exc.error_code.value if hasattr(exc.error_code, "value") else str(exc.error_code),
            request_id=request_id,
            status="error",
            status_code=exc.status_code,
            latency_ms=latency_ms,
        )

        metrics.record_request(
            latency_ms=latency_ms,
            inference_ms=0,
            success=False,
            error_code=exc.error_code.value if hasattr(exc.error_code, "value") else str(exc.error_code),
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code=exc.error_code.value if hasattr(exc.error_code, "value") else "MODEL_ERROR",
                message=str(exc),
                retryable=exc.retryable,
            ),
            metadata=ResponseMetadata(
                modelVersion=get_model_version(),
                inferenceMs=0,
                requestId=request_id,
            ),
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(by_alias=True),
        )

    except Exception:
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.error(
            "suggest_plan unexpected error",
            error_code="MODEL_ERROR",
            request_id=request_id,
            status="error",
            status_code=500,
            latency_ms=latency_ms,
        )

        metrics.record_request(
            latency_ms=latency_ms,
            inference_ms=0,
            success=False,
            error_code="MODEL_ERROR",
        )

        error_response = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_ERROR",
                message="Internal server error",
                retryable=True,
            ),
            metadata=ResponseMetadata(
                modelVersion=get_model_version(),
                inferenceMs=0,
                requestId=request_id,
            ),
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(by_alias=True),
        )
