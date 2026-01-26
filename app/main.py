"""
MedGemma Service - FastAPI Application Entry Point.

PHI-Safe medical transcription extraction service.
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.extract import router as extract_router
from app.api.health import router as health_router
from app.core.auth import init_firebase
from app.core.config import get_settings
from app.core.logging import get_safe_logger, setup_logging
from app.schemas.response import ErrorDetail, ErrorResponse, ResponseMetadata
from app.services.extractor import get_model_version
from app.api.jobs import router as jobs_router, metrics_router as jobs_metrics_router

from app.services.job_manager import JobManager


# Initialize logging first
setup_logging()
logger = get_safe_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    Initializes resources on startup and cleans up on shutdown.
    """
    # Startup
    settings = get_settings()
    logger.info(
        "Starting MedGemma Service",
        model_version=get_model_version()
    )

    # Initialize Firebase Admin SDK
    try:
        init_firebase()
        logger.info("Firebase Admin SDK initialized")
    except Exception:
        logger.error("Failed to initialize Firebase", error_code="FIREBASE_INIT_ERROR")
        # Don't prevent startup - auth will fail at request time
        # This allows health checks to work even if Firebase is misconfigured

    # Start Job Worker
    job_manager = JobManager.get_instance()
    worker_task = asyncio.create_task(job_manager.start_worker())

    yield

    # Shutdown
    job_manager._shutting_down = True
    # Allow worker a moment to see shutdown flag if needed, or just cancel
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


    # Shutdown
    logger.info("Shutting down MedGemma Service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="MedGemma Service",
        description="PHI-safe clinical fact extraction from medical transcripts",
        version="0.1.0",
        docs_url="/docs" if settings.service_env == "dev" else None,
        redoc_url="/redoc" if settings.service_env == "dev" else None,
        openapi_url="/openapi.json" if settings.service_env == "dev" else None,
        lifespan=lifespan
    )

    # Register routes
    app.include_router(health_router)
    app.include_router(extract_router)
    app.include_router(jobs_metrics_router)
    app.include_router(jobs_router)


    # Register exception handlers
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    return app


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors.
    PHI-safe: Don't include validation details that might contain PHI.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Log error without validation details (may contain PHI)
    logger.error(
        "Request validation failed",
        error_code="BAD_REQUEST",
        request_id=request_id,
        status_code=400
    )

    # Return generic validation error - don't leak field values
    error_response = ErrorResponse(
        success=False,
        error=ErrorDetail(
            code="BAD_REQUEST",
            message="Invalid request format",
            retryable=False
        ),
        metadata=ResponseMetadata(
            modelVersion=get_model_version(),
            inferenceMs=0,
            requestId=request_id
        )
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump(by_alias=True)
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """
    Handle HTTP exceptions (including auth errors and rate limiting).
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # If detail is already a dict (structured error from rate limiter), pass through
    if isinstance(exc.detail, dict):
        logger.error(
            "HTTP exception",
            error_code=exc.detail.get("error", {}).get("code", "UNKNOWN"),
            request_id=request_id,
            status_code=exc.status_code
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )

    # Map status codes to error codes
    if exc.status_code == 401:
        error_code = "UNAUTHORIZED"
    elif exc.status_code == 400:
        error_code = "BAD_REQUEST"
    elif exc.status_code == 429:
        error_code = "RATE_LIMITED"
    else:
        error_code = "MODEL_ERROR"

    logger.error(
        "HTTP exception",
        error_code=error_code,
        request_id=request_id,
        status_code=exc.status_code
    )

    error_response = ErrorResponse(
        success=False,
        error=ErrorDetail(
            code=error_code,
            message=exc.detail if isinstance(exc.detail, str) else "Request failed",
            retryable=exc.status_code >= 500 or exc.status_code == 429
        ),
        metadata=ResponseMetadata(
            modelVersion=get_model_version(),
            inferenceMs=0,
            requestId=request_id
        )
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(by_alias=True)
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle unexpected exceptions.
    PHI-safe: Never log exception details.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Log error without exception details (may contain PHI)
    logger.error(
        "Unexpected error",
        error_code="MODEL_ERROR",
        request_id=request_id,
        status_code=500
    )

    error_response = ErrorResponse(
        success=False,
        error=ErrorDetail(
            code="MODEL_ERROR",
            message="Internal server error",
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


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.service_env == "dev"
    )
