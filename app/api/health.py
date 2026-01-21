"""
Health check and metrics endpoints.
PHI-safe: no user data in responses.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any

from fastapi import APIRouter, status

from app.core.cache import get_extraction_cache
from app.core.config import get_settings
from app.core.metrics import get_metrics_collector
from app.services.extractor import check_backend_health, get_model_version

router = APIRouter(tags=["health"])


# === Response Models ===

class HealthResponse(BaseModel):
    """Basic health check response."""
    ok: bool


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    checks: Dict[str, bool]


class DetailedHealthResponse(BaseModel):
    """Detailed health response for /v1/health."""
    ok: bool
    service: str = Field(default="medgemma-service")
    version: str
    model_version: str = Field(alias="modelVersion")
    backend: str
    checks: Dict[str, bool]

    class Config:
        populate_by_name = True


class MetricsResponse(BaseModel):
    """Metrics response."""
    uptime_seconds: int = Field(alias="uptimeSeconds")
    total_requests: int = Field(alias="totalRequests")
    success_count: int = Field(alias="successCount")
    error_count: int = Field(alias="errorCount")
    error_codes: Dict[str, int] = Field(alias="errorCodes")
    latency: Dict[str, Any]
    inference_latency: Dict[str, Any] = Field(alias="inferenceLatency")
    cache: Dict[str, Any]
    rate_limited: int = Field(alias="rateLimited")

    class Config:
        populate_by_name = True


# === Endpoints ===

@router.get(
    "/healthz",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Returns 200 if the service is alive"
)
async def health_check() -> HealthResponse:
    """
    Liveness probe for Kubernetes/Cloud Run.
    Simply returns ok=true if the service is running.
    """
    return HealthResponse(ok=True)


@router.get(
    "/readyz",
    response_model=ReadyResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Returns 200 if the service is ready to handle requests"
)
async def readiness_check() -> ReadyResponse:
    """
    Readiness probe for Kubernetes/Cloud Run.
    Checks if all dependencies are ready.
    
    When backend=vllm, includes vllm_reachable check.
    When backend=openai_compat, includes openai_compat_reachable check.
    When backend=mock, always ready.
    """
    settings = get_settings()
    
    # Get backend-specific health checks
    checks = await check_backend_health()
    
    # Determine readiness based on backend
    if settings.extractor_backend == "vllm":
        all_ready = checks.get("vllm_reachable", False)
    elif settings.extractor_backend == "openai_compat":
        all_ready = checks.get("openai_compat_reachable", False)
    else:
        # Mock is always ready
        all_ready = all(checks.values())

    return ReadyResponse(
        ready=all_ready,
        checks=checks
    )


@router.get(
    "/v1/health",
    response_model=DetailedHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed health check",
    description="Returns detailed health status including model version and backend reachability"
)
async def detailed_health_check() -> DetailedHealthResponse:
    """
    Detailed health check with service info and backend status.
    """
    settings = get_settings()
    checks = await check_backend_health()
    
    # Determine overall health
    if settings.extractor_backend == "vllm":
        ok = checks.get("vllm_reachable", False)
    elif settings.extractor_backend == "openai_compat":
        ok = checks.get("openai_compat_reachable", False)
    else:
        ok = True  # Mock always ok

    return DetailedHealthResponse(
        ok=ok,
        service="medgemma-service",
        version="0.1.0",
        model_version=get_model_version(),
        backend=settings.extractor_backend,
        checks=checks
    )


@router.get(
    "/v1/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Service metrics",
    description="Returns aggregated service metrics (PHI-safe)"
)
async def get_metrics() -> MetricsResponse:
    """
    Get aggregated metrics.
    
    PHI-safe: No user identifiers or PHI in response.
    All metrics are aggregated counters/sums.
    """
    metrics = get_metrics_collector()
    snapshot = metrics.get_snapshot()
    
    # Add cache stats
    cache = get_extraction_cache()
    cache_stats = cache.get_stats()
    snapshot["cache"]["entries"] = cache_stats["entries"]
    
    return MetricsResponse(
        uptime_seconds=snapshot["uptime_seconds"],
        total_requests=snapshot["total_requests"],
        success_count=snapshot["success_count"],
        error_count=snapshot["error_count"],
        error_codes=snapshot["error_codes"],
        latency=snapshot["latency"],
        inference_latency=snapshot["inference_latency"],
        cache=snapshot["cache"],
        rate_limited=snapshot["rate_limited"]
    )
