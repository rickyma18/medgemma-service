from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Header, status
from fastapi.responses import JSONResponse

from app.core.auth import verify_auth_header
from app.core.config import get_settings
from app.core.logging import get_safe_logger
from app.schemas.request import ExtractRequest
from app.schemas.job import JobStatusResponse, JobSubmissionResponse
from app.services.job_manager import JobManager

router = APIRouter(prefix="/v1", tags=["jobs"], dependencies=[Depends(verify_auth_header)])
metrics_router = APIRouter(prefix="/v1", tags=["jobs"])
logger = get_safe_logger(__name__)


@metrics_router.get(
    "/jobs/metrics",
    summary="Get job queue metrics (Admin)",
    description="Returns aggregated PHI-safe metrics for the job queue system.",
)
async def get_job_metrics(
    x_admin_token: Annotated[Optional[str], Header(alias="X-Admin-Token")] = None,
):
    """
    Get observability metrics.
    Admin token required if ADMIN_API_KEY is set.
    """
    settings = get_settings()
    
    # Optional Admin Protection
    if settings.admin_api_key:
        if not x_admin_token or x_admin_token != settings.admin_api_key:
            logger.warning("Unauthorized access to metrics endpoint")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid admin token"
            )

    job_manager = JobManager.get_instance()
    return job_manager.get_observability_metrics()


@router.post(
    "/jobs",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue an extraction job",
    responses={
        202: {"description": "Job accepted and queued"},
        409: {"description": "User already has a job in progress"},
        429: {"description": "Daily quota exceeded"},
    }
)
async def submit_job(
    request: Request,
    request_body: ExtractRequest,
):
    """
    Submit a job to the queue.
    """
    uid = getattr(request.state, "uid", "unknown")
    job_manager = JobManager.get_instance()
    
    try:
        job_id = await job_manager.submit_job(uid, request_body)
        
        # Determine initial status
        position = job_manager.get_queue_position(job_id)
        eta = job_manager.get_eta_seconds(position) if position else 0
        
        return JobSubmissionResponse(
            success=True,
            jobId=job_id,
            status="queued",
            position=position,
            etaSeconds=eta
        )
        
    except ValueError as e:
        # User has job in progress
        logger.warning(f"Job submission conflict: {e}", user_id=uid)
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "success": False,
                # "existingJobId": ??? - we could parse it from exception or change exception 
                # but simple message is fine per spec unless stricter needed
                "message": str(e)
            }
        )
        
    except RuntimeError as e:
        # Quota exceeded
        logger.warning(f"Job quota exceeded: {e}", user_id=uid)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "success": False,
                "message": str(e)
            }
        )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
)
async def get_job_status(
    job_id: str,
    request: Request,
):
    """
    Get the status and result of a job.
    """
    # Verify ownership? The prompt doesn't strictly say, but usually good practice.
    # For beta, assume if you have ID you can check it? Or safer: check uid matches.
    uid = getattr(request.state, "uid", "unknown")
    job_manager = JobManager.get_instance()
    
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Ownership check
    if job.user_id != uid:
        # In PROD this should be 403 or 404, but let's stick to 404 to hide existence
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    position = None
    eta = None
    result = None
    
    if job.status == "queued":
        position = job_manager.get_queue_position(job_id)
        if position:
            eta = job_manager.get_eta_seconds(position)
    
    elif job.status == "done":
        # Only return result if done
        result = job.result
        
    return JobStatusResponse(
        success=True,
        jobId=job.id,
        status=job.status,
        position=position,
        etaSeconds=eta,
        fallbackUsed=job.fallback_used,
        contractWarnings=job.contract_warnings,
        result=result,
        error=job.error
    )
