from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class JobSubmissionResponse(BaseModel):
    success: bool = True
    jobId: str
    status: str
    position: Optional[int] = None
    etaSeconds: Optional[int] = None


class JobError(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    retryable: bool = False


class JobStatusResponse(BaseModel):
    success: bool = True
    jobId: str
    status: Literal["queued", "running", "done", "failed"]
    position: Optional[int] = None
    etaSeconds: Optional[int] = None
    fallbackUsed: bool = False
    contractWarnings: List[str] = Field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[JobError] = None
    # Optional backward-compat string for legacy clients.
    errorMessage: Optional[str] = None
