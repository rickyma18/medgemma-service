from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class JobSubmissionResponse(BaseModel):
    success: bool = True
    jobId: str
    status: str
    position: Optional[int] = None
    etaSeconds: Optional[int] = None


class JobStatusResponse(BaseModel):
    success: bool = True
    jobId: str
    status: Literal["queued", "running", "done", "failed"]
    position: Optional[int] = None
    etaSeconds: Optional[int] = None
    fallbackUsed: bool = False
    contractWarnings: List[str] = Field(default_factory=list)
    result: Optional[Any] = None
    
    # Error message if failed (PHI safe generic message usually, or specific error code)
    error: Optional[str] = None
