"""
Custom exceptions for extraction backends.
PHI-safe: These exceptions contain no PHI data.
"""
from enum import Enum
from typing import Optional


class ExtractorErrorCode(str, Enum):
    """PHI-safe error codes for extractor failures."""
    MODEL_ERROR = "MODEL_ERROR"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"


class ExtractorError(Exception):
    """
    Base exception for extractor errors.
    
    Attributes:
        error_code: PHI-safe error code for logging and response
        status_code: HTTP status code to return
        retryable: Whether the client should retry
        message: PHI-safe message (no sensitive data)
    """
    
    def __init__(
        self,
        error_code: ExtractorErrorCode,
        message: str = "Extraction failed",
        status_code: int = 500,
        retryable: bool = True
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(message)


class BackendUnavailableError(ExtractorError):
    """Raised when the backend is not reachable."""
    
    def __init__(self, backend: str = "unknown"):
        super().__init__(
            error_code=ExtractorErrorCode.BACKEND_UNAVAILABLE,
            message=f"Backend unavailable: {backend}",
            status_code=503,
            retryable=True
        )


class BackendTimeoutError(ExtractorError):
    """Raised when the backend request times out."""
    
    def __init__(self, timeout_ms: int = 0):
        super().__init__(
            error_code=ExtractorErrorCode.TIMEOUT,
            message=f"Backend timeout after {timeout_ms}ms",
            status_code=503,
            retryable=True
        )


class RateLimitedError(ExtractorError):
    """Raised when the backend returns 429."""
    
    def __init__(self):
        super().__init__(
            error_code=ExtractorErrorCode.RATE_LIMITED,
            message="Rate limited by backend",
            status_code=429,
            retryable=True
        )


class ModelError(ExtractorError):
    """Raised when the model returns invalid output."""
    
    def __init__(self, reason: str = "Invalid model output"):
        super().__init__(
            error_code=ExtractorErrorCode.MODEL_ERROR,
            message=reason,
            status_code=500,
            retryable=True
        )
