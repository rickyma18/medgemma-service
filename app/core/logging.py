"""
PHI-safe logging module.
CRITICAL: Never log transcript, segments, clinical data, or patient information.
Only log: requestId, latencyMs, status, errorCode.
"""
import logging
import sys
from typing import Any, Optional

from app.core.config import get_settings


def setup_logging() -> None:
    """Configure application logging with PHI-safe format."""
    settings = get_settings()

    # Determine log level based on environment
    log_level = logging.DEBUG if settings.service_env == "dev" else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)


class SafeLogger:
    """
    PHI-safe logger wrapper.
    Only allows logging of safe fields: requestId, latencyMs, status, errorCode.
    """

    SAFE_FIELDS = frozenset({
        "request_id",
        "latency_ms",
        "status",
        "status_code",
        "error_code",
        "method",
        "path",
        "model_version",
        "inference_ms",
        "credential_mode",
        "exception_class",
    })

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _format_safe_context(self, context: dict[str, Any]) -> str:
        """Format only safe fields from context."""
        safe_items = []
        for key, value in context.items():
            if key in self.SAFE_FIELDS:
                safe_items.append(f"{key}={value}")
        return " | ".join(safe_items) if safe_items else ""

    def info(self, message: str, **context: Any) -> None:
        """Log info with safe context only."""
        ctx = self._format_safe_context(context)
        full_message = f"{message} | {ctx}" if ctx else message
        self._logger.info(full_message)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning with safe context only."""
        ctx = self._format_safe_context(context)
        full_message = f"{message} | {ctx}" if ctx else message
        self._logger.warning(full_message)

    def error(
        self,
        message: str,
        error_code: Optional[str] = None,
        **context: Any
    ) -> None:
        """
        Log error with safe context only.
        NEVER log exception details that might contain PHI.
        """
        if error_code:
            context["error_code"] = error_code
        ctx = self._format_safe_context(context)
        full_message = f"{message} | {ctx}" if ctx else message
        self._logger.error(full_message)

    def debug(self, message: str, **context: Any) -> None:
        """Log debug with safe context only."""
        ctx = self._format_safe_context(context)
        full_message = f"{message} | {ctx}" if ctx else message
        self._logger.debug(full_message)


def get_safe_logger(name: str) -> SafeLogger:
    """Get a PHI-safe logger instance."""
    return SafeLogger(name)
