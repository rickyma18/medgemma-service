"""
PHI-safe telemetry module for emitting events.

This module provides a simple, extensible interface for emitting telemetry events.
Currently logs to structured JSON; can be extended to send to external services.

CRITICAL: Never include PHI (transcript text, patient data, segments) in payloads.
Only hashes, counts, flags, and metadata are allowed.
"""
import json
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from app.core.logging import get_safe_logger

logger = get_safe_logger(__name__)

# In-memory rate limiting state
_rate_limit_lock = threading.Lock()
_last_emit_times: Dict[str, float] = {}

# PHI-unsafe keys that must NEVER appear in payloads
PHI_FORBIDDEN_KEYS: Set[str] = {
    "text",
    "transcript",
    "segments",
    "segment",
    "patient",
    "content",
    "raw",
    "audio",
    "speech",
    "name",
    "diagnosis",
    "condition",
}


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively remove any PHI-unsafe keys from payload.

    This is a defense-in-depth measure. Callers should never include PHI,
    but this ensures it's stripped if accidentally included.
    """
    if not isinstance(payload, dict):
        return payload

    sanitized = {}
    for key, value in payload.items():
        key_lower = key.lower()
        if key_lower in PHI_FORBIDDEN_KEYS:
            # Skip this key entirely
            continue
        if isinstance(value, dict):
            sanitized[key] = _sanitize_payload(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_payload(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized


def _should_emit(event_name: str, cooldown_s: int) -> bool:
    """
    Check if we should emit this event based on rate limiting.

    Args:
        event_name: The event name to check
        cooldown_s: Cooldown period in seconds (0 = no cooldown)

    Returns:
        True if we should emit, False if rate-limited
    """
    if cooldown_s <= 0:
        return True

    now = time.time()

    with _rate_limit_lock:
        last_time = _last_emit_times.get(event_name, 0)
        if now - last_time >= cooldown_s:
            _last_emit_times[event_name] = now
            return True
        return False


def emit_event(
    name: str,
    payload: Dict[str, Any],
    cooldown_s: int = 0,
    force: bool = False
) -> bool:
    """
    Emit a PHI-safe telemetry event.

    Args:
        name: Event name (e.g., "contract_drift_detected")
        payload: Event payload (will be sanitized for PHI safety)
        cooldown_s: Rate limit cooldown in seconds (0 = no cooldown)
        force: If True, ignore cooldown and always emit

    Returns:
        True if event was emitted, False if rate-limited

    PHI Safety:
        - Payload is sanitized to remove any PHI-unsafe keys
        - Only structured metadata should be included
        - Never include transcript text, segments, or patient data
    """
    # Check rate limit (unless forced)
    if not force and not _should_emit(name, cooldown_s):
        logger.debug(
            "Telemetry event rate-limited",
            event_name=name,
            cooldown_s=cooldown_s
        )
        return False

    # Sanitize payload for PHI safety
    safe_payload = _sanitize_payload(payload)

    # Build event envelope
    event = {
        "event": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": safe_payload
    }

    # Log as structured JSON (can be extended to send to external service)
    logger.info(
        "Telemetry event emitted",
        telemetry_event=name,
        telemetry_payload=json.dumps(safe_payload, default=str)
    )

    return True


def reset_rate_limits() -> None:
    """
    Reset all rate limit state. Useful for testing.
    """
    global _last_emit_times
    with _rate_limit_lock:
        _last_emit_times = {}


def get_last_emit_time(event_name: str) -> Optional[float]:
    """
    Get the last emit time for an event. Useful for testing.

    Returns:
        Unix timestamp of last emit, or None if never emitted
    """
    with _rate_limit_lock:
        return _last_emit_times.get(event_name)
