"""
Firebase authentication module.
PHI-safe: never log tokens, uid, email, or user data.
"""
import json
import os
import socket
from enum import Enum
from typing import Optional

import firebase_admin
from firebase_admin import auth, credentials
from fastapi import HTTPException, Request, status

from app.core.config import get_settings
from app.core.logging import get_safe_logger

logger = get_safe_logger(__name__)

# Global Firebase app instance
_firebase_app: Optional[firebase_admin.App] = None


class CredentialMode(str, Enum):
    """Firebase credential initialization mode."""
    SERVICE_ACCOUNT_JSON = "service_account_json"
    SERVICE_ACCOUNT_FILE = "service_account_file"
    ADC = "adc"


class AuthErrorCode(str, Enum):
    """
    PHI-safe error codes for authentication failures.
    These codes are safe to log and return to clients.
    """
    CERT_FETCH_FAILED = "CERT_FETCH_FAILED"
    NETWORK_ERROR = "NETWORK_ERROR"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_REVOKED = "TOKEN_REVOKED"
    TOKEN_INVALID = "TOKEN_INVALID"
    PROJECT_MISMATCH = "PROJECT_MISMATCH"
    CLOCK_SKEW = "CLOCK_SKEW"
    UNKNOWN_AUTH_ERROR = "UNKNOWN_AUTH_ERROR"


def init_firebase() -> firebase_admin.App:
    """
    Initialize Firebase Admin SDK singleton.
    
    Priority order:
    1. FIREBASE_CREDENTIALS_JSON env var (JSON string)
    2. GOOGLE_APPLICATION_CREDENTIALS env var (file path)
    3. Application Default Credentials (ADC)
    
    Logs (PHI-safe): credential_mode used for initialization.
    """
    global _firebase_app

    if _firebase_app is not None:
        return _firebase_app

    settings = get_settings()
    credential_mode: CredentialMode = CredentialMode.ADC
    cred: Optional[credentials.Base] = None

    try:
        # Priority 1: FIREBASE_CREDENTIALS_JSON (JSON string from env)
        if settings.firebase_credentials_json:
            cred_dict = json.loads(settings.firebase_credentials_json)
            cred = credentials.Certificate(cred_dict)
            credential_mode = CredentialMode.SERVICE_ACCOUNT_JSON
        
        # Priority 2: GOOGLE_APPLICATION_CREDENTIALS (file path)
        elif settings.google_application_credentials:
            cred = credentials.Certificate(settings.google_application_credentials)
            credential_mode = CredentialMode.SERVICE_ACCOUNT_FILE
        
        # Priority 3: Check if GOOGLE_APPLICATION_CREDENTIALS env var is set directly
        elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            cred = credentials.Certificate(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
            credential_mode = CredentialMode.SERVICE_ACCOUNT_FILE
        
        # Priority 4: ADC (Application Default Credentials)
        # cred remains None, firebase_admin will use ADC

        if cred is not None:
            _firebase_app = firebase_admin.initialize_app(
                cred,
                {"projectId": settings.firebase_project_id}
            )
        else:
            # Use ADC
            _firebase_app = firebase_admin.initialize_app(
                options={"projectId": settings.firebase_project_id}
            )

        # PHI-safe log: only credential mode, no secrets
        logger.info(
            "Firebase initialized",
            credential_mode=credential_mode.value
        )

        return _firebase_app

    except Exception as e:
        # Log error without sensitive details
        logger.error(
            "Failed to initialize Firebase",
            error_code="FIREBASE_INIT_ERROR",
            exception_class=type(e).__name__
        )
        raise


def _classify_auth_exception(exc: Exception) -> AuthErrorCode:
    """
    Classify Firebase auth exceptions into PHI-safe error codes.
    
    This function NEVER logs exception messages (may contain PHI).
    Only the exception class name is used for classification.
    """
    exc_class_name = type(exc).__name__
    exc_message_lower = str(exc).lower() if exc else ""
    
    # Firebase SDK specific exceptions
    if isinstance(exc, auth.ExpiredIdTokenError):
        return AuthErrorCode.TOKEN_EXPIRED
    
    if isinstance(exc, auth.RevokedIdTokenError):
        return AuthErrorCode.TOKEN_REVOKED
    
    if isinstance(exc, auth.InvalidIdTokenError):
        # Check for specific sub-causes without logging the message
        if "wrong audience" in exc_message_lower or "aud" in exc_message_lower:
            return AuthErrorCode.PROJECT_MISMATCH
        if "issued in the future" in exc_message_lower or "iat" in exc_message_lower:
            return AuthErrorCode.CLOCK_SKEW
        if "has expired" in exc_message_lower:
            return AuthErrorCode.TOKEN_EXPIRED
        return AuthErrorCode.TOKEN_INVALID
    
    if isinstance(exc, auth.CertificateFetchError):
        return AuthErrorCode.CERT_FETCH_FAILED
    
    # Network-related errors
    if isinstance(exc, (socket.timeout, socket.gaierror, ConnectionError)):
        return AuthErrorCode.NETWORK_ERROR
    
    # Check class name for common patterns
    if "certificate" in exc_class_name.lower() or "cert" in exc_class_name.lower():
        return AuthErrorCode.CERT_FETCH_FAILED
    
    if "network" in exc_class_name.lower() or "connection" in exc_class_name.lower():
        return AuthErrorCode.NETWORK_ERROR
    
    if "timeout" in exc_class_name.lower():
        return AuthErrorCode.NETWORK_ERROR
    
    return AuthErrorCode.UNKNOWN_AUTH_ERROR


def verify_firebase_token(token: str) -> str:
    """
    Verify token and return the user's UID.
    
    Behavior depends on AUTH_MODE setting:
    - firebase: Uses Firebase Admin SDK to verify ID token
    - dev: Accepts DEV_BEARER_TOKEN and returns fixed uid "dev_uid"

    Args:
        token: Bearer token from Authorization header

    Returns:
        User UID (doctor_id) - NEVER LOGGED

    Raises:
        HTTPException: If token is invalid or expired, with PHI-safe error_code
    """
    settings = get_settings()
    
    # Dev auth mode - for local testing without Firebase
    if settings.auth_mode == "dev":
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DEBUG-AUTH: Temporary logging to diagnose TOKEN_INVALID issue
        # REMOVE THESE LOGS AFTER DEBUGGING (tokens should NEVER be logged)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        expected_token = settings.dev_bearer_token
        received_token = token.strip() if token else ""
        
        logger.warning(
            "DEBUG-AUTH: DEV mode token comparison",
            expected_token_repr=repr(expected_token),
            expected_len=len(expected_token) if expected_token else 0,
            received_token_repr=repr(received_token),
            received_len=len(received_token) if received_token else 0,
            tokens_equal=(received_token == expected_token),
            auth_mode=settings.auth_mode,
        )
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if received_token == expected_token:
            # Return fixed dev uid - DO NOT LOG
            return "dev_uid"
        else:
            # Token doesn't match - same error format as Firebase
            logger.warning(
                "Token verification failed",
                error_code=AuthErrorCode.TOKEN_INVALID.value,
                exception_class="DevAuthError"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed ({AuthErrorCode.TOKEN_INVALID.value})"
            )
    
    # Firebase auth mode (default)
    if _firebase_app is None:
        init_firebase()

    try:
        # Verify the token - this checks signature, expiration, etc.
        decoded_token = auth.verify_id_token(token)

        # Extract UID - DO NOT LOG THIS
        uid: str = decoded_token["uid"]

        return uid

    except Exception as exc:
        # Classify the exception into a PHI-safe error code
        error_code = _classify_auth_exception(exc)
        
        # PHI-safe log: error_code + exception class only (no message, no stacktrace)
        logger.warning(
            "Token verification failed",
            error_code=error_code.value,
            exception_class=type(exc).__name__
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed ({error_code.value})"
        )


def extract_bearer_token(request: Request) -> str:
    """
    Extract Bearer token from Authorization header.

    Args:
        request: FastAPI request object

    Returns:
        The token string

    Raises:
        HTTPException: If header is missing or malformed
    """
    auth_header = request.headers.get("Authorization")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DEBUG-AUTH: Log raw header to diagnose auth issues
    # REMOVE AFTER DEBUGGING (headers should NOT be logged in production)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.warning(
        "DEBUG-AUTH: Raw authorization header",
        auth_header_repr=repr(auth_header),
        auth_header_len=len(auth_header) if auth_header else 0,
        all_headers=list(request.headers.keys()),
    )
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    if not auth_header:
        logger.warning("Missing authorization header", error_code="NO_AUTH_HEADER")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header"
        )

    parts = auth_header.split(" ")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DEBUG-AUTH: Log parsed parts
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.warning(
        "DEBUG-AUTH: Header parts after split",
        parts_count=len(parts),
        parts_repr=[repr(p) for p in parts],
    )
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    if len(parts) != 2 or parts[0].lower() != "bearer":
        logger.warning("Invalid authorization header format", error_code="INVALID_AUTH_FORMAT")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: Bearer <token>"
        )

    # Apply .strip() to handle any trailing whitespace
    return parts[1].strip()


async def get_current_user(request: Request) -> str:
    """
    FastAPI dependency to get authenticated user ID.

    Returns:
        User UID (doctor_id) - DO NOT LOG
    """
    token = extract_bearer_token(request)
    return verify_firebase_token(token)


async def verify_auth_header(request: Request) -> None:
    """
    Router-level dependency that verifies auth BEFORE body parsing.

    This dependency only checks headers - it doesn't access the body,
    so FastAPI executes it BEFORE attempting to parse/validate JSON body.
    
    Also stores the authenticated uid in request.state for rate limiting.

    Raises:
        HTTPException 401: If token missing, malformed, or invalid.
        Response detail includes PHI-safe error_code.
    """
    token = extract_bearer_token(request)
    uid = verify_firebase_token(token)
    # Store uid in request.state for rate limiting (never logged)
    request.state.uid = uid

