"""
Unit tests for auth module.
PHI-safe: tests use mock tokens, never real credentials.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException, Request

from app.core.auth import (
    extract_bearer_token,
    verify_firebase_token,
    _classify_auth_exception,
    AuthErrorCode,
    init_firebase,
    CredentialMode,
)
from firebase_admin import auth


class TestExtractBearerToken:
    """Tests for extract_bearer_token function."""
    
    def test_missing_authorization_header(self):
        """Should return 401 with 'Missing Authorization header' message."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            extract_bearer_token(mock_request)
        
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Missing Authorization header"
    
    def test_invalid_format_no_bearer(self):
        """Should return 401 for auth header without 'Bearer' prefix."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = "Basic abc123"
        
        with pytest.raises(HTTPException) as exc_info:
            extract_bearer_token(mock_request)
        
        assert exc_info.value.status_code == 401
        assert "Invalid Authorization header format" in exc_info.value.detail
    
    def test_invalid_format_extra_parts(self):
        """Should return 401 for auth header with extra spaces/parts."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = "Bearer token extra"
        
        with pytest.raises(HTTPException) as exc_info:
            extract_bearer_token(mock_request)
        
        assert exc_info.value.status_code == 401
    
    def test_valid_bearer_token(self):
        """Should extract token from valid Bearer header."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = "Bearer valid_token_here"
        
        token = extract_bearer_token(mock_request)
        
        assert token == "valid_token_here"
    
    def test_bearer_case_insensitive(self):
        """Should accept 'bearer' in any case."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = "BEARER valid_token"
        
        token = extract_bearer_token(mock_request)
        
        assert token == "valid_token"


class TestVerifyFirebaseToken:
    """Tests for verify_firebase_token function."""
    
    @patch('app.core.auth.get_settings')
    @patch('app.core.auth._firebase_app', MagicMock())
    @patch('app.core.auth.auth.verify_id_token')
    def test_invalid_token_returns_401_with_error_code(self, mock_verify, mock_settings):
        """Should return 401 with TOKEN_INVALID error code for invalid tokens."""
        mock_settings.return_value.auth_mode = "firebase"
        mock_verify.side_effect = auth.InvalidIdTokenError("Invalid token")
        
        with pytest.raises(HTTPException) as exc_info:
            verify_firebase_token("invalid_mock_token")
        
        assert exc_info.value.status_code == 401
        assert "TOKEN_INVALID" in exc_info.value.detail
    
    @patch('app.core.auth.get_settings')
    @patch('app.core.auth._firebase_app', MagicMock())
    @patch('app.core.auth.auth.verify_id_token')
    def test_expired_token_returns_401_with_error_code(self, mock_verify, mock_settings):
        """Should return 401 with TOKEN_EXPIRED error code for expired tokens."""
        mock_settings.return_value.auth_mode = "firebase"
        mock_verify.side_effect = auth.ExpiredIdTokenError("Token expired", cause=None)
        
        with pytest.raises(HTTPException) as exc_info:
            verify_firebase_token("expired_mock_token")
        
        assert exc_info.value.status_code == 401
        assert "TOKEN_EXPIRED" in exc_info.value.detail
    
    @patch('app.core.auth.get_settings')
    @patch('app.core.auth._firebase_app', MagicMock())
    @patch('app.core.auth.auth.verify_id_token')
    def test_revoked_token_returns_401_with_error_code(self, mock_verify, mock_settings):
        """Should return 401 with TOKEN_REVOKED error code for revoked tokens."""
        mock_settings.return_value.auth_mode = "firebase"
        mock_verify.side_effect = auth.RevokedIdTokenError("Token revoked")
        
        with pytest.raises(HTTPException) as exc_info:
            verify_firebase_token("revoked_mock_token")
        
        assert exc_info.value.status_code == 401
        assert "TOKEN_REVOKED" in exc_info.value.detail
    
    @patch('app.core.auth.get_settings')
    @patch('app.core.auth._firebase_app', MagicMock())
    @patch('app.core.auth.auth.verify_id_token')
    def test_cert_fetch_error_returns_401_with_error_code(self, mock_verify, mock_settings):
        """Should return 401 with CERT_FETCH_FAILED error code."""
        mock_settings.return_value.auth_mode = "firebase"
        mock_verify.side_effect = auth.CertificateFetchError("Cannot fetch certs", cause=None)
        
        with pytest.raises(HTTPException) as exc_info:
            verify_firebase_token("any_mock_token")
        
        assert exc_info.value.status_code == 401
        assert "CERT_FETCH_FAILED" in exc_info.value.detail
    
    @patch('app.core.auth.get_settings')
    @patch('app.core.auth._firebase_app', MagicMock())
    @patch('app.core.auth.auth.verify_id_token')
    def test_valid_token_returns_uid(self, mock_verify, mock_settings):
        """Should return uid for valid tokens (uid not logged)."""
        mock_settings.return_value.auth_mode = "firebase"
        mock_verify.return_value = {"uid": "mock_uid_123"}
        
        uid = verify_firebase_token("valid_mock_token")
        
        assert uid == "mock_uid_123"
        mock_verify.assert_called_once_with("valid_mock_token")


class TestClassifyAuthException:
    """Tests for _classify_auth_exception helper."""
    
    def test_expired_token_exception(self):
        """Should classify ExpiredIdTokenError as TOKEN_EXPIRED."""
        exc = auth.ExpiredIdTokenError("expired", cause=None)
        assert _classify_auth_exception(exc) == AuthErrorCode.TOKEN_EXPIRED
    
    def test_revoked_token_exception(self):
        """Should classify RevokedIdTokenError as TOKEN_REVOKED."""
        exc = auth.RevokedIdTokenError("revoked")
        assert _classify_auth_exception(exc) == AuthErrorCode.TOKEN_REVOKED
    
    def test_invalid_token_exception(self):
        """Should classify InvalidIdTokenError as TOKEN_INVALID."""
        exc = auth.InvalidIdTokenError("invalid")
        assert _classify_auth_exception(exc) == AuthErrorCode.TOKEN_INVALID
    
    def test_invalid_token_wrong_audience(self):
        """Should classify InvalidIdTokenError with 'aud' as PROJECT_MISMATCH."""
        exc = auth.InvalidIdTokenError("wrong audience (aud)")
        assert _classify_auth_exception(exc) == AuthErrorCode.PROJECT_MISMATCH
    
    def test_invalid_token_clock_skew(self):
        """Should classify InvalidIdTokenError with 'iat' issue as CLOCK_SKEW."""
        exc = auth.InvalidIdTokenError("issued in the future (iat)")
        assert _classify_auth_exception(exc) == AuthErrorCode.CLOCK_SKEW
    
    def test_cert_fetch_exception(self):
        """Should classify CertificateFetchError as CERT_FETCH_FAILED."""
        exc = auth.CertificateFetchError("cannot fetch", cause=None)
        assert _classify_auth_exception(exc) == AuthErrorCode.CERT_FETCH_FAILED
    
    def test_connection_error_exception(self):
        """Should classify ConnectionError as NETWORK_ERROR."""
        exc = ConnectionError("connection refused")
        assert _classify_auth_exception(exc) == AuthErrorCode.NETWORK_ERROR
    
    def test_unknown_exception(self):
        """Should classify unknown exceptions as UNKNOWN_AUTH_ERROR."""
        exc = ValueError("some unknown error")
        assert _classify_auth_exception(exc) == AuthErrorCode.UNKNOWN_AUTH_ERROR


class TestAuthErrorCodesArePhiSafe:
    """Verify that error codes don't leak PHI."""
    
    def test_error_codes_contain_no_phi_patterns(self):
        """All error codes should be generic strings without PHI."""
        for code in AuthErrorCode:
            # Error codes should not contain patterns that could be PHI
            assert "@" not in code.value  # No emails
            assert len(code.value) < 30  # Short, generic codes
            assert code.value.isupper() or "_" in code.value  # Standard error format
    
    @patch('app.core.auth.get_settings')
    @patch('app.core.auth._firebase_app', MagicMock())
    @patch('app.core.auth.auth.verify_id_token')
    def test_exception_detail_does_not_contain_token(self, mock_verify, mock_settings):
        """HTTPException detail should not expose the token."""
        mock_settings.return_value.auth_mode = "firebase"
        mock_verify.side_effect = auth.InvalidIdTokenError("bad token content")
        
        with pytest.raises(HTTPException) as exc_info:
            verify_firebase_token("my_secret_token_12345")
        
        # The detail should not contain the actual token
        assert "my_secret_token_12345" not in exc_info.value.detail
        # Should only contain the error code pattern
        assert "Authentication failed" in exc_info.value.detail


class TestDevAuthMode:
    """Tests for dev auth mode (AUTH_MODE=dev)."""
    
    @patch('app.core.auth.get_settings')
    def test_valid_dev_token_returns_dev_uid(self, mock_settings):
        """Should return 'dev_uid' when token matches DEV_BEARER_TOKEN."""
        mock_settings.return_value.auth_mode = "dev"
        mock_settings.return_value.dev_bearer_token = "dev-token"
        
        uid = verify_firebase_token("dev-token")
        
        assert uid == "dev_uid"
    
    @patch('app.core.auth.get_settings')
    def test_invalid_dev_token_returns_401(self, mock_settings):
        """Should return 401 when token doesn't match DEV_BEARER_TOKEN."""
        mock_settings.return_value.auth_mode = "dev"
        mock_settings.return_value.dev_bearer_token = "dev-token"
        
        with pytest.raises(HTTPException) as exc_info:
            verify_firebase_token("wrong-token")
        
        assert exc_info.value.status_code == 401
        assert "TOKEN_INVALID" in exc_info.value.detail
    
    @patch('app.core.auth.get_settings')
    def test_dev_mode_does_not_call_firebase(self, mock_settings):
        """In dev mode, should not attempt Firebase verification."""
        mock_settings.return_value.auth_mode = "dev"
        mock_settings.return_value.dev_bearer_token = "dev-token"
        
        # This should work without any Firebase setup
        uid = verify_firebase_token("dev-token")
        
        assert uid == "dev_uid"
    
    @patch('app.core.auth.get_settings')
    def test_custom_dev_token_accepted(self, mock_settings):
        """Should accept custom DEV_BEARER_TOKEN value."""
        mock_settings.return_value.auth_mode = "dev"
        mock_settings.return_value.dev_bearer_token = "my-custom-secret"
        
        uid = verify_firebase_token("my-custom-secret")
        
        assert uid == "dev_uid"

