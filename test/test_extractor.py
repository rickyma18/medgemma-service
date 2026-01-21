"""
Unit tests for extractor service.
PHI-safe: tests use mock data, never real transcripts.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.schemas.request import Transcript, TranscriptSegment
from app.services.extractor import (
    mock_extract,
    extract,
    get_model_version,
    check_backend_health,
    MOCK_MODEL_VERSION,
)
from app.services.exceptions import (
    ExtractorError,
    ExtractorErrorCode,
    BackendUnavailableError,
    BackendTimeoutError,
    RateLimitedError,
    ModelError,
)


def create_test_transcript(segment_count: int = 2) -> Transcript:
    """Create a test transcript with given segment count."""
    segments = [
        TranscriptSegment(
            speaker="doctor",
            text="Test segment",
            start_ms=i * 1000,
            end_ms=(i + 1) * 1000
        )
        for i in range(segment_count)
    ]
    return Transcript(
        segments=segments,
        language="es",
        duration_ms=segment_count * 1000
    )


class TestMockExtract:
    """Tests for mock_extract function."""
    
    def test_returns_clinical_facts_and_timing(self):
        """Should return ClinicalFacts tuple with inference time."""
        transcript = create_test_transcript(2)
        
        facts, inference_ms = mock_extract(transcript)
        
        assert facts is not None
        assert inference_ms >= 0
        assert facts.chief_complaint.text == "Motivo de consulta registrado"
    
    def test_does_not_invent_diagnoses(self):
        """Mock should never return invented diagnoses."""
        transcript = create_test_transcript(5)
        
        facts, _ = mock_extract(transcript)
        
        assert facts.assessment.primary is None
        assert facts.assessment.differential == []
    
    def test_does_not_invent_symptoms(self):
        """Mock should never return invented symptoms."""
        transcript = create_test_transcript(5)
        
        facts, _ = mock_extract(transcript)
        
        assert facts.ros.positives == []
        assert facts.ros.negatives == []


class TestExtractFactory:
    """Tests for extract factory function."""
    
    @pytest.mark.asyncio
    @patch('app.services.extractor.get_settings')
    async def test_uses_mock_backend_by_default(self, mock_settings):
        """Should use mock backend when extractor_backend=mock."""
        mock_settings.return_value.extractor_backend = "mock"
        transcript = create_test_transcript(1)
        
        facts, inference_ms, model_version = await extract(transcript)
        
        assert facts is not None
        assert facts.chief_complaint.text == "Motivo de consulta registrado"
        assert model_version == MOCK_MODEL_VERSION
    
    @pytest.mark.asyncio
    @patch('app.services.extractor.get_settings')
    @patch('app.services.vllm_extractor.vllm_extract')
    @patch('app.services.vllm_extractor.get_vllm_model_version')
    async def test_uses_vllm_backend_when_configured(self, mock_version, mock_vllm, mock_settings):
        """Should use vLLM backend when extractor_backend=vllm."""
        from app.schemas.response import ClinicalFacts
        
        mock_settings.return_value.extractor_backend = "vllm"
        mock_vllm.return_value = (ClinicalFacts(), 100)
        mock_version.return_value = "vllm-test-model"
        transcript = create_test_transcript(1)
        
        facts, inference_ms, model_version = await extract(transcript)
        
        mock_vllm.assert_called_once()
        assert model_version == "vllm-test-model"
    
    @pytest.mark.asyncio
    @patch('app.services.extractor.get_settings')
    @patch('app.services.openai_compat_extractor.openai_compat_extract')
    async def test_uses_openai_compat_backend_when_configured(self, mock_compat, mock_settings):
        """Should use OpenAI-compat backend when extractor_backend=openai_compat."""
        from app.schemas.response import ClinicalFacts
        
        mock_settings.return_value.extractor_backend = "openai_compat"
        mock_compat.return_value = (ClinicalFacts(), 150, "openai-compat-test")
        transcript = create_test_transcript(1)
        
        facts, inference_ms, model_version = await extract(transcript)
        
        mock_compat.assert_called_once()
        assert model_version == "openai-compat-test"


class TestGetModelVersion:
    """Tests for get_model_version function."""
    
    @patch('app.services.extractor.get_settings')
    def test_returns_mock_version_for_mock_backend(self, mock_settings):
        """Should return mock version when backend=mock."""
        mock_settings.return_value.extractor_backend = "mock"
        
        version = get_model_version()
        
        assert version == MOCK_MODEL_VERSION
    
    @patch('app.services.extractor.get_settings')
    @patch('app.services.vllm_extractor.get_vllm_model_version')
    def test_returns_vllm_version_for_vllm_backend(self, mock_version, mock_settings):
        """Should return vllm version when backend=vllm."""
        mock_settings.return_value.extractor_backend = "vllm"
        mock_version.return_value = "vllm-test-model"
        
        version = get_model_version()
        
        assert version == "vllm-test-model"
    
    @patch('app.services.extractor.get_settings')
    @patch('app.services.openai_compat_extractor.get_openai_compat_model_version')
    def test_returns_compat_version_for_openai_compat_backend(self, mock_version, mock_settings):
        """Should return openai-compat version when backend=openai_compat."""
        mock_settings.return_value.extractor_backend = "openai_compat"
        mock_version.return_value = "openai-compat-test-model"
        
        version = get_model_version()
        
        assert version == "openai-compat-test-model"


class TestCheckBackendHealth:
    """Tests for check_backend_health function."""
    
    @pytest.mark.asyncio
    @patch('app.services.extractor.get_settings')
    async def test_mock_backend_always_healthy(self, mock_settings):
        """Mock backend should always report healthy."""
        mock_settings.return_value.extractor_backend = "mock"
        
        checks = await check_backend_health()
        
        assert checks["mock_extractor"] is True
    
    @pytest.mark.asyncio
    @patch('app.services.extractor.get_settings')
    @patch('app.services.vllm_extractor.check_vllm_health')
    async def test_vllm_backend_checks_health(self, mock_health, mock_settings):
        """vLLM backend should check vLLM health."""
        mock_settings.return_value.extractor_backend = "vllm"
        mock_health.return_value = True
        
        checks = await check_backend_health()
        
        assert checks["vllm_reachable"] is True
        mock_health.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.services.extractor.get_settings')
    @patch('app.services.openai_compat_extractor.check_openai_compat_health')
    async def test_openai_compat_backend_checks_health(self, mock_health, mock_settings):
        """OpenAI-compat backend should check health."""
        mock_settings.return_value.extractor_backend = "openai_compat"
        mock_health.return_value = True
        
        checks = await check_backend_health()
        
        assert checks["openai_compat_reachable"] is True
        mock_health.assert_called_once()


class TestExtractorExceptions:
    """Tests for extractor exception classes."""
    
    def test_backend_unavailable_error_has_correct_status(self):
        """BackendUnavailableError should have 503 status code."""
        error = BackendUnavailableError("test_backend")
        
        assert error.status_code == 503
        assert error.error_code == ExtractorErrorCode.BACKEND_UNAVAILABLE
        assert error.retryable is True
        assert "test_backend" in error.message
    
    def test_timeout_error_has_correct_status(self):
        """BackendTimeoutError should have 503 status code."""
        error = BackendTimeoutError(5000)
        
        assert error.status_code == 503
        assert error.error_code == ExtractorErrorCode.TIMEOUT
        assert error.retryable is True
        assert "5000" in error.message
    
    def test_rate_limited_error_has_correct_status(self):
        """RateLimitedError should have 429 status code."""
        error = RateLimitedError()
        
        assert error.status_code == 429
        assert error.error_code == ExtractorErrorCode.RATE_LIMITED
        assert error.retryable is True
    
    def test_model_error_has_correct_status(self):
        """ModelError should have 500 status code."""
        error = ModelError("Invalid JSON")
        
        assert error.status_code == 500
        assert error.error_code == ExtractorErrorCode.MODEL_ERROR
        assert error.retryable is True


class TestErrorCodeMapping:
    """Tests for error code to HTTP status mapping."""
    
    def test_all_error_codes_are_valid_response_codes(self):
        """All ExtractorErrorCode values should be valid in ErrorDetail."""
        from app.schemas.response import ErrorDetail
        
        for code in ExtractorErrorCode:
            # This should not raise ValidationError
            detail = ErrorDetail(
                code=code.value,
                message="test",
                retryable=True
            )
            assert detail.code == code.value
    
    def test_extractor_error_codes_match_response_schema(self):
        """ExtractorErrorCode values should all be allowed in response schema."""
        expected_codes = {
            "MODEL_ERROR",
            "BACKEND_UNAVAILABLE", 
            "TIMEOUT",
            "RATE_LIMITED"
        }
        
        actual_codes = {code.value for code in ExtractorErrorCode}
        
        # All extractor error codes should be in the response schema
        assert actual_codes <= expected_codes | {"UNAUTHORIZED", "BAD_REQUEST"}
