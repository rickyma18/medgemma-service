"""
Unit tests for ÉPICA 8: Rate Limiting, Cache, Metrics.
PHI-safe: tests use mock data, never real transcripts or user IDs.
"""
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import Request

from app.core.rate_limiter import RateLimiter, get_rate_limiter, WINDOW_SECONDS
from app.core.cache import ExtractionCache, get_extraction_cache, CACHE_TTL_SECONDS
from app.core.metrics import MetricsCollector, get_metrics_collector
from app.schemas.request import Transcript, TranscriptSegment, Context
from app.schemas.response import ClinicalFacts


def create_test_transcript(text: str = "Test segment") -> Transcript:
    """Create a test transcript with given text."""
    return Transcript(
        segments=[
            TranscriptSegment(
                speaker="doctor",
                text=text,
                start_ms=0,
                end_ms=1000
            )
        ],
        language="es",
        duration_ms=1000
    )


class TestRateLimiter:
    """Tests for rate limiter functionality."""

    def setup_method(self):
        """Reset rate limiter before each test."""
        get_rate_limiter().reset()
    
    def test_allows_first_request(self):
        """First request should be allowed."""
        limiter = get_rate_limiter()
        allowed, remaining = limiter.check_and_record("test_uid_1")
        
        assert allowed is True
        assert remaining >= 0

    def test_tracks_requests_per_uid(self):
        """Should track requests independently per UID."""
        limiter = get_rate_limiter()
        
        # User 1 makes 5 requests
        for _ in range(5):
            limiter.check_and_record("user_1")
        
        # User 2 should still have full quota
        remaining_user2 = limiter.get_remaining("user_2")
        remaining_user1 = limiter.get_remaining("user_1")
        
        assert remaining_user2 > remaining_user1

    def test_rate_limit_enforced(self):
        """Should block requests after limit exceeded."""
        limiter = get_rate_limiter()
        limiter.set_limit(5)  # Low limit for testing
        
        # Make 5 allowed requests
        for i in range(5):
            allowed, _ = limiter.check_and_record("test_uid")
            assert allowed is True
        
        # 6th request should be blocked
        allowed, remaining = limiter.check_and_record("test_uid")
        assert allowed is False
        assert remaining == 0

    def test_uid_is_hashed(self):
        """UID should be hashed to avoid storing PII."""
        limiter = get_rate_limiter()
        
        # Hash should be deterministic
        hash1 = limiter._hash_uid("test_uid")
        hash2 = limiter._hash_uid("test_uid")
        
        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA256
        assert hash1 != "test_uid"

    def test_get_reset_time_returns_seconds(self):
        """Should return seconds until rate limit resets."""
        limiter = get_rate_limiter()
        limiter.check_and_record("test_uid")
        
        reset_time = limiter.get_reset_time("test_uid")
        
        assert reset_time > 0
        assert reset_time <= WINDOW_SECONDS


class TestExtractionCache:
    """Tests for extraction cache functionality."""

    def setup_method(self):
        """Reset cache before each test."""
        get_extraction_cache().clear()

    def test_cache_miss_returns_none(self):
        """Should return None for cache miss."""
        cache = get_extraction_cache()
        transcript = create_test_transcript()
        
        result = cache.get(transcript, None, None)
        
        assert result is None

    def test_cache_hit_returns_cached_data(self):
        """Should return cached data for identical request."""
        cache = get_extraction_cache()
        transcript = create_test_transcript()
        facts = ClinicalFacts()
        
        # Store in cache
        cache.set(transcript, None, None, facts, 100, "test-model")
        
        # Should hit cache
        result = cache.get(transcript, None, None)
        
        assert result is not None
        cached_facts, inference_ms, model_version = result
        assert inference_ms == 100
        assert model_version == "test-model"

    def test_different_transcript_is_cache_miss(self):
        """Should be cache miss for different transcript."""
        cache = get_extraction_cache()
        transcript1 = create_test_transcript("Hello")
        transcript2 = create_test_transcript("Goodbye")
        facts = ClinicalFacts()
        
        cache.set(transcript1, None, None, facts, 100, "test-model")
        
        result = cache.get(transcript2, None, None)
        assert result is None

    def test_cache_key_is_deterministic(self):
        """Same request should produce same cache key."""
        cache = get_extraction_cache()
        transcript = create_test_transcript()
        
        key1 = cache._compute_cache_key(transcript, None, None)
        key2 = cache._compute_cache_key(transcript, None, None)
        
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length

    def test_context_affects_cache_key(self):
        """Different context should produce different cache key."""
        cache = get_extraction_cache()
        transcript = create_test_transcript()
        context1 = Context(specialty="cardiology")
        context2 = Context(specialty="neurology")
        
        key1 = cache._compute_cache_key(transcript, context1, None)
        key2 = cache._compute_cache_key(transcript, context2, None)
        
        assert key1 != key2

    def test_get_stats_returns_entry_count(self):
        """Should return cache statistics."""
        cache = get_extraction_cache()
        transcript = create_test_transcript()
        
        cache.set(transcript, None, None, ClinicalFacts(), 100, "test")
        
        stats = cache.get_stats()
        
        assert stats["entries"] == 1
        assert stats["ttl_seconds"] == CACHE_TTL_SECONDS


class TestMetricsCollector:
    """Tests for metrics collector functionality."""

    def setup_method(self):
        """Reset metrics before each test."""
        get_metrics_collector().reset()

    def test_records_successful_request(self):
        """Should record successful request metrics."""
        metrics = get_metrics_collector()
        
        metrics.record_request(
            latency_ms=150,
            inference_ms=120,
            success=True,
            cache_hit=False
        )
        
        snapshot = metrics.get_snapshot()
        
        assert snapshot["total_requests"] == 1
        assert snapshot["success_count"] == 1
        assert snapshot["error_count"] == 0
        assert snapshot["latency"]["sum_ms"] == 150
        assert snapshot["inference_latency"]["sum_ms"] == 120

    def test_records_failed_request(self):
        """Should record failed request with error code."""
        metrics = get_metrics_collector()
        
        metrics.record_request(
            latency_ms=50,
            inference_ms=0,
            success=False,
            error_code="MODEL_ERROR"
        )
        
        snapshot = metrics.get_snapshot()
        
        assert snapshot["total_requests"] == 1
        assert snapshot["success_count"] == 0
        assert snapshot["error_count"] == 1
        assert snapshot["error_codes"]["MODEL_ERROR"] == 1

    def test_records_rate_limited(self):
        """Should record rate limited requests."""
        metrics = get_metrics_collector()
        
        metrics.record_rate_limited()
        
        snapshot = metrics.get_snapshot()
        
        assert snapshot["rate_limited"] == 1
        assert snapshot["error_codes"]["RATE_LIMITED"] == 1
        assert snapshot["error_count"] == 1

    def test_tracks_cache_hits(self):
        """Should track cache hit rate."""
        metrics = get_metrics_collector()
        
        # 3 hits, 2 misses
        for _ in range(3):
            metrics.record_request(100, 80, True, cache_hit=True)
        for _ in range(2):
            metrics.record_request(100, 80, True, cache_hit=False)
        
        snapshot = metrics.get_snapshot()
        
        assert snapshot["cache"]["hits"] == 3
        assert snapshot["cache"]["misses"] == 2
        assert snapshot["cache"]["hit_rate"] == 0.6

    def test_calculates_average_latency(self):
        """Should calculate average latency correctly."""
        metrics = get_metrics_collector()
        
        metrics.record_request(100, 80, True)
        metrics.record_request(200, 180, True)
        metrics.record_request(300, 280, True)
        
        snapshot = metrics.get_snapshot()
        
        assert snapshot["latency"]["avg_ms"] == 200.0
        assert snapshot["inference_latency"]["avg_ms"] == 180.0

    def test_uptime_increases(self):
        """Should track uptime in seconds."""
        metrics = get_metrics_collector()
        
        snapshot = metrics.get_snapshot()
        
        assert "uptime_seconds" in snapshot
        assert snapshot["uptime_seconds"] >= 0

    def test_is_thread_safe(self):
        """Should be safe to use from multiple threads."""
        import threading
        
        metrics = get_metrics_collector()
        errors = []
        
        def record_requests():
            try:
                for _ in range(100):
                    metrics.record_request(100, 80, True)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=record_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        snapshot = metrics.get_snapshot()
        assert snapshot["total_requests"] == 1000


class TestMetricsPhiSafety:
    """Verify metrics don't leak PHI."""

    def test_snapshot_contains_no_uid_patterns(self):
        """Metrics snapshot should not contain UID-like patterns."""
        metrics = get_metrics_collector()
        metrics.reset()
        
        metrics.record_request(100, 80, True)
        snapshot = metrics.get_snapshot()
        
        # Convert to string and check for UID patterns
        snapshot_str = str(snapshot)
        
        assert "uid" not in snapshot_str.lower()
        assert "@" not in snapshot_str  # No email patterns

    def test_error_codes_are_generic(self):
        """Error codes should be generic, not contain PHI."""
        metrics = get_metrics_collector()
        metrics.reset()
        
        metrics.record_request(100, 0, False, error_code="MODEL_ERROR")
        snapshot = metrics.get_snapshot()
        
        error_codes = snapshot["error_codes"]
        
        for code in error_codes.keys():
            assert code.isupper() or "_" in code
            assert len(code) < 30


class TestOpenAICompatErrorHandling:
    """Tests for OpenAI-compat backend error handling."""

    @pytest.mark.asyncio
    @patch('app.services.openai_compat_extractor.httpx.AsyncClient')
    async def test_backend_error_response_raises_model_error(self, mock_client_class):
        """When backend returns {"error": {...}}, should raise ModelError."""
        from app.services.openai_compat_extractor import openai_compat_extract
        from app.services.exceptions import ModelError
        from app.schemas.request import Transcript, TranscriptSegment
        
        # Setup mock response with error payload
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error": {
                "message": "Some backend error",
                "type": "invalid_request_error",
                "code": "context_length_exceeded"
            }
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client
        
        transcript = Transcript(
            segments=[TranscriptSegment(speaker="doctor", text="test", start_ms=0, end_ms=1000)],
            language="es",
            duration_ms=1000
        )
        
        with pytest.raises(ModelError) as exc_info:
            await openai_compat_extract(transcript, None, None)
        
        # Check error message is generic (PHI-safe)
        assert "Backend returned an error response" in str(exc_info.value.message)

    @pytest.mark.asyncio
    @patch('app.services.openai_compat_extractor.httpx.AsyncClient')
    async def test_missing_choices_raises_model_error(self, mock_client_class):
        """When backend response has no 'choices', should raise ModelError."""
        from app.services.openai_compat_extractor import openai_compat_extract
        from app.services.exceptions import ModelError
        from app.schemas.request import Transcript, TranscriptSegment
        
        # Setup mock response without choices
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "abc", "object": "chat.completion"}
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client
        
        transcript = Transcript(
            segments=[TranscriptSegment(speaker="doctor", text="test", start_ms=0, end_ms=1000)],
            language="es",
            duration_ms=1000
        )
        
        with pytest.raises(ModelError) as exc_info:
            await openai_compat_extract(transcript, None, None)
        
        assert "Invalid response format" in str(exc_info.value.message)


class TestModelVersionOverride:
    """Tests for config.modelVersion override."""

    @pytest.mark.asyncio
    @patch('app.services.extractor.get_settings')
    async def test_extract_returns_model_version(self, mock_settings):
        """extract() should return model_version in tuple."""
        from app.services.extractor import extract
        from app.schemas.request import Transcript, TranscriptSegment
        
        mock_settings.return_value.extractor_backend = "mock"
        
        transcript = Transcript(
            segments=[TranscriptSegment(speaker="doctor", text="test", start_ms=0, end_ms=1000)],
            language="es",
            duration_ms=1000
        )
        
        facts, inference_ms, model_version = await extract(transcript)
        
        assert model_version == "mock-0"
        assert inference_ms >= 0
        assert facts is not None

    @pytest.mark.asyncio
    @patch('app.services.openai_compat_extractor.httpx.AsyncClient')
    async def test_openai_compat_uses_config_model_version(self, mock_client_class):
        """openai_compat should use config.modelVersion if provided."""
        from app.services.openai_compat_extractor import openai_compat_extract
        from app.schemas.request import Transcript, TranscriptSegment, ExtractConfig
        
        # Valid JSON response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"chiefComplaint":{"text":"dolor"},"hpi":{"narrative":"test"},"ros":{"positives":[],"negatives":[]},"physicalExam":{"findings":[],"vitals":[]},"assessment":{"primary":null,"differential":[]},"plan":{"diagnostics":[],"treatments":[],"followUp":null}}'
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client
        
        transcript = Transcript(
            segments=[TranscriptSegment(speaker="doctor", text="test", start_ms=0, end_ms=1000)],
            language="es",
            duration_ms=1000
        )
        
        config = ExtractConfig(model_version="my-custom-model")
        
        facts, inference_ms, model_version = await openai_compat_extract(transcript, None, config)
        
        assert model_version == "openai-compat-my-custom-model"
        
        # Verify the model name was sent to the backend
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "my-custom-model"
