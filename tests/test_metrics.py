import pytest
from fastapi import status
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import get_settings
from app.services.job_manager import JobManager

@pytest.mark.asyncio
async def test_metrics_endpoint_open_by_default():
    # Ensure no admin key set
    settings = get_settings()
    original_key = settings.admin_api_key
    settings.admin_api_key = None
    
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/v1/jobs/metrics")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Verify structure
            assert "jobs" in data
            assert "in_queue" in data["jobs"]
            assert "latency_ms" in data
            assert "rates" in data
            
            # Verify PHI safety (no lists of texts or user IDs)
            # Just ensuring no unexpected keys
            assert set(data.keys()) == {"jobs", "latency_ms", "rates", "updatedAt"}
    finally:
        settings.admin_api_key = original_key


@pytest.mark.asyncio
async def test_metrics_endpoint_protected():
    settings = get_settings()
    original_key = settings.admin_api_key
    secret = "secret-admin-key"
    settings.admin_api_key = secret
    
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # 1. No token -> 403
            response = await ac.get("/v1/jobs/metrics")
            assert response.status_code == status.HTTP_403_FORBIDDEN
            
            # 2. Wrong token -> 403
            response = await ac.get("/v1/jobs/metrics", headers={"X-Admin-Token": "wrong"})
            assert response.status_code == status.HTTP_403_FORBIDDEN
            
            # 3. Correct token -> 200
            response = await ac.get("/v1/jobs/metrics", headers={"X-Admin-Token": secret})
            assert response.status_code == status.HTTP_200_OK
            
    finally:
        settings.admin_api_key = original_key


@pytest.mark.asyncio
async def test_metrics_values():
    # Populate some dummy data in JobManager
    manager = JobManager.get_instance()
    # Reset for cleaner test
    manager._metrics = {
        "queue_time_ms": [1000, 2000, 3000],
        "inference_time_ms": [4000, 5000, 6000],
        "failures": 1,
        "fallbacks": 0
    }
    
    # We need to mock queue size, as we can't easily push without async loop logic interfering
    # But we can check empty queue
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/v1/jobs/metrics")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check calculation
        # Queue p50 of [1000, 2000, 3000] is 2000
        assert data["latency_ms"]["queue"]["p50"] == 2000
        # Fail rate: 3 successes (inference_times length), 1 failure. Total 4. Rate 0.25
        assert data["rates"]["fail_rate"] == 0.25
        assert data["jobs"]["completed"] == 3
        assert data["jobs"]["failed"] == 1
