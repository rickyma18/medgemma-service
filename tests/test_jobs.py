import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from app.schemas.request import ExtractRequest, Transcript, TranscriptSegment
from app.services.job_manager import JobManager


# Dummy request data
dummy_transcript = Transcript(
    segments=[
        TranscriptSegment(
            speaker="doctor",
            text="Test",
            startMs=0,
            endMs=1000
        )
    ],
    durationMs=1000,
    language="es"
)
dummy_request = ExtractRequest(transcript=dummy_transcript)


@pytest.fixture
def reset_job_manager():
    JobManager._instance = None
    yield
    JobManager._instance = None


@pytest.mark.asyncio
async def test_job_submission_and_status(reset_job_manager):
    manager = JobManager.get_instance()
    user_id = "user1"
    
    # Submit job
    job_id = await manager.submit_job(user_id, dummy_request)
    assert job_id is not None
    
    # Check status
    job = manager.get_job(job_id)
    assert job.status == "queued"
    assert job.user_id == user_id
    
    # Position
    pos = manager.get_queue_position(job_id)
    assert pos == 1


@pytest.mark.asyncio
async def test_one_job_per_user_limit(reset_job_manager):
    manager = JobManager.get_instance()
    user_id = "user1"
    
    # Submit first job
    await manager.submit_job(user_id, dummy_request)
    
    # Submit second job -> Should fail
    with pytest.raises(ValueError, match="User already has a job in progress"):
        await manager.submit_job(user_id, dummy_request)


@pytest.mark.asyncio
async def test_daily_quota(reset_job_manager):
    manager = JobManager.get_instance()
    user_id = "user_quota"
    
    # We need to simulate completed jobs to hit quota, 
    # because "in progress" limit prevents submitting 5 at once.
    # We can manually manipulate the counters for testing.
    
    from datetime import date
    today = date.today()
    manager._daily_counts[user_id] = {today: manager.max_daily_quota}
    
    # Submit should fail
    with pytest.raises(RuntimeError, match="Daily quota exceeded"):
        await manager.submit_job(user_id, dummy_request)


@pytest.mark.asyncio
async def test_global_semaphore_execution(reset_job_manager):
    manager = JobManager.get_instance()
    
    # Create 2 jobs from different users
    uid1 = "u1"
    uid2 = "u2"
    
    jid1 = await manager.submit_job(uid1, dummy_request)
    jid2 = await manager.submit_job(uid2, dummy_request)
    
    assert manager.get_queue_position(jid1) == 1
    assert manager.get_queue_position(jid2) == 2
    
    # Mock extractor to be slow so we can verify sequential execution
    async def slow_extract(*args, **kwargs):
        await asyncio.sleep(0.1) 
        return ({"mock": "result"}, 100, "v1")
    
    with patch("app.services.job_manager.extract_structured_v1", side_effect=slow_extract) as mock_extract:
        # Start worker as a task
        worker_task = asyncio.create_task(manager.start_worker())
        
        # Wait for first job to pick up
        # We need to wait a bit
        await asyncio.sleep(0.05)
        
        # Verify job 1 is running, job 2 is still queued
        j1 = manager.get_job(jid1)
        j2 = manager.get_job(jid2)
        
        # Note: Timing is tricky in asyncio tests. 
        # If worker picks up jid1 immediately, it sleeps 0.1s.
        # At 0.05s, jid1 should be running. jid2 should be queued (waiting for lock/worker).
        
        assert j1.status == "running" or j1.status == "done" # Depends on speed, but expected running
        
        # Crucial check: Semaphor means only 1 runs.
        # If j1 is running, j2 MUST be queued.
        if j1.status == "running":
            assert j2.status == "queued"
        
        # Wait for both to finish
        await asyncio.sleep(0.2)
        
        manager._shutting_down = True
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
            
        assert j1.status == "done"
        assert j2.status == "done"
        assert mock_extract.call_count == 2
