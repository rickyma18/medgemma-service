import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from app.core.logging import get_safe_logger
from app.core.config import get_settings
from app.core.circuit_breaker import get_circuit_breaker, PipelineState
from app.schemas.request import ExtractRequest
from app.services.structured_v1_extractor import extract_structured_v1

logger = get_safe_logger(__name__)

# Constants
JOB_TTL_SECONDS = 1800  # 30 minutes
ETA_WINDOW_SIZE = 10    # Number of past jobs to use for moving average


@dataclass
class Job:
    id: str
    user_id: str
    request: ExtractRequest
    status: str = "queued"  # queued, running, done, failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    fallback_used: bool = False
    contract_warnings: List[str] = field(default_factory=list)
    
    # Metadata for metrics
    inference_ms: int = 0
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class JobManager:
    _instance = None

    def __init__(self):
        # FIFO Queue
        self._queue: asyncio.Queue = asyncio.Queue()
        
        # Job Storage (In-memory)
        self._jobs: Dict[str, Job] = {}
        
        # Concurrency Control
        self._active_job_id: Optional[str] = None
        self._user_active_jobs: Dict[str, str] = {}  # user_id -> job_id
        
        # Quotas
        self._daily_counts: Dict[str, Dict[date, int]] = {}
        
        # Metrics / ETA
        self._inference_times: deque = deque(maxlen=ETA_WINDOW_SIZE)
        self._metrics = {
            "queue_time_ms": [],
            "inference_time_ms": [],
            "failures": 0,
            "fallbacks": 0,
        }
        
        self._shutting_down = False
        self._maintenance_mode = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def max_daily_quota(self) -> int:
        return get_settings().job_queue_daily_quota
    
    @property
    def maintenance_mode(self) -> bool:
        return self._maintenance_mode

    def set_maintenance_mode(self, enabled: bool):
        self._maintenance_mode = enabled
        logger.warning(
            f"Maintenance mode {'ENABLED' if enabled else 'DISABLED'}",
            action="maintenance_toggle"
        )

    def get_job(self, job_id: str) -> Optional[Job]:

        return self._jobs.get(job_id)

    async def submit_job(self, user_id: str, request: ExtractRequest) -> str:
        """
        Enqueues a new job if quotas allow.
        Returns job_id or raises exception.
        """
        if self._maintenance_mode:
            raise RuntimeError("Service is under maintenance")
            
        # Check for auto-recovery before rejecting
        self._check_recovery_conditions()
        
        cb_state = get_circuit_breaker().state
        if cb_state == PipelineState.DISABLED:
            raise RuntimeError("Service is disabled (Kill-Switch)")
            
        # 1. Check User Active Jobs Limit

        # 1. Check User Active Jobs Limit
        if user_id in self._user_active_jobs:
            existing_id = self._user_active_jobs[user_id]
            # Verify if it's actually still active
            job = self._jobs.get(existing_id)
            if job and job.status in ("queued", "running"):
                raise ValueError(f"User already has a job in progress: {existing_id}")
            else:
                # Cleanup state if stale
                del self._user_active_jobs[user_id]

        # 2. Check Daily Quota
        today = date.today()
        user_counts = self._daily_counts.setdefault(user_id, {})
        current_count = user_counts.get(today, 0)
        
        if current_count >= self.max_daily_quota:
            # Check if this user is a super admin or has overrides? 
            # For now, strict limit.
            raise RuntimeError("Daily quota exceeded")

        # 3. Create Job
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, user_id=user_id, request=request)
        self._jobs[job_id] = job
        
        # 4. Enqueue
        await self._queue.put(job_id)
        
        # 5. Update State
        self._user_active_jobs[user_id] = job_id
        user_counts[today] = current_count + 1
        
        logger.info(
            "Job submitted",
            job_id=job_id,
            user_id=user_id,  # PHI warning: user_id is usually safe (GUID), but check policy. 
                              # User prompt says "No incluir PHI en logs". User ID is typically system metadata.
            queue_size=self._queue.qsize()
        )
        return job_id

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Returns 1-based position in queue, or None if not queued."""
        if job_id not in self._jobs:
            return None
        
        job = self._jobs[job_id]
        if job.status != "queued":
            return None

        # This is O(N) but queue makes it hard to peek.
        # Since we use an asyncio.Queue, we can't easily iterate it without private access or copy.
        # For a simple Beta, we can trust the internal _queue._queue deque if standard asyncio.
        # Or, we can just iterate our valid jobs and sort by created_at for those that are queued.
        
        queued_jobs = [j for j in self._jobs.values() if j.status == "queued"]
        queued_jobs.sort(key=lambda x: x.created_at)
        
        try:
            return queued_jobs.index(job) + 1
        except ValueError:
            return None

    def get_eta_seconds(self, position: int) -> int:
        """Calculate approximate ETA."""
        avg_time = 30.0  # Default 30s
        if self._inference_times:
            avg_time = sum(self._inference_times) / len(self._inference_times)
            
        # If position is 1 (next), ETA is basically avg_time (or remaining of current + avg_time)
        # We simplify: pos * avg_time
        return int(position * avg_time)

    async def start_worker(self):
        logger.info("Starting Job Worker")
        while not self._shutting_down:
            try:
                # Wait for a job with timeout to allow periodic cleanup
                try:
                    job_id = await asyncio.wait_for(self._queue.get(), timeout=60.0)
                    await self._process_job(job_id)
                    self._queue.task_done()
                except asyncio.TimeoutError:
                    # No jobs in 60s, run cleanup and check recovery
                    self._cleanup_stale_jobs()
                    self._check_recovery_conditions()
            except Exception as e:
                logger.error("Worker error loop", error_code="WORKER_ERROR")
                # Prevent tight loop on error (though wait_for protects us mostly)
                await asyncio.sleep(1)

    async def _process_job(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job:
            return

        self._active_job_id = job_id
        job.status = "running"
        job.started_at = time.time()
        
        # Calculate queue time
        queue_duration = (job.started_at - job.created_at) * 1000
        self._metrics["queue_time_ms"].append(queue_duration)

        logger.info("Job started", job_id=job_id)

        try:
            # Execute Extraction
            # Check Circuit Breaker for DEGRADED state (Fallback)
            cb_state = get_circuit_breaker().state
            
            if cb_state == PipelineState.DEGRADED:
                # Force fallback (baseline / heuristic / mock if implemented)
                logger.warning("Executing in DEGRADED mode (Fallback)", job_id=job_id)
                
                # Mock heuristic fallback for Beta
                from app.schemas.structured_fields_v1 import StructuredFieldsV1
                result = StructuredFieldsV1(
                    motivoConsulta="[DEGRADED MODE] Servicio degradado temporalmente.",
                    diagnostico={"texto": "Servicio no disponible", "tipo": "sindromico"}
                )
                inference_ms = 0
                model_version = "fallback-heuristic"
                job.fallback_used = True
                
            else:
                # Normal Execution
                result, inference_ms, model_version = await extract_structured_v1(
                    transcript=job.request.transcript,
                    context=job.request.context
                )
            
            job.result = result
            job.inference_ms = inference_ms
            job.status = "done"
            
            # Record success metrics
            self._inference_times.append(inference_ms / 1000.0)
            self._metrics["inference_time_ms"].append(inference_ms)
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self._metrics["failures"] += 1
            logger.error("Job failed", job_id=job_id, error_code="JOB_FAILED")
            
        finally:
            job.completed_at = time.time()
            self._active_job_id = None
            
            # Remove from active user mapping so they can submit again
            if job.user_id in self._user_active_jobs and self._user_active_jobs[job.user_id] == job_id:
                del self._user_active_jobs[job.user_id]
            
            # Cleanup old jobs occasionaly?
            self._cleanup_stale_jobs()
            
            logger.info("Job finished", job_id=job_id, status=job.status)

    def _cleanup_stale_jobs(self):
        """Remove jobs older than TTL."""
        now = time.time()
        to_remove = []
        for jid, job in self._jobs.items():
            if (now - job.created_at) > JOB_TTL_SECONDS:
                to_remove.append(jid)
        
        for jid in to_remove:
            del self._jobs[jid]
            # Note: Do not delete from daily counts!

    def _check_recovery_conditions(self):
        """Helper to trigger circuit breaker recovery check."""
        # Only check if strictly needed to avoid overhead? 
        # But we need metrics.
        # Let's use the lightweight get_metrics() or construct ad-hoc?
        # evaluating recovery uses: in_queue, fail_rate, active.
        # get_observability_metrics() calculates these.
        metrics = self.get_observability_metrics()
        get_circuit_breaker().evaluate_recovery(metrics)

    def get_observability_metrics(self) -> Dict[str, Any]:

        """
        Get aggregated PHI-safe metrics for observability dashboard.
        Returns JSON-serializable dict compatible with /v1/jobs/metrics.
        """
        import statistics
        
        # Snapshots to avoid race conditions (roughly)
        q_times = list(self._metrics["queue_time_ms"])
        i_times = list(self._metrics["inference_time_ms"])
        failures = self._metrics["failures"]
        fallbacks = self._metrics["fallbacks"]
        
        completed = len(i_times)
        total_attempts = completed + failures
        
        # Rates
        fail_rate = (failures / total_attempts) if total_attempts > 0 else 0.0
        fallback_rate = (fallbacks / completed) if completed > 0 else 0.0
        
        def safe_percentile(data, p):
            if not data:
                return 0
            if len(data) == 1:
                return data[0]
            try:
                return sorted(data)[int(len(data) * p)]
            except IndexError:
                return data[-1]

        return {
            "jobs": {
                "in_queue": self._queue.qsize(),
                "active": 1 if self._active_job_id else 0,
                "completed": completed,
                "failed": failures,
                "active": 1 if self._active_job_id else 0,
                "completed": completed,
                "failed": failures,
                "maintenance_mode": self._maintenance_mode,
                "circuit_breaker_state": get_circuit_breaker().state.value
            },
            "latency_ms": {
                "queue": {
                    "p50": int(statistics.median(q_times)) if q_times else 0,
                    "p95": int(safe_percentile(q_times, 0.95))
                },
                "inference": {
                    "p50": int(statistics.median(i_times)) if i_times else 0,
                    "p95": int(safe_percentile(i_times, 0.95))
                }
            },
            "rates": {
                "fail_rate": round(fail_rate, 4),
                "fallback_rate": round(fallback_rate, 4)
            },
            "updatedAt": datetime.utcnow().isoformat() + "Z"
        }

    # Observability Getters
    def get_metrics(self):

        import statistics
        
        q_times = self._metrics["queue_time_ms"]
        i_times = self._metrics["inference_time_ms"]
        
        return {
            "queue_pending": self._queue.qsize(),
            "active_job": 1 if self._active_job_id else 0,
            "queue_time_p50": statistics.median(q_times) if q_times else 0,
            "queue_time_p95": sorted(q_times)[int(len(q_times)*0.95)] if q_times else 0,
            "inference_time_p50": statistics.median(i_times) if i_times else 0,
            "inference_time_p95": sorted(i_times)[int(len(i_times)*0.95)] if i_times else 0,
            "failures": self._metrics["failures"],
            "total_jobs_processed": len(i_times) + self._metrics["failures"]
        }
