from datetime import datetime, timedelta
import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any, Dict
from app.core.logging import get_safe_logger
from app.core.config import get_settings

logger = get_safe_logger(__name__)

class PipelineState(str, Enum):
    ENABLED = "enabled"    # Normal operation
    DEGRADED = "degraded"  # Force fallback (no LLM)
    DISABLED = "disabled"  # Reject new jobs

class PipelineCircuitBreaker:
    _instance = None
    
    def __init__(self):
        self._state = PipelineState.ENABLED
        # manual_override prevents auto-switching back
        self._manual_override: Optional[PipelineState] = None
        
        # Recovery state
        self._last_critical_alert_ts: Optional[datetime] = None
        self._state_entry_ts: datetime = datetime.utcnow()
        self._recovery_attempt_count: int = 0
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    @property
    def state(self) -> PipelineState:
        # Manual override takes precedence
        if self._manual_override:
            return self._manual_override
        return self._state
        
    def set_manual_override(self, state: Optional[PipelineState]):
        """
        Manually force a state. Set to None to resume auto-mode.
        """
        old_state = self.state
        self._manual_override = state
        new_state = self.state
        
        if old_state != new_state:
            # When forcing a state change manually, reset recovery counters mostly?
            # Or assume manual intervention fixes things? 
            # Let's reset timestamps to now to enforce fresh cooldowns if moving to restrictive
            self._state_entry_ts = datetime.utcnow()
            
            logger.warning(
                "Pipeline state manually changed",
                old_state=old_state.value,
                new_state=new_state.value,
                action="manual_override"
            )

    def transition(self, new_state: PipelineState, reason: str):
        """
        Attempt automatic transition (usually negative).
        Ignored if manual override is active.
        """
        if self._manual_override:
            return

        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            self._state_entry_ts = datetime.utcnow()
            
            # If moving to restricted state, record critical event time?
            # Actually, AlertEngine calls this. If it's a negative transition,
            # it implies a critical alert occurred.
            if new_state in (PipelineState.DEGRADED, PipelineState.DISABLED):
                self._last_critical_alert_ts = datetime.utcnow()
                # Do NOT reset attempt count here? Or do we? 
                # If we were recovering and failed again, exponential backoff continues.
                # If we were ENABLED and crashed, maybe reset?
                # Design says: "cooldown_effective = cooldown_seconds * (2 ** recovery_attempt_count)"
                # This implies attempt count grows on failures.
                # If we transition from ENABLED -> Degradation, it's a failure.
                # So we increment attempt count?
                # Actually, the spec says: "Reset recovery_attempt_count" ONLY on DEGRADED -> ENABLED.
                if old_state == PipelineState.ENABLED:
                     # Fresh failure sequence
                     self._recovery_attempt_count = 0
                else:
                     # Escestating failure or flapping
                     self._recovery_attempt_count += 1
            
            logger.warning(
                "Pipeline state automatically changed",
                old_state=old_state.value,
                new_state=new_state.value,
                reason=reason,
                action="auto_circuit_breaker",
                attempt_count=self._recovery_attempt_count
            )

    def evaluate_recovery(self, metrics: Dict[str, Any]):
        """
        Check conditions for auto-recovery.
        Should be called periodically (e.g. before processing jobs).
        """
        # 1. Manual override blocks recovery
        if self._manual_override:
            return

        # 2. If ENABLED, nothing to recover
        if self._state == PipelineState.ENABLED:
            return

        settings = get_settings()
        now = datetime.utcnow()
        cooldown_base = settings.circuit_breaker_cooldown_seconds
        
        # Calculate effective cooldown with exponential backoff
        # Cap exponent to avoid overflow, e.g. 2^10 is plenty
        exponent = min(self._recovery_attempt_count, 10)
        cooldown_effective = cooldown_base * (2 ** exponent)
        
        # CASE 1: DISABLED -> DEGRADED
        if self._state == PipelineState.DISABLED:
            # Condition A: Time since last critical alert
            if not self._last_critical_alert_ts:
                # Should not happen if transitioned correctly, but safety net
                self._last_critical_alert_ts = self._state_entry_ts
            
            time_since_alert = (now - self._last_critical_alert_ts).total_seconds()
            
            if time_since_alert > cooldown_effective:
                # Condition B: Queue Empty
                queue_size = metrics.get("jobs", {}).get("in_queue", 0)
                if queue_size == 0:
                    # Execute Transition
                    logger.info(
                        "Auto-recover: DISABLED -> DEGRADED",
                        cooldown_effective=cooldown_effective,
                        time_since_alert=time_since_alert
                    )
                    # Use internal set to bypass 'transition' logic of counting failures
                    self._state = PipelineState.DEGRADED
                    self._state_entry_ts = now
                    # Keep attempt count high until full recovery
                    return

        # CASE 2: DEGRADED -> ENABLED
        elif self._state == PipelineState.DEGRADED:
            # Condition A: Time since state entry
            time_in_state = (now - self._state_entry_ts).total_seconds()
            
            if time_in_state > cooldown_effective:
                # Condition B: Healthy Metrics
                rates = metrics.get("rates", {})
                fail_rate = rates.get("fail_rate", 0.0)
                
                # Active jobs (concurrency limit) - hardcoded to 1 for beta
                active_jobs = metrics.get("jobs", {}).get("active", 0)
                concurrency_limit = 1 
                
                safe_threshold = settings.circuit_breaker_recovery_failure_threshold
                
                if fail_rate < safe_threshold and active_jobs < concurrency_limit:
                    logger.info(
                        "Auto-recover: DEGRADED -> ENABLED",
                        cooldown_effective=cooldown_effective,
                        fail_rate=fail_rate
                    )
                    self._state = PipelineState.ENABLED
                    self._state_entry_ts = now
                    # SUCCESS: Reset backoff
                    self._recovery_attempt_count = 0


def get_circuit_breaker() -> PipelineCircuitBreaker:
    return PipelineCircuitBreaker.get_instance()
