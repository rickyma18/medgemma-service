import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from app.core.circuit_breaker import PipelineCircuitBreaker, PipelineState, get_circuit_breaker
from app.services.job_manager import JobManager

@pytest.fixture
def reset_circuit_breaker():
    cb = get_circuit_breaker()
    cb._state = PipelineState.ENABLED
    cb._manual_override = None
    cb._recovery_attempt_count = 0
    cb._last_critical_alert_ts = None
    yield
    cb._state = PipelineState.ENABLED
    cb._manual_override = None


def test_auto_recovery_disabled_to_degraded(reset_circuit_breaker):
    cb = get_circuit_breaker()
    
    # Simulate DISABLED state due to congestion
    cb.transition(PipelineState.DISABLED, "Critical congestion")
    cb._last_critical_alert_ts = datetime.utcnow() - timedelta(minutes=10) # 10 mins ago
    
    # Cooldown is default 300s (5m). So 10m > 5m.
    
    # Empty queue
    metrics = {"jobs": {"in_queue": 0}}
    
    cb.evaluate_recovery(metrics)
    
    assert cb.state == PipelineState.DEGRADED
    assert cb._state == PipelineState.DEGRADED


def test_auto_recovery_degraded_to_enabled(reset_circuit_breaker):
    cb = get_circuit_breaker()
    
    # Simulate DEGRADED state
    cb._state = PipelineState.DEGRADED
    cb._state_entry_ts = datetime.utcnow() - timedelta(minutes=6)
    
    # Healthy metrics
    metrics = {
        "rates": {"fail_rate": 0.01}, # < 5%
        "jobs": {"active": 0}
    }
    
    cb.evaluate_recovery(metrics)
    
    assert cb.state == PipelineState.ENABLED


def test_no_recovery_if_cooldown_not_met(reset_circuit_breaker):
    cb = get_circuit_breaker()
    cb.transition(PipelineState.DISABLED, "Crash")
    cb._last_critical_alert_ts = datetime.utcnow() - timedelta(seconds=10) # Only 10s ago
    
    metrics = {"jobs": {"in_queue": 0}}
    
    cb.evaluate_recovery(metrics)
    assert cb.state == PipelineState.DISABLED


def test_anti_flapping_backoff(reset_circuit_breaker):
    cb = get_circuit_breaker()
    cb._state = PipelineState.DISABLED
    cb._recovery_attempt_count = 3 # Backoff factor 2^3 = 8. 300*8 = 2400s (40m)
    
    cb._last_critical_alert_ts = datetime.utcnow() - timedelta(minutes=10) # 600s
    
    # 600s < 2400s, should NOT recover
    metrics = {"jobs": {"in_queue": 0}}
    cb.evaluate_recovery(metrics)
    assert cb.state == PipelineState.DISABLED
    
    # Forward time to 41 mins
    cb._last_critical_alert_ts = datetime.utcnow() - timedelta(minutes=41)
    cb.evaluate_recovery(metrics)
    assert cb.state == PipelineState.DEGRADED


def test_manual_override_blocks_recovery(reset_circuit_breaker):
    cb = get_circuit_breaker()
    cb.set_manual_override(PipelineState.DISABLED)
    
    # Even if metrics are perfect and time passed
    cb._last_critical_alert_ts = datetime.utcnow() - timedelta(days=1)
    metrics = {"jobs": {"in_queue": 0}}
    
    cb.evaluate_recovery(metrics)
    assert cb.state == PipelineState.DISABLED
