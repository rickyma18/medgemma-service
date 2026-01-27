import pytest
from unittest.mock import MagicMock, patch
from app.core.circuit_breaker import PipelineCircuitBreaker, PipelineState, get_circuit_breaker
from app.services.alerting import AlertEngine, AlertEvent, AlertSeverity, HighFailureRateRule

@pytest.fixture
def reset_circuit_breaker():
    cb = get_circuit_breaker()
    cb._state = PipelineState.ENABLED
    cb._manual_override = None
    yield
    cb._state = PipelineState.ENABLED
    cb._manual_override = None


def test_circuit_breaker_transitions(reset_circuit_breaker):
    cb = get_circuit_breaker()
    assert cb.state == PipelineState.ENABLED

    # Auto transition
    cb.transition(PipelineState.DEGRADED, "Testing")
    assert cb.state == PipelineState.DEGRADED

    # Auto transition further
    cb.transition(PipelineState.DISABLED, "Testing more")
    assert cb.state == PipelineState.DISABLED


def test_circuit_breaker_manual_override(reset_circuit_breaker):
    cb = get_circuit_breaker()
    
    # Enable manual override
    cb.set_manual_override(PipelineState.DISABLED)
    assert cb.state == PipelineState.DISABLED
    
    # Try auto transition -> Should be ignored
    cb.transition(PipelineState.ENABLED, "Should be ignored")
    assert cb.state == PipelineState.DISABLED
    
    # Clear override
    cb.set_manual_override(None)
    # State remains where the underlying state was? Or snap to last?
    # Our impl: self._state is the auto state. 
    # If we only had set manual override without changing self._state, it snaps back to _state.
    # In test setup _state was ENABLED.
    assert cb.state == PipelineState.ENABLED


def test_alert_engine_triggers_circuit_breaker(reset_circuit_breaker):
    engine = AlertEngine(sinks=[])
    cb = get_circuit_breaker()
    
    # Set up a critical alert
    metrics = {
        "jobs": {"completed": 100, "failed": 50}, # 33% fail rate
        "rates": {"fail_rate": 0.33, "fallback_rate": 0},
        "latency_ms": {"inference": {"p95": 100}, "queue": {"p95": 100}}
    }
    
    # Rule should trigger CRITICAL HighFailureRate
    # This should trigger CB transition to DEGRADED
    
    engine.evaluate(metrics)
    assert cb.state == PipelineState.DEGRADED


def test_alert_engine_escalates_to_disabled(reset_circuit_breaker):
    engine = AlertEngine(sinks=[])
    cb = get_circuit_breaker()
    
    # Already degraded
    cb.transition(PipelineState.DEGRADED, "Initial")
    
    # Now massive queue congestion -> Trigger DISABLED
    metrics = {
        "jobs": {"in_queue": 100, "completed": 0, "failed": 0}, 
        "rates": {"fail_rate": 0, "fallback_rate": 0},
        "latency_ms": {"inference": {"p95": 100}, "queue": {"p95": 100}}
    }
    
    engine.evaluate(metrics)
    assert cb.state == PipelineState.DISABLED
