from datetime import datetime, timedelta
import pytest
from app.services.alerting import (
    AlertEngine, 
    AlertEvent, 
    AlertSeverity,
    HighFailureRateRule, 
    HighFallbackRateRule, 
    HighLatencyRule, 
    QueueCongestionRule
)

# Mock Metrics Payload
def mock_metrics(
    completed=50, 
    failed=0, 
    in_queue=0, 
    fail_rate=0.0, 
    fallback_rate=0.0, 
    p95_inf=1000, 
    p95_queue=500
):
    return {
        "jobs": {
            "completed": completed,
            "failed": failed,
            "in_queue": in_queue,
            "active": 1
        },
        "rates": {
            "fail_rate": fail_rate,
            "fallback_rate": fallback_rate
        },
        "latency_ms": {
            "inference": {"p95": p95_inf},
            "queue": {"p95": p95_queue}
        },
        "updatedAt": datetime.utcnow().isoformat()
    }


def test_alert_failure_rate():
    rule = HighFailureRateRule(threshold=0.10, min_volume=10)
    
    # 1. No alert
    metrics = mock_metrics(completed=95, failed=5, fail_rate=0.05)
    assert rule.evaluate(metrics) is None
    
    # 2. Alert Triggered
    metrics = mock_metrics(completed=80, failed=20, fail_rate=0.20)
    event = rule.evaluate(metrics)
    assert event is not None
    assert event.name == "HighFailureRate"
    assert event.severity == AlertSeverity.CRITICAL
    assert event.value == 0.20
    assert event.threshold == 0.10

    # 3. Low volume -> No Alert
    metrics = mock_metrics(completed=0, failed=1, fail_rate=1.0) # 1 sample
    assert rule.evaluate(metrics) is None


def test_alert_fallback_rate():
    rule = HighFallbackRateRule(threshold=0.20, min_volume=10)
    
    # 1. Alert Triggered
    metrics = mock_metrics(completed=50, fallback_rate=0.25)
    event = rule.evaluate(metrics)
    assert event is not None
    assert event.name == "HighFallbackRate"
    assert event.severity == AlertSeverity.WARNING
    assert event.value == 0.25


def test_alert_latency():
    rule = HighLatencyRule(("latency_ms", "inference", "p95"), 5000, "SlowInference")
    
    # 1. No Alert
    metrics = mock_metrics(p95_inf=4000)
    assert rule.evaluate(metrics) is None
    
    # 2. Alert Triggered
    metrics = mock_metrics(p95_inf=6000)
    event = rule.evaluate(metrics)
    assert event is not None
    assert event.name == "SlowInference"
    assert event.value == 6000
    assert event.severity == AlertSeverity.WARNING


def test_queue_congestion():
    rule = QueueCongestionRule(threshold_count=10)
    
    # Alert
    metrics = mock_metrics(in_queue=15)
    event = rule.evaluate(metrics)
    assert event is not None
    assert event.name == "QueueCongestion"
    assert event.value == 15


def test_alert_engine_integration():
    engine = AlertEngine()
    
    # Scenario: Everything is bad
    metrics = mock_metrics(
        completed=100, 
        failed=20, 
        fail_rate=0.16, # > 10%
        fallback_rate=0.25, # > 20%
        p95_inf=60000, # > 45s (critical default usually)
        in_queue=30 # > 20
    )
    
    events = engine.evaluate(metrics)
    
    assert len(events) >= 4
    names = {e.name for e in events}
    assert "HighFailureRate" in names
    assert "HighFallbackRate" in names
    assert "SlowInference" in names
    assert "QueueCongestion" in names

