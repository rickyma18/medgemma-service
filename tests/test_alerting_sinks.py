import pytest
from unittest.mock import MagicMock, patch
from app.services.alerting import (
    AlertEngine, 
    AlertEvent, 
    AlertSeverity,
    AlertSink,
    LoggingAlertSink,
    DummySlackAlertSink,
    HighFailureRateRule
)

# Mock Metrics Payload
def mock_metrics(
    completed=50, 
    failed=0, 
    fail_rate=0.0, 
):
    return {
        "jobs": {
            "completed": completed,
            "failed": failed,
        },
        "rates": {
            "fail_rate": fail_rate,
        }
    }


def test_logging_alert_sink():
    sink = LoggingAlertSink()
    event = AlertEvent(
        name="TestAlert",
        severity=AlertSeverity.WARNING,
        message="Test message",
        metric="test",
        value=1.0,
        threshold=0.5
    )
    
    with patch("app.services.alerting.logger") as mock_logger:
        sink.emit(event)
        mock_logger.warning.assert_called_once()
        args, kwargs = mock_logger.warning.call_args
        assert "ALERT: Test message" in args[0]
        assert kwargs["alert_name"] == "TestAlert"


def test_dummy_slack_alert_sink():
    sink = DummySlackAlertSink()
    event = AlertEvent(
        name="TestAlert",
        severity=AlertSeverity.CRITICAL,
        message="Test message",
        metric="test",
        value=1.0,
        threshold=0.5
    )
    
    with patch("app.services.alerting.logger") as mock_logger:
        sink.emit(event)
        mock_logger.info.assert_called_once()
        args, kwargs = mock_logger.info.call_args
        assert "DummySlackAlertSink payload" in args[0]
        payload = kwargs["payload"]
        assert payload["attachments"][0]["color"] == "#ff0000" # critical red


def test_alert_engine_multiple_sinks():
    mock_sink1 = MagicMock(spec=AlertSink)
    mock_sink2 = MagicMock(spec=AlertSink)
    
    engine = AlertEngine(sinks=[mock_sink1, mock_sink2])
    
    # Override rules for predictability
    rule = MagicMock(spec=HighFailureRateRule)
    rule.evaluate.return_value = AlertEvent(
        name="MockEvent",
        severity=AlertSeverity.INFO,
        message="msg",
        metric="m",
        value=1,
        threshold=0
    )
    engine.rules = [rule]
    
    engine.evaluate({})
    
    mock_sink1.emit.assert_called_once()
    mock_sink2.emit.assert_called_once()


def test_alert_engine_sink_failure_resilience():
    bad_sink = MagicMock(spec=AlertSink)
    bad_sink.emit.side_effect = Exception("Boom")
    
    good_sink = MagicMock(spec=AlertSink)
    
    engine = AlertEngine(sinks=[bad_sink, good_sink])
    
    # Mock event
    rule = MagicMock()
    rule.evaluate.return_value = AlertEvent(name="E", severity="info", message="m", metric="m", value=0, threshold=0)
    engine.rules = [rule]
    
    # Should not raise
    with patch("app.services.alerting.logger") as mock_logger:
        engine.evaluate({})
        
        bad_sink.emit.assert_called_once()
        good_sink.emit.assert_called_once()
        
        # Verify error logged for bad sink
        errors = [call for call in mock_logger.error.call_args_list if "Alert sink failed" in call[0][0]]
        assert len(errors) == 1
