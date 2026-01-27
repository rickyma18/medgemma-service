from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import time

from app.core.logging import get_safe_logger

logger = get_safe_logger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertEvent:
    name: str
    severity: AlertSeverity
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat() + "Z"
        }

class AlertSink:
    """Base class for alert sinks."""
    def emit(self, event: AlertEvent) -> None:
        raise NotImplementedError


class LoggingAlertSink(AlertSink):
    """Sink that writes to structured logs."""
    
    def emit(self, event: AlertEvent) -> None:
        log_method = logger.warning
        if event.severity == AlertSeverity.CRITICAL:
            log_method = logger.error
        elif event.severity == AlertSeverity.INFO:
            log_method = logger.info
            
        log_method(
            f"ALERT: {event.message}",
            alert_name=event.name,
            severity=event.severity.value,
            metric=event.metric,
            val=event.value,
            threshold=event.threshold
        )


class DummySlackAlertSink(AlertSink):
    """Stub sink that formats Slack payload without making HTTP request."""
    
    def emit(self, event: AlertEvent) -> None:
        # Construct simplified Slack payload
        color = "#36a64f" # green/info
        if event.severity == AlertSeverity.WARNING:
            color = "#ffcc00"
        elif event.severity == AlertSeverity.CRITICAL:
            color = "#ff0000"
            
        payload = {
            "channel": "#alerts",
            "username": "MedGemma Alerts",
            "attachments": [{
                "color": color,
                "title": f"[{event.severity.value.upper()}] {event.name}",
                "text": event.message,
                "fields": [
                    {"title": "Metric", "value": event.metric, "short": True},
                    {"title": "Value", "value": str(event.value), "short": True},
                    {"title": "Threshold", "value": str(event.threshold), "short": True}
                ],
                "ts": int(event.timestamp.timestamp())
            }]
        }
        
        # Log that we WOULD have sent this
        logger.info("DummySlackAlertSink payload", payload=payload)


class AlertRule:
    """Base class for alert rules."""
    
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[AlertEvent]:
        """
        Evaluate metrics against the rule.
        Returns AlertEvent if threshold breached, else None.
        """
        raise NotImplementedError


class HighFailureRateRule(AlertRule):
    """Alert if failure rate exceeds threshold (e.g. > 10%)."""
    
    def __init__(self, threshold: float = 0.10, min_volume: int = 5):
        self.threshold = threshold
        self.min_volume = min_volume

    def evaluate(self, metrics: Dict[str, Any]) -> Optional[AlertEvent]:
        # Extract job counts
        jobs = metrics.get("jobs", {})
        completed = jobs.get("completed", 0)
        failed = jobs.get("failed", 0)
        total = completed + failed
        
        # Avoid noise on low volume
        if total < self.min_volume:
            return None
            
        fail_rate = metrics.get("rates", {}).get("fail_rate", 0.0)
        
        if fail_rate > self.threshold:
            return AlertEvent(
                name="HighFailureRate",
                severity=AlertSeverity.CRITICAL,
                message=f"Failure rate {fail_rate:.1%} exceeds threshold {self.threshold:.1%}",
                metric="fail_rate",
                value=fail_rate,
                threshold=self.threshold
            )
        return None


class HighFallbackRateRule(AlertRule):
    """Alert if fallback rate exceeds threshold (e.g. > 20%)."""
    
    def __init__(self, threshold: float = 0.20, min_volume: int = 5):
        self.threshold = threshold
        self.min_volume = min_volume

    def evaluate(self, metrics: Dict[str, Any]) -> Optional[AlertEvent]:
        jobs = metrics.get("jobs", {})
        completed = jobs.get("completed", 0)
        
        if completed < self.min_volume:
            return None
            
        rate = metrics.get("rates", {}).get("fallback_rate", 0.0)
        
        if rate > self.threshold:
            return AlertEvent(
                name="HighFallbackRate",
                severity=AlertSeverity.WARNING,
                message=f"Fallback rate {rate:.1%} exceeds threshold {self.threshold:.1%}",
                metric="fallback_rate",
                value=rate,
                threshold=self.threshold
            )
        return None


class HighLatencyRule(AlertRule):
    """Alert if P95 latency exceeds threshold."""
    
    def __init__(self, metric_path: tuple, threshold_ms: int, name: str, severity: AlertSeverity = AlertSeverity.WARNING):
        self.metric_path = metric_path
        self.threshold = threshold_ms
        self.name = name
        self.severity = severity

    def evaluate(self, metrics: Dict[str, Any]) -> Optional[AlertEvent]:
        # Drill down into metrics dict
        val = metrics
        try:
            for key in self.metric_path:
                val = val[key]
        except (KeyError, TypeError):
            return None
            
        if val > self.threshold:
            return AlertEvent(
                name=self.name,
                severity=self.severity,
                message=f"{self.name} {val}ms exceeds threshold {self.threshold}ms",
                metric=".".join(self.metric_path),
                value=float(val),
                threshold=float(self.threshold)
            )
        return None


class QueueCongestionRule(AlertRule):
    """Alert if queue size exceeds limits."""
    
class QueueCongestionRule(AlertRule):
    """Alert if queue size exceeds limits."""
    
    def __init__(self, threshold_count: int = 10, severity: AlertSeverity = AlertSeverity.WARNING):
        self.threshold = threshold_count
        self.severity = severity

    def evaluate(self, metrics: Dict[str, Any]) -> Optional[AlertEvent]:
        in_queue = metrics.get("jobs", {}).get("in_queue", 0)
        
        if in_queue > self.threshold:
            return AlertEvent(
                name="QueueCongestion",
                severity=self.severity,
                message=f"Queue size {in_queue} exceeds threshold {self.threshold}",
                metric="in_queue",
                value=float(in_queue),
                threshold=float(self.threshold)
            )
        return None


class AlertEngine:
    """
    Evaluates rules and emits alerts.
    Designed to be stateless (rules receive current snapshot).
    """
    
    def __init__(self, sinks: Optional[List[AlertSink]] = None):
        self.rules: List[AlertRule] = [
            HighFailureRateRule(threshold=0.10, min_volume=5),
            HighFallbackRateRule(threshold=0.20, min_volume=5),
            HighLatencyRule(("latency_ms", "inference", "p95"), 45000, "SlowInference", AlertSeverity.WARNING),
            HighLatencyRule(("latency_ms", "queue", "p95"), 60000, "SlowQueue", AlertSeverity.WARNING),
            QueueCongestionRule(threshold_count=20, severity=AlertSeverity.CRITICAL)
        ]
        
        # Default to logging sink if none provided
        self.sinks = sinks if sinks is not None else [LoggingAlertSink()]

    def evaluate(self, metrics: Dict[str, Any]) -> List[AlertEvent]:
        events = []
        for rule in self.rules:
            try:
                event = rule.evaluate(metrics)
                if event:
                    events.append(event)
                    self._emit(event)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {type(rule).__name__}", error=str(e))
        return events

    def _emit(self, event: AlertEvent):
        """
        Emit alert to all configured sinks.
        PHI-safe: AlertEvent contains no sensitive data.
        
        Also triggers Pipeline Circuit Breaker if CRITICAL.
        """
        # Kill-Switch Logic
        if event.severity == AlertSeverity.CRITICAL:
            try:
                from app.core.circuit_breaker import get_circuit_breaker, PipelineState
                cb = get_circuit_breaker()
                # Determine target state based on alert type?
                # Per spec: ENABLED -> DEGRADED -> DISABLED
                # If critical failure rate, maybe disable?
                # If congestion, maybe disable?
                # Let's be aggressive for Beta: switch to DEGRADED first, unless already there?
                # Spec says: ENABLED -> DEGRADED -> DISABLED
                
                if cb.state == PipelineState.ENABLED:
                    cb.transition(PipelineState.DEGRADED, reason=f"CRITICAL ALERT: {event.name}")
                elif cb.state == PipelineState.DEGRADED:
                    # If heavily congested while degraded, maybe disable?
                    if event.name == "QueueCongestion":
                         cb.transition(PipelineState.DISABLED, reason=f"CRITICAL ALERT: {event.name}")
            except Exception as e:
                 logger.error("Failed to trigger circuit breaker", error=str(e))

        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception as e:
                # Sink failure should not crash the engine
                logger.error(
                    f"Alert sink failed: {type(sink).__name__}", 
                    error=str(e),
                    alert_name=event.name
                )

