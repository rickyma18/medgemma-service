"""
In-memory metrics collector for PHI-safe observability.
Thread-safe singleton – no PHI is ever stored or logged.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Literal


@dataclass
class LatencyStats:
    """Aggregated latency statistics (sum/count for average calculation)."""
    sum_ms: int = 0
    count: int = 0

    def record(self, ms: int) -> None:
        self.sum_ms += ms
        self.count += 1

    @property
    def avg_ms(self) -> float:
        return self.sum_ms / self.count if self.count > 0 else 0.0


@dataclass
class MetricsData:
    """Container for all aggregated metrics."""
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    error_codes: Dict[str, int] = field(default_factory=dict)
    latency: LatencyStats = field(default_factory=LatencyStats)
    inference_latency: LatencyStats = field(default_factory=LatencyStats)
    cache_hits: int = 0
    cache_misses: int = 0
    rate_limited: int = 0
    started_at: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Thread-safe singleton for collecting PHI-safe metrics.
    
    Usage:
        metrics = get_metrics_collector()
        metrics.record_request(latency_ms=150, inference_ms=120, success=True)
    """
    _instance: "MetricsCollector | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._data = MetricsData()
                    instance._data_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def record_request(
        self,
        latency_ms: int,
        inference_ms: int,
        success: bool,
        error_code: str | None = None,
        cache_hit: bool = False,
    ) -> None:
        """
        Record a request completion.
        
        Args:
            latency_ms: Total request latency
            inference_ms: LLM inference time (0 if cached/errored)
            success: Whether request succeeded
            error_code: Error code if not success (PHI-safe codes only)
            cache_hit: Whether response was served from cache
        """
        with self._data_lock:
            self._data.total_requests += 1
            self._data.latency.record(latency_ms)
            
            if success:
                self._data.success_count += 1
                self._data.inference_latency.record(inference_ms)
            else:
                self._data.error_count += 1
                if error_code:
                    self._data.error_codes[error_code] = (
                        self._data.error_codes.get(error_code, 0) + 1
                    )
            
            if cache_hit:
                self._data.cache_hits += 1
            else:
                self._data.cache_misses += 1

    def record_rate_limited(self) -> None:
        """Record a rate-limited request (429)."""
        with self._data_lock:
            self._data.rate_limited += 1
            self._data.total_requests += 1
            self._data.error_count += 1
            self._data.error_codes["RATE_LIMITED"] = (
                self._data.error_codes.get("RATE_LIMITED", 0) + 1
            )

    def get_snapshot(self) -> dict:
        """
        Get a snapshot of current metrics.
        Returns a plain dict suitable for JSON serialization.
        """
        with self._data_lock:
            uptime_seconds = int(time.time() - self._data.started_at)
            return {
                "uptime_seconds": uptime_seconds,
                "total_requests": self._data.total_requests,
                "success_count": self._data.success_count,
                "error_count": self._data.error_count,
                "error_codes": dict(self._data.error_codes),
                "latency": {
                    "sum_ms": self._data.latency.sum_ms,
                    "count": self._data.latency.count,
                    "avg_ms": round(self._data.latency.avg_ms, 2),
                },
                "inference_latency": {
                    "sum_ms": self._data.inference_latency.sum_ms,
                    "count": self._data.inference_latency.count,
                    "avg_ms": round(self._data.inference_latency.avg_ms, 2),
                },
                "cache": {
                    "hits": self._data.cache_hits,
                    "misses": self._data.cache_misses,
                    "hit_rate": round(
                        self._data.cache_hits / max(self._data.cache_hits + self._data.cache_misses, 1),
                        4
                    ),
                },
                "rate_limited": self._data.rate_limited,
            }

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._data_lock:
            self._data = MetricsData()


def get_metrics_collector() -> MetricsCollector:
    """Get the singleton metrics collector instance."""
    return MetricsCollector()
