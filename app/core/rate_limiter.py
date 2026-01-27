"""
In-memory rate limiter by user UID.
MVP implementation: sliding window, 60 requests/hour per UID.
PHI-safe: UIDs are hashed before storage.
"""
import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict


# Default limits
DEFAULT_REQUESTS_PER_HOUR = 600
WINDOW_SECONDS = 3600  # 1 hour


@dataclass
class RateLimitEntry:
    """Track request timestamps for a single UID."""
    timestamps: list[float] = field(default_factory=list)


class RateLimiter:
    """
    Sliding window rate limiter by UID.
    
    Uses UID hash (not raw UID) to avoid storing PII.
    """
    _instance: "RateLimiter | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "RateLimiter":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._entries: Dict[str, RateLimitEntry] = {}
                    instance._data_lock = threading.Lock()
                    instance._requests_per_hour = DEFAULT_REQUESTS_PER_HOUR
                    cls._instance = instance
        return cls._instance

    @staticmethod
    def _hash_uid(uid: str) -> str:
        """Hash UID to avoid storing PII."""
        return hashlib.sha256(uid.encode()).hexdigest()[:16]

    def _cleanup_old_entries(self, entry: RateLimitEntry, now: float) -> None:
        """Remove timestamps older than the window."""
        cutoff = now - WINDOW_SECONDS
        entry.timestamps = [ts for ts in entry.timestamps if ts > cutoff]

    def check_and_record(self, uid: str) -> tuple[bool, int]:
        """
        Check if request is allowed and record it.
        
        Args:
            uid: User ID (will be hashed)
            
        Returns:
            Tuple of (allowed: bool, remaining: int)
            - allowed: True if request is within limit
            - remaining: Number of requests remaining in window
        """
        uid_hash = self._hash_uid(uid)
        now = time.time()

        with self._data_lock:
            if uid_hash not in self._entries:
                self._entries[uid_hash] = RateLimitEntry()
            
            entry = self._entries[uid_hash]
            self._cleanup_old_entries(entry, now)
            
            current_count = len(entry.timestamps)
            remaining = max(0, self._requests_per_hour - current_count)
            
            if current_count >= self._requests_per_hour:
                return False, 0
            
            # Record this request
            entry.timestamps.append(now)
            return True, remaining - 1

    def get_remaining(self, uid: str) -> int:
        """Get remaining requests for a UID without recording."""
        uid_hash = self._hash_uid(uid)
        now = time.time()

        with self._data_lock:
            if uid_hash not in self._entries:
                return self._requests_per_hour
            
            entry = self._entries[uid_hash]
            self._cleanup_old_entries(entry, now)
            return max(0, self._requests_per_hour - len(entry.timestamps))

    def get_reset_time(self, uid: str) -> int:
        """
        Get seconds until rate limit resets (oldest entry expires).
        Returns 0 if no entries exist.
        """
        uid_hash = self._hash_uid(uid)
        now = time.time()

        with self._data_lock:
            if uid_hash not in self._entries:
                return 0
            
            entry = self._entries[uid_hash]
            self._cleanup_old_entries(entry, now)
            
            if not entry.timestamps:
                return 0
            
            oldest = min(entry.timestamps)
            return max(0, int((oldest + WINDOW_SECONDS) - now))

    def reset(self) -> None:
        """Reset all rate limit entries (for testing)."""
        with self._data_lock:
            self._entries.clear()

    def set_limit(self, requests_per_hour: int) -> None:
        """Update the requests per hour limit (for testing)."""
        with self._data_lock:
            self._requests_per_hour = requests_per_hour


def get_rate_limiter() -> RateLimiter:
    """Get the singleton rate limiter instance."""
    return RateLimiter()
