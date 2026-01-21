"""
In-memory response cache for extraction results.
Cache key: SHA256 hash of normalized request (transcript + context + modelVersion).
TTL: 24 hours.
PHI-safe: Only hashes are stored as keys, values are serialized ClinicalFacts.
"""
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from app.schemas.request import Context, ExtractConfig, Transcript
from app.schemas.response import ClinicalFacts


# Cache TTL in seconds (24 hours)
CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass
class CacheEntry:
    """A cached extraction result with expiration."""
    facts: ClinicalFacts
    inference_ms: int
    model_version: str
    created_at: float

    def is_expired(self) -> bool:
        return time.time() - self.created_at > CACHE_TTL_SECONDS


class ExtractionCache:
    """
    Thread-safe in-memory cache for extraction results.
    
    Key generation:
        SHA256(normalize(transcript) + normalize(context) + modelVersion)
    
    Usage:
        cache = get_extraction_cache()
        result = cache.get(transcript, context, config)
        if result is None:
            # perform extraction
            cache.set(transcript, context, config, facts, inference_ms, model_version)
    """
    _instance: "ExtractionCache | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ExtractionCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._entries: Dict[str, CacheEntry] = {}
                    instance._data_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    @staticmethod
    def _normalize_transcript(transcript: Transcript) -> str:
        """Normalize transcript to a canonical string representation."""
        segments = []
        for seg in transcript.segments:
            segments.append({
                "speaker": seg.speaker,
                "text": seg.text.strip().lower(),
                "startMs": seg.start_ms,
                "endMs": seg.end_ms,
            })
        return json.dumps({
            "segments": segments,
            "language": transcript.language,
            "durationMs": transcript.duration_ms,
        }, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _normalize_context(context: Optional[Context]) -> str:
        """Normalize context to a canonical string representation."""
        if context is None:
            return "{}"
        return json.dumps({
            "specialty": context.specialty,
            "encounterType": context.encounter_type,
            "patientAge": context.patient_age,
            "patientGender": context.patient_gender,
        }, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _get_model_version(config: Optional[ExtractConfig]) -> str:
        """Extract model version from config or return empty string."""
        if config is None or config.model_version is None:
            return ""
        return config.model_version

    def _compute_cache_key(
        self,
        transcript: Transcript,
        context: Optional[Context],
        config: Optional[ExtractConfig],
    ) -> str:
        """Compute SHA256 cache key from request components."""
        normalized = (
            self._normalize_transcript(transcript) +
            self._normalize_context(context) +
            self._get_model_version(config)
        )
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _cleanup_expired(self) -> None:
        """Remove expired entries (called within lock)."""
        expired_keys = [
            key for key, entry in self._entries.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._entries[key]

    def get(
        self,
        transcript: Transcript,
        context: Optional[Context],
        config: Optional[ExtractConfig],
    ) -> Optional[Tuple[ClinicalFacts, int, str]]:
        """
        Get cached result if available and not expired.
        
        Returns:
            Tuple of (ClinicalFacts, inference_ms, model_version) or None
        """
        cache_key = self._compute_cache_key(transcript, context, config)

        with self._data_lock:
            self._cleanup_expired()
            
            entry = self._entries.get(cache_key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._entries[cache_key]
                return None
            
            return entry.facts, entry.inference_ms, entry.model_version

    def set(
        self,
        transcript: Transcript,
        context: Optional[Context],
        config: Optional[ExtractConfig],
        facts: ClinicalFacts,
        inference_ms: int,
        model_version: str,
    ) -> None:
        """
        Cache an extraction result.
        """
        cache_key = self._compute_cache_key(transcript, context, config)

        with self._data_lock:
            self._entries[cache_key] = CacheEntry(
                facts=facts,
                inference_ms=inference_ms,
                model_version=model_version,
                created_at=time.time(),
            )

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._data_lock:
            self._cleanup_expired()
            return {
                "entries": len(self._entries),
                "ttl_seconds": CACHE_TTL_SECONDS,
            }

    def clear(self) -> None:
        """Clear all cache entries (for testing)."""
        with self._data_lock:
            self._entries.clear()


def get_extraction_cache() -> ExtractionCache:
    """Get the singleton extraction cache instance."""
    return ExtractionCache()
