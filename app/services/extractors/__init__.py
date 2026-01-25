"""
Extractors module for Epic 15.
Contains lite and full extractors for chunked pipeline.
"""
from app.services.extractors.lite_extractor import (
    extract_chunk_lite,
    LITE_EXTRACTOR_VERSION,
)

__all__ = ["extract_chunk_lite", "LITE_EXTRACTOR_VERSION"]
