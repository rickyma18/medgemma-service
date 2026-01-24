"""
ORL Extraction Pipeline Orchestrator.
Implements Map-Reduce pattern for clinical extraction with Concurrency Control & Fallbacks.

Stages:
1. Normalize (Medicalization)
2. Clean (PII/Format)
3. Chunk (Split long audio)
4. Map (Extract per chunk)
5. Reduce (Aggregate results)
6. Finalize (Refine/Summary)

PHI-safe: Metrics only, no content logging.
"""
import time
import asyncio
import json
import httpx
from typing import Optional, List, Any, Dict

from app.core.config import get_settings
from app.core.logging import get_safe_logger
from app.schemas.request import Transcript, Context
from app.schemas.structured_fields_v1 import StructuredFieldsV1
from app.services.structured_v1_extractor import extract_structured_v1, _parse_v1_output
from app.services.pipeline_prompts import _build_finalize_prompt

logger = get_safe_logger(__name__)

# Defaults (should come from config)
PIPELINE_MAX_CONCURRENCY = 1
PIPELINE_TIMEOUT_S = 120.0
CHUNK_TIMEOUT_S = 60.0
FINALIZE_TIMEOUT_S = 45.0

# Global semaphore (initialized on module load)
_pipeline_semaphore = asyncio.Semaphore(PIPELINE_MAX_CONCURRENCY)

async def run_orl_pipeline(
    transcript: Transcript,
    context: Optional[Context] = None
) -> tuple[StructuredFieldsV1, Dict[str, Any]]:
    """
    Executes the full ORL extraction pipeline with safeguards.
    - Max concurrency: 1
    - Global timeout
    - Fallback to baseline extraction on error/timeout
    
    Returns:
        tuple[StructuredFieldsV1, dict]: The extracted fields and PHI-safe metrics.
    """
    pipeline_start = time.perf_counter()
    metrics = {
        "pipelineUsed": "orl_pipeline_stub",
        "chunksCount": 0,
        "normalizationReplacements": 0,
        "stageMs": {},
        "fallbackReason": None
    }
    
    try:
        # Enforce concurrency limit
        async with _pipeline_semaphore:
            # Enforce global timeout
            return await asyncio.wait_for(
                _run_pipeline_logic(transcript, context, metrics, pipeline_start),
                timeout=PIPELINE_TIMEOUT_S
            )

    except asyncio.TimeoutError:
        logger.warning("Pipeline timeout exceeded, falling back to baseline", timeout=PIPELINE_TIMEOUT_S)
        return await _fallback_to_baseline(transcript, context, metrics, pipeline_start, "timeout_pipeline")
        
    except Exception as e:
        logger.error("Pipeline error, falling back to baseline", error=str(e))
        return await _fallback_to_baseline(transcript, context, metrics, pipeline_start, f"error_{type(e).__name__}")


async def _run_pipeline_logic(
    transcript: Transcript, 
    context: Optional[Context],
    metrics: Dict[str, Any],
    start_time: float
) -> tuple[StructuredFieldsV1, Dict[str, Any]]:
    
    # helper to track stage time
    def mark_stage(name: str, start_t: float):
        ms = int((time.perf_counter() - start_t) * 1000)
        metrics["stageMs"][name] = ms
        return time.perf_counter()

    current_t = start_time

    # 1. Normalize
    from app.services.text_normalizer_orl import normalize_transcript_orl
    normalized_transcript, replacements = normalize_transcript_orl(transcript)
    metrics["normalizationReplacements"] = replacements
    current_t = mark_stage("normalize", current_t)

    # 2. Clean
    from app.services.transcript_cleaner import clean_transcript
    cleaned_transcript = clean_transcript(normalized_transcript)
    current_t = mark_stage("clean", current_t)

    # 3. Chunk
    from app.services.chunking import chunk_transcript, DEFAULT_MAX_CHUNK_DURATION_MS
    
    # Use real chunking logic
    chunks = chunk_transcript(cleaned_transcript, DEFAULT_MAX_CHUNK_DURATION_MS)
    metrics["chunksCount"] = len(chunks)
    current_t = mark_stage("chunk", current_t)

    # 4. Map (Extract per chunk)
    # Using semaphore inside loop is redundant if the whole pipeline is semaphore-locked.
    # But useful if we increase PIPELINE_MAX_CONCURRENCY in future.
    extracted_results: List[StructuredFieldsV1] = []
    
    # We need to capture the model version from the first extraction at least
    last_model_version = "stub"
    
    for chunk in chunks:
        # Time-boxed extraction per chunk
        try:
            fields, infra_ms, model_ver = await asyncio.wait_for(
                extract_structured_v1(chunk, context),
                timeout=CHUNK_TIMEOUT_S
            )
            extracted_results.append(fields)
            last_model_version = model_ver
        except asyncio.TimeoutError:
            # If a single chunk times out, what do we do?
            # Option A: Fail pipeline -> Fallback (Safest for now)
            logger.warning("Chunk extraction timeout")
            raise 

    current_t = mark_stage("map", current_t)

    # 5. Reduce (Aggregate)
    from app.services.aggregator import aggregate_structured_fields_v1
    
    final_fields = aggregate_structured_fields_v1(extracted_results)
    current_t = mark_stage("reduce", current_t)

    # 6. Finalize (Refine)
    # Only run if we actually have data and it's not a trivial pass-through
    # If chunks > 1, refinement is highly recommended.
    # If chunks == 1, we might skip to save latency? User req says: "La llamada finalize NO debe correr si chunksCount==1 y pipelineUsed ya es fallback_baseline".
    # Here we are in the main pipeline logic, so pipelineUsed is "orl_pipeline_stub" (or normal).
    # We should run it to ensure high quality dedupe.
    
    try:
        final_fields = await asyncio.wait_for(
            _finalize_refine_fields(final_fields),
            timeout=FINALIZE_TIMEOUT_S
        )
    except Exception as e:
        # Fallback to aggregated
        metrics["fallbackReason"] = f"finalize_failed:{type(e).__name__}"
        logger.warning("Finalize stage failed, returning aggregated fields", error=str(e))
        # final_fields remains as aggregated
    
    current_t = mark_stage("finalize", current_t)

    # Metrics finalization
    metrics["modelVersion"] = last_model_version
    metrics["totalPipelineMs"] = int((time.perf_counter() - start_time) * 1000)
    
    return final_fields, metrics


async def _fallback_to_baseline(
    transcript: Transcript,
    context: Optional[Context],
    metrics: Dict[str, Any],
    start_time: float,
    reason: str
) -> tuple[StructuredFieldsV1, Dict[str, Any]]:
    """Fallback: Execute single-shot extraction ignoring chunks."""
    
    metrics["pipelineUsed"] = "fallback_baseline"
    metrics["fallbackReason"] = reason
    
    # Try simple extraction (may fail too, but distinct path)
    try:
        fields, _, model_ver = await extract_structured_v1(transcript, context)
        metrics["modelVersion"] = model_ver
        metrics["totalPipelineMs"] = int((time.perf_counter() - start_time) * 1000)
        return fields, metrics
    except Exception as e:
        # Only re-raise if fallback also fails
        raise e

async def _finalize_refine_fields(fields: StructuredFieldsV1) -> StructuredFieldsV1:
    """
    Calls LLM to refine and deduplicate the aggregated fields.
    """
    settings = get_settings()
    
    # 1. Helper to serialize current fields to JSON for prompt
    # exclude_none=True to reduce noise, but V1 schema has many optionals.
    input_json = fields.model_dump_json(exclude_none=True, indent=2)
    
    prompt = _build_finalize_prompt(input_json)
    
    # 2. Call LLM (similar to extractor)
    base_url = settings.openai_compat_base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    timeout_s = settings.openai_compat_timeout_ms / 1000.0
    # Allow slightly less than global if needed, but we use FINALIZE_TIMEOUT_S wrapping this call
    
    payload = {
        "model": settings.openai_compat_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
    # 3. Parse Response
    result = response.json()
    choices = result.get("choices", [])
    if not choices:
        raise ValueError("No choices in LLM response")
        
    choice = choices[0]
    content = choice.get("message", {}).get("content", "") or choice.get("text", "")
    
    # Reuse existing V1 parser/repair logic from extractor service?
    # Importing `_parse_v1_output` from `structured_v1_extractor` (we added it to imports)
    parsed_fields = _parse_v1_output(content)
    
    return parsed_fields
