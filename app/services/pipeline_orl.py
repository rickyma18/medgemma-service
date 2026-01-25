"""
ORL Extraction Pipeline Orchestrator.
Implements Map-Reduce pattern for clinical extraction with Concurrency Control & Fallbacks.

Stages:
1. Normalize (Medicalization)
2. Clean (PII/Format)
3. Chunk (Split long audio)
4. Map (Extract per chunk) - Epic 15: lite extractor by default
5. Reduce (Aggregate results) - Epic 15: uses ChunkExtractionResult
6. Finalize (Refine/Summary) - FULL MedGemma (1 call)

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
from app.schemas.chunk_extraction_result import ChunkExtractionResult, ChunkEvidenceSummary
from app.services.structured_v1_extractor import extract_structured_v1, _parse_v1_output
from app.services.extractors.lite_extractor import extract_chunk_lite
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
        "medicalizationReplacements": 0,
        "negationSpans": 0,
        "stageMs": {},
        "fallbackReason": None,
        "medicalizationVersion": None,
        "medicalizationGlossaryHash": None,
        "normalizationVersion": None,
        "normalizationRulesHash": None,
        "contractWarnings": [],
        "contractStatus": "ok",
        "contractDetails": None
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
    
    # 0. Medicalization (Step 0)
    # Apply medicalization to each segment
    try:
        from app.services.medicalization.medicalization_service import apply_medicalization
        from app.services.medicalization.medicalization_glossary import (
            get_glossary_hash,
            get_glossary_version
        )

        # We modify transcript in place? No, create new one to be safe/functional
        # Actually pipeline flows are usually sequential transformations.

        total_med_replacements = 0
        total_neg_spans = 0

        medicalized_transcript = transcript.model_copy(deep=True)

        for seg in medicalized_transcript.segments:
            med_text, res_metrics = apply_medicalization(seg.text)
            seg.text = med_text
            total_med_replacements += res_metrics.get("replacementsCount", 0)
            total_neg_spans += res_metrics.get("negationSpansCount", 0)

        metrics["medicalizationReplacements"] = total_med_replacements
        metrics["negationSpans"] = total_neg_spans

        # Inject version/hash for contract freeze (PHI-safe)
        glossary_hash = get_glossary_hash()
        if glossary_hash:
            metrics["medicalizationVersion"] = get_glossary_version()
            metrics["medicalizationGlossaryHash"] = glossary_hash

        # Pass to next stage
        # But wait, we used local variable 'medicalized_transcript'.
        # We should update 'transcript' variable for next steps?
        # Or rename flow variable. Let's use 'current_transcript'.
        current_transcript = medicalized_transcript

    except Exception as e:
        # Soft fail - leave version/hash as None
        logger.warning("Medicalization step failed", error=str(e))
        metrics["fallbackReason"] = f"medicalization_failed:{type(e).__name__}"
        current_transcript = transcript # fallback to original

    current_t = mark_stage("medicalization", current_t)

    # 1. Normalize
    try:
        from app.services.text_normalizer_orl import normalize_transcript_orl
        from app.services.normalization.normalization_contract import (
            get_normalization_hash,
            get_normalization_version
        )

        normalized_transcript, replacements = normalize_transcript_orl(current_transcript)
        metrics["normalizationReplacements"] = replacements

        # Inject version/hash for contract freeze (PHI-safe)
        norm_hash = get_normalization_hash()
        if norm_hash:
            metrics["normalizationVersion"] = get_normalization_version()
            metrics["normalizationRulesHash"] = norm_hash

    except Exception as e:
        # Soft fail - leave version/hash as None, use original transcript
        logger.warning("Normalization step failed", error=str(e))
        if not metrics.get("fallbackReason"):
            metrics["fallbackReason"] = f"normalization_failed:{type(e).__name__}"
        normalized_transcript = current_transcript

    current_t = mark_stage("normalize", current_t)

    # 1.5. Contract Guard - Drift detection (PHI-safe)
    contract_warnings: List[str] = []
    contract_details: Optional[Dict[str, Any]] = None

    try:
        from app.contracts.contract_guard import check_contracts

        contract_result = check_contracts()
        contract_warnings = contract_result.get("warnings", []) or []
        contract_details = contract_result.get("details")

    except Exception as e:
        # Soft fail - never break pipeline
        logger.warning("Contract guard check failed", error=str(e))

    # Always set list (never None)
    metrics["contractWarnings"] = contract_warnings
    metrics["contractDetails"] = contract_details
    
    # Analyze drift nature
    has_drift = any(w.startswith("DRIFT:") for w in contract_warnings)
    
    # Initial status logic (pending safe mode decision)
    if not contract_warnings:
        metrics["contractStatus"] = "ok"
    else:
        # Default to warning - only becomes drift if we force fallback below
        metrics["contractStatus"] = "warning"
        
    current_t = mark_stage("contract_guard", current_t)

    # 1.6. Drift Guard - Telemetry & Safe Mode (PHI-safe)
    settings = get_settings()
    drift_guard_mode = settings.drift_guard_mode
    drift_guard_cooldown = settings.drift_guard_cooldown_s

    if contract_warnings and drift_guard_mode in ("warn", "safe"):
        # Emit telemetry event (rate-limited)
        try:
            from app.services.telemetry import emit_event

            # Build PHI-safe payload (NO transcript/segments/text!)
            drift_payload = {
                "warnings": contract_warnings,
                "details": contract_details,
                "pipelineUsed": metrics.get("pipelineUsed"),
                "medicalizationVersion": metrics.get("medicalizationVersion"),
                "medicalizationGlossaryHash": metrics.get("medicalizationGlossaryHash"),
                "normalizationVersion": metrics.get("normalizationVersion"),
                "normalizationRulesHash": metrics.get("normalizationRulesHash"),
                "driftGuardMode": drift_guard_mode,
            }

            emit_event(
                name="contract_drift_detected",
                payload=drift_payload,
                cooldown_s=drift_guard_cooldown
            )

        except Exception as e:
            # Telemetry failure should never break pipeline
            logger.warning("Drift telemetry emit failed", error=str(e))

        # Safe mode: force fallback to prevent serving with drifted contracts
        if drift_guard_mode == "safe" and has_drift:
            logger.warning(
                "Drift guard safe mode triggered - forcing fallback",
                warnings=contract_warnings
            )
            # Logic: If safe mode forces fallback -> drift
            metrics["contractStatus"] = "drift"
            
            return await _fallback_to_baseline(
                transcript, context, metrics, start_time, "contract_drift"
            )

    current_t = mark_stage("drift_guard", current_t)

    # 2. Clean
    from app.services.transcript_cleaner import clean_transcript
    cleaned_transcript = clean_transcript(normalized_transcript)
    current_t = mark_stage("clean", current_t)

    # 3. Chunk (Intelligent or Legacy based on config)
    from app.services.chunking import (
        chunk_transcript,
        estimate_segment_tokens,
        DEFAULT_MAX_CHUNK_DURATION_MS
    )

    # Determine chunking mode from config
    chunking_enabled = settings.chunking_enabled

    if chunking_enabled:
        # Intelligent chunking with token + duration limits
        chunks = chunk_transcript(
            cleaned_transcript,
            max_duration_ms=settings.chunking_max_duration_ms,
            hard_token_limit=settings.chunking_hard_token_limit,
            soft_duration_limit_ms=settings.chunking_soft_duration_limit_ms,
            min_segments_per_chunk=settings.chunking_min_segments_per_chunk,
        )

        # Calculate PHI-safe metrics for telemetry
        total_tokens_est = sum(
            estimate_segment_tokens(seg)
            for seg in cleaned_transcript.segments
        )
        total_duration_ms = (
            cleaned_transcript.segments[-1].end_ms - cleaned_transcript.segments[0].start_ms
            if cleaned_transcript.segments else 0
        )

        # Emit chunking telemetry (PHI-safe: no text/segments)
        try:
            from app.services.telemetry import emit_event

            chunking_payload = {
                "numChunks": len(chunks),
                "totalTokensEst": total_tokens_est,
                "hardTokenLimit": settings.chunking_hard_token_limit,
                "softDurationLimitMs": settings.chunking_soft_duration_limit_ms,
                "maxDurationMs": settings.chunking_max_duration_ms,
                "totalDurationMs": total_duration_ms,
                "minSegmentsPerChunk": settings.chunking_min_segments_per_chunk,
                "totalSegments": len(cleaned_transcript.segments),
            }

            # Only emit if chunking actually occurred (more than 1 chunk)
            if len(chunks) > 1:
                emit_event(
                    name="chunking_applied",
                    payload=chunking_payload,
                    cooldown_s=0  # No cooldown for chunking events
                )
        except Exception as e:
            # Telemetry failure should never break pipeline
            logger.warning("Chunking telemetry emit failed", error=str(e))
    else:
        # Legacy chunking: duration-only (backward compatible)
        chunks = chunk_transcript(
            cleaned_transcript,
            max_duration_ms=DEFAULT_MAX_CHUNK_DURATION_MS,
            soft_duration_limit_ms=None  # Disable soft limit for legacy
        )

    metrics["chunksCount"] = len(chunks)
    metrics["chunkingEnabled"] = chunking_enabled
    current_t = mark_stage("chunk", current_t)

    # 4. Map (Extract per chunk) - Epic 15: lite extractor by default
    # Using semaphore inside loop is redundant if the whole pipeline is semaphore-locked.
    # But useful if we increase PIPELINE_MAX_CONCURRENCY in future.
    chunk_results: List[ChunkExtractionResult] = []

    # Determine extractor mode from config
    map_extractor_mode = settings.map_extractor_mode

    # We need to capture the model version from the first extraction at least
    last_model_version = "lite-v1" if map_extractor_mode == "lite" else "stub"

    for chunk_idx, chunk in enumerate(chunks):
        # Time-boxed extraction per chunk
        try:
            if map_extractor_mode == "lite":
                # Epic 15: Use lite extractor (cheap/fast)
                chunk_result, infra_ms = await asyncio.wait_for(
                    extract_chunk_lite(chunk, chunk_idx, context),
                    timeout=CHUNK_TIMEOUT_S
                )
                chunk_results.append(chunk_result)
                last_model_version = "lite-v1"
            else:
                # Full extractor (expensive, legacy behavior)
                fields, infra_ms, model_ver = await asyncio.wait_for(
                    extract_structured_v1(chunk, context),
                    timeout=CHUNK_TIMEOUT_S
                )
                # Wrap in ChunkExtractionResult for uniform handling
                chunk_result = ChunkExtractionResult(
                    chunkIndex=chunk_idx,
                    fields=fields,
                    evidence=[],  # Full extractor doesn't produce evidence
                    extractorUsed="full"
                )
                chunk_results.append(chunk_result)
                last_model_version = model_ver

        except asyncio.TimeoutError:
            # If a single chunk times out, fail pipeline -> Fallback (Safest for now)
            logger.warning("Chunk extraction timeout", chunk_index=chunk_idx)
            raise

    metrics["mapExtractorMode"] = map_extractor_mode
    current_t = mark_stage("map", current_t)

    # 5. Reduce (Aggregate) - Epic 15: uses ChunkExtractionResult wrapper
    from app.services.aggregator import aggregate_chunk_results

    final_fields, evidence_summaries = aggregate_chunk_results(chunk_results)

    # Store evidence summaries for optional response inclusion
    metrics["_evidence_summaries"] = evidence_summaries  # Internal, stripped before response

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
