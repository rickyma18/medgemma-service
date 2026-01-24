"""
Extractor service with backend selection.
Supports mock (testing), vLLM, and OpenAI-compatible backends.

PHI-safe: NEVER log transcript or extracted facts.
"""
import time
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_safe_logger
from app.schemas.request import Context, Transcript, ExtractConfig
from app.schemas.response import (
    Assessment,
    ChiefComplaint,
    ClinicalFacts,
    HPI,
    PhysicalExam,
    Plan,
    ROS,
)
from app.services.exceptions import ExtractorError

logger = get_safe_logger(__name__)

# Mock model version identifier
MOCK_MODEL_VERSION = "mock-0"

# Placeholder values to filter from vitals
_PLACEHOLDER_VALUES = frozenset({
    "no especificado",
    "sin información",
    "sin informacion",
    "pendiente",
    "n/a",
    "na",
    "none",
    "null",
})


# Non-findings to remove from physicalExam.findings
_NON_FINDINGS = frozenset({
    "a la exploración",
    "a la exploracion",
})

# ROS term normalization (english -> spanish)
_ROS_NORMALIZE = {
    "fever": "fiebre",
}

def _sanitize_facts(facts: ClinicalFacts, transcript: Transcript) -> ClinicalFacts:
    sanitized_count = 0

    # 1) Sanitize vitals
    if facts.physical_exam and facts.physical_exam.vitals:
        original = len(facts.physical_exam.vitals)
        cleaned = []
        for v in facts.physical_exam.vitals:
            val = (getattr(v, "value", None) or "").strip().lower()
            if not val:
                continue
            if val in _PLACEHOLDER_VALUES:
                continue
            cleaned.append(v)
        facts.physical_exam.vitals = cleaned
        sanitized_count += original - len(cleaned)

    # 2) Sanitize findings
    if facts.physical_exam and facts.physical_exam.findings:
        original = len(facts.physical_exam.findings)
        cleaned = []
        for f in facts.physical_exam.findings:
            txt = (f or "").strip().lower()
            if not txt:
                continue
            if txt in _NON_FINDINGS:
                continue
            cleaned.append(f)
        facts.physical_exam.findings = cleaned
        sanitized_count += original - len(cleaned)

    # 3) Normalize ROS
    if facts.ros:
        if facts.ros.positives:
            facts.ros.positives = [
                _ROS_NORMALIZE.get((p or "").strip().lower(), p)
                for p in facts.ros.positives
                if (p or "").strip()
            ]
        if facts.ros.negatives:
            facts.ros.negatives = [
                _ROS_NORMALIZE.get((n or "").strip().lower(), n)
                for n in facts.ros.negatives
                if (n or "").strip()
            ]

    if sanitized_count > 0:
        logger.debug("sanitize_facts applied", items_removed=sanitized_count)
    # 4) If no DOCTOR exam cues in transcript -> clear physicalExam
    has_exam_cues = any(
        cue in (seg.text or "").lower()
        for seg in transcript.segments
        if seg.speaker == "doctor"
        for cue in [
            "a la exploración",
            "a la exploracion",
            "a la otoscopia",
            "otoscopia",
            "se observa",
            "orofaringe",
            "amígd",
            "amigd",
            "adenopat",
            "auscult",
            "ta ",
            "fc ",
            "temperatura",
        ]
    )

    if not has_exam_cues:
        # Patient-reported stuff must not be in physicalExam if doctor didn't examine in transcript
        if facts.physical_exam:
            if facts.physical_exam.findings:
                sanitized_count += len(facts.physical_exam.findings)
            if facts.physical_exam.vitals:
                sanitized_count += len(facts.physical_exam.vitals)
            facts.physical_exam.findings = []
            facts.physical_exam.vitals = []

    return facts

def mock_extract(
    transcript: Transcript,
    context: Optional[Context] = None
) -> tuple[ClinicalFacts, int]:
    """
    Mock extraction function that returns deterministic, neutral clinical facts.

    IMPORTANT:
    - Does NOT invent diagnoses or treatments
    - Returns neutral placeholder text
    - Deterministic: same input structure -> same output
    - Does NOT use actual transcript content in any meaningful way

    Args:
        transcript: The clinical transcript (NOT logged, NOT used for content)
        context: Optional clinical context

    Returns:
        Tuple of (ClinicalFacts, inference_ms)
    """
    start_time = time.perf_counter()

    # Generate deterministic "inference" based on structural properties only
    segment_count = len(transcript.segments)

    # Simulate inference time: base + small amount per segment
    simulated_delay_ms = 50 + (segment_count * 10)
    simulated_delay_ms = min(simulated_delay_ms, 500)

    # Build neutral clinical facts - NO invented medical data
    facts = ClinicalFacts(
        chief_complaint=ChiefComplaint(
            text="Motivo de consulta registrado" if segment_count > 0 else None
        ),
        hpi=HPI(
            narrative="Historia clínica registrada" if segment_count > 0 else None
        ),
        ros=ROS(
            positives=[],
            negatives=[]
        ),
        physical_exam=PhysicalExam(
            findings=[],
            vitals=[]
        ),
        assessment=Assessment(
            primary=None,
            differential=[]
        ),
        plan=Plan(
            diagnostics=[],
            treatments=[],
            follow_up=None
        )
    )

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    inference_ms = max(elapsed_ms, simulated_delay_ms)

    return facts, inference_ms


async def extract(
    transcript: Transcript,
    context: Optional[Context] = None,
    config: Optional[ExtractConfig] = None,
) -> tuple[ClinicalFacts, int, str]:
    """
    Extract clinical facts using configured backend.
    
    PHI-safe: NEVER logs transcript or output.
    
    Args:
        transcript: Clinical transcript (PHI)
        context: Optional clinical context
        config: Optional extraction config (includes modelVersion)
        
    Returns:
        Tuple of (ClinicalFacts, inference_ms, model_version)
        
    Raises:
        ExtractorError: On backend errors (with proper error_code and status_code)
    """
    settings = get_settings()
    
    if settings.extractor_backend == "vllm":
        from app.services.vllm_extractor import vllm_extract, get_vllm_model_version
        facts, ms = await vllm_extract(transcript, context)
        facts = _sanitize_facts(facts, transcript)
        return facts, ms, get_vllm_model_version()

    elif settings.extractor_backend == "openai_compat":
        from app.services.openai_compat_extractor import openai_compat_extract
        facts, ms, model_version = await openai_compat_extract(transcript, context, config)
        facts = _sanitize_facts(facts, transcript)
        return facts, ms, model_version
    else:
        # Default to mock
        facts, ms = mock_extract(transcript, context)
        return facts, ms, MOCK_MODEL_VERSION


def get_model_version() -> str:
    """Get the current model version string based on backend."""
    settings = get_settings()
    
    if settings.extractor_backend == "vllm":
        from app.services.vllm_extractor import get_vllm_model_version
        return get_vllm_model_version()
    
    elif settings.extractor_backend == "openai_compat":
        from app.services.openai_compat_extractor import get_openai_compat_model_version
        return get_openai_compat_model_version()
    
    else:
        return MOCK_MODEL_VERSION


async def check_backend_health() -> dict[str, bool]:
    """
    Check health of the configured backend.
    
    Returns:
        Dict with backend-specific health checks
    """
    settings = get_settings()
    
    if settings.extractor_backend == "vllm":
        from app.services.vllm_extractor import check_vllm_health
        vllm_ok = await check_vllm_health()
        return {
            "extractor_backend": True,
            "vllm_reachable": vllm_ok
        }
    
    elif settings.extractor_backend == "openai_compat":
        from app.services.openai_compat_extractor import check_openai_compat_health
        compat_ok = await check_openai_compat_health()
        return {
            "extractor_backend": True,
            "openai_compat_reachable": compat_ok
        }
    
    else:
        return {
            "mock_extractor": True
        }
