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
        return facts, ms, get_vllm_model_version()
    
    elif settings.extractor_backend == "openai_compat":
        from app.services.openai_compat_extractor import openai_compat_extract
        return await openai_compat_extract(transcript, context, config)
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
