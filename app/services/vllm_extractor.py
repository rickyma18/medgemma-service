"""
vLLM-based clinical facts extractor.
Uses OpenAI-compatible API from vLLM.

PHI-safe: NEVER log transcript, prompt, or model output.
"""
import json
import time
from typing import Optional

import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.logging import get_safe_logger
from app.schemas.request import Context, Transcript
from app.schemas.response import ClinicalFacts
from app.services.exceptions import (
    BackendUnavailableError,
    BackendTimeoutError,
    ModelError,
    RateLimitedError,
)

logger = get_safe_logger(__name__)

# vLLM model version - updated when model changes
VLLM_MODEL_VERSION = "vllm-medgemma-1"


def _build_system_prompt() -> str:
    """
    Build the system prompt for clinical fact extraction.
    
    IMPORTANT: This prompt enforces:
    - Output ONLY valid JSON matching ClinicalFacts schema
    - NO invented diagnoses or treatments
    - null/[] for missing data
    """
    return """You are an expert clinical documentation specialist. Extract clinical facts from a medical consultation transcript using a concise, pro-doctor style.

OUTPUT FORMAT - Return ONLY valid JSON:
{
  "chiefComplaint": {"text": string | null},
  "hpi": {"narrative": string | null},
  "ros": {"positives": string[], "negatives": string[]},
  "physicalExam": {
    "findings": string[],
    "vitals": [{"name": string, "value": string, "unit": string | null}]
  },
  "assessment": {
    "primary": {"description": string, "icd10": string | null} | null,
    "differential": [{"description": string, "icd10": string | null}]
  },
  "plan": {
    "diagnostics": [{"name": string, "reason": string | null}],
    "treatments": [{"name": string, "dosage": string | null, "instructions": string | null}],
    "followUp": string | null
  }
}

EXTRACTION RULES:

1. chiefComplaint.text: Brief (3-10 words), not a long patient quote.

2. hpi.narrative: 1-3 sentences with onset, duration, severity. Only facts present.

3. ros: ONLY symptoms explicitly confirmed/denied by patient. No allergies/meds/history.

4. physicalExam: CRITICAL - ONLY findings from DOCTOR'S physical examination.
   - Valid: "a la exploración", "a la otoscopia", "se observa", "orofaringe", "amígdalas", "adenopatías", "temperatura", "TA", "FC"
   - Patient perception ("siento", "refiere", "me duele", "oído tapado") → hpi/ros, NOT physicalExam
   - No exam evidence → findings: [], vitals: []

5. assessment: MANDATORY - ALWAYS populate assessment.primary
   A) Explicit diagnosis stated → use it, icd10 may be null
   B) NO explicit diagnosis → use SYNDROMIC LABEL below, icd10=null, differential=[]

   SYNDROMIC MAPPING (use ONE label, never concatenate with "y"):
   | Symptom | Label |
   |---------|-------|
   | dolor/ardor garganta, odinofagia | "Odinofagia a estudio" |
   | vértigo rotatorio, mareo al girar | "Vértigo posicional a estudio" |
   | congestión + rinorrea + dolor facial | "Síndrome rinosinusal a estudio" |
   | otalgia, oído tapado | "Otalgia a estudio" |
   | disfonía, ronquera | "Disfonía a estudio" |
   | tinnitus | "Tinnitus a estudio" |
   | Other → "<main symptom> a estudio" (max 6 words) |

6. plan: ONLY tests/treatments explicitly ordered.

FORBIDDEN: placeholders ("registrado", "no especificado"), concatenated symptoms in assessment, patient perceptions in physicalExam.

Respond in Spanish if transcript is Spanish. ONLY JSON, no markdown."""


def _build_user_prompt(transcript: Transcript, context: Optional[Context]) -> str:
    """
    Build the user prompt with transcript.
    PHI-safe: This function is internal; prompt is NEVER logged.
    """
    # Build transcript text from segments
    text_parts = []
    for seg in transcript.segments:
        speaker_label = {
            "doctor": "Doctor",
            "patient": "Paciente",
            "unknown": "Desconocido"
        }.get(seg.speaker, seg.speaker)
        text_parts.append(f"[{speaker_label}]: {seg.text}")
    
    transcript_text = "\n".join(text_parts)
    
    # Build context section if provided
    context_section = ""
    if context:
        context_parts = []
        if context.specialty:
            context_parts.append(f"Specialty: {context.specialty}")
        if context.encounter_type:
            context_parts.append(f"Encounter type: {context.encounter_type}")
        if context.patient_age is not None:
            context_parts.append(f"Patient age: {context.patient_age}")
        if context.patient_gender:
            context_parts.append(f"Patient gender: {context.patient_gender}")
        
        if context_parts:
            context_section = f"\n\nCONTEXT:\n" + "\n".join(context_parts)
    
    return f"""TRANSCRIPT:
{transcript_text}{context_section}

Extract clinical facts as JSON:"""


def _parse_model_output(output: str) -> ClinicalFacts:
    """
    Parse and validate model output as ClinicalFacts.
    
    Raises:
        ValueError: If output is not valid JSON or doesn't match schema
    """
    # Try to extract JSON from output (model may add extra text)
    output = output.strip()
    
    # Find JSON object boundaries
    start_idx = output.find("{")
    end_idx = output.rfind("}") + 1
    
    if start_idx == -1 or end_idx == 0:
        raise ValueError("No JSON object found in output")
    
    json_str = output[start_idx:end_idx]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ModelError("Invalid JSON in model output")
    
    try:
        # Validate against Pydantic schema
        return ClinicalFacts.model_validate(data)
    except ValidationError:
        # Don't include validation details (may contain PHI)
        raise ModelError("Output does not match ClinicalFacts schema")


async def vllm_extract(
    transcript: Transcript,
    context: Optional[Context] = None
) -> tuple[ClinicalFacts, int]:
    """
    Extract clinical facts using vLLM.
    
    PHI-safe:
    - NEVER logs transcript, prompt, or model output
    - Only logs: latency_ms, error_code, status
    
    Args:
        transcript: Clinical transcript (PHI - not logged)
        context: Optional clinical context
        
    Returns:
        Tuple of (ClinicalFacts, inference_ms)
        
    Raises:
        Exception: On network/parsing errors (caught by caller)
    """
    settings = get_settings()
    start_time = time.perf_counter()
    
    # Build prompts (PHI - not logged)
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(transcript, context)
    
    # Prepare request
    url = f"{settings.vllm_base_url.rstrip('/')}/v1/chat/completions"
    timeout_s = settings.vllm_timeout_ms / 1000.0
    
    payload = {
        "model": settings.vllm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,  # Low temperature for deterministic extraction
        "max_tokens": 2048,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Handle specific status codes
            if response.status_code == 429:
                raise RateLimitedError()
            
            if response.status_code >= 500:
                raise BackendUnavailableError("vllm")
            
            response.raise_for_status()
            
    except httpx.ConnectError:
        raise BackendUnavailableError("vllm")
    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        raise BackendTimeoutError(elapsed_ms)
    except (RateLimitedError, BackendUnavailableError, BackendTimeoutError):
        raise  # Re-raise our custom exceptions
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise RateLimitedError()
        raise BackendUnavailableError("vllm")
    
    # Parse response
    try:
        result = response.json()
        model_output = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError):
        raise ModelError("Invalid response format from backend")
    
    # Parse and validate
    facts = _parse_model_output(model_output)
    
    # Calculate inference time
    inference_ms = int((time.perf_counter() - start_time) * 1000)
    
    return facts, inference_ms


async def check_vllm_health() -> bool:
    """
    Check if vLLM is reachable.
    
    Returns:
        True if vLLM responds to health check, False otherwise.
    """
    settings = get_settings()
    
    try:
        url = f"{settings.vllm_base_url.rstrip('/')}/v1/models"
        timeout_s = 2.0  # Quick health check timeout
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
            return response.status_code == 200
    except Exception:
        # Any error means vLLM is not reachable
        return False


def get_vllm_model_version() -> str:
    """Get the vLLM model version string."""
    settings = get_settings()
    if settings.vllm_model:
        return f"vllm-{settings.vllm_model}"
    return VLLM_MODEL_VERSION
