"""
OpenAI-compatible backend for clinical facts extraction.
Works with LM Studio, Ollama, and other OpenAI-compatible servers.

PHI-safe: NEVER log transcript, prompt, or model output.
"""
import json
import time
from typing import Optional

import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.logging import get_safe_logger
from app.schemas.request import Context, Transcript, ExtractConfig
from app.schemas.response import ClinicalFacts
from app.services.exceptions import (
    BackendUnavailableError,
    BackendTimeoutError,
    ModelError,
    RateLimitedError,
)

logger = get_safe_logger(__name__)


def _build_system_prompt() -> str:
    """
    Build the system prompt for clinical fact extraction.
    
    IMPORTANT: This prompt enforces:
    - Output ONLY valid JSON matching ClinicalFacts schema
    - NO invented diagnoses or treatments
    - NO placeholder text (e.g., "registrado", "no especificado")
    - null/[] for truly missing data
    """
    return """You are an expert clinical documentation specialist. Your task is to extract clinical facts from a medical consultation transcript.

## CRITICAL ANTI-PLACEHOLDER RULE
NEVER use generic placeholder phrases like:
- "Motivo de consulta registrado"
- "Historia clínica registrada"
- "No especificado"
- "Sin información"
- "Pendiente"
- "Registrado"

If the transcript contains clinical information, you MUST extract the ACTUAL content.
If a field is truly NOT mentioned in the transcript, use null (for optional fields) or [] (for lists).

## EXTRACTION EXAMPLES

Example 1 - Patient statement:
Transcript: "[Paciente]: Me arde la garganta desde hace 4 días"
Extract:
- chiefComplaint.text: "Ardor de garganta desde hace 4 días"
- hpi.narrative: "Paciente refiere ardor de garganta con 4 días de evolución."

Example 2 - Negative symptom:
Transcript: "[Paciente]: No me falta el aire"
Extract:
- ros.negatives: ["disnea"]

Example 3 - Missing data:
If blood pressure is NOT mentioned anywhere → vitals: [] (empty array, NOT a placeholder)

## OUTPUT FORMAT
Return ONLY one JSON object conforming EXACTLY to this schema:
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

## EXTRACTION RULES
1. chiefComplaint: Extract the patient's main reason for visit in their own words or paraphrased
2. hpi: Build a narrative with onset, duration, severity, and relevant details mentioned
3. ros: Include ONLY symptoms explicitly confirmed (positives) or denied (negatives) by the patient
4. physicalExam: Include ONLY findings explicitly stated by the doctor during examination
5. assessment: Include ONLY diagnoses explicitly stated by the doctor; do NOT infer diagnoses
6. plan: Include ONLY diagnostic tests, treatments, and follow-up explicitly ordered

## LANGUAGE
- If the transcript is in Spanish, respond in Spanish
- Use medical terminology appropriately

## FINAL INSTRUCTION
Respond with ONLY the JSON object. No explanations, no markdown, no additional text."""


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

def _ensure_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    # a veces el modelo regresa string único
    return [str(v)]


def _ensure_obj(v):
    return v if isinstance(v, dict) else {}


def _coerce_str_or_none(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    # si viene lista/obj raro, lo ignoramos
    return None


def _repair_clinical_facts_dict(data: dict) -> dict:
    """
    Attempt to coerce model output into the ClinicalFacts schema shape.
    PHI-safe: do not log data content.
    """
    if not isinstance(data, dict):
        return {}

    # Allow some common key variants
    key_map = {
        "chief_complaint": "chiefComplaint",
        "cc": "chiefComplaint",
        "hpi_text": "hpi",
        "physical_exam": "physicalExam",
        "physicalexam": "physicalExam",
        "assessment_plan": "plan",
        "follow_up": "followUp",
    }
    for k, v in list(data.items()):
        if k in key_map and key_map[k] not in data:
            data[key_map[k]] = v

    # Top-level required objects
    data.setdefault("chiefComplaint", {})
    data.setdefault("hpi", {})
    data.setdefault("ros", {})
    data.setdefault("physicalExam", {})
    data.setdefault("assessment", {})
    data.setdefault("plan", {})

    # chiefComplaint
    cc = data["chiefComplaint"]
    if isinstance(cc, str):
        data["chiefComplaint"] = {"text": _coerce_str_or_none(cc)}
    else:
        cc = _ensure_obj(cc)
        cc["text"] = _coerce_str_or_none(cc.get("text"))
        data["chiefComplaint"] = cc

    # hpi
    hpi = data["hpi"]
    if isinstance(hpi, str):
        data["hpi"] = {"narrative": _coerce_str_or_none(hpi)}
    else:
        hpi = _ensure_obj(hpi)
        hpi["narrative"] = _coerce_str_or_none(hpi.get("narrative"))
        data["hpi"] = hpi

    # ros
    ros = _ensure_obj(data["ros"])
    ros["positives"] = _ensure_list(ros.get("positives"))
    ros["negatives"] = _ensure_list(ros.get("negatives"))
    # si vino "ros": "niega fiebre", lo metemos a negatives como string
    if not ros["positives"] and not ros["negatives"] and isinstance(data.get("ros"), str):
        ros["negatives"] = [data["ros"]]
    data["ros"] = ros

    # physicalExam
    pe = _ensure_obj(data["physicalExam"])
    pe["findings"] = _ensure_list(pe.get("findings"))
    vitals = pe.get("vitals")
    vitals = _ensure_list(vitals)
    # asegurar objetos con name/value/unit
    fixed_vitals = []
    for item in vitals:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            value = str(item.get("value", "")).strip()
            unit = item.get("unit")
            unit = _coerce_str_or_none(unit)
            if name and value:
                fixed_vitals.append({"name": name, "value": value, "unit": unit})
        # si viene string, lo ignoramos (no sabemos parsearlo sin inventar)
    pe["vitals"] = fixed_vitals
    data["physicalExam"] = pe

    # assessment
    a = _ensure_obj(data["assessment"])
    primary = a.get("primary")
    if isinstance(primary, str):
        a["primary"] = {"description": primary.strip(), "icd10": None} if primary.strip() else None
    elif primary is None:
        a["primary"] = None
    else:
        primary = _ensure_obj(primary)
        desc = str(primary.get("description", "")).strip()
        icd10 = _coerce_str_or_none(primary.get("icd10"))
        a["primary"] = {"description": desc, "icd10": icd10} if desc else None

    diff = _ensure_list(a.get("differential"))
    fixed_diff = []
    for item in diff:
        if isinstance(item, str):
            s = item.strip()
            if s:
                fixed_diff.append({"description": s, "icd10": None})
        elif isinstance(item, dict):
            desc = str(item.get("description", "")).strip()
            icd10 = _coerce_str_or_none(item.get("icd10"))
            if desc:
                fixed_diff.append({"description": desc, "icd10": icd10})
    a["differential"] = fixed_diff
    data["assessment"] = a

    # plan
    p = _ensure_obj(data["plan"])

    diagnostics = _ensure_list(p.get("diagnostics"))
    fixed_diag = []
    for item in diagnostics:
        if isinstance(item, str):
            s = item.strip()
            if s:
                fixed_diag.append({"name": s, "reason": None})
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            reason = _coerce_str_or_none(item.get("reason"))
            if name:
                fixed_diag.append({"name": name, "reason": reason})
    p["diagnostics"] = fixed_diag

    treatments = _ensure_list(p.get("treatments"))
    fixed_tx = []
    for item in treatments:
        if isinstance(item, str):
            s = item.strip()
            if s:
                fixed_tx.append({"name": s, "dosage": None, "instructions": None})
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            dosage = _coerce_str_or_none(item.get("dosage"))
            instr = _coerce_str_or_none(item.get("instructions"))
            if name:
                fixed_tx.append({"name": name, "dosage": dosage, "instructions": instr})
    p["treatments"] = fixed_tx

    p["followUp"] = _coerce_str_or_none(p.get("followUp"))
    data["plan"] = p

    return data


def _parse_model_output(output: str) -> ClinicalFacts:
    """
    Parse and validate model output as ClinicalFacts.
    
    Raises:
        ModelError: If output is not valid JSON or doesn't match schema
    """
    output = output.strip()
    
    # Remove markdown code blocks if present
    if output.startswith("```"):
        lines = output.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        output = "\n".join(lines)
    
    # Find JSON object boundaries
    start_idx = output.find("{")
    end_idx = output.rfind("}") + 1
    
    if start_idx == -1 or end_idx == 0:
        raise ModelError("No JSON object found in output")
    
    json_str = output[start_idx:end_idx]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ModelError("Invalid JSON in model output")
    
    try:
        # Repair + validate against Pydantic schema
        repaired = _repair_clinical_facts_dict(data)
        return ClinicalFacts.model_validate(repaired)
    except ValidationError:
        raise ModelError("Output does not match ClinicalFacts schema")


def _looks_placeholder(text: str) -> bool:
    """
    PHI-safe helper: detect if text looks like a generic placeholder.
    
    Returns True if the text contains common placeholder patterns in Spanish/English.
    Does NOT log or expose the actual text content.
    
    Common placeholders detected:
    - "registrado", "no especificado", "sin información"
    - "recorded", "not specified", "no information"
    - Generic phrases like "motivo de consulta registrado"
    """
    if not text:
        return False
    
    lower = text.lower().strip()
    
    # Common Spanish placeholders
    placeholder_keywords = [
        "registrado",
        "no especificado",
        "sin información",
        "sin datos",
        "no aplica",
        "pendiente",
        "no disponible",
        # Common English placeholders
        "recorded",
        "not specified",
        "no information",
        "not available",
        "n/a",
        "pending",
        # Generic short phrases that are likely placeholders
        "historia clínica registrada",
        "motivo de consulta registrado",
        "examen físico registrado",
        "clinical history recorded",
        "chief complaint recorded",
    ]
    
    for keyword in placeholder_keywords:
        if keyword in lower:
            return True
    
    # Very short text (< 10 chars) that doesn't look like real content
    if len(lower) < 10 and not any(c.isdigit() for c in lower):
        # Short text without numbers is suspicious
        return True
    
    return False


async def openai_compat_extract(
    transcript: Transcript,
    context: Optional[Context] = None,
    config: Optional[ExtractConfig] = None,
) -> tuple[ClinicalFacts, int, str]:
    """
    Extract clinical facts using OpenAI-compatible API.
    
    PHI-safe:
    - NEVER logs transcript, prompt, or model output
    - Only logs: latency_ms, error_code, status
    
    Args:
        transcript: Clinical transcript (PHI - not logged)
        context: Optional clinical context
        
    Returns:
        Tuple of (ClinicalFacts, inference_ms, model_version)
        
    Raises:
        BackendUnavailableError: If backend is not reachable
        BackendTimeoutError: If request times out
        RateLimitedError: If backend returns 429
        ModelError: If model output is invalid
    """
    settings = get_settings()
    start_time = time.perf_counter()
    
    # Build prompts (PHI - not logged)
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(transcript, context)
    
    # Prepare request
    base_url = settings.openai_compat_base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    timeout_s = settings.openai_compat_timeout_ms / 1000.0
    
    # Determine model name from config or settings
    model_name = config.model_version if config and config.model_version else settings.openai_compat_model
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,  # Deterministic for clinical extraction
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
                raise BackendUnavailableError("openai_compat")
            
            response.raise_for_status()
            
    except httpx.ConnectError:
        raise BackendUnavailableError("openai_compat")
    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        raise BackendTimeoutError(elapsed_ms)
    except (RateLimitedError, BackendUnavailableError):
        raise  # Re-raise our custom exceptions
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise RateLimitedError()
        raise BackendUnavailableError("openai_compat")
    
    # Parse response (PHI-safe)
    try:
        result = response.json()
    except json.JSONDecodeError:
        raise ModelError("Invalid JSON response from backend")

    # OpenAI-style error object
    if isinstance(result, dict) and "error" in result:
        err = result.get("error") or {}
        # PHI-safe: NO message, NO transcript, NO output
        logger.error(
            "openai_compat backend returned error object",
            error_code="MODEL_ERROR",
            http_status=response.status_code,
            backend_error_type=err.get("type"),
            backend_error_code=err.get("code"),
        )
        raise ModelError("Backend returned an error response")

    # Normal OpenAI chat.completions
    try:
        choices = result.get("choices", []) if isinstance(result, dict) else []
        if not choices:
            logger.error(
                "openai_compat missing choices",
                error_code="MODEL_ERROR",
                http_status=response.status_code,
                result_keys=list(result.keys()) if isinstance(result, dict) else None,
            )
            raise KeyError("choices")

        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message")

        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            model_output = msg["content"]
        elif isinstance(first.get("text"), str):
            # fallback for some servers
            model_output = first["text"]
        else:
            logger.error(
                "openai_compat unexpected choice shape",
                error_code="MODEL_ERROR",
                http_status=response.status_code,
                choice_keys=list(first.keys()) if isinstance(first, dict) else None,
            )
            raise KeyError("content")

    except (KeyError, IndexError, TypeError):
        raise ModelError("Invalid response format from backend")

    # Parse and validate output
    facts = _parse_model_output(model_output)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DEBUG-FACTS: PHI-safe instrumentation for diagnosing placeholder/empty outputs
    # Logs ONLY booleans and counts, NEVER actual text content
    # REMOVE after debugging is complete
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        # Extract counts and booleans (PHI-safe)
        cc_text = facts.chiefComplaint.text if facts.chiefComplaint else None
        hpi_text = facts.hpi.narrative if facts.hpi else None
        
        debug_metrics = {
            # Presence booleans
            "cc_present": cc_text is not None and len(cc_text.strip()) > 0,
            "hpi_present": hpi_text is not None and len(hpi_text.strip()) > 0,
            "assessment_primary_present": facts.assessment.primary is not None if facts.assessment else False,
            "followup_present": facts.plan.followUp is not None if facts.plan else False,
            
            # Counts
            "ros_pos_count": len(facts.ros.positives) if facts.ros else 0,
            "ros_neg_count": len(facts.ros.negatives) if facts.ros else 0,
            "pe_findings_count": len(facts.physicalExam.findings) if facts.physicalExam else 0,
            "vitals_count": len(facts.physicalExam.vitals) if facts.physicalExam else 0,
            "differential_count": len(facts.assessment.differential) if facts.assessment else 0,
            "plan_dx_count": len(facts.plan.diagnostics) if facts.plan else 0,
            "plan_tx_count": len(facts.plan.treatments) if facts.plan else 0,
            
            # Placeholder detection (PHI-safe: only reports boolean, not the text)
            "cc_placeholder": _looks_placeholder(cc_text) if cc_text else False,
            "hpi_placeholder": _looks_placeholder(hpi_text) if hpi_text else False,
        }
        
        logger.warning(
            "DEBUG-FACTS: Extraction output metrics (PHI-safe)",
            **debug_metrics
        )
    except Exception:
        # Never break the extraction flow due to debug logging
        pass
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Calculate inference time
    inference_ms = int((time.perf_counter() - start_time) * 1000)
    
    # Return model version used
    model_version = f"openai-compat-{model_name}"
    
    return facts, inference_ms, model_version


async def check_openai_compat_health() -> bool:
    """
    Check if OpenAI-compatible backend is reachable.
    
    Tries /models endpoint first, then falls back to base URL.
    Returns True if we get any successful response (200) or 404 (server is up).
    
    Returns:
        True if backend is reachable, False otherwise.
    """
    settings = get_settings()
    base_url = settings.openai_compat_base_url.rstrip("/")
    
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Try /models first (standard OpenAI endpoint)
            try:
                response = await client.get(f"{base_url}/models")
                if response.status_code in (200, 404):
                    return True
            except httpx.HTTPError:
                pass
            
            # Fallback: try base URL
            try:
                # Remove /v1 suffix if present for base health check
                health_url = base_url.replace("/v1", "")
                response = await client.get(health_url)
                return response.status_code in (200, 404)
            except httpx.HTTPError:
                pass
            
            return False
            
    except Exception:
        return False


def get_openai_compat_model_version() -> str:
    """Get the OpenAI-compatible model version string."""
    settings = get_settings()
    if settings.openai_compat_model:
        return f"openai-compat-{settings.openai_compat_model}"
    return "openai-compat-unknown"
