"""
Lite Extractor for Epic 15 - Cheap baseline extraction per chunk.

Uses a shorter prompt optimized for speed/cost while still producing
valid StructuredFieldsV1 output with evidence snippets.

PHI-safe: NEVER log transcript, prompt, or model output.
"""
import json
import time
from typing import List, Optional, Tuple

import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.logging import get_safe_logger
from app.schemas.request import Context, Transcript
from app.schemas.structured_fields_v1 import StructuredFieldsV1, Diagnostico
from app.schemas.chunk_extraction_result import (
    ChunkExtractionResult,
    EvidenceSnippet,
)
from app.services.evidence_sanitizer import (
    sanitize_evidence,
    extract_evidence_from_text,
)
from app.services.exceptions import (
    BackendUnavailableError,
    BackendTimeoutError,
    ModelError,
    RateLimitedError,
)

logger = get_safe_logger(__name__)

# Version identifier
LITE_EXTRACTOR_VERSION = "lite-v1"


def _build_lite_system_prompt() -> str:
    """
    Minimal system prompt for lite extraction.
    ~40% shorter than full prompt to reduce latency/cost.
    Focuses on essential fields only.
    """
    return '''Extrae informacion clinica ORL a JSON. Reglas:
1. Si NO se menciona -> null (nunca inventar)
2. diagnostico.texto OBLIGATORIO (si no hay -> "Consulta ORL en estudio")
3. exploracionFisica = SOLO hallazgos del medico

JSON Schema:
{
  "motivoConsulta": "string|null",
  "padecimientoActual": "string|null",
  "antecedentes": {"heredofamiliares":null,"personalesNoPatologicos":null,"personalesPatologicos":null},
  "exploracionFisica": {"signosVitales":null,"rinoscopia":null,"orofaringe":null,"cuello":null,"otoscopia":null},
  "diagnostico": {"texto":"OBLIGATORIO","tipo":"definitivo|presuntivo|sindromico","cie10":null},
  "planTratamiento": "string|null"
}

Responde SOLO JSON valido.'''


def _build_lite_user_prompt(transcript: Transcript, context: Optional[Context]) -> str:
    """
    Construye prompt de usuario simplificado.
    PHI-safe: nunca se loguea.
    """
    # Concatenar texto sin etiquetas detalladas
    text_parts = [seg.text for seg in transcript.segments]
    transcript_text = " ".join(text_parts)

    # Truncar si es muy largo (lite = rapido)
    max_chars = 4000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "..."

    return f"TRANSCRIPCION:\n{transcript_text}\n\nJSON:"


def _repair_lite_output(data: dict) -> dict:
    """
    Minimal repair for lite extractor output.
    Less comprehensive than full repair - prioritizes speed.
    """
    if not isinstance(data, dict):
        return {}

    # Essential key normalization only
    key_map = {
        "motivo_consulta": "motivoConsulta",
        "padecimiento_actual": "padecimientoActual",
        "plan_tratamiento": "planTratamiento",
        "exploracion_fisica": "exploracionFisica",
        "signos_vitales": "signosVitales",
    }

    def normalize_keys(obj):
        if isinstance(obj, dict):
            return {key_map.get(k, k): normalize_keys(v) for k, v in obj.items()}
        return obj

    data = normalize_keys(data)

    # Ensure required structures
    data.setdefault("antecedentes", {})
    data.setdefault("exploracionFisica", {})

    # Ensure diagnostico
    dx = data.get("diagnostico")
    if not dx or (isinstance(dx, dict) and not dx.get("texto")):
        motivo = data.get("motivoConsulta") or ""
        data["diagnostico"] = {
            "texto": f"{motivo} en estudio" if motivo else "Consulta ORL en estudio",
            "tipo": "sindromico",
            "cie10": None
        }
    elif isinstance(dx, str):
        data["diagnostico"] = {"texto": dx, "tipo": "presuntivo", "cie10": None}

    return data


def _parse_lite_output(output: str) -> StructuredFieldsV1:
    """
    Parse lite extractor output to StructuredFieldsV1.
    """
    output = output.strip()

    # Remove markdown if present
    if output.startswith("```"):
        lines = output.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        output = "\n".join(lines)

    # Find JSON boundaries
    start_idx = output.find("{")
    end_idx = output.rfind("}") + 1

    if start_idx == -1 or end_idx == 0:
        raise ModelError("No JSON object found in lite extractor output")

    json_str = output[start_idx:end_idx]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ModelError("Invalid JSON from lite extractor")

    try:
        repaired = _repair_lite_output(data)
        return StructuredFieldsV1.model_validate(repaired)
    except ValidationError:
        raise ModelError("Lite output does not match StructuredFieldsV1 schema")


def _extract_evidence_from_fields(
    fields: StructuredFieldsV1,
    transcript: Transcript,
    max_snippets: int
) -> List[EvidenceSnippet]:
    """
    Extract evidence snippets from the transcript based on extracted fields.
    Maps field values back to source text segments.

    PHI-safe: evidence is sanitized before storage.
    """
    evidence: List[EvidenceSnippet] = []

    # Build full transcript text for matching
    full_text = " ".join(seg.text for seg in transcript.segments)

    # Extract evidence for non-null fields
    field_checks = [
        (fields.motivo_consulta, "motivoConsulta"),
        (fields.padecimiento_actual, "padecimientoActual"),
        (fields.diagnostico.texto if fields.diagnostico else None, "diagnostico.texto"),
        (fields.plan_tratamiento, "planTratamiento"),
        (fields.exploracion_fisica.orofaringe, "exploracionFisica.orofaringe"),
        (fields.exploracion_fisica.rinoscopia, "exploracionFisica.rinoscopia"),
        (fields.exploracion_fisica.otoscopia, "exploracionFisica.otoscopia"),
        (fields.exploracion_fisica.cuello, "exploracionFisica.cuello"),
    ]

    snippets_per_field = max(1, max_snippets // len([f for f, _ in field_checks if f]))

    for field_value, field_path in field_checks:
        if not field_value:
            continue

        # Simple heuristic: find segments containing keywords from the field value
        keywords = field_value.lower().split()[:3]  # First 3 words

        for seg in transcript.segments:
            seg_lower = seg.text.lower()
            if any(kw in seg_lower for kw in keywords if len(kw) > 3):
                sanitized = sanitize_evidence(seg.text)
                if sanitized and len(sanitized) >= 10:
                    evidence.append(EvidenceSnippet(
                        text=sanitized,
                        fieldPath=field_path
                    ))
                    break  # One snippet per field for lite

        if len(evidence) >= max_snippets:
            break

    return evidence[:max_snippets]


async def extract_chunk_lite(
    transcript: Transcript,
    chunk_index: int,
    context: Optional[Context] = None,
) -> Tuple[ChunkExtractionResult, int]:
    """
    Extract structured fields from a single chunk using lite extractor.

    This is the cheap/fast extractor for the MAP stage.
    Returns ChunkExtractionResult with partial fields and evidence.

    PHI-safe:
    - NEVER logs transcript, prompt, or output
    - Only logs: latency_ms, error_code, chunk_index

    Args:
        transcript: Chunk transcript (PHI - never logged)
        chunk_index: 0-based index of this chunk
        context: Optional clinical context

    Returns:
        Tuple of (ChunkExtractionResult, inference_ms)

    Raises:
        BackendUnavailableError, BackendTimeoutError, RateLimitedError, ModelError
    """
    settings = get_settings()
    start_time = time.perf_counter()

    # Build prompts (PHI - never logged)
    system_prompt = _build_lite_system_prompt()
    user_prompt = _build_lite_user_prompt(transcript, context)

    # Prepare request
    base_url = settings.openai_compat_base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    timeout_s = settings.openai_compat_timeout_ms / 1000.0
    model_name = settings.openai_compat_model

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": settings.lite_extractor_max_tokens,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

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
        raise
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise RateLimitedError()
        raise BackendUnavailableError("openai_compat")

    # Parse response (PHI-safe)
    try:
        result = response.json()
    except json.JSONDecodeError:
        raise ModelError("Invalid JSON response from backend")

    if isinstance(result, dict) and "error" in result:
        raise ModelError("Backend returned error")

    try:
        choices = result.get("choices", []) if isinstance(result, dict) else []
        if not choices:
            raise KeyError("choices")

        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message")

        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            model_output = msg["content"]
        elif isinstance(first.get("text"), str):
            model_output = first["text"]
        else:
            raise KeyError("content")

    except (KeyError, IndexError, TypeError):
        raise ModelError("Invalid response format from backend")

    # Parse and validate output
    fields = _parse_lite_output(model_output)

    # Extract evidence (PHI-safe: sanitized)
    evidence = _extract_evidence_from_fields(
        fields,
        transcript,
        max_snippets=settings.evidence_max_snippets_per_chunk
    )

    inference_ms = int((time.perf_counter() - start_time) * 1000)

    # Build result
    chunk_result = ChunkExtractionResult(
        chunkIndex=chunk_index,
        fields=fields,
        evidence=evidence,
        extractorUsed="lite"
    )

    # PHI-safe logging
    logger.info(
        "lite_extraction_complete",
        chunk_index=chunk_index,
        inference_ms=inference_ms,
        evidence_count=len(evidence),
        has_diagnostico=fields.diagnostico is not None
    )

    return chunk_result, inference_ms
