"""
Service for generating treatment plan suggestions via LLM.

Lightweight single-call inference — no extraction, no contracts, no chunking.
PHI-safe: NEVER log motivo_consulta, diagnostico, or generated plan text.
"""
import json
import time
from typing import Literal

import httpx

from app.core.config import get_settings
from app.core.logging import get_safe_logger
from app.services.exceptions import (
    BackendUnavailableError,
    BackendTimeoutError,
    ModelError,
    RateLimitedError,
)

logger = get_safe_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(language: str, style: Literal["bullets", "paragraph"]) -> str:
    """Build the system prompt for treatment plan generation."""

    if style == "bullets":
        format_rule = (
            "Responde SOLO con una lista de viñetas (máximo 8 ítems). "
            "Cada viñeta inicia con '- '. No numeres."
        )
    else:
        format_rule = (
            "Responde SOLO con un párrafo de 3 a 6 frases."
        )

    lang_instruction = (
        "Responde en español clínico claro y conciso."
        if language == "es"
        else f"Respond in clear, concise clinical language ({language})."
    )

    return f"""Eres un especialista clínico experimentado.
Tu tarea es generar ÚNICAMENTE el plan de tratamiento para el paciente, basándote en el motivo de consulta y el diagnóstico proporcionados.

## REGLAS ESTRICTAS
- Genera SOLO el plan de tratamiento. Nada más.
- NO incluyas encabezados como "Plan de tratamiento:" ni títulos.
- NO incluyas disclaimers largos ni advertencias legales.
- NO inventes datos del paciente (edad, peso, alergias) que no se proporcionaron.
- Puedes incluir UNA sola línea breve de advertencia clínica al final si es relevante (ej. "Valorar ajuste en caso de alergia a penicilinas.").
- {format_rule}
- {lang_instruction}"""


def _build_user_prompt(motivo_consulta: str, diagnostico: str) -> str:
    """Build user prompt with clinical context. PHI — never logged."""
    return (
        f"Motivo de consulta: {motivo_consulta}\n"
        f"Diagnóstico: {diagnostico}\n\n"
        f"Genera el plan de tratamiento:"
    )


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

_MOCK_PLAN_BULLETS = (
    "- Reposo relativo y adecuada hidratación oral\n"
    "- Analgésico/antipirético según dolor y fiebre\n"
    "- Vigilar signos de alarma y acudir a urgencias si empeora"
)

_MOCK_PLAN_PARAGRAPH = (
    "Se recomienda reposo relativo y adecuada hidratación oral. "
    "Indicar analgésico/antipirético en caso de dolor o fiebre. "
    "Vigilar signos de alarma y acudir a urgencias si presenta deterioro clínico."
)


def _mock_suggest(style: Literal["bullets", "paragraph"]) -> tuple[str, int]:
    """Return deterministic mock plan for testing."""
    plan = _MOCK_PLAN_BULLETS if style == "bullets" else _MOCK_PLAN_PARAGRAPH
    return plan, 30


# ---------------------------------------------------------------------------
# LLM call (openai_compat / vllm)
# ---------------------------------------------------------------------------

async def _llm_suggest(
    motivo_consulta: str,
    diagnostico: str,
    language: str,
    style: Literal["bullets", "paragraph"],
) -> tuple[str, int]:
    """
    Call the configured LLM backend for plan generation.

    Returns (plan_text, inference_ms).
    Raises ExtractorError subclasses on failure.
    """
    settings = get_settings()
    start = time.perf_counter()

    system_prompt = _build_system_prompt(language, style)
    user_prompt = _build_user_prompt(motivo_consulta, diagnostico)

    # Select backend URL / model / timeout
    if settings.extractor_backend == "vllm":
        base_url = settings.vllm_base_url.rstrip("/")
        model_name = settings.vllm_model
        timeout_s = settings.vllm_timeout_ms / 1000.0
    else:
        # openai_compat (default for real backends)
        base_url = settings.openai_compat_base_url.rstrip("/")
        model_name = settings.openai_compat_model
        timeout_s = settings.openai_compat_timeout_ms / 1000.0

    url = f"{base_url}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 429:
                raise RateLimitedError()
            if response.status_code >= 500:
                raise BackendUnavailableError(
                    "vllm" if settings.extractor_backend == "vllm" else "openai_compat"
                )
            response.raise_for_status()

    except httpx.ConnectError:
        raise BackendUnavailableError(
            "vllm" if settings.extractor_backend == "vllm" else "openai_compat"
        )
    except httpx.TimeoutException:
        elapsed = int((time.perf_counter() - start) * 1000)
        raise BackendTimeoutError(elapsed)
    except (RateLimitedError, BackendUnavailableError):
        raise
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise RateLimitedError()
        raise BackendUnavailableError(
            "vllm" if settings.extractor_backend == "vllm" else "openai_compat"
        )

    # Parse response
    try:
        result = response.json()
    except json.JSONDecodeError:
        raise ModelError("Invalid JSON response from backend")

    if isinstance(result, dict) and "error" in result:
        raise ModelError("Backend returned an error response")

    try:
        choices = result.get("choices", []) if isinstance(result, dict) else []
        if not choices:
            raise KeyError("choices")

        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message")

        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            plan_text = msg["content"].strip()
        elif isinstance(first.get("text"), str):
            plan_text = first["text"].strip()
        else:
            raise KeyError("content")

    except (KeyError, IndexError, TypeError):
        raise ModelError("Invalid response format from backend")

    if not plan_text:
        raise ModelError("Empty plan returned by model")

    inference_ms = int((time.perf_counter() - start) * 1000)
    return plan_text, inference_ms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def suggest_plan(
    motivo_consulta: str,
    diagnostico: str,
    language: str = "es",
    style: Literal["bullets", "paragraph"] = "bullets",
) -> tuple[str, int]:
    """
    Generate a treatment plan suggestion.

    Returns (plan_text, inference_ms).
    PHI-safe: never logs inputs or outputs.
    """
    settings = get_settings()

    if settings.extractor_backend == "mock":
        return _mock_suggest(style)

    return await _llm_suggest(motivo_consulta, diagnostico, language, style)
