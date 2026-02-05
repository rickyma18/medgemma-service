"""
StructuredFieldsV1 extractor for ORL clinical documentation.
Uses OpenAI-compatible API (MedGemma/vLLM).

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
from app.schemas.structured_fields_v1 import StructuredFieldsV1
from app.services.exceptions import (
    BackendUnavailableError,
    BackendTimeoutError,
    ModelError,
    RateLimitedError,
)
from app.services.text_normalizer_orl import normalize_transcript_orl
from app.services.orl_postprocess import postprocess_orl_mapping

logger = get_safe_logger(__name__)

# Version identifier for this extractor
V1_EXTRACTOR_VERSION = "structured-v1"

# Scope -> allowed fields mapping
# Fields outside scope will be masked to null/empty POST-LLM
SCOPE_ALLOWED_FIELDS: dict[str, set[str]] = {
    "interview": {
        "motivoConsulta",
        "padecimientoActual",
        "antecedentes",  # includes all nested: heredofamiliares, personalesNoPatologicos, personalesPatologicos
    },
    "exam": {
        "exploracionFisica",  # includes all nested: signosVitales, rinoscopia, orofaringe, cuello, etc.
    },
    "studies": {
        "estudiosIndicados",
    },
    "assessment": {
        "diagnostico",
        "planTratamiento",
        "pronostico",
    },
}


def _build_v1_system_prompt(scope: str | None = None) -> str:
    """
    System prompt optimizado para MedGemma.

    - Mas conciso (modelos pequenos pierden contexto en prompts largos)
    - Incluye few-shot example (critico para extraccion precisa)
    - Reglas criticas al INICIO (primacy effect)
    - Campos alineados a nomenclatura medica mexicana

    Args:
        scope: Optional extraction scope (interview, exam, studies, assessment).
               If provided, instructs LLM to only fill scoped fields.
    """
    base_prompt = '''Eres un asistente de documentacion clinica ORL. Extrae informacion de transcripciones medicas a JSON.

## REGLAS CRITICAS
1. Si NO se menciona → null (nunca inventar, nunca "no especificado")
2. exploracionFisica = SOLO hallazgos del MEDICO ("se observa", "a la exploracion")
   - NO incluir sintomas del paciente ("me duele", "siento") → eso va en padecimientoActual
   - "placas/pus/exudado" pertenecen a orofaringe, NO a cuello.
3. diagnostico es OBLIGATORIO: si no hay explicito → "[sintoma principal] en estudio"

## SCHEMA
{
  "motivoConsulta": "string 3-15 palabras | null",
  "padecimientoActual": "narrativa 1-4 oraciones | null",
  "antecedentes": {
    "heredofamiliares": "enfermedades en familia | null",
    "personalesNoPatologicos": "tabaquismo, alcoholismo, ocupacion | null",
    "personalesPatologicos": "enfermedades, cirugias, alergias, medicamentos | null"
  },
  "exploracionFisica": {
    "signosVitales": "TA, FC, Temp, SpO2 | null",
    "rinoscopia": "mucosa, cornetes, septum | null",
    "orofaringe": "amigdalas, faringe | null",
    "cuello": "adenopatias, tiroides | null",
    "laringoscopia": "cuerdas vocales | null",
    "otoscopia": "CAE, membrana timpanica | null"
  },
  "diagnostico": {
    "texto": "OBLIGATORIO",
    "tipo": "definitivo|presuntivo|sindromico",
    "cie10": "codigo | null"
  },
  "planTratamiento": "medicamentos con dosis y duracion | null",
  "pronostico": "solo si se menciona | null",
  "estudiosIndicados": "laboratorios, imagen | null",
  "notasAdicionales": "seguimiento, referencias | null"
}

## EJEMPLO

ENTRADA:
[Medico]: Que lo trae?
[Paciente]: Llevo 5 dias con dolor de garganta y me cuesta tragar.
[Medico]: Fiebre?
[Paciente]: Ayer tuve 38.5.
[Medico]: Enfermedades cronicas?
[Paciente]: Soy diabetico, tomo metformina. Sin alergias.
[Medico]: Familiares con diabetes o hipertension?
[Paciente]: Mi mama es hipertensa.
[Medico]: A la orofaringe: amigdalas hiperhemicas grado II con exudado. Cuello con adenopatia submandibular izquierda 1cm. Es faringoamigdalitis aguda. Amoxicilina 500mg c/8h por 7 dias, ibuprofeno PRN.

SALIDA:
{
  "motivoConsulta": "Dolor de garganta de 5 dias con disfagia",
  "padecimientoActual": "Odinofagia de 5 dias, progresiva, con disfagia. Fiebre 38.5C ayer.",
  "antecedentes": {
    "heredofamiliares": "Madre con HTA",
    "personalesNoPatologicos": null,
    "personalesPatologicos": "DM2 con metformina. Niega alergias"
  },
  "exploracionFisica": {
    "signosVitales": null,
    "rinoscopia": null,
    "orofaringe": "Amigdalas hiperhemicas grado II con exudado",
    "cuello": "Adenopatia submandibular izquierda 1cm",
    "laringoscopia": null,
    "otoscopia": null
  },
  "diagnostico": {
    "texto": "Faringoamigdalitis aguda",
    "tipo": "definitivo",
    "cie10": null
  },
  "planTratamiento": "Amoxicilina 500mg VO c/8h x7 dias; Ibuprofeno PRN",
  "pronostico": null,
  "estudiosIndicados": null,
  "notasAdicionales": null
}

## DX SINDROMICOS (si no hay dx explicito)
- Dolor garganta → "Odinofagia en estudio"
- Vertigo → "Sindrome vertiginoso en estudio"
- Congestion nasal → "Rinosinusitis en estudio"
- Dolor oido → "Otalgia en estudio"
- Ronquera → "Disfonia en estudio"

Responde SOLO con JSON valido, sin explicaciones ni markdown.'''

    # Add scope instruction if provided
    if scope:
        scope_instructions = {
            "interview": "SCOPE: Solo extrae motivoConsulta, padecimientoActual y antecedentes. Deja el resto como null.",
            "exam": "SCOPE: Solo extrae exploracionFisica (todos sus subcampos). Deja el resto como null.",
            "studies": "SCOPE: Solo extrae estudiosIndicados. Deja el resto como null.",
            "assessment": "SCOPE: Solo extrae diagnostico, planTratamiento y pronostico. Deja el resto como null.",
        }
        if scope in scope_instructions:
            base_prompt += f"\n\n{scope_instructions[scope]}"

    return base_prompt


def _apply_scope_mask(data: dict, scope: str) -> dict:
    """
    Force all fields outside the scope to null/empty, POST-LLM.

    This is a security measure to ensure the LLM cannot "contaminate"
    fields outside the requested scope, regardless of what it returns.

    Args:
        data: The repaired dict from _repair_v1_dict
        scope: The extraction scope (interview, exam, studies, assessment)

    Returns:
        Dict with only scoped fields populated; all others null/empty.
    """
    allowed = SCOPE_ALLOWED_FIELDS.get(scope, set())
    if not allowed:
        # Unknown scope or "studies" with no fields in schema -> return minimal
        return {
            "antecedentes": {},
            "exploracionFisica": {},
        }

    # Define all top-level field keys
    all_fields = {
        "motivoConsulta",
        "padecimientoActual",
        "antecedentes",
        "exploracionFisica",
        "diagnostico",
        "planTratamiento",
        "pronostico",
        "estudiosIndicados",
        "notasAdicionales",
    }

    masked = {}
    for field in all_fields:
        if field in allowed:
            # Keep the value from data
            masked[field] = data.get(field)
        else:
            # Mask to null/empty based on field type
            if field == "antecedentes":
                masked[field] = {}
            elif field == "exploracionFisica":
                masked[field] = {}
            else:
                masked[field] = None

    return masked


def _build_v1_user_prompt(transcript: Transcript, context: Optional[Context]) -> str:
    """
    Construye el prompt de usuario.
    Detecta automaticamente si es dictado (1 speaker) o conversacion.
    PHI-safe: Esta funcion es interna; el prompt NUNCA se loguea.
    """
    # Detectar modo
    speakers = set(seg.speaker for seg in transcript.segments)
    is_dictation = len(speakers) == 1 or all(s in ("doctor", "unknown") for s in speakers)

    # Construir texto
    text_parts = []
    for seg in transcript.segments:
        if is_dictation:
            text_parts.append(seg.text)
        else:
            label = {"doctor": "Medico", "patient": "Paciente"}.get(seg.speaker, "")
            if label:
                text_parts.append(f"[{label}]: {seg.text}")
            else:
                text_parts.append(seg.text)

    transcript_text = "\n".join(text_parts)

    # Contexto clinico opcional
    context_parts = []
    if context:
        if context.patient_age is not None:
            context_parts.append(f"Edad: {context.patient_age} anos")
        if context.patient_gender:
            gender = {"male": "M", "female": "F"}.get(context.patient_gender, "")
            if gender:
                context_parts.append(f"Sexo: {gender}")

    context_line = f"[{', '.join(context_parts)}] " if context_parts else ""
    mode = "DICTADO" if is_dictation else "CONSULTA"

    return f"""{mode}: {context_line}
{transcript_text}

JSON:"""


def _repair_v1_dict(data: dict) -> dict:
    """
    Normaliza el output del modelo al schema V1.
    Maneja variantes comunes que MedGemma puede generar.
    PHI-safe: no loguea contenido.
    """
    if not isinstance(data, dict):
        return {}

    # Mapeo de variantes -> keys canonicas
    key_map = {
        # Top level
        "motivo_consulta": "motivoConsulta",
        "motivoDeConsulta": "motivoConsulta",
        "motivo": "motivoConsulta",
        "padecimiento_actual": "padecimientoActual",
        "padecimiento": "padecimientoActual",
        "plan_tratamiento": "planTratamiento",
        "plan": "planTratamiento",
        "tratamiento": "planTratamiento",
        "estudios_indicados": "estudiosIndicados",
        "estudios": "estudiosIndicados",
        "notas_adicionales": "notasAdicionales",
        "notas": "notasAdicionales",
        # Antecedentes
        "antecedentes_heredofamiliares": "heredofamiliares",
        "heredoFamiliares": "heredofamiliares",
        "familiares": "heredofamiliares",
        "personales_no_patologicos": "personalesNoPatologicos",
        "noPatologicos": "personalesNoPatologicos",
        "no_patologicos": "personalesNoPatologicos",
        "personales_patologicos": "personalesPatologicos",
        "patologicos": "personalesPatologicos",
        # Exploracion
        "exploracion_fisica": "exploracionFisica",
        "exploración_física": "exploracionFisica",
        "exploracion_orl": "exploracionFisica",
        "exploracionOrl": "exploracionFisica",
        "exploracion": "exploracionFisica",
        "signos_vitales": "signosVitales",
        "endoscopia_nasal": "endoscopiaNasal",
    }

    def normalize_keys(obj):
        if isinstance(obj, dict):
            return {key_map.get(k, k): normalize_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [normalize_keys(item) for item in obj]
        return obj

    data = normalize_keys(data)

    # Asegurar estructuras requeridas
    data.setdefault("antecedentes", {})
    data.setdefault("exploracionFisica", {})

    # Limpiar valores placeholder a null
    placeholder_values = {
        "", "null", "none", "n/a", "no especificado", "pendiente",
        "no se menciona", "sin datos", "no aplica", "no referido",
        "no mencionado", "sin información", "sin informacion"
    }

    def clean_values(obj):
        if isinstance(obj, dict):
            return {k: clean_values(v) for k, v in obj.items()}
        elif isinstance(obj, str):
            stripped = obj.strip()
            if stripped.lower() in placeholder_values:
                return None
            return stripped or None
        return obj

    data = clean_values(data)

    # Asegurar diagnostico (OBLIGATORIO)
    dx = data.get("diagnostico")
    if not dx or (isinstance(dx, dict) and not dx.get("texto")):
        # Generar diagnostico sindromico basado en motivo
        motivo = data.get("motivoConsulta") or ""
        texto_dx = f"{motivo} en estudio" if motivo else "Consulta ORL en estudio"
        data["diagnostico"] = {
            "texto": texto_dx,
            "tipo": "sindromico",
            "cie10": None
        }
    elif isinstance(dx, str):
        # Si el modelo devolvio string en vez de objeto
        data["diagnostico"] = {
            "texto": dx,
            "tipo": "presuntivo",
            "cie10": None
        }
    elif isinstance(dx, dict):
        dx.setdefault("tipo", "presuntivo")
        dx.setdefault("cie10", None)

    return data


def _parse_v1_output(output: str, scope: str | None = None) -> StructuredFieldsV1:
    """
    Parsea y valida el output del modelo como StructuredFieldsV1.

    Args:
        output: Raw model output string
        scope: Optional extraction scope. If provided, applies mask POST-repair.

    Raises:
        ModelError: Si el output no es JSON valido o no cumple el schema
    """
    output = output.strip()

    # Remover markdown code blocks si existen
    if output.startswith("```"):
        lines = output.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        output = "\n".join(lines)

    # Encontrar boundaries del JSON
    start_idx = output.find("{")
    end_idx = output.rfind("}") + 1

    if start_idx == -1 or end_idx == 0:
        raise ModelError("No se encontro objeto JSON en la salida del modelo")

    json_str = output[start_idx:end_idx]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ModelError("JSON invalido en la salida del modelo")

    try:
        repaired = _repair_v1_dict(data)

        # Apply scope mask POST-repair to prevent LLM contamination
        if scope:
            repaired = _apply_scope_mask(repaired, scope)

        return StructuredFieldsV1.model_validate(repaired)
    except ValidationError:
        raise ModelError("La salida no cumple el schema StructuredFieldsV1")


async def extract_structured_v1(
    transcript: Transcript,
    context: Optional[Context] = None,
) -> tuple[StructuredFieldsV1, int, str]:
    """
    Extrae campos estructurados V1 usando API OpenAI-compatible.

    PHI-safe:
    - NUNCA loguea transcript, prompt, o output del modelo
    - Solo loguea: latency_ms, error_code, status

    Args:
        transcript: Transcripcion clinica (PHI - no se loguea)
        context: Contexto clinico opcional

    Returns:
        Tupla de (StructuredFieldsV1, inference_ms, model_version)

    Raises:
        BackendUnavailableError: Si el backend no esta disponible
        BackendTimeoutError: Si hay timeout
        RateLimitedError: Si hay rate limiting
        ModelError: Si el output es invalido
    """
    settings = get_settings()
    start_time = time.perf_counter()

    # 1. Normalizacion STT (Whitelist ORL)
    transcript, nrep = normalize_transcript_orl(transcript)
    if nrep > 0:
        # Log solo conteo, PHI-safe
        logger.info("v1_normalization_applied", replacements=nrep)

    # Extract scope from context (if provided)
    scope = context.scope if context else None
    if scope:
        logger.info("v1_scoped_extraction", scope=scope)

    # Construir prompts (PHI - no se loguea)
    system_prompt = _build_v1_system_prompt(scope)
    user_prompt = _build_v1_user_prompt(transcript, context)

    # Preparar request
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
        "temperature": 0.0,  # Deterministico para extraccion clinica
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

    # Parsear response (PHI-safe)
    try:
        result = response.json()
    except json.JSONDecodeError:
        raise ModelError("Respuesta JSON invalida del backend")

    if isinstance(result, dict) and "error" in result:
        logger.error(
            "v1_extractor backend error",
            error_code="MODEL_ERROR",
            http_status=response.status_code,
        )
        raise ModelError("El backend retorno un error")

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
        raise ModelError("Formato de respuesta invalido del backend")

    # Parsear y validar output (with scope mask if applicable)
    fields = _parse_v1_output(model_output, scope)

    # 2. Post-procesamiento deterministico (Cuello <-> Orofaringe)
    fields = postprocess_orl_mapping(fields)

    # Calcular tiempo de inferencia
    inference_ms = int((time.perf_counter() - start_time) * 1000)

    # Version del modelo
    model_version = f"structured-v1-{model_name}"

    # PHI-safe debug metrics
    try:
        debug_metrics = {
            "has_motivo": fields.motivo_consulta is not None,
            "has_padecimiento": fields.padecimiento_actual is not None,
            "has_diagnostico": fields.diagnostico is not None,
            "has_plan": fields.plan_tratamiento is not None,
            "has_otoscopia": fields.exploracion_fisica.otoscopia is not None,
            "has_rinoscopia": fields.exploracion_fisica.rinoscopia is not None,
            "has_orofaringe": fields.exploracion_fisica.orofaringe is not None,
            "has_cuello": fields.exploracion_fisica.cuello is not None,
            "diagnostico_tipo": fields.diagnostico.tipo if fields.diagnostico else None,
        }
        logger.info("v1_extraction_metrics", **debug_metrics)
    except Exception:
        pass

    return fields, inference_ms, model_version


def get_v1_model_version() -> str:
    """Retorna la version del extractor V1."""
    settings = get_settings()
    if settings.openai_compat_model:
        return f"structured-v1-{settings.openai_compat_model}"
    return V1_EXTRACTOR_VERSION
