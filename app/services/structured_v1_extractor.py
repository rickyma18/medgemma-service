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
        "negations",
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


# FIX #5: Threshold below which few-shot examples are injected
# to prevent hallucination on sparse/short transcripts.
SHORT_TRANSCRIPT_THRESHOLD = 150


def _build_short_transcript_fewshot(scope: str | None = None) -> str:
    """
    Scope-aware few-shot examples for short transcripts (<150 chars).

    Examples MUST strictly respect the active scope.
    For scope=interview: only include interview fields (motivoConsulta, padecimientoActual, antecedentes).
    No placeholders like '[sin diagnostico]', 'sin datos', 'no referido' are ever shown.
    Fields outside the scope must NOT appear at all in the examples.

    PHI-safe: all examples are synthetic.
    """
    if scope == "interview":
        return '''

## EJEMPLOS PARA TRANSCRIPTS CORTOS – INTERVIEW

ENTRADA: "Padre diabetico. Niega alergias."
SALIDA:
{
  "motivoConsulta": null,
  "padecimientoActual": null,
  "antecedentes": {
    "heredofamiliares": "Padre con diabetes mellitus",
    "personalesNoPatologicos": null,
    "personalesPatologicos": "Niega alergias"
  },
  "negations": []
}

ENTRADA: "No fuma, no toma. Circuncision a los 5 años."
SALIDA:
{
  "motivoConsulta": null,
  "padecimientoActual": null,
  "antecedentes": {
    "heredofamiliares": null,
    "personalesNoPatologicos": "Niega tabaquismo. Niega alcoholismo",
    "personalesPatologicos": "Circuncisión a los 5 años"
  },
  "negations": []
}

ENTRADA: "Alergico a sulfas. Madre hipertensa."
SALIDA:
{
  "motivoConsulta": null,
  "padecimientoActual": null,
  "antecedentes": {
    "heredofamiliares": "Madre con hipertensión arterial",
    "personalesNoPatologicos": null,
    "personalesPatologicos": "Alergia a sulfas"
  },
  "negations": []
}'''

    # Fallback for non-scoped or other scopes: return empty (no few-shot needed)
    # Full extraction has its own comprehensive examples in the base prompt.
    return ""


def _build_v1_system_prompt(
    scope: str | None = None,
    transcript_len: int = 0,
) -> str:
    """
    System prompt optimizado para MedGemma.

    (a) Prompt base: built in this function.
    (b) Few-shot branch: injected when transcript_len < SHORT_TRANSCRIPT_THRESHOLD.
    (c) Scope instructions: appended when scope is provided.

    Args:
        scope: Optional extraction scope (interview, exam, studies, assessment).
               If provided, instructs LLM to only fill scoped fields.
        transcript_len: Effective transcript text length in chars.
                        When < SHORT_TRANSCRIPT_THRESHOLD, few-shot examples are added.
    """
    base_prompt = '''Eres un asistente de documentacion clinica ORL.

MISION: EXTRAER informacion del transcript a JSON. NO REDACTAR. NO INFERIR. NO INVENTAR.
- Copia los datos tal como aparecen en el texto.
- Si un campo no se menciona → null (o {} para objetos).
- Preserva negaciones como evidencia ("niega diabetes" → registrar "Niega diabetes", NO null).

## REGLAS CRITICAS (OBEDECE SIEMPRE)

0. RESPETA EL CONTENIDO DEL TRANSCRIPT:
   - Extrae SOLO lo que se menciona. NO inventes motivos ni sintomas.
   - Si el transcript solo contiene antecedentes (p. ej. "no fuma, alergia a sulfas"), entonces:
     motivoConsulta = null
     padecimientoActual = null
     diagnostico = "[sin datos de padecimiento actual]" SOLO si el schema lo obliga; si no, null.

1. Si NO se menciona → null (nunca inventar, nunca "no especificado")
   - EXCEPTO negativos pertinentes: "no fuma", "niega alergias", "sin cirugias", "no toma alcohol", "no drogas", "sin mascotas"
     SON datos validos y DEBEN registrarse.
   - Solo usar null si el tema NO se toco en la conversacion.

1B. ENRUTAMIENTO DE ANTECEDENTES (MUY IMPORTANTE):
   - Si el texto menciona habitos/negativos (fuma, alcohol, drogas, mascotas, vivienda) → va a antecedentes.personalesNoPatologicos.
   - Si el texto menciona alergias ("alergico a", "alergia a", "reaccion a", "anafilaxia", "penicilina", "sulfas") → va a antecedentes.personalesPatologicos.
   - Si el texto menciona cirugias/procedimientos ("operaron", "cirugia", "amigdalectomia", "circuncision") → va a antecedentes.personalesPatologicos.
   - NO conviertas antecedentes en "dolor de garganta" u otros sintomas si no estan en el texto.

2. exploracionFisica = hallazgos OBJETIVOS (lo que se ve/palpa/ausculta, NO sintomas subjetivos del paciente)
   - En DICTADO (1 speaker): todo lo que describe el medico ES exploracion (ej. "cornetes hipertroficos, septum desviado")
   - En CONSULTA: diferenciar medico ("se observa X") vs paciente ("me duele X" → padecimientoActual)
   - "placas/pus/exudado" pertenecen a orofaringe, NO a cuello.

3. diagnostico es OBLIGATORIO:
   - Si NO hay diagnostico explicito y NO hay padecimiento actual → "[sin diagnostico por falta de datos clinicos]".
   - Si hay sintomas pero no diagnostico → "[sintoma principal] en estudio".

4. "impresion" o "impresion diagnostica" = diagnostico (NO confundir con depresion ni estado de animo)

## MAPEO DE CAMPOS
- motivoConsulta: queja principal, 3-15 palabras. SOLO si el paciente/medico dice explicitamente por que viene.
- padecimientoActual: cronologia de sintomas (inicio, evolucion, intensidad). 1-4 oraciones.
- antecedentes.heredofamiliares: enfermedades de familiares directos. Ej: "Padre con DM2".
- antecedentes.personalesNoPatologicos: habitos y negativos pertinentes. Ej: "Niega tabaquismo, niega alcoholismo".
- antecedentes.personalesPatologicos: enfermedades cronicas, cirugias, alergias, medicamentos actuales.

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

FORMATO:
- Devuelve JSON valido EXACTAMENTE con el schema. No agregues texto afuera del JSON.'''

    # FIX #5: Inject scope-aware few-shot examples for short transcripts
    if 0 < transcript_len < SHORT_TRANSCRIPT_THRESHOLD:
        base_prompt += _build_short_transcript_fewshot(scope)

    # Add scope instruction if provided
    if scope:
        scope_instructions = {
        "interview": (
            "SCOPE: PASO = INTERVIEW (anamnesis). Extrae ÚNICAMENTE y SOLO estos campos. Deja TODO lo demás como null/{}:\n"
            "1) motivoConsulta\n"
            "2) padecimientoActual\n"
            "3) antecedentes.heredofamiliares\n"
            "4) antecedentes.personalesNoPatologicos\n"
            "5) antecedentes.personalesPatologicos\n"
            "\n"
            "REGLAS:\n"
            "- Si el transcript menciona CUALQUIER dato de estos campos, extráelo aunque sea mínimo.\n"
            "- Convierte negaciones ('niega', 'sin', 'no') en texto clínico útil dentro del campo correcto.\n"
            "- No inventes información.\n"
            "- Prohibido placeholders: 'sin datos', 'no refiere', 'N/A', '-', 'pendiente'. Si no hay info, usa null.\n"
            "- Responde SOLO JSON válido con este shape EXACTO:\n"
            "{\n"
            "  \"motivoConsulta\": string|null,\n"
            "  \"padecimientoActual\": string|null,\n"
            "  \"antecedentes\": {\n"
            "    \"heredofamiliares\": string|null,\n"
            "    \"personalesNoPatologicos\": string|null,\n"
            "    \"personalesPatologicos\": string|null\n"
            "  },\n"
            "  \"negations\": []\n"
            "}\n"
        ),

        "exam": (
            "SCOPE: PASO = EXAM (exploración física). Extrae ÚNICAMENTE exploracionFisica (con todos sus subcampos). "
            "Deja TODO lo demás como null/{}.\n"
            "\n"
            "REGLAS:\n"
            "- Incluye signosVitales si aparecen (TA/FC/FR/T/SpO2), si no: null.\n"
            "- No inventes hallazgos.\n"
            "- Prohibido placeholders (sin datos/no refiere/N/A). Si no hay info, usa null.\n"
            "- Responde SOLO JSON válido con este shape EXACTO:\n"
            "{\n"
            "  \"exploracionFisica\": object,\n"
            "  \"negations\": []\n"
            "}\n"
        ),

        "studies": (
            "SCOPE: PASO = STUDIES (estudios/indicaciones). Extrae ÚNICAMENTE estudiosIndicados. "
            "Deja TODO lo demás como null/{}.\n"
            "\n"
            "REGLAS:\n"
            "- Si hay estudios mencionados (labs, imagen, gabinete), listalos tal cual en estudiosIndicados.\n"
            "- No inventes estudios.\n"
            "- Prohibido placeholders (sin datos/no refiere/N/A). Si no hay info, usa null.\n"
            "- Responde SOLO JSON válido con este shape EXACTO:\n"
            "{\n"
            "  \"estudiosIndicados\": string|null,\n"
            "  \"negations\": []\n"
            "}\n"
        ),

        "assessment": (
            "SCOPE: PASO = ASSESSMENT (cierre clínico). Extrae ÚNICAMENTE:\n"
            "- diagnostico\n"
            "- planTratamiento\n"
            "- pronostico\n"
            "Deja TODO lo demás como null/{}.\n"
            "\n"
            "REGLAS:\n"
            "- NO inventes diagnósticos ni tratamientos. Si no están en el transcript, usa null.\n"
            "- Prohibido placeholders (sin datos/no refiere/N/A). Si no hay info, usa null.\n"
            "- Responde SOLO JSON válido con este shape EXACTO:\n"
            "{\n"
            "  \"diagnostico\": string|null,\n"
            "  \"planTratamiento\": string|null,\n"
            "  \"pronostico\": string|null,\n"
            "  \"negations\": []\n"
            "}\n"
        ),
        }

        if scope in scope_instructions:
            base_prompt += f"\n\n{scope_instructions[scope]}"

    return base_prompt


def _is_effectively_empty(value) -> bool:
    """Check if a value carries no useful information."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, dict) and not any(
        not _is_effectively_empty(v) for v in value.values()
    ):
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def compute_extraction_meta(fields: StructuredFieldsV1) -> dict:
    """
    Compute PHI-safe extraction metadata for client sparse-detection.

    Returns a dict with:
        hasContent (bool): True if any clinical field is non-null or negations non-empty.
        negatedFindingsCount (int): Number of items in negations list.

    This is safe to include in API responses (no PHI, only booleans/counts).
    """
    negations = fields.negations or []
    negated_count = len(negations)

    has_content = (
        fields.motivo_consulta is not None
        or fields.padecimiento_actual is not None
        or (fields.antecedentes is not None and (
            fields.antecedentes.heredofamiliares is not None
            or fields.antecedentes.personales_no_patologicos is not None
            or fields.antecedentes.personales_patologicos is not None
        ))
        or negated_count > 0
    )

    return {
        "hasContent": has_content,
        "negatedFindingsCount": negated_count,
    }


def _apply_scope_mask(data: dict, scope: str) -> dict:
    """
    Non-destructive scope mask applied POST-LLM.

    In-scope fields are always preserved as-is. Out-of-scope fields are
    kept when they contain actual data (cross-scope "bonus" data the LLM
    extracted from the transcript), but normalized to null/{} when empty.

    Args:
        data: The repaired dict from _repair_v1_dict
        scope: The extraction scope (interview, exam, studies, assessment)

    Returns:
        Dict with scoped fields preserved; out-of-scope fields preserved
        only when non-empty, otherwise normalized to null/{}.
    """
    allowed = SCOPE_ALLOWED_FIELDS.get(scope, set())

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
        "negations",
    }

    # Dict-typed fields get {} instead of None when empty
    dict_fields = {"antecedentes", "exploracionFisica"}
    list_fields = {"negations"}

    masked = {}
    for field in all_fields:
        if field in list_fields:
            value = data.get(field, [])
        else:
            value = data.get(field)
        if field in allowed:
            # In-scope: always keep as-is
            masked[field] = value
        else:
            # Out-of-scope: keep if non-empty, normalize if empty
            if _is_effectively_empty(value):
                if field in dict_fields:
                    masked[field] = {}
                elif field in list_fields:
                    masked[field] = []
                else:
                    masked[field] = None
            else:
                masked[field] = value

    return masked


def _merge_negations_into_antecedentes(data: dict, scope: str | None) -> dict:
    """
    Merge negations into appropriate antecedentes subfields for interview scope.

    For scope="interview", negations should not remain as a standalone top-level field.
    Instead, they are routed to the correct antecedentes subfield based on content:

    1. Habits (fuma, alcohol, drogas, tabaco) → personalesNoPatologicos
    2. Allergies, diseases, surgeries, conditions → personalesPatologicos
    3. Family relations + disease (padre, madre, hermano + diabetes, HTA) → heredofamiliares

    Args:
        data: The repaired dict from _repair_v1_dict
        scope: The extraction scope

    Returns:
        Dict with negations merged into antecedentes (only for interview scope)

    PHI-safe: No logging of content.
    """
    # Only apply for interview scope
    if scope != "interview":
        return data

    negations = data.get("negations", [])
    if not negations or not isinstance(negations, list):
        return data

    # Ensure antecedentes exists
    antecedentes = data.get("antecedentes", {})
    if not isinstance(antecedentes, dict):
        antecedentes = {}

    heredofam = antecedentes.get("heredofamiliares") or ""
    apnp = antecedentes.get("personalesNoPatologicos") or ""
    app = antecedentes.get("personalesPatologicos") or ""

    # Keyword sets for routing (case-insensitive matching)
    # Habits → personalesNoPatologicos
    habit_keywords = {
        "fuma", "fumar", "tabaco", "tabaquismo", "cigarro", "cigarrillo",
        "alcohol", "alcoholismo", "toma", "bebe", "bebedor",
        "droga", "drogas", "toxicomanía", "toxicomania", "marihuana", "cocaína", "cocaina",
        "mascota", "mascotas", "perro", "gato", "animales",
    }

    # Allergies, diseases, surgeries → personalesPatologicos
    patologicos_keywords = {
        "alergia", "alergias", "alérgico", "alergico", "alérgica", "alergica",
        "diabetes", "diabético", "diabetico", "dm", "dm2",
        "hipertensión", "hipertension", "hta", "hipertenso", "hipertensa",
        "asma", "asmático", "asmatico",
        "cirugía", "cirugia", "cirugías", "cirugias", "operación", "operacion",
        "enfermedad", "enfermedades", "patología", "patologia",
        "medicamento", "medicamentos", "fármaco", "farmaco",
        "transfusión", "transfusion", "hospitalización", "hospitalizacion",
        "cáncer", "cancer", "tumor",
    }

    # Family relations → heredofamiliares (need relation + disease)
    family_keywords = {
        "padre", "papá", "papa", "madre", "mamá", "mama",
        "hermano", "hermana", "abuelo", "abuela",
        "tío", "tio", "tía", "tia", "primo", "prima",
        "familia", "familiar", "familiares", "heredo",
    }

    routed_to_apnp = []
    routed_to_app = []
    routed_to_heredofam = []

    for neg in negations:
        if not isinstance(neg, str) or not neg.strip():
            continue

        neg_lower = neg.lower()

        # Check for family relation + disease patterns
        has_family = any(kw in neg_lower for kw in family_keywords)
        has_disease = any(kw in neg_lower for kw in patologicos_keywords)

        if has_family and has_disease:
            routed_to_heredofam.append(neg.strip())
        elif any(kw in neg_lower for kw in habit_keywords):
            routed_to_apnp.append(neg.strip())
        elif any(kw in neg_lower for kw in patologicos_keywords):
            routed_to_app.append(neg.strip())
        else:
            # Default: route to personalesPatologicos (most common for negations)
            routed_to_app.append(neg.strip())

    # Merge routed negations into existing fields
    if routed_to_heredofam:
        new_text = ". ".join(routed_to_heredofam)
        heredofam = f"{heredofam}. {new_text}".strip(". ") if heredofam else new_text

    if routed_to_apnp:
        new_text = ". ".join(routed_to_apnp)
        apnp = f"{apnp}. {new_text}".strip(". ") if apnp else new_text

    if routed_to_app:
        new_text = ". ".join(routed_to_app)
        app = f"{app}. {new_text}".strip(". ") if app else new_text

    # Update antecedentes
    antecedentes["heredofamiliares"] = heredofam if heredofam else None
    antecedentes["personalesNoPatologicos"] = apnp if apnp else None
    antecedentes["personalesPatologicos"] = app if app else None

    data["antecedentes"] = antecedentes

    # Clear negations after merging (for interview scope only)
    data["negations"] = []

    return data


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
        "negaciones": "negations",
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
    data.setdefault("negations", [])

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

        # Merge negations into antecedentes for interview scope (before scope mask)
        repaired = _merge_negations_into_antecedentes(repaired, scope)

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

    # Compute effective transcript length for few-shot decision (FIX #5)
    transcript_len = sum(len(seg.text) for seg in transcript.segments)

    # Construir prompts (PHI - no se loguea)
    system_prompt = _build_v1_system_prompt(scope, transcript_len)
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

    # Propagate upstream negations from context when provided.
    if context and context.negations:
        upstream_negations = [n for n in context.negations if isinstance(n, str) and n.strip()]
        fields.negations = upstream_negations

        # For interview scope, merge context negations into antecedentes
        # so clients see populated antecedentes fields (prevents false sparse).
        if scope == "interview" and upstream_negations:
            data_dict = fields.model_dump(by_alias=True)
            data_dict["negations"] = upstream_negations
            data_dict = _merge_negations_into_antecedentes(data_dict, scope)
            # Restore negations (merge clears them); keep in response for clients.
            data_dict["negations"] = upstream_negations
            fields = StructuredFieldsV1.model_validate(data_dict)

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
