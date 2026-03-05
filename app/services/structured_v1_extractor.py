"""
StructuredFieldsV1 extractor for ORL clinical documentation.
Uses OpenAI-compatible API (MedGemma/vLLM).

PHI-safe: NEVER log transcript, prompt, or model output.
"""
import json
import re
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
# Interview scope benefits from examples at higher thresholds (smaller model).
SHORT_TRANSCRIPT_THRESHOLD_INTERVIEW = 400


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
    "personalesNoPatologicos": "No fuma. No toma alcohol",
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
}

ENTRADA: "No fumo, no tomo alcohol. Me operaron de apéndice hace 5 años. No tengo diabetes ni hipertensión."
SALIDA:
{
  "motivoConsulta": null,
  "padecimientoActual": null,
  "antecedentes": {
    "heredofamiliares": null,
    "personalesNoPatologicos": "No fuma. No toma alcohol",
    "personalesPatologicos": "Apendicectomía hace 5 años. Niega diabetes. Niega hipertensión"
  },
  "negations": []
}

ENTRADA: "Colesistectomía hace 8 años. Niega alergias."
SALIDA:
{
  "motivoConsulta": null,
  "padecimientoActual": null,
  "antecedentes": {
    "heredofamiliares": null,
    "personalesNoPatologicos": null,
    "personalesPatologicos": "Colecistectomía hace 8 años. Niega alergias"
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
- motivoConsulta: queja principal, 1-2 oraciones cortas completas. Nunca cortar a mitad de frase. Si termina en preposicion (de/del/con/y) o numero suelto, reescribir como frase completa. SOLO si el paciente/medico dice explicitamente por que viene.
- padecimientoActual: cronologia de sintomas (inicio, evolucion, intensidad). 1-4 oraciones.
- antecedentes.heredofamiliares: enfermedades de familiares directos. Ej: "Padre con DM2".
- antecedentes.personalesNoPatologicos: habitos y negativos pertinentes. Ej: "Niega tabaquismo, niega alcoholismo".
- antecedentes.personalesPatologicos: enfermedades cronicas, cirugias, alergias, medicamentos actuales.

## SCHEMA
{
  "motivoConsulta": "string 1-2 oraciones cortas completas | null",
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
    # Interview scope uses a higher threshold because MedGemma benefits more from examples.
    effective_threshold = SHORT_TRANSCRIPT_THRESHOLD_INTERVIEW if scope == "interview" else SHORT_TRANSCRIPT_THRESHOLD
    if 0 < transcript_len < effective_threshold:
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
            "6) negations\n"
            "\n"
            "SIGNIFICADO DE CAMPOS DE ANTECEDENTES:\n"
            "- personalesNoPatologicos: SOLO hábitos (tabaco, alcohol, drogas, mascotas, vivienda, ocupación).\n"
            "- personalesPatologicos: SOLO antecedentes médicos crónicos/enfermedades previas + alergias + cirugías.\n"
            "- heredofamiliares: enfermedades de familiares directos.\n"
            "- PROHIBIDO colocar síntomas en antecedentes. Los síntomas (fiebre, tos, disnea, dolor, rinorrea, nausea, diarrea, cefalea, odinofagia, etc.) NO van en ningún campo de antecedentes. Si son clínicamente relevantes van en padecimientoActual; si fueron negados, pueden ir a negations[] pero NUNCA a antecedentes.\n"
            "\n"
            "REGLAS:\n"
            "- Si el transcript menciona CUALQUIER dato de estos campos, extráelo aunque sea mínimo.\n"
            "- Convierte negaciones en texto clínico útil dentro del campo correcto.\n"
            "- Cada hallazgo o negación va en ORACIÓN SEPARADA terminada en punto.\n"
            "- Formato de negaciones de ENFERMEDADES: usar 'Niega …' (Ej: 'Niega diabetes. Niega hipertensión.').\n"
            "- Formato de negaciones de HÁBITOS: usar 'No …' (Ej: 'No fuma. No toma alcohol.').\n"
            "- Cirugías e intervenciones quirúrgicas → personalesPatologicos.\n"
            "  Incluye variantes ortográficas: colecistectomía/colesistectomía, apendicectomía,\n"
            "  amigdalectomía, circuncisión, histerectomía, cesárea, hernioplastía,\n"
            "  y frases genéricas como 'me operaron', 'cirugía previa'.\n"
            "  Ejemplo: 'colesistectomía hace 8 años' → personalesPatologicos: 'Colecistectomía hace 8 años.'\n"
            "- motivoConsulta: 1-2 oraciones cortas COMPLETAS. NUNCA cortar a mitad de frase. Si termina en 'de/del/con/y' o número suelto, reescribir como frase completa.\n"
            "- No inventes información.\n"
            "- Prohibido placeholders: 'sin datos', 'no refiere', 'N/A', '-', 'pendiente'. Si no hay info, usa null.\n"
            "\n"
            "REGLAS DE negations[]:\n"
            "- negations es una lista de strings. Cada elemento debe ser UN concepto limpio y atómico.\n"
            "- PROHIBIDO conjunciones colgantes: nunca terminar un item con 'ni', 'y', 'e', 'o'. Ejemplo: si el paciente dice 'no fuma ni toma alcohol', negations debe ser [\"fuma\", \"toma alcohol\"], NUNCA [\"fuma ni\"].\n"
            "- PROHIBIDO fragmentos de puntuación, preposiciones sueltas o conectores.\n"
            "- negations[] debe incluir SOLO items negados que NO estén ya claramente capturados en campos de antecedentes (evitar duplicación).\n"
            "- Nunca inventar items; sin placeholders.\n"
            "\n"
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


def compute_extraction_meta(fields: StructuredFieldsV1, scope: str | None = None) -> dict:
    """
    Compute PHI-safe extraction metadata for client sparse-detection.

    Scope-aware: when scope is provided, only checks fields relevant to that scope.
    For interview: motivoConsulta, padecimientoActual, antecedentes, negations.

    Returns a dict with:
        hasContent (bool): True if any in-scope clinical field is non-null or negations non-empty.
        negatedFindingsCount (int): Number of items in negations list.

    This is safe to include in API responses (no PHI, only booleans/counts).
    """
    negations = fields.negations or []
    negated_count = len(negations)

    if scope == "interview":
        # Interview scope: check only anamnesis fields
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
    elif scope == "exam":
        has_content = fields.exploracion_fisica is not None and (
            fields.exploracion_fisica.otoscopia is not None
            or fields.exploracion_fisica.rinoscopia is not None
            or fields.exploracion_fisica.orofaringe is not None
            or fields.exploracion_fisica.cuello is not None
            or fields.exploracion_fisica.signos_vitales is not None
        )
    elif scope == "assessment":
        has_content = (
            fields.diagnostico is not None
            or fields.plan_tratamiento is not None
            or fields.pronostico is not None
        )
    else:
        # Full extraction or unknown scope: check all
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
    Strict scope mask applied POST-LLM.

    In-scope fields are preserved as-is. Out-of-scope fields are ALWAYS
    nulled/emptied regardless of content — the backend enforces that each
    scope produces ONLY its own fields, so the client never has to trim.

    Args:
        data: The repaired dict from _repair_v1_dict
        scope: The extraction scope (interview, exam, studies, assessment)

    Returns:
        Dict with in-scope fields preserved; out-of-scope fields strictly null/empty.
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

    list_fields = {"negations"}

    masked = {}
    for field in all_fields:
        if field in allowed:
            # In-scope: always keep as-is
            if field in list_fields:
                masked[field] = data.get(field, [])
            else:
                masked[field] = data.get(field)
        else:
            # Out-of-scope: strictly null (no cross-scope leaking)
            if field in list_fields:
                masked[field] = []
            else:
                masked[field] = None

    return masked


# ── Interview postprocess: reroute misplaced antecedentes from padecimientoActual ──

_HABIT_KEYWORDS_PA = {
    "fuma", "fumar", "tabaco", "tabaquismo", "cigarro",
    "alcohol", "alcoholismo", "bebe",
    "droga", "drogas", "toxicomania", "toxicomanía",
    "mascota", "mascotas",
}

_SURGERY_KEYWORDS_PA = [
    "apendicectom", "apendisectom", "amigdalectom",
    "cirugía", "cirugia", "cirugía previa", "cirugia previa",
    "operad", "operaci", "operaron", "operó", "opero", "me operaron",
    "circuncis",
    "colecistectom", "colesistectom", "colecistectomía", "colesistectomía",
    "histerectom", "histerectomía", "histerectomia",
    "cesárea", "cesarea", "hernioplast", "hernioplastía", "hernioplastia",
    "artroscop", "artroscopía", "artroscopia",
    "tiroidectom", "mastectom", "prostatectom", "nefrectom",
    # ORL-specific
    "rinoplast",        # rinoplastia (cirugía nasal)
    "septoplast",       # septoplastia (corrección de septum)
    "turbinoplast",     # turbinoplastia
    "timpanoplast",     # timpanoplastia (cirugía de oído)
    "adenoidectom",     # adenoidectomía
    "traqueotom",       # traqueotomía / traqueostomía
]

_HF_KEYWORDS_PA = {"familiares", "heredofamiliares", "en familia"}

# Regex: "no" followed by 0-2 words followed by a habit keyword, ANYWHERE in phrase
_RE_HABIT_IN_TEXT = re.compile(
    r'\bno\s+(?:\w+\s+){0,2}(?:'
    + '|'.join(sorted(_HABIT_KEYWORDS_PA, key=len, reverse=True))
    + r')',
    re.IGNORECASE,
)

# Regex: "niega" ANYWHERE in phrase
_RE_NEGATION_IN_TEXT = re.compile(r'\bniega\b', re.IGNORECASE)


def _postprocess_interview_fields(data: dict, scope: str | None) -> dict:
    """
    Move misplaced antecedentes data from padecimientoActual to the correct
    antecedentes subfields.  Only applies when scope == "interview".

    Detection is done ANYWHERE in each phrase (not just at the start):
      (a) Habits ("no fuma", "refiere que no toma alcohol") → personalesNoPatologicos
      (b) Disease negations ("niega diabetes")              → personalesPatologicos
      (c) Surgeries ("apendicectomía …")                    → personalesPatologicos
      (d) Family history ("en familiares …")                → heredofamiliares

    Remaining phrases stay in padecimientoActual.

    PHI-safe: only logs counts.
    """
    if scope != "interview":
        return data

    pa = data.get("padecimientoActual")
    if not pa or not isinstance(pa, str):
        return data

    # Ensure antecedentes structure
    antecedentes = data.get("antecedentes") or {}
    if not isinstance(antecedentes, dict):
        antecedentes = {}

    heredofam = antecedentes.get("heredofamiliares") or ""
    apnp = antecedentes.get("personalesNoPatologicos") or ""
    app_field = antecedentes.get("personalesPatologicos") or ""

    # Split into phrases by period, semicolon, or comma+space
    phrases = [p.strip() for p in re.split(r'(?:\.\s*|;\s*|,\s+)', pa) if p.strip()]

    remaining: list[str] = []
    to_apnp: list[str] = []
    to_app: list[str] = []
    to_hf: list[str] = []

    for phrase in phrases:
        lower = phrase.lower()

        # (d) HF
        if any(kw in lower for kw in _HF_KEYWORDS_PA):
            to_hf.append(phrase)
        # (a) Habits: "no" + habit keyword anywhere in phrase
        elif _RE_HABIT_IN_TEXT.search(lower):
            to_apnp.append(phrase)
        # (b) Disease negation: "niega" anywhere in phrase
        elif _RE_NEGATION_IN_TEXT.search(lower):
            to_app.append(phrase)
        # (c) Surgeries / procedures
        elif any(kw in lower for kw in _SURGERY_KEYWORDS_PA):
            to_app.append(phrase)
        else:
            remaining.append(phrase)

    moved_count = len(to_apnp) + len(to_app) + len(to_hf)
    if moved_count == 0:
        return data

    logger.info(
        "interview_postprocess_rerouted",
        moved_to_apnp=len(to_apnp),
        moved_to_app=len(to_app),
        moved_to_hf=len(to_hf),
        remaining=len(remaining),
    )

    # Update padecimientoActual
    if remaining:
        new_pa = ". ".join(remaining)
        if not new_pa.endswith("."):
            new_pa += "."
        data["padecimientoActual"] = new_pa
    else:
        data["padecimientoActual"] = None

    # Helper: concatenate with ". "
    def _append(existing: str, new_phrases: list[str]) -> str:
        new_text = ". ".join(new_phrases)
        return f"{existing}. {new_text}" if existing else new_text

    if to_hf:
        heredofam = _append(heredofam, to_hf)
    if to_apnp:
        apnp = _append(apnp, to_apnp)
    if to_app:
        app_field = _append(app_field, to_app)

    antecedentes["heredofamiliares"] = heredofam or None
    antecedentes["personalesNoPatologicos"] = apnp or None
    antecedentes["personalesPatologicos"] = app_field or None
    data["antecedentes"] = antecedentes

    return data


# ── Deterministic surgery rescue from transcript ──

# Tolerant regex patterns for common surgical procedures (handles typos).
# Each tuple: (compiled regex, canonical label used for dedup key).
_SURGERY_RESCUE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # colecistectomía / colesistectomía (and common typos)
    (re.compile(
        r'\bcole[sc]istectom[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "colecistectomia"),
    # apendicectomía / apendisectomía
    (re.compile(
        r'\bapend[ií][cs]ectom[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "apendicectomia"),
    # amigdalectomía
    (re.compile(
        r'\bamigdalectom[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "amigdalectomia"),
    # circuncisión
    (re.compile(
        r'\bcircuncisi[oó]n\b[^.;]*',
        re.IGNORECASE,
    ), "circuncision"),
    # histerectomía
    (re.compile(
        r'\bhisterectom[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "histerectomia"),
    # cesárea
    (re.compile(
        r'\bces[aá]rea\b[^.;]*',
        re.IGNORECASE,
    ), "cesarea"),
    # hernioplastía / hernioplastia
    (re.compile(
        r'\bhernioplast[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "hernioplastia"),
    # tiroidectomía
    (re.compile(
        r'\btiroidectom[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "tiroidectomia"),
    # ── ORL-specific procedures ──
    # rinoplastia
    (re.compile(
        r'\brinoplast[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "rinoplastia"),
    # septoplastia — tolerant regex also captures ASR typo "esceptoplastía":
    #   "septoplastia"   → e? s  c?  e  pt  o  plast ia ✓
    #   "esceptoplastía" → e  s  c   e  pt  o  plast ía ✓
    #   "siptoplastía"   → e? s  c?  i  pt  o  plast ía ✓
    (re.compile(
        r'\be?sc?[ei]pt?o?plast[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "septoplastia"),
    # timpanoplastia
    (re.compile(
        r'\btimpanoplast[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "timpanoplastia"),
    # adenoidectomía
    (re.compile(
        r'\badenoidectom[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "adenoidectomia"),
    # traqueotomía / traqueostomía
    (re.compile(
        r'\btraqueo(?:t|st)om[ií]a\b[^.;]*',
        re.IGNORECASE,
    ), "traqueotomia"),
    # generic "me operaron …" / "lo operaron …" / "fue operado/a …"
    (re.compile(
        r'(?:me|lo|la|le|fue)\s+operar?on\b[^.;]*',
        re.IGNORECASE,
    ), "operaron_generico"),
    # generic "cirugía previa" / "cirugía de …"
    (re.compile(
        r'\bcirug[ií]a\s+(?:previa|de\b)[^.;]*',
        re.IGNORECASE,
    ), "cirugia_generica"),
]

# Extra substrings checked in the existing personalesPatologicos text to avoid
# duplicate rescue when ASR typos produce a non-canonical spelling.
# key = canonical_key from _SURGERY_RESCUE_PATTERNS
# value = tuple of additional lowercase fragments; if ANY is present, skip rescue.
_SURGERY_RESCUE_EXTRA_CHECKS: dict[str, tuple[str, ...]] = {
    "septoplastia": ("septoplast", "esceptoplast", "siptoplast"),
}

# Post-capture normalization: correct ASR typos in the raw matched phrase.
# key = canonical_key; value = (pattern to replace, canonical replacement string)
_SURGERY_RESCUE_NORMALIZATIONS: dict[str, tuple[re.Pattern, str]] = {
    "septoplastia": (
        re.compile(r'\be?sc?[ei]pt?o?plast[ií]a\b', re.IGNORECASE),
        "Septoplastia",
    ),
}


def rescue_surgeries_from_transcript(data: dict, transcript_text: str, scope: str | None) -> dict:
    """
    Deterministic safety-net: scan *transcript_text* for surgical procedures
    and ensure they appear in antecedentes.personalesPatologicos.

    Only runs for scope == "interview".
    Skips any procedure whose canonical key is already present in the field
    (case-insensitive substring check) to avoid duplicates.

    Each rescued phrase is a complete sentence ending in a period.

    PHI-safe: only logs counts.
    """
    if scope != "interview" or not transcript_text:
        return data

    antecedentes = data.get("antecedentes") or {}
    if not isinstance(antecedentes, dict):
        antecedentes = {}

    app_field = (antecedentes.get("personalesPatologicos") or "").strip()
    app_lower = app_field.lower()

    rescued: list[str] = []

    for pattern, canonical_key in _SURGERY_RESCUE_PATTERNS:
        # Skip if canonical key already present in antecedentes.
        # Primary check: canonical key minus trailing 'a' covers most
        # inflections (e.g. "colecistectomi" matches both spellings).
        # Extra checks: handle ASR typo variants (e.g. "esceptoplast" for
        # "septoplast") defined per canonical_key in _SURGERY_RESCUE_EXTRA_CHECKS.
        check_key = canonical_key.rstrip("a")
        extra_checks = _SURGERY_RESCUE_EXTRA_CHECKS.get(canonical_key, ())
        if check_key in app_lower or any(ec in app_lower for ec in extra_checks):
            continue

        match = pattern.search(transcript_text)
        if match:
            phrase = match.group(0).strip().rstrip(".,;: ")
            # Normalize known ASR typos to canonical spelling
            # (e.g. "esceptoplastía" → "Septoplastia")
            if canonical_key in _SURGERY_RESCUE_NORMALIZATIONS:
                norm_pat, norm_repl = _SURGERY_RESCUE_NORMALIZATIONS[canonical_key]
                phrase = norm_pat.sub(norm_repl, phrase, count=1)
            # Capitalize first letter
            if phrase:
                phrase = phrase[0].upper() + phrase[1:]
                if not phrase.endswith("."):
                    phrase += "."
                rescued.append(phrase)

    if not rescued:
        return data

    logger.info("interview_surgery_rescue", rescued_count=len(rescued))

    new_text = " ".join(rescued)
    if app_field:
        # Ensure existing text ends with period before appending
        if not app_field.endswith("."):
            app_field += "."
        app_field = f"{app_field} {new_text}"
    else:
        app_field = new_text

    antecedentes["personalesPatologicos"] = app_field
    data["antecedentes"] = antecedentes

    return data


# ── Negation normalization ──

_INCOMPLETE_NEGATION_RE = re.compile(
    r'^('
    # single-token stopwords and pronouns
    r'e|y|o|ni|a|de|del|con|en|ha|las?|los?|un[ao]?s?'
    r'|se|me|te|mi|su|sus|por'
    r'|ha presentado|presentado'
    # two-token pronoun/stopword sequences (ASR noise)
    r'|se me|me la|me lo|me las|me los|te la|te lo|te las|te los'
    r'|se le|se lo|se la|se les|no me|no se'
    r')$',
    re.IGNORECASE,
)

_TRAILING_DANGLING_RE = re.compile(
    r'[\s,/]+(ni|y|e|o|a|de|del|con|en|por|la|el|me|te|mi|su)$',
    re.IGNORECASE,
)


# First-person singular verb prefixes that add no clinical information to a
# negation item: "tomo medicamentos" → "medicamentos", "uso drogas" → "drogas".
# Third-person ("toma", "usa") is intentionally excluded to preserve items like
# "toma alcohol" which an existing test explicitly checks for.
_NEGATION_VERB_PREFIX_RE = re.compile(
    r'^\b(?:tomo|uso|consumo)\s+',
    re.IGNORECASE,
)

# Multi-word sequences composed entirely of stopwords/pronouns.
# Catches ASR noise like "se me", "me lo", "te la" that slip through as full
# negation items.  Must anchor both ends.
_PURE_STOPWORD_SEQ_RE = re.compile(
    r'^(?:(?:se|me|te|le|nos|os|la|lo|las|los)\s+)+'
    r'(?:se|me|te|le|nos|os|la|lo|las|los)$',
    re.IGNORECASE,
)


def _normalize_negations(negations: list | None) -> list:
    """
    Clean negation entries: drop noise, trim dangling prepositions, merge
    fragments, deduplicate, and remove word-set subsumptions.

    Phases:
      0.5 Strip first-person verb prefixes ("tomo X" → "X").
      1.  Drop incomplete tokens, pure-stopword sequences, trim dangling
          prepositions/conjunctions.
      2.  Merge adjacent fragments ("alergias" + "medicamentos" → "alergias a
          medicamentos").
      3.  Deduplicate preserving order (case-insensitive exact match).
      3.5 Subsumption dedup: if every word of item A appears in item B AND all
          of A's words are ≥ 4 chars, drop A (B is the more specific form).
          Guard: items with any word < 4 chars (e.g. "tos", "asma") are never
          subsumed so that "tos" and "tos productiva" both survive.

    PHI-safe: no content logging.
    """
    if not negations:
        return []

    # Phase 0.5: strip first-person verb prefixes
    stripped: list[str] = []
    for neg in negations:
        if not isinstance(neg, str):
            continue
        neg = _NEGATION_VERB_PREFIX_RE.sub('', neg.strip())
        stripped.append(neg)

    # Phase 1: clean individual entries
    cleaned: list[str] = []
    for neg in stripped:
        neg = neg.strip().rstrip(".,;:/")
        if not neg or len(neg) < 3:
            continue
        if _INCOMPLETE_NEGATION_RE.match(neg):
            continue
        # Drop pure multi-token stopword sequences ("se me", "me la", ...)
        if _PURE_STOPWORD_SEQ_RE.match(neg):
            continue
        # Trim trailing dangling preposition/conjunction (applied once)
        neg = _TRAILING_DANGLING_RE.sub('', neg).strip()
        if neg and len(neg) >= 3:
            cleaned.append(neg)

    # Phase 2: merge adjacent fragments
    merged: list[str] = []
    skip_next = False
    for i, neg in enumerate(cleaned):
        if skip_next:
            skip_next = False
            continue

        if i + 1 < len(cleaned):
            lower = neg.lower()
            next_lower = cleaned[i + 1].lower()
            if lower in ("alergias", "alergia") and next_lower.startswith("medicamento"):
                merged.append(f"{neg} a {cleaned[i + 1]}")
                skip_next = True
                continue

        merged.append(neg)

    # Phase 3: deduplicate preserving order (case-insensitive)
    seen: set[str] = set()
    deduped: list[str] = []
    for neg in merged:
        key = neg.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(neg)

    # Phase 3.5: word-set subsumption dedup.
    # If words(A) ⊆ words(B) strictly and every word of A is ≥ 4 chars,
    # drop A (B is more specific / already contains A).
    # The ≥ 4-char guard protects short clinical terms like "tos", "asma".
    words_sets = [frozenset(neg.lower().split()) for neg in deduped]
    final: list[str] = []
    for i, neg in enumerate(deduped):
        words_i = words_sets[i]
        # Skip subsumption check when any word is short (≤ 3 chars)
        if not words_i or any(len(w) <= 3 for w in words_i):
            final.append(neg)
            continue
        # Drop if words_i is a strict subset of any other item's word-set
        subsumed = any(
            words_i < words_sets[j]
            for j in range(len(deduped))
            if j != i
        )
        if not subsumed:
            final.append(neg)

    return final


# Terms that are almost always part of the chief complaint and should never
# appear as negation items (even if the LLM sees them in negated secondary phrases).
_CHIEF_COMPLAINT_EXCLUSIONS: frozenset[str] = frozenset({
    "dolor", "molestia", "malestar", "consulta", "visita", "motivo",
})


def _filter_negations_against_positive_fields(data: dict) -> dict:
    """
    Remove from negations[] any item that represents a *positive* clinical
    finding already captured in motivoConsulta or padecimientoActual.

    This prevents the LLM from putting the chief-complaint symptom (e.g.
    "dolor") into negations[] just because the word also appears in a
    negated secondary phrase in the transcript.

    Three filters applied (in order):
      1. Exact exclusion list (_CHIEF_COMPLAINT_EXCLUSIONS).
      2. Term appears verbatim in motivoConsulta or padecimientoActual text.
      3. 2+ content words (≥4 chars) of the negation item all appear in
         the positive text — catches short paraphrases.

    PHI-safe: no content is logged.
    """
    negations = data.get("negations", [])
    if not negations:
        return data

    positive_text = " ".join(p for p in [
        (data.get("motivoConsulta") or "").lower(),
        (data.get("padecimientoActual") or "").lower(),
    ] if p)

    filtered: list[str] = []
    for neg in negations:
        if not isinstance(neg, str):
            continue
        neg_lower = neg.lower().strip()

        # Filter 1: known chief-complaint exclusions
        if neg_lower in _CHIEF_COMPLAINT_EXCLUSIONS:
            continue

        # Filter 2: verbatim substring of positive clinical text
        if positive_text and neg_lower in positive_text:
            continue

        # Filter 3: majority of content words appear in positive text
        content_words = [w for w in neg_lower.split() if len(w) >= 4]
        if positive_text and len(content_words) >= 2:
            if sum(1 for w in content_words if w in positive_text) >= 2:
                continue

        filtered.append(neg)

    data["negations"] = filtered
    return data


def _to_clinical_phrase(item: str, prefix: str) -> str:
    """
    Format a raw negation item as a self-contained clinical sentence.

    If the item already begins with a clinical negation marker ("niega",
    "no", "sin") the prefix is not added.  A trailing period is always
    ensured.

    Args:
        item:   Raw negation string, e.g. "otras cirugías".
        prefix: "Niega" for diseases/surgeries/allergies, "No" for habits.

    Returns:
        Formatted clinical sentence, e.g. "Niega otras cirugías."
    """
    clean = item.strip()
    lower = clean.lower()
    if lower.startswith(("niega ", "no ", "sin ")):
        phrase = clean
    else:
        phrase = f"{prefix} {clean}"
    return phrase if phrase.endswith(".") else phrase + "."


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

    routed_to_apnp: list[str] = []
    routed_to_app: list[str] = []
    routed_to_heredofam: list[str] = []

    for neg in negations:
        if not isinstance(neg, str) or not neg.strip():
            continue

        neg_lower = neg.lower()

        # Check for family relation + disease patterns
        has_family = any(kw in neg_lower for kw in family_keywords)
        has_disease = any(kw in neg_lower for kw in patologicos_keywords)

        if has_family and has_disease:
            routed_to_heredofam.append(_to_clinical_phrase(neg.strip(), "Niega"))
        elif any(kw in neg_lower for kw in habit_keywords):
            # Habits use "No" (e.g. "No fuma.", "No toma alcohol.")
            routed_to_apnp.append(_to_clinical_phrase(neg.strip(), "No"))
        elif any(kw in neg_lower for kw in patologicos_keywords):
            routed_to_app.append(_to_clinical_phrase(neg.strip(), "Niega"))
        else:
            # Default: personalesPatologicos with "Niega" prefix
            routed_to_app.append(_to_clinical_phrase(neg.strip(), "Niega"))

    def _append_phrases(existing: str, phrases: list[str]) -> str:
        """Concatenate clinical phrases onto existing field text."""
        new_text = " ".join(phrases)
        if not existing:
            return new_text
        sep = " " if existing.rstrip().endswith(".") else ". "
        return existing.rstrip() + sep + new_text

    # Merge routed negations into existing fields
    if routed_to_heredofam:
        heredofam = _append_phrases(heredofam, routed_to_heredofam)

    if routed_to_apnp:
        apnp = _append_phrases(apnp, routed_to_apnp)

    if routed_to_app:
        app = _append_phrases(app, routed_to_app)

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
    scope = context.scope if context else None
    speakers = set(seg.speaker for seg in transcript.segments)
    # For interview scope, never force DICTADO: a transcript where all speakers
    # are "unknown" (no diarisation) is still a doctor-patient conversation.
    # Other scopes (exam/dictado) keep the existing detection logic.
    is_dictation = (
        len(speakers) == 1 or all(s in ("doctor", "unknown") for s in speakers)
    ) and scope != "interview"

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

    if is_dictation:
        mode = "DICTADO"
    elif scope == "interview" and all(
        seg.speaker in ("unknown", "doctor") for seg in transcript.segments
    ):
        # Interview without speaker diarisation: make it explicit so the LLM
        # knows both speaker turns are present in the undifferentiated stream.
        mode = "CONSULTA (sin segmentación de turnos — el texto mezcla preguntas del médico y respuestas del paciente)"
    else:
        mode = "CONSULTA"

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

        # Normalize negations list (clean incomplete tokens)
        repaired["negations"] = _normalize_negations(repaired.get("negations", []))

        # Filter negations[] against positive clinical fields to remove false
        # positives (e.g. chief-complaint symptom "dolor" appearing as negation).
        repaired = _filter_negations_against_positive_fields(repaired)

        # Post-process interview: move misplaced antecedentes from padecimientoActual
        repaired = _postprocess_interview_fields(repaired, scope)

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
        upstream_negations = _normalize_negations(upstream_negations)
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
        def _field_counts(val):
            """Return PHI-safe (char_count, line_count) for a string field."""
            if not val:
                return 0, 0
            return len(val), val.count(".") + val.count("\n")

        motivo_len, motivo_lines = _field_counts(fields.motivo_consulta)
        padec_len, padec_lines = _field_counts(fields.padecimiento_actual)
        apnp_len, apnp_lines = _field_counts(
            fields.antecedentes.personales_no_patologicos if fields.antecedentes else None
        )
        app_len, app_lines = _field_counts(
            fields.antecedentes.personales_patologicos if fields.antecedentes else None
        )
        heredofam_len, heredofam_lines = _field_counts(
            fields.antecedentes.heredofamiliares if fields.antecedentes else None
        )

        # Compute scope-aware extraction quality metadata
        extraction_meta = compute_extraction_meta(fields, scope)

        debug_metrics = {
            "scope": scope or "full",
            "has_motivo": fields.motivo_consulta is not None,
            "motivo_len": motivo_len,
            "motivo_lines": motivo_lines,
            "has_padecimiento": fields.padecimiento_actual is not None,
            "padecimiento_len": padec_len,
            "padecimiento_lines": padec_lines,
            "has_diagnostico": fields.diagnostico is not None,
            "has_plan": fields.plan_tratamiento is not None,
            "has_heredofam": fields.antecedentes.heredofamiliares is not None if fields.antecedentes else False,
            "heredofam_len": heredofam_len,
            "heredofam_lines": heredofam_lines,
            "has_apnp": fields.antecedentes.personales_no_patologicos is not None if fields.antecedentes else False,
            "apnp_len": apnp_len,
            "apnp_lines": apnp_lines,
            "has_app": fields.antecedentes.personales_patologicos is not None if fields.antecedentes else False,
            "app_len": app_len,
            "app_lines": app_lines,
            "has_otoscopia": fields.exploracion_fisica.otoscopia is not None if fields.exploracion_fisica else False,
            "has_rinoscopia": fields.exploracion_fisica.rinoscopia is not None if fields.exploracion_fisica else False,
            "has_orofaringe": fields.exploracion_fisica.orofaringe is not None if fields.exploracion_fisica else False,
            "has_cuello": fields.exploracion_fisica.cuello is not None if fields.exploracion_fisica else False,
            "diagnostico_tipo": fields.diagnostico.tipo if fields.diagnostico else None,
            "useful_flag": extraction_meta["hasContent"],
            "sparse_flag": not extraction_meta["hasContent"],
            "negated_findings_count": extraction_meta["negatedFindingsCount"],
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
