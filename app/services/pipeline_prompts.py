
# Constant for Finalize timeout
FINALIZE_TIMEOUT_S = 45.0

def _build_finalize_prompt(data_json: str) -> str:
    """Prompt para la etapa de finalización (Refine)."""
    return f"""Eres un asistente clínico experto. Tienes un JSON con datos extraídos y agregados de una consulta ORL.
Tu tarea es REFINAR y DEDUPLICAR la información, respetando estrictamente el schema.

ENTRADA (JSON RAW):
{data_json}

INSTRUCCIONES:
1. Devuelve SOLO JSON válido. Sin markdown.
2. Si hay campos concatenados con ' | ' o duplicados semánticos, fusiónalos en una frase coherente o lista única.
3. Si hay contradicciones (ej: "Niega alergias | Alérico a penicilina"), prioriza la afirmación positiva o la más específica, pero si dice "Niega X", respétalo si no hay evidencia contraria.
4. En 'diagnostico.tipo', conserva el de mayor certeza (definitivo > presuntivo > sindromico).
5. NO inventes información. Si algo es null, déjalo null.
6. Schema EXACTO a respetar:
{{
  "motivoConsulta": "string | null",
  "padecimientoActual": "string | null",
  "antecedentes": {{ "heredofamiliares": "...", "personalesNoPatologicos": "...", "personalesPatologicos": "..." }},
  "exploracionFisica": {{ "signosVitales": "...", "rinoscopia": "...", "orofaringe": "...", "cuello": "...", "laringoscopia": "...", "otoscopia": "...", "endoscopiaNasal": "...", "otomicroscopia": "..." }},
  "diagnostico": {{ "texto": "...", "tipo": "definitivo|presuntivo|sindromico", "cie10": "..." }},
  "planTratamiento": "...",
  "pronostico": "...",
  "estudiosIndicados": "...",
  "notasAdicionales": "..."
}}

JSON REFINADO:"""
