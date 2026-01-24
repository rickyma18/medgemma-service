"""
Módulo de post-procesamiento para extracción ORL.
Objetivo: Reubicar hallazgos mal clasificados entre 'cuello' y 'orofaringe' 
debido a ambigüedades del modelo o del habla.

PHI-safe:
- Opera sobre campos estructurados ya extraídos.
- No loguea contenido específico.
"""
import re
from app.schemas.structured_fields_v1 import StructuredFieldsV1

# Keywords para identificar hallazgos típicamente de Orofaringe
# (placas, pus, exudado, amígdala, faringe, garganta)
RE_OROFARINGE_KEYWORDS = re.compile(
    r'\b(placas?|pus|exudados?|am[íi]gdalas?|faringe[as]?|garganta)\b', 
    re.IGNORECASE
)

# Keywords para identificar hallazgos típicamente de Cuello
# (adenopatía, ganglio, cervical, submandibular, cuello, bolita)
RE_CUELLO_KEYWORDS = re.compile(
    r'\b(adenopat[íi]as?|ganglios?|cervical(?:es)?|submandibular(?:es)?|cuello|bolitas?)\b', 
    re.IGNORECASE
)

# Separadores de frases: punto, punto y coma, saltos de línea.
RE_SPLIT = re.compile(r'[.;\n]+')

def postprocess_orl_mapping(fields: StructuredFieldsV1) -> StructuredFieldsV1:
    """
    Revisa y reasigna frases entre exploracionFisica.cuello y exploracionFisica.orofaringe
    basándose en keywords clínicas fuertes.
    
    Reglas:
    1. Si 'cuello' tiene frases de orofaringe (y no de cuello), mover a orofaringe.
    2. Si 'orofaringe' tiene frases de cuello (y no de orofaringe), mover a cuello.
    
    Args:
        fields: Objeto StructuredFieldsV1 con la extracción cruda del modelo.
        
    Returns:
        StructuredFieldsV1: Mismo objeto (o copia) con campos modificados in-place si aplica.
    """
    ef = fields.exploracion_fisica
    
    # 1. Obtener contenidos actuales (normalizar None -> "")
    cuello_content = ef.cuello or ""
    oro_content = ef.orofaringe or ""

    # Si ambos están vacíos, no hay nada que hacer
    if not cuello_content and not oro_content:
        return fields

    # Listas para reconstruir el contenido final
    final_cuello_phrases = []
    final_oro_phrases = []

    # --- Procesar CUELLO (buscar fugas hacia Orofaringe) ---
    if cuello_content:
        # Dividir en frases/oraciones
        phrases = [p.strip() for p in RE_SPLIT.split(cuello_content) if p.strip()]
        
        for phrase in phrases:
            has_oro_kw = bool(RE_OROFARINGE_KEYWORDS.search(phrase))
            has_cuello_kw = bool(RE_CUELLO_KEYWORDS.search(phrase))
            
            # Regla: Mover a orofaringe SI tiene keywords de orofaringe Y NO tiene de cuello
            if has_oro_kw and not has_cuello_kw:
                final_oro_phrases.append(phrase)
            else:
                final_cuello_phrases.append(phrase)
    
    # --- Procesar OROFARINGE (buscar fugas hacia Cuello) ---
    if oro_content:
        phrases = [p.strip() for p in RE_SPLIT.split(oro_content) if p.strip()]
        
        for phrase in phrases:
            has_oro_kw = bool(RE_OROFARINGE_KEYWORDS.search(phrase))
            has_cuello_kw = bool(RE_CUELLO_KEYWORDS.search(phrase))
            
            # Regla: Mover a cuello SI tiene keywords de cuello Y NO tiene de orofaringe
            if has_cuello_kw and not has_oro_kw:
                final_cuello_phrases.append(phrase)
            else:
                final_oro_phrases.append(phrase)

    # --- Reconstruir campos ---
    # Unir con ". " y asegurar mayúscula inicial si se rompió la estructura
    
    new_cuello = ". ".join(final_cuello_phrases)
    new_oro = ". ".join(final_oro_phrases)
    
    # Añadir punto final si no tiene y no está vacío
    if new_cuello and not new_cuello.strip().endswith('.'): 
        new_cuello += "."
    if new_oro and not new_oro.strip().endswith('.'):
        new_oro += "."

    ef.cuello = new_cuello if new_cuello else None
    ef.orofaringe = new_oro if new_oro else None
    
    return fields
