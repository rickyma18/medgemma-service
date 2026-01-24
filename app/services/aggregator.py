"""
Módulo para agregar (merge) múltiples resultados de StructuredFieldsV1.
Estrategia determinística:
- Strings (campos libres): Concatenar " | " + Dedupe
- Listas: Union + Dedupe
- Objetos (Exploración, Antecedentes): Union de campos. Si conflicto -> Concatenar.
- Diagnóstico: Conservar el más específico (ej. definitivo > presuntivo > sindromico) o lista.
  NOTA: En V1 diagnostico es single. Estrategia: "Append" textos si diferentes.
"""
from typing import List, Optional, Any, Dict
from copy import deepcopy
from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1, 
    ExploracionFisica, 
    Antecedentes, 
    Diagnostico
)

SEPARATOR = " | "

def _normalize_str(s: str) -> str:
    return s.strip().lower() if s else ""

def _merge_str_fields(values: List[str]) -> Optional[str]:
    """Combina lista de strings, eliminando duplicados y Nones."""
    valid_values = [v for v in values if v and v.strip()]
    if not valid_values:
        return None
    
    # Dedupe preservando orden
    seen = set()
    unique = []
    for val in valid_values:
        norm = _normalize_str(val)
        if norm not in seen:
            seen.add(norm)
            unique.append(val.strip())
            
    if not unique:
        return None
    return SEPARATOR.join(unique)

def _merge_exploracion(items: List[ExploracionFisica]) -> ExploracionFisica:
    """Merge de sub-objeto ExploracionFisica."""
    # Como todos son strings opcionales, usamos _merge_str_fields para cada campo
    merged = ExploracionFisica()
    
    fields = merged.model_dump().keys()
    for f in fields:
        vals = [getattr(item, f) for item in items]
        setattr(merged, f, _merge_str_fields(vals))
        
    return merged

def _merge_antecedentes(items: List[Antecedentes]) -> Antecedentes:
    """Merge de sub-objeto Antecedentes."""
    merged = Antecedentes()
    fields = merged.model_dump().keys()
    
    for f in fields:
        vals = [getattr(item, f) for item in items]
        setattr(merged, f, _merge_str_fields(vals))
        
    return merged

def _merge_diagnostico(items: List[Optional[Diagnostico]]) -> Optional[Diagnostico]:
    """
    Merge de Diagnostico.
    Si hay multiple diagnosticos, concatenar texto.
    Tipo: tomar el de mayor certeza: definitivo > presuntivo > sindromico
    CIE10: tomar el primero no nulo o concat.
    """
    valid_items = [x for x in items if x]
    if not valid_items:
        return None

    # Merge Texto
    textos = [x.texto for x in valid_items]
    final_texto = _merge_str_fields(textos)
    if not final_texto:
        return None # Should not happen if items valid
        
    # Merge CIE10
    cies = [x.cie10 for x in valid_items]
    final_cie = _merge_str_fields(cies)
    
    # Merge Tipo (Hierarchy)
    priorities = {"definitivo": 3, "presuntivo": 2, "sindromico": 1}
    best_tipo = "sindromico"
    max_prio = 0
    
    for x in valid_items:
        p = priorities.get(x.tipo, 0)
        if p > max_prio:
            max_prio = p
            best_tipo = x.tipo
            
    return Diagnostico(
        texto=final_texto,
        tipo=best_tipo, # type: ignore
        cie10=final_cie
    )

def aggregate_structured_fields_v1(results: List[StructuredFieldsV1]) -> StructuredFieldsV1:
    """
    Reduce una lista de resultados parciales en uno solo unificado.
    """
    if not results:
        return StructuredFieldsV1() # Empty
        
    if len(results) == 1:
        return results[0]
        
    # Agrupar valores
    motivos = [r.motivo_consulta for r in results]
    padecimientos = [r.padecimiento_actual for r in results]
    
    antecedentes_list = [r.antecedentes for r in results]
    exploracion_list = [r.exploracion_fisica for r in results]
    dx_list = [r.diagnostico for r in results]
    
    planes = [r.plan_tratamiento for r in results]
    pronosticos = [r.pronostico for r in results]
    estudios = [r.estudios_indicados for r in results]
    notas = [r.notas_adicionales for r in results]
    
    merged = StructuredFieldsV1(
        motivoConsulta=_merge_str_fields(motivos),
        padecimientoActual=_merge_str_fields(padecimientos),
        antecedentes=_merge_antecedentes(antecedentes_list),
        exploracionFisica=_merge_exploracion(exploracion_list),
        diagnostico=_merge_diagnostico(dx_list),
        planTratamiento=_merge_str_fields(planes),
        pronostico=_merge_str_fields(pronosticos),
        estudiosIndicados=_merge_str_fields(estudios),
        notasAdicionales=_merge_str_fields(notas)
    )
    
    return merged
