
import pytest
from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1, ExploracionFisica, Antecedentes, Diagnostico
)
from app.services.aggregator import aggregate_structured_fields_v1

def test_aggregator_strings():
    v1 = StructuredFieldsV1(motivoConsulta="Dolor", padecimientoActual="Inicio ayer.")
    v2 = StructuredFieldsV1(motivoConsulta="Fiebre", padecimientoActual="Inicio ayer.") # Duplicate padecimiento
    v3 = StructuredFieldsV1(motivoConsulta=None, padecimientoActual="Empeoro hoy.")
    
    merged = aggregate_structured_fields_v1([v1, v2, v3])
    
    # Motivo: concat "Dolor | Fiebre"
    assert "Dolor" in merged.motivo_consulta
    assert "Fiebre" in merged.motivo_consulta
    assert " | " in merged.motivo_consulta
    
    # Padecimiento: dedupe "Inicio ayer." + "Empeoro hoy."
    # Should contain "Inicio ayer. | Empeoro hoy." (order preserved)
    assert merged.padecimiento_actual == "Inicio ayer. | Empeoro hoy."

def test_aggregator_nested_objects():
    # Antecedentes merge
    v1 = StructuredFieldsV1()
    v1.antecedentes.heredofamiliares = "Padre DM2"
    
    v2 = StructuredFieldsV1()
    v2.antecedentes.personales_patologicos = "Asma"
    
    merged = aggregate_structured_fields_v1([v1, v2])
    
    assert merged.antecedentes.heredofamiliares == "Padre DM2"
    assert merged.antecedentes.personales_patologicos == "Asma"

def test_aggregator_diagnostico_prio():
    d1 = Diagnostico(texto="Gripe", tipo="presuntivo")
    d2 = Diagnostico(texto="Influenza A", tipo="definitivo")
    d3 = Diagnostico(texto="Tos", tipo="sindromico")
    
    v1 = StructuredFieldsV1(diagnostico=d1)
    v2 = StructuredFieldsV1(diagnostico=d2)
    v3 = StructuredFieldsV1(diagnostico=d3)
    
    merged = aggregate_structured_fields_v1([v1, v2, v3])
    
    assert merged.diagnostico is not None
    # Tipo definitivo wins
    assert merged.diagnostico.tipo == "definitivo"
    # Texts concatenated
    assert "Gripe" in merged.diagnostico.texto
    assert "Influenza A" in merged.diagnostico.texto

def test_aggregator_empty_input():
    merged = aggregate_structured_fields_v1([])
    assert isinstance(merged, StructuredFieldsV1)
    assert merged.motivo_consulta is None

def test_aggregator_single_input():
    v1 = StructuredFieldsV1(motivoConsulta="Single")
    merged = aggregate_structured_fields_v1([v1])
    assert merged == v1
