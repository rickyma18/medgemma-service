import json
import pytest
from app.schemas.structured_fields_v1 import (
    StructuredFieldsV1,
    Antecedentes,
    ExploracionFisica,
    Diagnostico
)

# == Fixtures without PHI ==

@pytest.fixture
def empty_structured_fields():
    """Returns an empty StructuredFieldsV1 (all defaults)."""
    return StructuredFieldsV1()

@pytest.fixture
def full_structured_fields():
    """Returns a fully populated StructuredFieldsV1 with dummy non-PHI data."""
    return StructuredFieldsV1(
        motivo_consulta="dolor de garganta",
        padecimiento_actual="El paciente refiere dolor desde ayer.",
        antecedentes=Antecedentes(
            heredofamiliares="Madre con diabetes",
            personales_no_patologicos="Niega tabaquismo",
            personales_patologicos="Alergia a penicilina"
        ),
        exploracion_fisica=ExploracionFisica(
            signos_vitales="TA 120/80",
            rinoscopia="Normal",
            orofaringe="Hiperemia",
            cuello="Sin adenopatias",
            laringoscopia="No realizada",
            otoscopia="Membrana integra",
            otomicroscopia="No realizada",
            endoscopia_nasal="No realizada"
        ),
        diagnostico=Diagnostico(
            texto="Faringitis aguda",
            tipo="presuntivo",
            cie10="J02.9"
        ),
        plan_tratamiento="Paracetamol 500mg cada 8 horas",
        pronostico="Bueno para la vida",
        estudios_indicados="Ninguno",
        notas_adicionales="Revisar en 3 dias"
    )

@pytest.fixture
def garbage_input_dict():
    """Dictionary satisfying minimum required fields but mostly garbage/extra keys."""
    return {
        "motivoConsulta": None,
        "unexpected_key": "Should not be here",
        "conflicts": ["conflict1"],
        "evidence": {"internal": "data"},
        # Diagnostico inside logic might require checks if it's optional in schema?
        # In schema: diagnostico: Optional[Diagnostico] = None
        # So it can be missing or None.
    }

# == Tests ==

def test_root_keys_expected_shape(full_structured_fields):
    """
    Validation 1: Confirm expected root keys in camelCase when by_alias=True.
    Validates compatibility with Flutter structured_fields_schema_v1.dart.
    """
    dump = full_structured_fields.model_dump(by_alias=True, exclude_none=False)
    
    expected_keys = {
        "motivoConsulta",
        "padecimientoActual",
        "antecedentes",
        "exploracionFisica",
        "diagnostico",
        "planTratamiento",
        "pronostico",
        "estudiosIndicados",
        "notasAdicionales"
    }
    
    current_keys = set(dump.keys())
    
    # Check that all expected keys are present
    missing = expected_keys - current_keys
    assert not missing, f"Missing expected root keys: {missing}"
    
    # Optional: Verify no extra keys if we want strictness, 
    # but the primary goal is ensuring the expected contract is met.
    # The user instruction 3 says "No filtrar keys internas", implying we SHOULD NOT see internal keys.
    # So checking for absence of internal keys happens in another test or implicitly here if we check exact keys.
    
def test_default_empty_schema_validity(empty_structured_fields):
    """
    Validation 2: Default empty schema.
    Ensure strict shape even if everything is None/Default.
    """
    dump = empty_structured_fields.model_dump(by_alias=True, exclude_none=False)
    
    # Verify root structure exists
    assert "antecedentes" in dump
    assert "exploracionFisica" in dump
    assert "motivoConsulta" in dump
    
    # Verify nested objects are present even if empty (default_factory used in schema)
    assert isinstance(dump["antecedentes"], dict)
    assert isinstance(dump["exploracionFisica"], dict)
    
    # Diagnostico is optional and defaults to None
    assert dump["diagnostico"] is None

def test_no_internal_keys_leak(garbage_input_dict):
    """
    Validation 3: No internal keys (conflicts, evidence, etc) should leak into output.
    Simulates passing a 'dirty' dict (like from reducer) into the model.
    """
    # Pydantic v2 ignores extra fields by default, ensuring they are stripped.
    model = StructuredFieldsV1(**garbage_input_dict)
    dump = model.model_dump(by_alias=True, exclude_none=False)
    
    forbidden_substrings = ["conflict", "evidence", "marker", "internal", "intermediate"]
    
    for key in dump.keys():
        for forbidden in forbidden_substrings:
            assert forbidden not in key, f"Found forbidden internal key '{key}' in output"

def test_serialization_determinism(full_structured_fields):
    """
    Validation 4: Stability. Same inputs => Identical Dict/JSON.
    """
    dump1 = full_structured_fields.model_dump(by_alias=True, exclude_none=False)
    dump2 = full_structured_fields.model_dump(by_alias=True, exclude_none=False)
    
    assert dump1 == dump2
    
    # Check JSON string stability (key order)
    # Pydantic model_dump_json uses consistent ordering by field definition
    json1 = full_structured_fields.model_dump_json(by_alias=True, exclude_none=False)
    json2 = full_structured_fields.model_dump_json(by_alias=True, exclude_none=False)
    
    assert json1 == json2

def test_nested_aliases(full_structured_fields):
    """
    Extra check: Verify nested aliases (e.g. personalesNoPatologicos) are respected.
    """
    dump = full_structured_fields.model_dump(by_alias=True, exclude_none=False)
    
    antecedentes = dump["antecedentes"]
    assert "personalesNoPatologicos" in antecedentes
    assert "personalesPatologicos" in antecedentes
    # "heredofamiliares" has no alias so it remains snake_case ?? 
    # Let's check schema: heredofamiliares: Optional[str] = Field(default=None...)
    # No alias defined. So it should be "heredofamiliares".
    assert "heredofamiliares" in antecedentes
    
    exploracion = dump["exploracionFisica"]
    assert "signosVitales" in exploracion
    assert "endoscopiaNasal" in exploracion
