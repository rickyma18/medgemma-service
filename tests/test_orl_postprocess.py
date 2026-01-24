import pytest
from app.schemas.structured_fields_v1 import StructuredFieldsV1, ExploracionFisica
from app.services.orl_postprocess import postprocess_orl_mapping

def test_move_cuello_to_orofaringe():
    """
    Test: Frases en Cuello con keywords de Orofaringe (placas, pus) deben moverse a Orofaringe.
    """
    fields = StructuredFieldsV1(
        exploracionFisica=ExploracionFisica(
            cuello="No adenopatías. Se observan placas blanquecinas en amígdalas.",
            orofaringe="Hiperémica."
        )
    )
    
    processed = postprocess_orl_mapping(fields)
    
    cuello = processed.exploracion_fisica.cuello
    oro = processed.exploracion_fisica.orofaringe
    
    # Cuello debe conservar lo suyo
    assert "No adenopatías" in cuello
    # Cuello NO debe tener placas ni amigdalas
    assert "placas" not in cuello
    assert "amígdalas" not in cuello
    
    # Orofaringe debe recibir lo movido
    assert "placas blanquecinas en amígdalas" in oro
    assert "Hiperémica" in oro

def test_move_orofaringe_to_cuello():
    """
    Test: Frases en Orofaringe con keywords de Cuello (ganglios, adenopatías) deben moverse a Cuello.
    """
    fields = StructuredFieldsV1(
        exploracionFisica=ExploracionFisica(
            cuello=None,
            orofaringe="Faringe normal. Se palpa ganglio submandibular doloroso."
        )
    )
    
    processed = postprocess_orl_mapping(fields)
    
    cuello = processed.exploracion_fisica.cuello
    oro = processed.exploracion_fisica.orofaringe
    
    # Orofaringe pierde el ganglio
    assert "Faringe normal" in oro
    assert "ganglio" not in oro
    
    # Cuello gana el ganglio
    assert "ganglio submandibular doloroso" in cuello

def test_no_change_if_correct():
    """
    Test: Si todo está en su lugar, no debe haber cambios.
    """
    fields = StructuredFieldsV1(
        exploracionFisica=ExploracionFisica(
            cuello="Cuello cilíndrico sin adenopatías.",
            orofaringe="Amígdalas grado I sin exudado."
        )
    )
    
    # Copia para comparar
    original_cuello = fields.exploracion_fisica.cuello
    original_oro = fields.exploracion_fisica.orofaringe
    
    processed = postprocess_orl_mapping(fields)
    
    assert processed.exploracion_fisica.cuello == original_cuello
    assert processed.exploracion_fisica.orofaringe == original_oro

def test_mixed_sentences_split():
    """
    Test: Separación correcta de frases por puntuación.
    """
    # Caso complejo: en cuello mezclado.
    input_cuello = "Sin adenopatías; amígdalas cripticas. Cuello móvil."
    
    fields = StructuredFieldsV1(
        exploracionFisica=ExploracionFisica(
            cuello=input_cuello,
            orofaringe=None
        )
    )
    
    processed = postprocess_orl_mapping(fields)
    
    cuello = processed.exploracion_fisica.cuello
    oro = processed.exploracion_fisica.orofaringe
    
    # Cuello conserva: "Sin adenopatías", "Cuello móvil"
    assert "Sin adenopatías" in cuello
    assert "Cuello móvil" in cuello
    assert "amígdalas" not in cuello
    
    # Orofaringe recibe: "amígdalas cripticas"
    assert "amígdalas cripticas" in oro
