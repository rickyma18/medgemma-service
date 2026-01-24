import pytest
from app.schemas.request import Transcript, TranscriptSegment
from app.services.text_normalizer_orl import normalize_transcript_orl

def test_normalize_orl_basic_replacements():
    """Prueba reemplazos basicos de la whitelist."""
    # input con typos tipicos
    # migda -> amígdala (singular)
    # exsudado -> exudado
    # vertigo -> vértigo
    text = "La migda derecha tiene exsudado y el paciente siente vertigo."
    
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                speaker="doctor",
                text=text,
                startMs=0,
                endMs=1000
            )
        ],
        durationMs=1000,
        language="es"
    )
    
    normalized, count = normalize_transcript_orl(transcript)
    
    assert count == 3
    new_text = normalized.segments[0].text
    
    # Verificar correcciones exactas
    assert "amígdala" in new_text
    assert "amígdalas" not in new_text  # Check NO plural intrusion
    assert "/" not in new_text # Check NO slash intrusion from bug
    assert "exudado" in new_text
    assert "vértigo" in new_text
    
    # El texto exacto esperado
    assert new_text == "La amígdala derecha tiene exudado y el paciente siente vértigo."

def test_normalize_orl_plurals_and_gender():
    """Prueba variantes de plural y genero especificas."""
    # migdalas -> amígdalas
    # faringea -> faríngea
    text = "Las migdalas estan faringea." 
    
    transcript = Transcript(
        segments=[TranscriptSegment(speaker="doctor", text=text, startMs=0, endMs=1000)],
        durationMs=1000
    )
    
    normalized, count = normalize_transcript_orl(transcript)
    
    assert count == 2
    new_text = normalized.segments[0].text
    
    # Verificar texto exacto para evitar falsos positivos por substrings (ej. amígdala dentro de amígdalas)
    assert new_text == "Las amígdalas estan faríngea."
    assert "amígdalas" in new_text
    assert "faríngea" in new_text

def test_normalize_orl_case_insensitive_and_boundaries():
    """Prueba case insensitivity y bordes de palabra."""
    # "timpanica" -> timpánica, "NASO FARINGE" -> nasofaringe
    text = "Membrana timpanica integra. OBSERVA NASO FARINGE libre."
    
    transcript = Transcript(
        segments=[
            TranscriptSegment(speaker="doctor", text=text, startMs=0, endMs=1000)
        ],
        durationMs=1000
    )
    
    normalized, count = normalize_transcript_orl(transcript)
    
    assert count == 2
    new_text = normalized.segments[0].text
    assert "timpánica" in new_text
    assert "nasofaringe" in new_text

def test_normalize_orl_multiple_segments():
    """Prueba normalizacion a traves de multiples segmentos."""
    segments = [
        TranscriptSegment(speaker="doctor", text="Tiene rinoria.", startMs=0, endMs=100),
        TranscriptSegment(speaker="patient", text="Y acufenos.", startMs=100, endMs=200)
    ]
    transcript = Transcript(segments=segments, durationMs=200)
    
    normalized, count = normalize_transcript_orl(transcript)
    
    assert count == 2
    assert "rinorrea" in normalized.segments[0].text
    assert "acúfenos" in normalized.segments[1].text
