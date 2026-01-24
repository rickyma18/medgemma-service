"""
Módulo para dividir transcripciones largas en chunks manejables.
PHI-Safe: No loguea contenido de texto.
"""
from typing import List
from app.schemas.request import Transcript

# Default chunk duration: 5 minutes (300,000 ms)
DEFAULT_MAX_CHUNK_DURATION_MS = 300000

def chunk_transcript(
    transcript: Transcript, 
    max_duration_ms: int = DEFAULT_MAX_CHUNK_DURATION_MS
) -> List[Transcript]:
    """
    Divide un Transcript en una lista de Transcripts más pequeños.
    
    Reglas:
    - Respeta fronteras de segmentos (no corta texto a la mitad).
    - Mantiene metadata original (language, etc.).
    - Recalcula durationMs para cada chunk.
    - Si el transcript es corto, devuelve una lista con el transcript original.
    
    Args:
        transcript: Transcript completo.
        max_duration_ms: Duración máxima aproximada de cada chunk en ms.
        
    Returns:
        List[Transcript]: Lista de chunks.
    """
    # Si no hay segmentos, devolver tal cual
    if not transcript.segments:
        return [transcript]

    # Calcular duración real total basada en el último segmento si no coincide
    last_end = transcript.segments[-1].end_ms
    total_duration = max(transcript.duration_ms, last_end)
    
    # Caso 1: Duración total dentro del límite -> 1 chunk
    if total_duration <= max_duration_ms:
        return [transcript]

    chunks: List[Transcript] = []
    
    current_segments = []
    current_chunk_start_ms = transcript.segments[0].start_ms
    
    # Iterar y agrupar segmentos
    for segment in transcript.segments:
        # Calcular dónde terminaría este segmento relativo al inicio del chunk actual
        # Usamos end_ms absoluto para decidir el corte
        
        # Si agregar este segmento excede el max_duration desde el inicio del chunk...
        # Y YA tenemos segmentos acumulados (para no dejar chunks vacíos si un solo seg es enorme)
        if (segment.end_ms - current_chunk_start_ms > max_duration_ms) and current_segments:
            # Cerrar chunk actual
            chunk_duration = current_segments[-1].end_ms - current_segments[0].start_ms
            
            new_chunk = Transcript(
                segments=current_segments,
                language=transcript.language,
                durationMs=chunk_duration  # type: ignore (alias populated by validator or init)
            )
            chunks.append(new_chunk)
            
            # Reset para el siguiente
            current_segments = []
            current_chunk_start_ms = segment.start_ms
            
        current_segments.append(segment)
    
    # Agregar último chunk remanente
    if current_segments:
        chunk_duration = current_segments[-1].end_ms - current_segments[0].start_ms
        new_chunk = Transcript(
            segments=current_segments,
            language=transcript.language,
            durationMs=chunk_duration # type: ignore
        )
        chunks.append(new_chunk)

    return chunks
