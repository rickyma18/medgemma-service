import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from app.schemas.request import Transcript, TranscriptSegment
from app.services.pipeline_orl import run_orl_pipeline, _fallback_to_baseline

# Minimal factory
def make_transcript():
    return Transcript(
        segments=[TranscriptSegment(text="test", start_ms=0, end_ms=1000, speaker="doctor")],
        duration_ms=1000
    )

@pytest.fixture
def mock_deps():
    """
    Mocks for dependencies of run_orl_pipeline.
    We patch where the objects are IMPORTED from if they are imported inside functions,
    or in the module if imported at top level.
    """
    with patch("app.services.pipeline_orl.get_settings") as mock_settings, \
         patch("app.contracts.contract_guard.check_contracts") as mock_contracts, \
         patch("app.services.telemetry.emit_event") as mock_emit, \
         patch("app.services.transcript_cleaner.clean_transcript") as mock_clean, \
         patch("app.services.chunking.chunk_transcript") as mock_chunk, \
         patch("app.services.pipeline_orl.extract_chunk_lite") as mock_lite, \
         patch("app.services.aggregator.aggregate_chunk_results") as mock_agg, \
         patch("app.services.pipeline_orl._finalize_refine_fields", new_callable=MagicMock) as mock_finalize, \
         patch("app.services.pipeline_orl._fallback_to_baseline", new_callable=AsyncMock) as mock_fallback, \
         patch("app.services.medicalization.medicalization_service.apply_medicalization") as mock_med, \
         patch("app.services.text_normalizer_orl.normalize_transcript_orl") as mock_norm:
         
        # Basic Success Setup
        mock_settings.return_value.chunking_enabled = False
        mock_settings.return_value.map_extractor_mode = "lite"
        mock_settings.return_value.drift_guard_mode = "warn"
        mock_settings.return_value.drift_guard_cooldown_s = 60
        
        # Stages
        mock_med.return_value = ("test", {})
        mock_norm.return_value = (make_transcript(), 0)
        mock_clean.return_value = make_transcript()
        mock_chunk.return_value = [make_transcript()]
        
        # Async mocks need return_value setting
        mock_lite.return_value = (MagicMock(), 0)
        mock_agg.return_value = (MagicMock(), {})
        
        async def async_finalize(x): return x
        mock_finalize.side_effect = async_finalize
        
        yield {
            "settings": mock_settings,
            "contracts": mock_contracts,
            "fallback": mock_fallback,
            "finalize": mock_finalize
        }

@pytest.mark.asyncio
async def test_contract_status_ok(mock_deps):
    """Case A: ok when warnings=[]"""
    mock_deps["contracts"].return_value = {"warnings": [], "details": None}
    
    fields, metrics = await run_orl_pipeline(make_transcript())
    
    assert metrics["contractWarnings"] == []
    assert metrics["contractStatus"] == "ok"

@pytest.mark.asyncio
async def test_contract_status_warning_benign_safe(mock_deps):
    """Case B: warning when warnings present (no drift) in SAFE mode.
    Should NOT force fallback because it is not real drift."""
    # Warning without DRIFT: prefix
    mock_deps["contracts"].return_value = {
        "warnings": ["medicalization_snapshot_missing"], "details": {}
    }
    mock_deps["settings"].return_value.drift_guard_mode = "safe"
    
    fields, metrics = await run_orl_pipeline(make_transcript())
    
    assert metrics["contractWarnings"] == ["medicalization_snapshot_missing"]
    assert metrics["contractStatus"] == "warning"
    
    # Fallback should NOT be called because it is not real drift
    mock_deps["fallback"].assert_not_called()

@pytest.mark.asyncio
async def test_contract_status_drift_safe(mock_deps):
    """Case C: drift when safe mode triggers fallback on REAL drift."""
    # Warning WITH DRIFT: prefix
    mock_deps["contracts"].return_value = {"warnings": ["DRIFT:medicalization_drift"]}
    mock_deps["settings"].return_value.drift_guard_mode = "safe"
    
    # Simulate fallback behavior: return fields + metrics
    fallback_result = (MagicMock(), {"status": "fallback_metrics"})
    mock_deps["fallback"].return_value = fallback_result
    
    fields, metrics = await run_orl_pipeline(make_transcript())
    
    # Assert return is what fallback returned
    assert metrics == {"status": "fallback_metrics"}
    
    # Assert fallback called with contractStatus="drift" in metrics
    mock_deps["fallback"].assert_called_once()
    call_args = mock_deps["fallback"].call_args
    passed_metrics = call_args[0][2] # 3rd arg is metrics
    
    assert passed_metrics["contractStatus"] == "drift"
    assert passed_metrics["contractWarnings"] == ["DRIFT:medicalization_drift"]

@pytest.mark.asyncio
async def test_contract_status_drift_warn(mock_deps):
    """Case D: drift present but mode is WARN -> Warning status, NO fallback."""
    mock_deps["contracts"].return_value = {"warnings": ["DRIFT:medicalization_drift"]}
    mock_deps["settings"].return_value.drift_guard_mode = "warn"
    
    fields, metrics = await run_orl_pipeline(make_transcript())
    
    assert metrics["contractStatus"] == "warning"
    assert metrics["contractWarnings"] == ["DRIFT:medicalization_drift"]
    
    mock_deps["fallback"].assert_not_called()

