import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.schemas.structured_fields_v1 import StructuredFieldsV1
from app.services.pipeline_orl import _finalize_refine_fields

@pytest.fixture
def mock_settings():
    with patch("app.services.pipeline_orl.get_settings") as mock:
        mock.return_value.openai_compat_base_url = "http://test"
        mock.return_value.openai_compat_model = "test-model"
        yield mock

@pytest.mark.asyncio
async def test_rewrite_hpi_success(mock_settings):
    """Test successful rewrite of HPI."""
    fields = StructuredFieldsV1(
        padecimiento_actual="Dolor de oido derecho desde hace 3 dias, intensidad 7/10."
    )
    
    # ... rest of validation

    
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "Otalgia derecha de 3 días de evolución, EVA 7/10."
                }
            }
        ]
    }
    
    
    
    # Define simple mock class for Async Context Manager and Client
    class MockClientContext:
        async def __aenter__(self):
             return self
        async def __aexit__(self, exc_type, exc, tb):
             pass
        
        async def post(self, *args, **kwargs):
             # Return a mock response object
             m = MagicMock()
             m.json.return_value = mock_response
             # raise_for_status is sync
             m.raise_for_status = MagicMock()
             return m

    with patch("app.services.pipeline_orl.httpx") as mock_httpx:
        # Create instance of our manual mock
        mock_instance = MockClientContext()
        
        # When AsyncClient() is called, return our mock instance
        mock_httpx.AsyncClient.return_value = mock_instance
        
        # Need to spy on 'post' to verify calls later?
        # We can wrap it or just rely on logic for now. 
        # But to use assert called, we need a spy.
        # Let's use a side_effect on a MagicMock if we want to assert.
        
        # Simpler approaches for verification:
        # Create a MagicMock that we call inside the async function
        post_spy = MagicMock(return_value=MagicMock())
        post_spy.return_value.json.return_value = mock_response
        
        async def mock_post(*args, **kwargs):
             return post_spy(*args, **kwargs)
             
        mock_instance.post = mock_post

        
        result = await _finalize_refine_fields(fields)
        
        # Should be rewritten
        assert result.padecimiento_actual == "Otalgia derecha de 3 días de evolución, EVA 7/10."
        
        # Verify
        assert post_spy.called
        call_kwargs = post_spy.call_args[1]
        payload = call_kwargs["json"]
        assert "INPUT_HPI" in payload["messages"][1]["content"]
        assert "Dolor de oido derecho" in payload["messages"][1]["content"]
        # System prompt should be present
        assert "Eres médico ORL" in payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_rewrite_hpi_null_or_empty(mock_settings):
    """Test skippping rewrite if HPI is null or empty."""
    # Case 1: None
    fields_none = StructuredFieldsV1(padecimiento_actual=None)
    
    with patch("app.services.pipeline_orl.httpx") as mock_httpx:
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client
        
        result = await _finalize_refine_fields(fields_none)
        
        # Should not have called LLM
        mock_client.post.assert_not_called()
        assert result.padecimiento_actual is None

    # Case 2: Empty string
    fields_empty = StructuredFieldsV1(padecimiento_actual="   ")
    
    with patch("app.services.pipeline_orl.httpx") as mock_httpx:
        # Note: Using different variable names or new context ensures fresh mock
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client
        
        result = await _finalize_refine_fields(fields_empty)
        
        mock_client.post.assert_not_called()
        assert result.padecimiento_actual == "   " 


@pytest.mark.asyncio
async def test_rewrite_hpi_fallback_on_error(mock_settings):
    """Test fallback to original text if LLM call fails."""
    fields = StructuredFieldsV1(padecimiento_actual="Original text")
    
    with patch("app.services.pipeline_orl.httpx") as mock_httpx:
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client
        # Simulate network error
        mock_client.post.side_effect = Exception("Network error")
        
        result = await _finalize_refine_fields(fields)
        
        # Should keep original text
        assert result.padecimiento_actual == "Original text"


@pytest.mark.asyncio
async def test_rewrite_hpi_fallback_on_empty_output(mock_settings):
    """Test fallback if LLM returns empty string."""
    fields = StructuredFieldsV1(padecimiento_actual="Original text")
    
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "   " # Empty response
                }
            }
        ]
    }
    
    with patch("app.services.pipeline_orl.httpx") as mock_httpx:
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value.json.return_value = mock_response
        
        result = await _finalize_refine_fields(fields)
        
        # Should keep original text
        assert result.padecimiento_actual == "Original text"
