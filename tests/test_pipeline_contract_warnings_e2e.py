"""
E2E tests for contractWarnings flow through pipeline.

Validates that:
1. contractWarnings is ALWAYS a list (never None)
2. Warnings from contract_guard appear in final pipeline metrics
3. No warnings results in empty list (not None)

PHI-safe: No transcript content logged.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.schemas.request import Transcript, TranscriptSegment
from app.schemas.structured_fields_v1 import StructuredFieldsV1, V1ResponseMetadata


@pytest.fixture
def minimal_transcript():
    """Create minimal valid transcript for testing."""
    return Transcript(
        segments=[
            TranscriptSegment(
                speaker="doctor",
                text="Paciente con cefalea.",
                startMs=0,
                endMs=3000
            )
        ],
        durationMs=3000
    )


@pytest.fixture
def mock_settings():
    """Mock settings for pipeline."""
    mock = MagicMock()
    mock.drift_guard_mode = "warn"
    mock.drift_guard_cooldown_s = 3600
    mock.openai_compat_base_url = "http://localhost:1234/v1"
    mock.openai_compat_model = "test-model"
    mock.openai_compat_timeout_ms = 30000
    return mock


@pytest.fixture
def mock_structured_fields():
    """Create minimal valid StructuredFieldsV1 for mocking extractor."""
    return StructuredFieldsV1(
        motivo_consulta="Cefalea",
        padecimiento_actual="Dolor de cabeza desde hace 2 dias."
    )


class TestContractWarningsE2E:
    """E2E tests for contractWarnings in pipeline output."""

    @pytest.mark.asyncio
    async def test_warnings_appear_in_pipeline_metrics(
        self, minimal_transcript, mock_settings, mock_structured_fields
    ):
        """
        When contract_guard returns warnings, they must appear in metrics["contractWarnings"].
        """
        expected_warnings = ["medicalization_drift"]

        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                mock_check.return_value = {
                    "medicalizationDrift": True,
                    "normalizationDrift": False,
                    "warnings": expected_warnings,
                    "details": {
                        "medicalization": {
                            "expected": "abc123",
                            "actual": "def456",
                            "match": False
                        }
                    }
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_fields, 100, "stub-model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_fields

                        with patch("app.services.telemetry.emit_event"):
                            from app.services.pipeline_orl import run_orl_pipeline

                            fields, metrics = await run_orl_pipeline(minimal_transcript)

                            # Core assertion: warnings in metrics
                            assert metrics["contractWarnings"] == expected_warnings
                            assert isinstance(metrics["contractWarnings"], list)

    @pytest.mark.asyncio
    async def test_no_warnings_returns_empty_list_not_none(
        self, minimal_transcript, mock_settings, mock_structured_fields
    ):
        """
        When contract_guard returns no warnings, metrics["contractWarnings"] must be [] (not None).
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                mock_check.return_value = {
                    "medicalizationDrift": False,
                    "normalizationDrift": False,
                    "warnings": [],
                    "details": {}
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_fields, 100, "stub-model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_fields

                        from app.services.pipeline_orl import run_orl_pipeline

                        fields, metrics = await run_orl_pipeline(minimal_transcript)

                        # Core assertion: empty list, NOT None
                        assert metrics["contractWarnings"] == []
                        assert metrics["contractWarnings"] is not None
                        assert isinstance(metrics["contractWarnings"], list)

    @pytest.mark.asyncio
    async def test_contract_guard_exception_returns_empty_list(
        self, minimal_transcript, mock_settings, mock_structured_fields
    ):
        """
        When contract_guard raises exception, soft-fail with empty list (not None).
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                # Simulate contract guard failure
                mock_check.side_effect = RuntimeError("Contract guard crashed")

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_fields, 100, "stub-model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_fields

                        from app.services.pipeline_orl import run_orl_pipeline

                        # Pipeline should NOT crash
                        fields, metrics = await run_orl_pipeline(minimal_transcript)

                        # Core assertion: soft-fail with empty list
                        assert metrics["contractWarnings"] == []
                        assert isinstance(metrics["contractWarnings"], list)

    @pytest.mark.asyncio
    async def test_multiple_warnings_all_appear(
        self, minimal_transcript, mock_settings, mock_structured_fields
    ):
        """
        Multiple warnings from contract_guard must all appear in metrics.
        """
        expected_warnings = ["medicalization_drift", "normalization_drift"]

        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                mock_check.return_value = {
                    "medicalizationDrift": True,
                    "normalizationDrift": True,
                    "warnings": expected_warnings,
                    "details": {}
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_fields, 100, "stub-model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_fields

                        with patch("app.services.telemetry.emit_event"):
                            from app.services.pipeline_orl import run_orl_pipeline

                            fields, metrics = await run_orl_pipeline(minimal_transcript)

                            assert metrics["contractWarnings"] == expected_warnings
                            assert len(metrics["contractWarnings"]) == 2


class TestContractWarningsSchemaDefault:
    """Tests for V1ResponseMetadata.contract_warnings default value."""

    def test_schema_default_is_empty_list(self):
        """
        V1ResponseMetadata.contract_warnings should default to [] (not None).
        """
        metadata = V1ResponseMetadata(
            model_version="test-v1",
            inference_ms=100,
            request_id="req-123"
        )

        # Default should be empty list
        assert metadata.contract_warnings == []
        assert isinstance(metadata.contract_warnings, list)

    def test_schema_serializes_empty_list(self):
        """
        Empty contract_warnings should serialize to [] in JSON, not null.
        """
        metadata = V1ResponseMetadata(
            model_version="test-v1",
            inference_ms=100,
            request_id="req-123"
        )

        # Serialize with alias
        json_data = metadata.model_dump(by_alias=True)

        assert "contractWarnings" in json_data
        assert json_data["contractWarnings"] == []

    def test_schema_serializes_warnings_list(self):
        """
        Non-empty contract_warnings should serialize correctly.
        """
        metadata = V1ResponseMetadata(
            model_version="test-v1",
            inference_ms=100,
            request_id="req-123",
            contract_warnings=["medicalization_drift", "normalization_drift"]
        )

        json_data = metadata.model_dump(by_alias=True)

        assert json_data["contractWarnings"] == ["medicalization_drift", "normalization_drift"]


class TestContractWarningsNullCoercion:
    """Tests to verify null values are coerced to empty list."""

    @pytest.mark.asyncio
    async def test_contract_guard_returns_null_warnings_coerced_to_list(
        self, minimal_transcript, mock_settings, mock_structured_fields
    ):
        """
        If contract_guard returns {"warnings": None}, pipeline should coerce to [].
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                # Simulate malformed response with None
                mock_check.return_value = {
                    "medicalizationDrift": False,
                    "normalizationDrift": False,
                    "warnings": None,  # <-- Malformed!
                    "details": {}
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_fields, 100, "stub-model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_fields

                        from app.services.pipeline_orl import run_orl_pipeline

                        fields, metrics = await run_orl_pipeline(minimal_transcript)

                        # Pipeline should coerce None to []
                        assert metrics["contractWarnings"] == []
                        assert metrics["contractWarnings"] is not None

    @pytest.mark.asyncio
    async def test_contract_guard_missing_warnings_key_coerced_to_list(
        self, minimal_transcript, mock_settings, mock_structured_fields
    ):
        """
        If contract_guard returns dict without 'warnings' key, pipeline should default to [].
        """
        with patch("app.services.pipeline_orl.get_settings", return_value=mock_settings):
            with patch("app.contracts.contract_guard.check_contracts") as mock_check:
                # Simulate response missing 'warnings' key entirely
                mock_check.return_value = {
                    "medicalizationDrift": False,
                    "normalizationDrift": False,
                    "details": {}
                    # 'warnings' key missing!
                }

                with patch("app.services.pipeline_orl.extract_structured_v1") as mock_extract:
                    mock_extract.return_value = (mock_structured_fields, 100, "stub-model-v1")

                    with patch("app.services.pipeline_orl._finalize_refine_fields") as mock_finalize:
                        mock_finalize.return_value = mock_structured_fields

                        from app.services.pipeline_orl import run_orl_pipeline

                        fields, metrics = await run_orl_pipeline(minimal_transcript)

                        # Pipeline should default to []
                        assert metrics["contractWarnings"] == []
