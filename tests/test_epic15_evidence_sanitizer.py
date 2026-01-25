"""
Unit tests for Epic 15 Evidence Sanitizer.
Tests PHI-safe sanitization of evidence snippets.
"""
import pytest
from app.services.evidence_sanitizer import (
    sanitize_evidence,
    sanitize_evidence_list,
    is_potentially_phi,
    MAX_EVIDENCE_LENGTH,
)


class TestSanitizeEvidence:
    """Tests for sanitize_evidence function."""

    def test_empty_string_returns_empty(self):
        assert sanitize_evidence("") == ""
        assert sanitize_evidence(None) == ""

    def test_short_text_unchanged(self):
        text = "Dolor de garganta por 5 dias"
        result = sanitize_evidence(text)
        assert result == text

    def test_truncates_long_text(self):
        long_text = "A" * 200
        result = sanitize_evidence(long_text)
        assert len(result) <= MAX_EVIDENCE_LENGTH
        assert result.endswith("...")

    def test_normalizes_whitespace(self):
        text = "Dolor   de   garganta\n\tpor   5   dias"
        result = sanitize_evidence(text)
        assert result == "Dolor de garganta por 5 dias"

    def test_replaces_full_names(self):
        # Names preceded by explicit markers like "paciente" are detected
        text = "El paciente Juan Carlos Perez presenta dolor"
        result = sanitize_evidence(text)
        assert "Juan Carlos Perez" not in result
        assert "[NOMBRE]" in result

    def test_replaces_curp(self):
        text = "CURP: PEGJ850101HDFRRL09 verificado"
        result = sanitize_evidence(text)
        assert "PEGJ850101HDFRRL09" not in result
        assert "[CURP]" in result

    def test_replaces_phone_numbers(self):
        text = "Contacto: 55-1234-5678 o 5512345678"
        result = sanitize_evidence(text)
        assert "55-1234-5678" not in result
        assert "5512345678" not in result
        assert "[TEL]" in result

    def test_replaces_email(self):
        text = "Email: paciente@ejemplo.com para contacto"
        result = sanitize_evidence(text)
        assert "paciente@ejemplo.com" not in result
        assert "[EMAIL]" in result

    def test_replaces_dates(self):
        text = "Fecha de nacimiento: 15/03/1985"
        result = sanitize_evidence(text)
        assert "15/03/1985" not in result
        assert "[FECHA]" in result

    def test_replaces_patient_ids(self):
        # ID pattern requires explicit keyword + number
        text = "Expediente No. 123456789 del paciente"
        result = sanitize_evidence(text)
        assert "[ID]" in result

    def test_clinical_text_preserved(self):
        """Clinical content without PII should be preserved."""
        text = "Amigdalas hiperhemicas grado II con exudado"
        result = sanitize_evidence(text)
        assert result == text

    def test_multiple_pii_types(self):
        # Multiple PII types detected together
        text = "Tel 5512345678, nac 01/01/1990"
        result = sanitize_evidence(text)
        assert "5512345678" not in result
        assert "01/01/1990" not in result
        assert "[TEL]" in result
        assert "[FECHA]" in result


class TestSanitizeEvidenceList:
    """Tests for sanitize_evidence_list function."""

    def test_empty_list(self):
        assert sanitize_evidence_list([]) == []

    def test_filters_empty_strings(self):
        result = sanitize_evidence_list(["texto", "", "otro", None])
        assert len(result) == 2
        assert "texto" in result
        assert "otro" in result

    def test_sanitizes_each_item(self):
        # Use patterns that will be detected
        texts = ["Tel 5512345678 de contacto", "Texto normal"]
        result = sanitize_evidence_list(texts)
        assert "[TEL]" in result[0]
        assert result[1] == "Texto normal"


class TestIsPotentiallyPhi:
    """Tests for is_potentially_phi function."""

    def test_clean_text_returns_false(self):
        assert not is_potentially_phi("Dolor de garganta")
        assert not is_potentially_phi("Amigdalitis aguda")

    def test_name_returns_true(self):
        # Names preceded by explicit markers are detected
        assert is_potentially_phi("Paciente Juan Carlos Perez")

    def test_phone_returns_true(self):
        assert is_potentially_phi("Tel: 5512345678")

    def test_email_returns_true(self):
        assert is_potentially_phi("email@ejemplo.com")

    def test_empty_returns_false(self):
        assert not is_potentially_phi("")
        assert not is_potentially_phi(None)
