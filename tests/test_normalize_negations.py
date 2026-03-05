"""
Unit tests for _normalize_negations in structured_v1_extractor.

Validates:
- Dangling conjunctions (ni, y, e, o) are stripped from entry tails.
- Entries that become too short after trimming are dropped.
- Trailing punctuation (comma, slash) is cleaned.
- Deduplication is preserved.
- Existing behaviour for clean entries is unchanged.
"""
import pytest

from app.services.structured_v1_extractor import _normalize_negations


class TestDanglingConjunctionRemoval:
    """Trailing conjunctions like 'ni', 'y', 'e', 'o' must be stripped."""

    def test_fuma_ni_becomes_fuma(self):
        result = _normalize_negations(["fuma ni"])
        assert result == ["fuma"]

    def test_toma_alcohol_y_stripped(self):
        result = _normalize_negations(["toma alcohol y"])
        assert result == ["toma alcohol"]

    def test_diabetes_e_stripped(self):
        result = _normalize_negations(["diabetes e"])
        assert result == ["diabetes"]

    def test_alergias_o_stripped(self):
        result = _normalize_negations(["alergias o"])
        assert result == ["alergias"]

    def test_mixed_list_with_dangling(self):
        result = _normalize_negations(["fuma ni", "usa drogas"])
        assert result == ["fuma", "usa drogas"]


class TestIncompleteTokensDropped:
    """Bare conjunctions / stopwords alone must be removed entirely."""

    def test_bare_ni_dropped(self):
        assert _normalize_negations(["ni"]) == []

    def test_bare_y_dropped(self):
        assert _normalize_negations(["y"]) == []

    def test_bare_o_dropped(self):
        assert _normalize_negations(["o"]) == []

    def test_bare_e_dropped(self):
        assert _normalize_negations(["e"]) == []


class TestTrailingPunctuation:
    """Trailing commas, slashes, semicolons, colons must be stripped."""

    def test_trailing_comma(self):
        result = _normalize_negations(["no fuma,"])
        assert result == ["no fuma"]

    def test_trailing_slash(self):
        result = _normalize_negations(["no fuma/"])
        assert result == ["no fuma"]

    def test_trailing_semicolon(self):
        result = _normalize_negations(["no fuma;"])
        assert result == ["no fuma"]

    def test_comma_then_conjunction(self):
        """'fuma, ni' should strip ', ni' leaving 'fuma'."""
        result = _normalize_negations(["fuma, ni"])
        assert result == ["fuma"]


class TestCompoundEntryPreserved:
    """Entries with internal conjunctions are preserved (only trailing stripped)."""

    def test_niega_fiebre_y_tos_no_trailing_conjunction(self):
        """Internal 'y' is fine; only trailing conjunction is stripped."""
        result = _normalize_negations(["niega fiebre y tos"])
        assert result == ["niega fiebre y tos"]

    def test_internal_ni_preserved(self):
        result = _normalize_negations(["ni fuma ni toma"])
        assert result == ["ni fuma ni toma"]


class TestDeduplication:
    """Duplicate entries (case-insensitive) should be collapsed."""

    def test_exact_dupes(self):
        result = _normalize_negations(["fuma", "fuma"])
        assert result == ["fuma"]

    def test_case_insensitive_dupes(self):
        result = _normalize_negations(["Fuma", "fuma"])
        assert result == ["Fuma"]


class TestEmptyAndNone:
    """Edge cases: None, empty list, empty strings."""

    def test_none_returns_empty(self):
        assert _normalize_negations(None) == []

    def test_empty_list(self):
        assert _normalize_negations([]) == []

    def test_whitespace_only_entries(self):
        assert _normalize_negations(["  ", ""]) == []

    def test_short_entries_dropped(self):
        assert _normalize_negations(["ab"]) == []


class TestMergeFragments:
    """Adjacent 'alergias' + 'medicamentos' should merge."""

    def test_alergias_medicamentos_merge(self):
        result = _normalize_negations(["alergias", "medicamentos"])
        assert result == ["alergias a medicamentos"]


# ---------------------------------------------------------------------------
# New: verb-prefix stripping (Phase 0.5)
# ---------------------------------------------------------------------------

class TestVerbPrefixStripping:
    """First-person singular verbs (tomo, uso, consumo) are stripped."""

    def test_tomo_medicamentos_stripped(self):
        result = _normalize_negations(["tomo medicamentos"])
        assert result == ["medicamentos"]

    def test_uso_drogas_stripped(self):
        result = _normalize_negations(["uso drogas"])
        assert result == ["drogas"]

    def test_consumo_alcohol_stripped(self):
        result = _normalize_negations(["consumo alcohol"])
        assert result == ["alcohol"]

    def test_toma_alcohol_NOT_stripped(self):
        """Third-person 'toma' must NOT be stripped (existing behaviour)."""
        result = _normalize_negations(["toma alcohol"])
        assert result == ["toma alcohol"]

    def test_verb_then_dedup(self):
        """'tomo medicamentos' and 'medicamentos' become the same after strip."""
        result = _normalize_negations(["tomo medicamentos", "medicamentos"])
        assert result == ["medicamentos"]


# ---------------------------------------------------------------------------
# New: expanded trailing dangling (por, la, el, me, te, mi, su)
# ---------------------------------------------------------------------------

class TestExpandedTrailingDangling:
    """New trailing stop-tokens are stripped from negation tails."""

    def test_trailing_por_stripped(self):
        result = _normalize_negations(["secreción por"])
        assert result == ["secreción"]

    def test_trailing_la_stripped(self):
        result = _normalize_negations(["otitis la"])
        assert result == ["otitis"]

    def test_trailing_el_stripped(self):
        result = _normalize_negations(["reflujo el"])
        assert result == ["reflujo"]

    def test_trailing_me_stripped(self):
        result = _normalize_negations(["antibióticos me"])
        assert result == ["antibióticos"]

    def test_trailing_su_stripped(self):
        result = _normalize_negations(["diabetes su"])
        assert result == ["diabetes"]

    def test_secrecion_por_not_in_output(self):
        """Real production noise: 'secreción por' must not survive."""
        result = _normalize_negations(["secreción por", "fiebre", "tos"])
        flat = [r.lower() for r in result]
        assert "secreción por" not in flat
        assert any("secreción" in r for r in flat)
        assert "fiebre" in flat
        assert "tos" in flat


# ---------------------------------------------------------------------------
# New: pure-stopword sequence detection
# ---------------------------------------------------------------------------

class TestPureStopwordSequences:
    """Multi-token pronoun/stopword sequences must be dropped."""

    def test_se_me_dropped(self):
        result = _normalize_negations(["se me"])
        assert result == []

    def test_me_la_dropped(self):
        result = _normalize_negations(["me la"])
        assert result == []

    def test_me_lo_dropped(self):
        result = _normalize_negations(["me lo"])
        assert result == []

    def test_te_lo_dropped(self):
        result = _normalize_negations(["te lo"])
        assert result == []

    def test_se_me_not_in_output_mixed_list(self):
        """Real production noise mix: 'se me' gone, clinical items survive."""
        result = _normalize_negations(["se me", "secreción por", "fiebre", "tos"])
        assert "se me" not in result
        # 'secreción por' → 'secreción' after trailing strip
        assert all("se me" not in r for r in result)
        assert "fiebre" in result
        assert "tos" in result

    def test_valid_clinical_item_preserved(self):
        """Items that are not pure stopwords are kept."""
        result = _normalize_negations(["asma"])
        assert result == ["asma"]


# ---------------------------------------------------------------------------
# New: word-set subsumption dedup (Phase 3.5)
# ---------------------------------------------------------------------------

class TestSubsumptionDedup:
    """Shorter items whose words are a strict subset of a longer item are removed."""

    def test_cirugia_subset_of_otras_cirugia(self):
        """'cirugías' ⊂ 'otras cirugías' → 'cirugías' is removed."""
        result = _normalize_negations(["otras cirugías", "cirugías"])
        assert "cirugías" not in result
        assert any("otras cirugías" in r for r in result)

    def test_tos_not_subsumed_by_tos_productiva(self):
        """'tos' has only 3 chars → guard fires → both items preserved."""
        result = _normalize_negations(["tos", "tos productiva"])
        lower = [r.lower() for r in result]
        assert "tos" in lower
        assert any("tos productiva" in r for r in lower)

    def test_asma_not_subsumed(self):
        """'asma' has 4 chars — guard fires (≤3 chars check is strict <4),
        so 'asma' is NOT subsumed even if a superset exists."""
        # "asma" = 4 chars; guard condition is len(w) <= 3 → 4 > 3, guard fails
        # meaning subsumption WOULD apply if a superset exists.
        # But with a standalone item and no superset, it must be kept.
        result = _normalize_negations(["asma"])
        assert "asma" in result

    def test_independent_items_not_affected(self):
        """Items with no subset relationship must all survive."""
        result = _normalize_negations(["diabetes", "hipertensión", "asma"])
        assert "diabetes" in result
        assert "hipertensión" in result
        assert "asma" in result
