"""Tests for filter_stopwords in tract/stopwords.py.

The filter rebuilt text by joining regex matches of alphabetic runs, which
silently discarded every digit and every punctuation mark. In a corpus of
security controls the identifiers and thresholds are among the most
distinctive tokens present, so the filter was destroying exactly what stop
word removal is supposed to leave behind.
"""
from __future__ import annotations

from tract.stopwords import filter_stopwords

STOPWORDS = frozenset({"the", "organization", "standards", "which", "within"})


class TestIdentifiersSurvive:
    """Each case is a real identifier shape from the corpus."""

    def test_nist_control_id(self) -> None:
        assert "AC-1" in filter_stopwords("AC-1 Policy for the organization", STOPWORDS)

    def test_cwe_id(self) -> None:
        assert "CWE-79" in filter_stopwords("CWE-79 within the standards", STOPWORDS)

    def test_asvs_dotted_requirement(self) -> None:
        assert "V1.1.1" in filter_stopwords("V1.1.1 which the organization", STOPWORDS)

    def test_owasp_colon_year_id(self) -> None:
        assert "ML01:2023" in filter_stopwords("ML01:2023 the standards", STOPWORDS)

    def test_numeric_thresholds(self) -> None:
        out = filter_stopwords("Retain within 90 days at 99.9% the standards", STOPWORDS)
        assert "90" in out
        assert "99.9%" in out

    def test_the_hub_separator_survives(self) -> None:
        """firewall.py recovers hub names by splitting on " | "."""
        out = filter_stopwords("Architecture | Adversarial training", STOPWORDS)
        assert " | " in out


class TestFilteringStillHappens:

    def test_listed_words_are_removed(self) -> None:
        out = filter_stopwords("the organization standards apply", STOPWORDS)
        assert "organization" not in out
        assert "standards" not in out
        assert "apply" in out

    def test_removal_is_case_insensitive(self) -> None:
        assert "Organization" not in filter_stopwords("The Organization acts", STOPWORDS)

    def test_a_word_containing_a_stopword_is_kept(self) -> None:
        """Substring matching would eat "theory" for "the"."""
        out = filter_stopwords("theory of thewall", STOPWORDS)
        assert "theory" in out
        assert "thewall" in out

    def test_no_double_spaces_are_left_behind(self) -> None:
        out = filter_stopwords("alpha the organization beta", STOPWORDS)
        assert "  " not in out
        assert out == "alpha beta"


class TestDegenerateInput:

    def test_empty_text_round_trips(self) -> None:
        assert filter_stopwords("", STOPWORDS) == ""

    def test_text_that_would_empty_is_returned_unchanged(self) -> None:
        """A control reduced to nothing is unusable, not cheaper."""
        assert filter_stopwords("the organization", STOPWORDS) == "the organization"

    def test_text_reduced_to_punctuation_is_returned_unchanged(self) -> None:
        """Bare punctuation is not a usable anchor either."""
        original = "the organization, standards."
        assert filter_stopwords(original, STOPWORDS) == original

    def test_an_empty_stopword_set_is_a_no_op(self) -> None:
        text = "AC-1: retain 90 days."
        assert filter_stopwords(text, frozenset()) == text
