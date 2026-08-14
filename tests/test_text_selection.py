"""Tests for prose-first text selection and corpus-derived stop words.

Both implement standing rules, so the tests assert the rules rather than the
implementation: prose always wins over a title, a fallback is always counted,
and a word that names a CRE hub is never treated as boilerplate.
"""
from __future__ import annotations

import pytest

from tract.stopwords import (
    PROTECTED_WORDS,
    filter_stopwords,
    generate_stopwords,
    tokenize,
)
from tract.text_selection import (
    ProseIndex,
    SelectionStats,
    canonical_framework,
    select_control_text,
)

CONTROLS = [
    {
        "framework_name": "MITRE ATLAS",
        "controls": [
            {
                "control_id": "AML.T0001",
                "title": "Data Poisoning",
                "description": "An adversary contaminates the training corpus "
                               "so the resulting model behaves as they intend.",
            },
            {
                "control_id": "AML.T0002",
                "title": "Model Exfiltration",
                "description": "Model Exfiltration",   # title restated, not prose
            },
        ],
    },
    {
        "framework_name": "OWASP Top 10 for LLM Applications 2025",
        "controls": [
            {
                "control_id": "LLM01:2025",
                "title": "Prompt Injection",
                "description": "short",
                "full_text": "A prompt injection vulnerability occurs when user "
                             "prompts alter the model's behaviour in unintended ways.",
            },
        ],
    },
]


class TestCanonicalFramework:

    @pytest.mark.parametrize("link_side,control_side", [
        ("OWASP Top10 for LLM", "OWASP Top 10 for LLM Applications 2025"),
        ("NIST 800-53 v5", "NIST 800-53"),
        ("DevSecOps Maturity Model (DSOMM)", "DSOMM"),
        ("OWASP Web Security Testing Guide (WSTG)", "WSTG"),
    ])
    def test_link_side_names_map_to_control_side(self, link_side, control_side) -> None:
        """These four disagreements silently cost 645 links their prose."""
        assert canonical_framework(link_side) == control_side

    def test_unknown_names_pass_through(self) -> None:
        assert canonical_framework("MITRE ATLAS") == "MITRE ATLAS"

    def test_matching_ignores_case_and_spacing(self) -> None:
        assert canonical_framework("  owasp   top10 for llm  ") == (
            "OWASP Top 10 for LLM Applications 2025"
        )


class TestProseIndex:

    @pytest.fixture
    def index(self) -> ProseIndex:
        return ProseIndex(CONTROLS)

    def test_prefers_full_text_over_description(self, index: ProseIndex) -> None:
        hit = index.lookup("OWASP Top10 for LLM", "LLM01:2025", None)
        assert hit is not None
        assert hit.source == "full_text"
        assert "unintended ways" in hit.text

    def test_finds_by_control_id(self, index: ProseIndex) -> None:
        hit = index.lookup("MITRE ATLAS", "AML.T0001", None)
        assert hit is not None and hit.source == "description"

    def test_finds_by_title_when_id_does_not_match(self, index: ProseIndex) -> None:
        """OpenCRE's section_id matches control ids for some frameworks only."""
        hit = index.lookup("MITRE ATLAS", "not-an-id", "Data Poisoning")
        assert hit is not None and hit.source == "description"

    def test_a_description_that_restates_the_title_is_not_prose(
        self, index: ProseIndex
    ) -> None:
        """Nineteen frameworks arrive from OpenCRE shaped exactly like this."""
        assert index.lookup("MITRE ATLAS", "AML.T0002", "Model Exfiltration") is None


class TestSelectControlText:

    @pytest.fixture
    def index(self) -> ProseIndex:
        return ProseIndex(CONTROLS)

    def test_prose_wins_over_the_title(self, index: ProseIndex) -> None:
        chosen = select_control_text(index, "MITRE ATLAS", "AML.T0001", "Data Poisoning")
        assert chosen.is_prose
        assert chosen.text.startswith("An adversary")

    def test_falls_back_to_the_title_when_there_is_no_prose(
        self, index: ProseIndex
    ) -> None:
        chosen = select_control_text(index, "ISO 27001", "A.5.1", "Policies for information security")
        assert chosen.source == "title"
        assert chosen.text == "Policies for information security"

    def test_fallbacks_are_counted_per_framework(self, index: ProseIndex) -> None:
        """A run that silently fell back to titles must not look like a clean one."""
        stats = SelectionStats()
        select_control_text(index, "MITRE ATLAS", "AML.T0001", "Data Poisoning", stats)
        select_control_text(index, "ISO 27001", "A.5.1", "Policies", stats)
        select_control_text(index, "ISO 27001", "A.5.2", "Roles", stats)

        assert stats.total == 3
        assert stats.by_source["title"] == 2
        assert stats.fallback_by_framework["ISO 27001"] == 2
        assert "MITRE ATLAS" not in stats.fallback_by_framework
        assert stats.prose_fraction == pytest.approx(1 / 3)

    def test_no_text_at_all_raises(self, index: ProseIndex) -> None:
        with pytest.raises(ValueError, match="No text of any kind"):
            select_control_text(index, "ISO 27001", None, None)

    def test_works_without_an_index(self) -> None:
        chosen = select_control_text(None, "ISO 27001", "A.5.1", "Policies")
        assert chosen.source == "title"


class TestStopwords:

    DOCS = [
        "The organization shall establish documented guidelines for access control.",
        "The organization shall establish documented procedures for data retention.",
        "The organization shall establish documented standards for encryption.",
        "Adversarial training hardens a model against evasion.",
    ]

    def test_frequent_boilerplate_is_selected(self) -> None:
        words = generate_stopwords(self.DOCS, min_doc_freq=0.5)
        assert "documented" in words
        assert "establish" in words

    def test_negations_and_modals_are_never_removed(self) -> None:
        """'shall not permit' and 'shall permit' must not collapse together."""
        docs = ["shall not permit access"] * 10
        words = generate_stopwords(docs, min_doc_freq=0.1)
        assert "shall" not in words
        assert "not" not in words
        for protected in ("no", "never", "must", "only"):
            assert protected in PROTECTED_WORDS

    def test_hub_vocabulary_is_protected(self) -> None:
        """A word that names a hub is never boilerplate, whatever its frequency."""
        docs = ["access control policy"] * 10
        unprotected = generate_stopwords(docs, min_doc_freq=0.5)
        protected = generate_stopwords(docs, min_doc_freq=0.5, protect={"access", "control"})
        assert "access" in unprotected
        assert "access" not in protected
        assert "control" not in protected

    def test_short_tokens_are_excluded(self) -> None:
        """List markers and initials are not words."""
        words = generate_stopwords(["a b c the item"] * 10, min_doc_freq=0.5)
        assert not any(len(w) < 3 for w in words)

    def test_output_is_deterministic_and_sorted(self) -> None:
        first = generate_stopwords(self.DOCS, min_doc_freq=0.5)
        second = generate_stopwords(self.DOCS, min_doc_freq=0.5)
        assert first == second == sorted(first)

    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(ValueError, match="empty corpus"):
            generate_stopwords([])

    def test_filter_drops_only_stop_words(self) -> None:
        out = filter_stopwords("The documented policy for access", frozenset({"documented", "the"}))
        assert out == "policy for access"

    def test_filter_returns_the_original_rather_than_nothing(self) -> None:
        """A control reduced to an empty string is unusable, not cheaper."""
        text = "the and or"
        assert filter_stopwords(text, frozenset({"the", "and", "or"})) == text

    def test_tokenize_lowercases_and_drops_digits(self) -> None:
        assert tokenize("AC-2 Account Management") == ["ac", "account", "management"]


class TestCommittedStopwordList:

    def test_the_committed_list_loads_and_is_sane(self) -> None:
        """The list is a versioned input to every downstream metric."""
        from tract.stopwords import load_stopwords

        words = load_stopwords()
        assert len(words) > 10
        assert not (words & PROTECTED_WORDS), "a protected word reached the list"
        assert all(len(w) >= 3 for w in words)
