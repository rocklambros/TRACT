"""Tests for prose-first text selection and corpus-derived stop words.

Both implement standing rules, so the tests assert the rules rather than the
implementation: prose always wins over a title, a fallback is always counted,
and a word that names a CRE hub is never treated as boilerplate.
"""
from __future__ import annotations

from dataclasses import dataclass

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


class TestArmIdentityInvariant:
    """The property every arm comparison depends on.

    build_evaluation_corpus de-duplicates on control text, so building it once
    per arm lets the anchor decide how many items exist. Substituting prose
    collapsed 147 items to 144, because several NIST AI 100-2 sections share
    wording once expanded and three distinct items became one. Arms measured
    over different item sets cannot be compared, and paired_bootstrap_delta
    raises outright on unequal per-fold lengths.

    apply_prose_to_corpus fixes identity first and swaps text second.
    """

    @dataclass
    class Item:
        control_text: str
        ground_truth_hub_id: str
        valid_hub_ids: frozenset
        ground_truth_hub_name: str
        framework_name: str
        section_id: str
        track: str

    def _corpus(self):
        return [
            self.Item("Data Poisoning", "h1", frozenset({"h1"}), "Poisoning",
                      "MITRE ATLAS", "AML.T0001", "all"),
            self.Item("Model Exfiltration", "h2", frozenset({"h2"}), "Exfil",
                      "MITRE ATLAS", "AML.T0002", "all"),
            self.Item("Prompt Injection", "h3", frozenset({"h3"}), "Injection",
                      "OWASP Top10 for LLM", "LLM01:2025", "all"),
        ]

    def test_item_count_and_identity_survive_substitution(self) -> None:
        from tract.text_selection import ProseIndex, apply_prose_to_corpus

        index = ProseIndex(CONTROLS)
        before = self._corpus()
        after = apply_prose_to_corpus(before, index)

        assert len(after) == len(before)
        assert [i.section_id for i in after] == [i.section_id for i in before]
        assert [i.ground_truth_hub_id for i in after] == [
            i.ground_truth_hub_id for i in before
        ]

    def test_two_items_sharing_prose_do_not_collapse(self) -> None:
        """The exact failure: distinct items whose prose happens to match."""
        from tract.text_selection import ProseIndex, apply_prose_to_corpus

        shared = "The same paragraph describes both of these sections."
        index = ProseIndex([{
            "framework_name": "NIST AI 100-2",
            "controls": [
                {"control_id": "2.2", "title": "Alpha", "description": shared},
                {"control_id": "2.3", "title": "Beta", "description": shared},
            ],
        }])
        corpus = [
            self.Item("Alpha", "h1", frozenset({"h1"}), "A", "NIST AI 100-2", "2.2", "all"),
            self.Item("Beta", "h2", frozenset({"h2"}), "B", "NIST AI 100-2", "2.3", "all"),
        ]
        after = apply_prose_to_corpus(corpus, index)

        assert len(after) == 2, "identical prose must not merge two eval items"
        assert after[0].control_text == after[1].control_text
        assert after[0].ground_truth_hub_id != after[1].ground_truth_hub_id

    def test_no_index_is_a_passthrough(self) -> None:
        from tract.text_selection import apply_prose_to_corpus

        before = self._corpus()
        assert apply_prose_to_corpus(before, None) is before

    def test_items_without_prose_keep_their_title(self) -> None:
        from tract.text_selection import ProseIndex, apply_prose_to_corpus

        index = ProseIndex(CONTROLS)
        after = apply_prose_to_corpus(self._corpus(), index)
        exfil = next(i for i in after if i.section_id == "AML.T0002")
        # description restates the title, so it is not prose
        assert exfil.control_text == "Model Exfiltration"
        assert exfil.track == "all"

    def test_stopword_arm_preserves_identity_too(self) -> None:
        from tract.text_selection import ProseIndex, apply_prose_to_corpus

        index = ProseIndex(CONTROLS)
        plain = apply_prose_to_corpus(self._corpus(), index)
        filtered = apply_prose_to_corpus(self._corpus(), index, frozenset({"the", "a"}))
        assert [i.section_id for i in plain] == [i.section_id for i in filtered]
        assert len(plain) == len(filtered)


class TestLookupPrecedence:

    def test_section_name_wins_over_a_coarser_section_id(self) -> None:
        """NIST links three techniques under one Mitigations section id.

        Resolving by id first handed all three the same paragraph.
        """
        from tract.text_selection import ProseIndex

        index = ProseIndex([{
            "framework_name": "NIST AI 100-2",
            "controls": [
                {"control_id": "2.2.4", "title": "Mitigations",
                 "description": "A general discussion of evasion mitigations, at length."},
                {"control_id": "technique:adversarial_training",
                 "title": "Adversarial training",
                 "description": "Introduced by Goodfellow et al., training on adversarial examples."},
            ],
        }])
        hit = index.lookup("NIST AI 100-2", "2.2.4", "Adversarial training")
        assert hit is not None
        assert "Goodfellow" in hit.text, "the specific name must beat the coarser id"
