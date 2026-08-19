"""Tests for prose-first text selection and corpus-derived stop words.

Both implement standing rules, so the tests assert the rules rather than the
implementation: prose always wins over a title, a fallback is always counted,
and a word that names a CRE hub is never treated as boilerplate.
"""
from __future__ import annotations

import logging
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
        """A run that silently fell back to titles must not look like a clean one.

        Counted under the canonical name, not the link-side spelling. That is
        deliberate: select_control_text records
        `canonical_framework(framework)`, so a framework OpenCRE spells more
        than one way reports one total instead of several partial ones. "ISO
        27001" gained an alias to "ISO/IEC 27001:2022 Annex A" when its 94
        links turned out to resolve to nothing, and this key moved with it.
        """
        stats = SelectionStats()
        select_control_text(index, "MITRE ATLAS", "AML.T0001", "Data Poisoning", stats)
        select_control_text(index, "ISO 27001", "A.5.1", "Policies", stats)
        select_control_text(index, "ISO 27001", "A.5.2", "Roles", stats)

        assert stats.total == 3
        assert stats.by_source["title"] == 2
        assert stats.fallback_by_framework["ISO/IEC 27001:2022 Annex A"] == 2
        assert "ISO 27001" not in stats.fallback_by_framework
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

    def test_two_items_sharing_prose_stay_distinguishable(self) -> None:
        """Distinct items whose prose happens to match.

        Originally this asserted only that the two items did not merge, and
        accepted that both ended up carrying the same anchor. That is not
        enough: two items with one anchor and two different correct answers
        cap accuracy at one of the pair no matter what the model does, and the
        cap applies to the prose arms and not to the title arm, so the arm
        comparison measures the collision rather than the anchor. They must
        stay both present AND separable.
        """
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
        assert after[0].ground_truth_hub_id != after[1].ground_truth_hub_id
        assert after[0].control_text != after[1].control_text, (
            "two anchors with different correct answers must not be identical"
        )
        # Reverting to the title is how they stay separable; titles are unique
        # by construction because the corpus was de-duplicated on them.
        assert after[0].control_text == "Alpha"
        assert after[1].control_text == "Beta"

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

    def test_stopword_arm_preserves_identity_and_actually_filters(self) -> None:
        """Asserting section_id equality alone proves nothing here.

        dataclasses.replace cannot change section_id, so that assertion holds
        even if the stopword argument were dropped entirely. The test has to
        show the filter reached the text.
        """
        from tract.text_selection import ProseIndex, apply_prose_to_corpus

        index = ProseIndex(CONTROLS)
        plain = apply_prose_to_corpus(self._corpus(), index)
        filtered = apply_prose_to_corpus(
            self._corpus(), index, frozenset({"adversary", "the"}),
        )
        assert [i.section_id for i in plain] == [i.section_id for i in filtered]
        assert len(plain) == len(filtered)

        poisoning_plain = next(i for i in plain if i.section_id == "AML.T0001")
        poisoning_filtered = next(i for i in filtered if i.section_id == "AML.T0001")
        assert "adversary" in poisoning_plain.control_text.lower()
        assert "adversary" not in poisoning_filtered.control_text.lower(), (
            "the stopword set never reached the anchor"
        )


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


class TestAlternateTitlePrecedence:
    """An alternate must never take a slot belonging to a real title.

    "First writer wins" holds within one control and says nothing across
    controls. NIST AI 100-2 section 2.3's generated alternate "Poisoning
    Attacks" claimed the key before section 3.2.2, whose actual title is
    "Poisoning Attacks", so the Generative-AI eval item resolved to the
    Predictive-AI chapter's text. A wrong anchor, not a fallback, and invisible
    downstream.
    """

    CONTROLS = [{
        "framework_name": "NIST AI 100-2",
        "controls": [
            {"control_id": "2.3", "title": "Poisoning Attacks and Mitigations",
             "description": "Predictive AI chapter: poisoning of training data at scale.",
             "metadata": {"alt_titles": ["Poisoning Attacks"]}},
            {"control_id": "3.2.2", "title": "Poisoning Attacks",
             "description": "Generative AI chapter: poisoning through the model supply chain."},
        ],
    }]

    def test_real_title_beats_another_controls_alternate(self) -> None:
        from tract.text_selection import ProseIndex

        index = ProseIndex(self.CONTROLS)
        hit = index.lookup("NIST AI 100-2", None, "Poisoning Attacks")
        assert hit is not None
        assert "Generative AI chapter" in hit.text, (
            "an alternate displaced a real title from another control"
        )

    def test_alternate_still_resolves_when_no_real_title_claims_it(self) -> None:
        from tract.text_selection import ProseIndex

        index = ProseIndex(self.CONTROLS)
        hit = index.lookup("NIST AI 100-2", None, "Poisoning Attacks and Mitigations")
        assert hit is not None and "Predictive AI chapter" in hit.text


class TestSectionIdNormalization:

    def test_opencre_section_prefix_is_stripped(self) -> None:
        """NIST links carry "Sec. 2.2"; the parser emits "2.2"."""
        from tract.text_selection import normalize_section_id

        for raw in ("Sec. 2.2", "Sec 2.4.2", "Section 2.2", "  2.2  "):
            assert normalize_section_id(raw) == "2.2" or normalize_section_id(raw) == "2.4.2"

    def test_id_fallback_fires_across_the_prefix(self) -> None:
        from tract.text_selection import ProseIndex

        index = ProseIndex([{
            "framework_name": "NIST AI 100-2",
            "controls": [{"control_id": "2.2", "title": "Evasion",
                          "description": "A long enough body to count as prose here."}],
        }])
        assert index.lookup("NIST AI 100-2", "Sec. 2.2", None) is not None


class TestAnchorBudget:

    def test_anchors_are_cut_to_the_encoder_budget_and_flagged(self) -> None:
        """13,007-character anchors were being handed to a 512-token encoder."""
        from tract.config import MAX_ANCHOR_CHARS
        from tract.text_selection import prepare_anchor

        text, truncated = prepare_anchor("x" * (MAX_ANCHOR_CHARS + 500))
        assert truncated is True
        assert len(text) <= MAX_ANCHOR_CHARS

        short, untruncated = prepare_anchor("a short control")
        assert untruncated is False and short == "a short control"

    def test_anchors_are_nfc_normalised_and_null_stripped(self) -> None:
        """The corpus builder sanitises titles; substitution was bypassing it."""
        from tract.text_selection import prepare_anchor

        text, _ = prepare_anchor("café \x00control")
        assert "\x00" not in text
        assert "café" in text


class TestMarkupStripping:
    """Markdown and site furniture reach the encoder as tokens otherwise.

    OWASP source carries "#### **Example Attack Scenarios**" and AI Exchange
    carries ">Category:" and ">Permalink: https://owaspai.org/...". Removing it
    cut the median prose anchor from 585 characters to 182, so most of what
    looked like prose was syntax.
    """

    def test_removes_markdown_headings_and_emphasis(self) -> None:
        from tract.text_selection import strip_markup

        out = strip_markup("#### **Example Attack Scenarios** **Scenario #1** text")
        assert "####" not in out and "*" not in out
        assert "Example Attack Scenarios" in out
        # "Scenario #1" is content, not markup, so the lone # stays.
        assert "Scenario #1" in out

    def test_removes_site_furniture_and_urls(self) -> None:
        """A framework-branded URL is a shortcut a bi-encoder can learn."""
        from tract.text_selection import strip_markup

        out = strip_markup(
            ">Category: runtime control\n>Permalink: https://owaspai.org/go/ratelimit/\n"
            "Limit the rate of requests."
        )
        assert "owaspai.org" not in out
        assert "Permalink" not in out
        assert "Limit the rate of requests." in out

    def test_keeps_link_text_and_drops_the_target(self) -> None:
        from tract.text_selection import strip_markup

        assert "OWASP guidance" in strip_markup("[OWASP guidance](https://example.com/x)")
        assert "example.com" not in strip_markup("[OWASP guidance](https://example.com/x)")

    def test_plain_prose_is_unchanged(self) -> None:
        from tract.text_selection import strip_markup

        text = "An adversary contaminates the training corpus."
        assert strip_markup(text) == text


class TestRemediationStripping:

    def test_cuts_at_the_first_remediation_heading(self) -> None:
        from tract.text_selection import strip_remediation

        body = (
            "Prompt injection occurs when user input alters the model's behaviour "
            "in ways the developer did not intend, which is the defining risk here. "
            "How to Prevent Constrain model behaviour with explicit instructions."
        )
        head, was_cut = strip_remediation(body)
        assert was_cut
        assert "Prompt injection occurs" in head
        assert "Constrain model behaviour" not in head

    def test_survives_markdown_around_the_heading(self) -> None:
        """The real corpus writes '#### **Example Attack Scenarios**'."""
        from tract.text_selection import strip_remediation

        body = (
            "Retrieval augmented generation can surface stale documents to the user, "
            "which is the substance of this weakness and what it maps to. "
            "#### **Example Attack Scenarios** **Scenario #1** An attacker uploads."
        )
        head, was_cut = strip_remediation(body)
        assert was_cut and "Scenario" not in head

    def test_does_not_cut_on_the_word_used_in_ordinary_prose(self) -> None:
        """'a mitigation for this is' must not be read as a section boundary."""
        from tract.text_selection import strip_remediation

        body = ("Model inversion lets an adversary recover training records, and "
                "a mitigation for this is differential privacy applied at training time.")
        head, was_cut = strip_remediation(body)
        assert not was_cut and head == body

    def test_refuses_to_leave_a_stub(self) -> None:
        """A short description plus its remediation beats a fragment."""
        from tract.text_selection import strip_remediation

        body = "Short. How to Prevent Do the thing that prevents it properly."
        head, was_cut = strip_remediation(body)
        assert not was_cut and head == body


class TestArmProcessingOrder:

    def test_stopword_arm_does_not_inherit_url_fragments(self) -> None:
        """Ordering bug: filtering before markup removal shredded URLs.

        filter_stopwords rebuilds text from alphabetic tokens, so a URL became
        "https owaspai org go ratelimit" and survived as tokens. The stopword
        arm then differed from the prose arm by markdown handling as well as by
        stop words, which is a confound rather than an ablation.
        """
        from tract.text_selection import ProseIndex, apply_prose_to_corpus
        from tests.test_text_selection import TestArmIdentityInvariant as T

        index = ProseIndex([{
            "framework_name": "MITRE ATLAS",
            "controls": [{
                "control_id": "AML.T0001", "title": "Data Poisoning",
                "description": ">Permalink: https://owaspai.org/go/ratelimit/ "
                               "An adversary contaminates the training corpus badly.",
            }],
        }])
        corpus = [T.Item("Data Poisoning", "h1", frozenset({"h1"}), "P",
                         "MITRE ATLAS", "AML.T0001", "all")]
        filtered = apply_prose_to_corpus(corpus, index, frozenset({"the"}))
        text = filtered[0].control_text.lower()
        assert "owaspai" not in text and "https" not in text
        assert "adversary contaminates" in text


class TestAlternateIds:
    """A retired or malformed id may add a key. It may never displace one."""

    LONG = "A control statement long enough to clear the prose bar easily. " * 3

    def _index(self) -> ProseIndex:
        return ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "PS.1.1", "title": "Protect code",
                 "description": self.LONG + " Real PS.1.1.",
                 "metadata": {"alt_ids": ["code, executable code"]}},
                {"control_id": "PW.8.1", "title": "Test executable",
                 "description": self.LONG + " Real PW.8.1."},
            ],
        }])

    def test_alternate_id_resolves(self) -> None:
        hit = self._index().lookup("Demo", "code, executable code", None)
        assert hit is not None
        assert hit.text.endswith("Real PS.1.1.")

    def test_alternate_never_displaces_a_real_id(self) -> None:
        """The alternate is declared first, in corpus order, on purpose."""
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "A-1", "title": "First",
                 "description": self.LONG + " First.",
                 "metadata": {"alt_ids": ["A-2"]}},
                {"control_id": "A-2", "title": "Second",
                 "description": self.LONG + " Second."},
            ],
        }])
        hit = index.lookup("Demo", "A-2", None)
        assert hit is not None
        assert hit.text.endswith("Second.")

    def test_alternate_never_displaces_a_real_id_declared_earlier(self) -> None:
        """Order must not matter. The second pass is what carries this."""
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "A-2", "title": "Second",
                 "description": self.LONG + " Second."},
                {"control_id": "A-1", "title": "First",
                 "description": self.LONG + " First.",
                 "metadata": {"alt_ids": ["A-2"]}},
            ],
        }])
        hit = index.lookup("Demo", "A-2", None)
        assert hit is not None
        assert hit.text.endswith("Second.")

    def test_two_alternates_claiming_one_key_keep_the_first(self) -> None:
        """Alternates are first-writer-wins among themselves, and counted."""
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "B-1", "title": "One",
                 "description": self.LONG + " One.",
                 "metadata": {"alt_ids": ["shared"]}},
                {"control_id": "B-2", "title": "Two",
                 "description": self.LONG + " Two.",
                 "metadata": {"alt_ids": ["shared"]}},
            ],
        }])
        hit = index.lookup("Demo", "shared", None)
        assert hit is not None
        assert hit.text.endswith("One.")
        assert index.alternate_id_collisions == 1

    def test_two_real_ids_claiming_one_key_keep_the_last(self) -> None:
        """Real ids are last-writer-wins and unguarded. Pinned, not fixed.

        Changing this would move the join measured in Task 1's BEFORE
        artifact. The current corpus carries zero such collisions among
        indexed controls. [measured]
        """
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "C-1", "title": "One",
                 "description": self.LONG + " One."},
                {"control_id": "C-1", "title": "Two",
                 "description": self.LONG + " Two."},
            ],
        }])
        hit = index.lookup("Demo", "C-1", None)
        assert hit is not None
        assert hit.text.endswith("Two.")
        assert index.real_id_collisions == 1

    def test_two_real_titles_claiming_one_key_keep_the_first(self) -> None:
        """The other half of the asymmetry, pinned for the same reason.

        82 of these exist in the corpus today: AIUC-1 Standard 73,
        NIST AI 100-2 7, EU AI Act 1, NIST AI RMF 1. [measured]
        """
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "D-1", "title": "Same title",
                 "description": self.LONG + " One."},
                {"control_id": "D-2", "title": "Same title",
                 "description": self.LONG + " Two."},
            ],
        }])
        hit = index.lookup("Demo", None, "Same title")
        assert hit is not None
        assert hit.text.endswith("One.")

    def test_a_clean_index_reports_no_collisions_of_either_kind(self) -> None:
        """The zero direction of both counters, so neither can only rise."""
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "E-1", "title": "One",
                 "description": self.LONG + " One.",
                 "metadata": {"alt_ids": ["E-legacy"]}},
                {"control_id": "E-2", "title": "Two",
                 "description": self.LONG + " Two."},
            ],
        }])
        assert index.real_id_collisions == 0
        assert index.alternate_id_collisions == 0

    def test_alternate_id_is_normalised_like_a_real_one(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "X-1", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": ["2.2"]}},
            ],
        }])
        assert index.lookup("Demo", "Sec. 2.2", None) is not None

    def test_a_prefixed_alternate_is_normalised_when_it_is_stored(self) -> None:
        """The half of normalisation the lookup side cannot cover.

        The brief's case stores a bare "2.2" and looks up "Sec. 2.2", which
        passes whether or not the store side normalises, because lookup
        normalises anyway. This one stores the prefixed spelling and looks up
        the bare one, so it fails unless normalize_section_id runs on the way
        in. OpenCRE writes both spellings, so both directions are live.
        """
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "X-2", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": ["Sec. 3.4"]}},
            ],
        }])
        assert index.lookup("Demo", "3.4", None) is not None

    def test_a_bare_string_alternate_is_accepted(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "Y-1", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": "Y-legacy"}},
            ],
        }])
        assert index.lookup("Demo", "Y-legacy", None) is not None

    def test_an_empty_alternate_is_ignored(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "Z-1", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": ["", "   "]}},
            ],
        }])
        assert len(index) == 1
        assert index.alternate_id_collisions == 0

    def test_a_dead_alternate_is_reported_rather_than_dropped_in_silence(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An alternate losing to a real id is correct, and still worth a line.

        The join counts cannot say which of several declarations did nothing,
        so a parser author who writes an alternate a real control already
        spells needs the log to find it. Not counted: the two counters are the
        interface later parser tasks read.
        """
        with caplog.at_level(logging.WARNING, logger="tract.text_selection"):
            index = ProseIndex([{
                "framework_name": "Demo",
                "controls": [
                    {"control_id": "F-1", "title": "One",
                     "description": self.LONG + " One.",
                     "metadata": {"alt_ids": ["F-2"]}},
                    {"control_id": "F-2", "title": "Two",
                     "description": self.LONG + " Two."},
                ],
            }])
        assert index.alternate_id_collisions == 0
        assert sum(
            "already a real control id" in record.message
            for record in caplog.records
        ) == 1

    def test_a_live_alternate_is_not_reported_as_dead(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The other direction, so the warning above cannot fire on everything."""
        with caplog.at_level(logging.WARNING, logger="tract.text_selection"):
            ProseIndex([{
                "framework_name": "Demo",
                "controls": [
                    {"control_id": "G-1", "title": "One",
                     "description": self.LONG + " One.",
                     "metadata": {"alt_ids": ["G-legacy"]}},
                ],
            }])
        assert not [
            record for record in caplog.records
            if "already a real control id" in record.message
        ]

    def test_an_alternate_on_a_control_the_prose_rule_drops_adds_nothing(
        self,
    ) -> None:
        """The channel cannot rescue a control ProseIndex never indexed.

        This is why CWE-937 needs its alt_id attached to a different control
        rather than to itself, and the corpus test in
        tests/test_corpus_report.py depends on it.
        """
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "W-1", "title": "Restated",
                 "description": "Restated",
                 "metadata": {"alt_ids": ["W-legacy"]}},
            ],
        }])
        assert len(index) == 0
        assert index.lookup("Demo", "W-legacy", None) is None
