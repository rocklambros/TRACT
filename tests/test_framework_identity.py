"""The framework-identity set, its gates, and the symmetry it exists to hold.

The defect these tests ratchet: `data/processed/stopwords.json` nominated
"owasp" on document frequency and nothing else, so OWASP controls lost their
publisher token while thirteen other frameworks kept theirs. Under
leave-one-framework-out that is a per-fold inconsistency, and no metric in this
repository reports it.

Every assertion here was checked in both directions before it was written. The
attainable ranges are quoted beside the numbers, because an assertion whose
range is a single point is a restatement rather than a check.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tract.config import PROCESSED_DIR
from tract.framework_identity import (
    FRAMEWORK_IDENTITY_PATH,
    MIN_UPPERCASE_FRACTION,
    FrameworkCorpus,
    assert_identity_symmetry,
    derive_framework_identity_tokens,
    filter_set,
    identity_candidates,
    load_framework_corpora,
    load_framework_identity_tokens,
    load_hub_vocabulary,
    self_acronym,
)
from tract.stopwords import filter_stopwords, load_stopwords, tokenize
from tract.text_selection import strip_markup

# Surface-form counts over the corpus, shared by the range checks below.
# Module scope because every test that quotes a fraction needs the same
# denominator, and recomputing per test costs about a second each.
_SURFACE = __import__("tract.framework_identity", fromlist=["_SURFACE_TOKEN"])


def _corpus_surface_counts(
    corpora: list[FrameworkCorpus],
) -> tuple[Counter[str], Counter[str]]:
    total: Counter[str] = Counter()
    capitalised: Counter[str] = Counter()
    for corpus in corpora:
        for document in corpus.documents:
            for match in _SURFACE._SURFACE_TOKEN.finditer(strip_markup(document)):
                surface = match.group(0)
                total[surface.lower()] += 1
                if surface.isupper():
                    capitalised[surface.lower()] += 1
    return total, capitalised


@pytest.fixture(scope="module")
def corpora() -> list[FrameworkCorpus]:
    return load_framework_corpora()


@pytest.fixture(scope="module")
def hub_vocabulary() -> set[str]:
    return load_hub_vocabulary()


@pytest.fixture(scope="module")
def counts(
    corpora: list[FrameworkCorpus],
) -> tuple[Counter[str], Counter[str]]:
    return _corpus_surface_counts(corpora)


@pytest.fixture(scope="module")
def derived() -> frozenset[str]:
    return load_framework_identity_tokens()


# A corpus small enough to reason about, shaped like the real one. Used
# wherever a test needs to move a single measurable property and see the
# derivation react.
TOY = [
    FrameworkCorpus(
        framework_id="owasp_asvs",
        framework_name="OWASP ASVS",
        documents=(
            "OWASP ASVS requires the top of each session to rotate.",
            "OWASP guidance on session handling. Top priority.",
        ),
    ),
    FrameworkCorpus(
        framework_id="cwe",
        framework_name="CWE",
        documents=("CWE describes a weakness in session handling.",),
    ),
    FrameworkCorpus(
        framework_id="nist_ssdf",
        framework_name="NIST SSDF",
        documents=("NIST SSDF covers the build pipeline.",),
    ),
]
TOY_HUB_WORDS = {"session", "handling", "nist"}


class TestTheCommittedSetReproduces:
    """Staleness is invisible in the metrics and changes every one of them."""

    def test_the_artifact_is_what_the_gates_derive_from_the_corpus(
        self, corpora: list[FrameworkCorpus], hub_vocabulary: set[str],
    ) -> None:
        committed = json.loads(
            FRAMEWORK_IDENTITY_PATH.read_text(encoding="utf-8")
        )
        derivation = derive_framework_identity_tokens(
            corpora, hub_vocabulary,
            min_uppercase_fraction=committed["min_uppercase_fraction"],
        )
        assert sorted(derivation.tokens) == committed["tokens"]
        assert len(corpora) == committed["n_frameworks"]

    def test_the_set_is_the_eighteen_measured_acronyms(
        self, derived: frozenset[str],
    ) -> None:
        """Named, so a silent shrink to one token is not a green test.

        The count is a fact about this corpus at this commit, not a target.
        It moves when a framework is added, and the diff should say so.
        """
        assert derived == {
            "aicm", "asvs", "atlas", "biml", "capec", "ccm", "csa", "cwe",
            "dsgai", "enisa", "eu", "gpai", "mitre", "owasp", "rmf", "samm",
            "ssdf", "wstg",
        }


class TestGateOneBoundsTheCandidates:
    """The machine id, not the human title, and not the whole vocabulary."""

    def test_a_word_that_only_sits_in_a_long_title_is_not_a_candidate(
        self, corpora: list[FrameworkCorpus], derived: frozenset[str],
    ) -> None:
        """"Cloud Controls Matrix" must not cost every control the word "matrix".

        Range: title_only holds 20 words on this corpus and the set holds 18,
        so the intersection could be anything from 0 to 18. It is 0.
        """
        ids: set[str] = set()
        for corpus in corpora:
            ids.update(identity_candidates(corpus.framework_id))
        title_only: set[str] = set()
        for corpus in corpora:
            title_only.update(tokenize(corpus.framework_name))
        title_only -= ids

        assert {"matrix", "regulation", "profile", "landscape"} <= title_only
        assert not (title_only & derived), (
            "A word that merely occurs inside a framework's title reached the "
            "strip set. Removing 'regulation' from every control because one "
            "framework is named a regulation deletes real signal."
        )

    def test_the_candidate_universe_is_the_machine_ids_and_nothing_else(
        self, corpora: list[FrameworkCorpus], hub_vocabulary: set[str],
    ) -> None:
        """Pinned on the universe, not on the survivors.

        Written after a mutation survived. Widening gate 1 to the human titles
        leaves the derived set unchanged on this corpus, because every
        title-only word fails one of the other two gates anyway, so a test that
        looks only at the output cannot see the widening. It would be seen the
        first time a framework is named something like "OWASP TOP" -- and by
        then the widened gate is load-bearing and nobody knows it.

        The four buckets partition the candidates, so their union is what the
        derivation considered.

        Range: 34 tokens today. The union of the human titles is 51 and the
        whole corpus vocabulary is five figures, so the assertion has room to
        be wrong in both directions.
        """
        derivation = derive_framework_identity_tokens(corpora, hub_vocabulary)
        considered = (
            set(derivation.tokens)
            | set(derivation.rejected_absent)
            | set(derivation.rejected_hub_vocabulary)
            | set(derivation.rejected_not_capitalised)
        )
        expected: set[str] = set()
        for corpus in corpora:
            expected.update(identity_candidates(corpus.framework_id))
        assert considered == expected, sorted(considered ^ expected)
        assert len(expected) == 34
        assert not ({"matrix", "regulation", "profile", "landscape"} & considered)

    def test_the_id_restriction_keeps_out_a_thousand_security_acronyms(
        self, corpora: list[FrameworkCorpus], hub_vocabulary: set[str],
        counts: tuple[Counter[str], Counter[str]], derived: frozenset[str],
    ) -> None:
        """Without gate 1 the other two admit every all-caps technical term.

        This is what makes gate 1 load-bearing, and it is not the title case
        above: "JWT", "SIEM", "CVE" and "FIPS" are capitalised in almost every
        occurrence and name no hub, so gates 2 and 3 pass them. They are
        security content, and an assignment that has lost the token "JWT" has
        lost the control.

        Range: 0 to the whole vocabulary. Measured at 1,137.
        """
        total, capitalised = counts
        ids: set[str] = set()
        for corpus in corpora:
            ids.update(identity_candidates(corpus.framework_id))

        would_pass = {
            token for token, n in total.items()
            if token not in hub_vocabulary and len(token) >= 2
            and capitalised[token] / n >= MIN_UPPERCASE_FRACTION
        }
        blocked = would_pass - ids
        assert len(blocked) > 500, (
            f"Only {len(blocked)} tokens are held out by the machine-id gate. "
            f"That gate is what stops the set from swallowing every acronym "
            f"in a security corpus."
        )
        assert {"jwt", "siem", "cve", "fips"} <= blocked
        assert not ({"jwt", "siem", "cve", "fips"} & derived)


class TestGateTwoSeparatesAcronymsFromWords:
    """A majority-capitals rule, sitting in an empty band."""

    def test_the_threshold_falls_between_the_two_classes(
        self, corpora: list[FrameworkCorpus], hub_vocabulary: set[str],
        counts: tuple[Counter[str], Counter[str]], derived: frozenset[str],
    ) -> None:
        """Range: accepted floor 0.690, rejected ceiling 0.000.

        The threshold is 0.5 and both edges are far from it, so this fails if
        the rule is moved to either end of the band as well as if a token
        crosses it.
        """
        total, capitalised = counts
        ids: set[str] = set()
        for corpus in corpora:
            ids.update(identity_candidates(corpus.framework_id))

        accepted = {t: capitalised[t] / total[t] for t in derived}
        rejected = {
            t: capitalised[t] / total[t] for t in ids
            if t not in derived and t not in hub_vocabulary and total[t] > 0
        }
        assert accepted and rejected
        assert min(accepted.values()) >= 0.65, min(accepted.items())
        assert max(rejected.values()) <= 0.25, max(rejected.items())
        assert max(rejected.values()) < MIN_UPPERCASE_FRACTION < min(accepted.values())

    def test_the_casing_gate_stands_alone_without_the_hub_gate(
        self, corpora: list[FrameworkCorpus],
    ) -> None:
        """Gate 3's cover for ordinary words is incidental. Gate 2's is not.

        "act", "cheat", "exchange", "proactive", "sheets" and "top" are all
        rejected as hub vocabulary today, because 400 generated hub
        descriptions happen to use them. Descriptions get regenerated -- this
        repository carries a hub_descriptions_reviewed.pre_regen.json to prove
        it -- and the day one of those words leaves the descriptions, gate 2 is
        the only thing between it and every OWASP anchor.

        Measured with hub vocabulary narrowed to names and hierarchy paths,
        which is the vocabulary scripts/build_stopwords.py protects.
        """
        narrow = load_hub_vocabulary(description_paths=())
        derivation = derive_framework_identity_tokens(corpora, narrow)
        assert {"act", "cheat", "exchange", "proactive", "sheets", "top"} <= set(
            derivation.rejected_not_capitalised
        )
        assert not ({"act", "cheat", "top"} & set(derivation.tokens))

    def test_lowering_the_threshold_admits_an_ordinary_word(
        self, corpora: list[FrameworkCorpus],
    ) -> None:
        """The gate has to be capable of a different answer, or it is decoration.

        "top" is written TOP in 6 of its 443 occurrences, so a threshold of
        0.001 lets it through and every OWASP anchor loses the word. "cheat"
        and "agentic" are never capitalised at all and stay out at any
        threshold above zero, which is the shape of an ordinary word.
        """
        narrow = load_hub_vocabulary(description_paths=())
        loose = derive_framework_identity_tokens(
            corpora, narrow, min_uppercase_fraction=0.001,
        )
        assert "top" in loose.tokens
        assert not ({"cheat", "agentic"} & set(loose.tokens))
        assert loose.uppercase_fraction["cheat"] == 0.0

    def test_markup_is_stripped_before_the_casing_is_measured(self) -> None:
        """URLs carry lowercase spellings the encoder never reads.

        "cwe.mitre.org" and "owaspai.org" alone dragged "mitre" from 0.690 to
        0.152 when the raw field was counted, which is below the threshold: the
        MITRE fold would have kept its acronym while OWASP lost its.
        """
        document = (
            "MITRE ATLAS covers adversarial machine learning. MITRE ATLAS "
            "techniques. MITRE publishes the mitre index at "
            "https://attack.mitre.org/mitre/mitre/mitre/mitre/mitre."
        )
        corpora = [FrameworkCorpus(
            framework_id="mitre_atlas",
            framework_name="MITRE ATLAS",
            documents=(document,),
        )]
        derivation = derive_framework_identity_tokens(corpora, {"learning"})
        assert "mitre" in derivation.tokens
        assert derivation.uppercase_fraction["mitre"] >= MIN_UPPERCASE_FRACTION

        # The same document counted without stripping the URL, which is what
        # the raw corpus field holds. Below the threshold, so the URL alone
        # decides whether MITRE keeps its name.
        raw = Counter(
            m.group(0) for m in _SURFACE._SURFACE_TOKEN.finditer(document)
        )
        capitals = sum(n for f, n in raw.items() if f.lower() == "mitre" and f.isupper())
        occurrences = sum(n for f, n in raw.items() if f.lower() == "mitre")
        assert capitals / occurrences < MIN_UPPERCASE_FRACTION


class TestGateThreeProtectsHubVocabulary:
    """A token an assignment has to match on is never stripped."""

    def test_the_live_examples_are_hub_words_and_survive(
        self, hub_vocabulary: set[str], derived: frozenset[str],
    ) -> None:
        """nist, ai, llm, ml and iso all appear in hub data.

        "nist" appears in no hub name and no hierarchy path, only in two hub
        descriptions, which is why load_hub_vocabulary reads them.
        """
        stopwords = load_stopwords()
        for word in ("nist", "ai", "llm", "ml", "iso"):
            assert word in hub_vocabulary, f"{word} left the hub vocabulary"
            assert word not in derived, f"{word} reached the strip set"
            assert word not in stopwords, f"{word} reached the stop word list"

    def test_nist_is_protected_only_by_the_descriptions(self) -> None:
        """Reading names and paths alone is not enough, and this says so.

        Range: with descriptions, "nist" is rejected as hub vocabulary; with
        names and paths alone it clears both remaining gates at 1.000 capitals
        and is stripped. Two different answers from one input change.
        """
        narrow = load_hub_vocabulary(description_paths=())
        assert "nist" not in narrow
        assert "nist" in load_hub_vocabulary()

        corpora = load_framework_corpora()
        assert "nist" in derive_framework_identity_tokens(corpora, narrow).tokens
        assert "nist" not in derive_framework_identity_tokens(
            corpora, load_hub_vocabulary()
        ).tokens

    def test_a_hub_word_is_never_stripped_however_the_arm_is_set(
        self, derived: frozenset[str], hub_vocabulary: set[str],
    ) -> None:
        """The identity set against every hub word, the stop word list against
        the matching surface.

        The two are held to different vocabularies on purpose. Hub text is
        "{hierarchy_path} | {name}", so that is what the stop word list must
        not eat. The identity set is held to the wider vocabulary including
        descriptions, because it is a protection rather than a matching
        surface and "nist" lives only in the descriptions.
        """
        assert not (derived & hub_vocabulary), sorted(derived & hub_vocabulary)
        narrow = load_hub_vocabulary(description_paths=())
        stopwords = load_stopwords()
        assert not (stopwords & narrow), sorted(stopwords & narrow)
        assert not ((derived | stopwords) & narrow)

    def test_filtering_a_hub_text_changes_nothing(
        self, derived: frozenset[str],
    ) -> None:
        """The firewall compares control text to hub text by exact substring.

        Both sides get the same filter, so the identity arm must be a no-op on
        the hub side or the comparison stops comparing like with like.
        """
        hierarchy = json.loads(
            (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
        )
        checked = 0
        for node in hierarchy["hubs"].values():
            text = f"{node['hierarchy_path']} | {node['name']}"
            # Compared on tokens rather than characters: filter_stopwords
            # collapses runs of whitespace even when it removes nothing, and
            # four hub names carry a double space.
            assert tokenize(filter_stopwords(text, derived)) == tokenize(text)
            checked += 1
        assert checked == 522


class TestTheSetIsSymmetricAcrossFrameworks:
    """The defect, stated as an invariant that fails on the world before it."""

    def test_the_committed_arms_are_all_symmetric(
        self, corpora: list[FrameworkCorpus], hub_vocabulary: set[str],
        derived: frozenset[str],
    ) -> None:
        stopwords = load_stopwords()
        for active in (
            frozenset(), derived, stopwords, derived | stopwords,
        ):
            assert_identity_symmetry(frozenset(active), corpora, hub_vocabulary)

    def test_the_pre_fix_stopword_list_is_rejected(
        self, corpora: list[FrameworkCorpus], hub_vocabulary: set[str],
    ) -> None:
        """This is the ruling, executable.

        The committed list held "owasp" and nothing else framework-shaped, so
        OWASP anchors were scrubbed of their publisher and CWE, CAPEC, ASVS,
        BIML, MITRE, CSA, ENISA, SAMM, WSTG and EU anchors were not.
        """
        with pytest.raises(ValueError, match="asymmetric across frameworks"):
            assert_identity_symmetry(
                frozenset(load_stopwords() | {"owasp"}), corpora, hub_vocabulary,
            )

    def test_dropping_any_single_acronym_is_rejected(
        self, corpora: list[FrameworkCorpus], hub_vocabulary: set[str],
        derived: frozenset[str],
    ) -> None:
        """Eleven frameworks name themselves in the corpus. All or none.

        Range: 11 eligible acronyms, so 11 distinct ways to be asymmetric and
        one way to be whole. Every one of the 11 is exercised.
        """
        eligible = {
            self_acronym(c.framework_id, c.framework_name) for c in corpora
        } & derived
        assert len(eligible) == 11, sorted(eligible)
        for acronym in sorted(eligible):
            with pytest.raises(ValueError, match="asymmetric across frameworks"):
                assert_identity_symmetry(
                    derived - {acronym}, corpora, hub_vocabulary,
                )

    def test_stripping_a_hub_acronym_is_rejected_from_the_other_side(
        self, corpora: list[FrameworkCorpus], hub_vocabulary: set[str],
        derived: frozenset[str],
    ) -> None:
        """The protection direction. "nist" names hubs and eleven frameworks."""
        with pytest.raises(ValueError, match="name or describe a CRE hub"):
            assert_identity_symmetry(
                derived | {"nist"}, corpora, hub_vocabulary,
            )

    def test_the_error_names_both_sides(
        self, corpora: list[FrameworkCorpus], hub_vocabulary: set[str],
    ) -> None:
        """A failure a reader cannot act on is a failure they will silence."""
        with pytest.raises(ValueError) as caught:
            assert_identity_symmetry(
                frozenset({"owasp"}), corpora, hub_vocabulary,
            )
        message = str(caught.value)
        assert "'owasp'" in message
        assert "'cwe'" in message
        assert "owasp_top10_2021" in message

    def test_an_absent_acronym_is_exempt_and_the_exemption_is_a_no_op(
        self, corpora: list[FrameworkCorpus],
        counts: tuple[Counter[str], Counter[str]], derived: frozenset[str],
    ) -> None:
        """AIUC, CoSAI and DSOMM are outside the set, and it costs nothing.

        The set carries one asymmetry on purpose: a framework whose acronym
        never appears in its own control text is not covered. That is
        measurable rather than a judgement -- there is no occurrence to strip.
        """
        total, _ = counts
        absent = {
            self_acronym(c.framework_id, c.framework_name) for c in corpora
        } - derived - {"nist"}
        assert absent == {"aiuc", "cosai", "dsomm"}
        for acronym in absent:
            assert total[acronym] == 0, (
                f"{acronym} occurs {total[acronym]} times and is not stripped, "
                f"so the exemption is no longer a no-op."
            )


class TestTheStopWordListCannotNominateAnAcronym:
    """Where the defect entered. Document frequency has no vote here."""

    def test_no_framework_identity_token_is_in_the_committed_stop_words(
        self, derived: frozenset[str],
    ) -> None:
        """Fails against the artifact as committed before this change.

        Range: the overlap could be 0 to 18. It was 1 ("owasp") and is 0.
        """
        overlap = derived & load_stopwords()
        assert not overlap, (
            f"{sorted(overlap)} are in the stop word list, so they are "
            f"stripped whenever the stop word arm runs and the other "
            f"framework acronyms are not."
        )

    def test_the_build_script_protects_them(self) -> None:
        """The protection is in the derivation, not in remembering to check."""
        from scripts.build_stopwords import collect_documents

        _, protected = collect_documents()
        assert load_framework_identity_tokens() <= protected


class TestSelfAcronymIsAnIndependentDerivation:
    """The symmetry check must not be the set restated against itself."""

    @pytest.mark.parametrize(("framework_id", "name", "expected"), [
        ("owasp_llm_top10", "OWASP Top 10 for LLM Applications 2025", "owasp"),
        ("csa_ccm", "Cloud Controls Matrix", "csa"),
        ("aiuc_1", "AIUC-1 Standard", "aiuc"),
        ("mitre_atlas", "MITRE ATLAS", "mitre"),
        ("eu_gpai_cop", "EU GPAI Code of Practice", "eu"),
        ("nist_800_53", "NIST 800-53", "nist"),
        # The trap: the first CAPITALISED token anywhere in the name is "AI",
        # which names hubs and would have exempted CoSAI from the invariant.
        ("cosai", "CoSAI Landscape of AI Security Risk Map", "cosai"),
    ])
    def test_it_reads_the_leading_token_or_falls_back_to_the_id(
        self, framework_id: str, name: str, expected: str,
    ) -> None:
        assert self_acronym(framework_id, name) == expected

    def test_it_refuses_a_framework_it_cannot_name(self) -> None:
        with pytest.raises(ValueError, match="yields no identity token"):
            self_acronym("800_53", "2024/1689")


class TestTheDerivationFailsLoud:
    """No empty set, no silent disable."""

    def test_no_frameworks(self) -> None:
        with pytest.raises(ValueError, match="no frameworks"):
            derive_framework_identity_tokens([], set())

    @pytest.mark.parametrize("threshold", [0.0, -0.5, 1.5])
    def test_a_threshold_outside_the_unit_interval(
        self, threshold: float,
    ) -> None:
        with pytest.raises(ValueError, match="min_uppercase_fraction"):
            derive_framework_identity_tokens(
                TOY, TOY_HUB_WORDS, min_uppercase_fraction=threshold,
            )

    def test_gates_that_admit_nothing(self) -> None:
        """An empty set would filter nothing while claiming the arm ran."""
        with pytest.raises(ValueError, match="No framework-identity token"):
            derive_framework_identity_tokens(
                TOY, {"owasp", "asvs", "cwe", "nist", "ssdf", "top"},
            )

    def test_a_corpus_file_with_no_frameworks(self, tmp_path: Path) -> None:
        empty = tmp_path / "all_controls.json"
        empty.write_text('{"frameworks": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="No framework records"):
            load_framework_corpora(empty)

    def test_a_missing_artifact_says_how_to_build_it(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="build_stopwords"):
            load_framework_identity_tokens(tmp_path / "absent.json")


class TestTheToggleDefaultsToCurrentBehaviour:
    """Nothing strips these until a measurement says it should."""

    def test_the_config_default_is_off(self) -> None:
        from tract.training.config import TrainingConfig

        config = TrainingConfig(name="probe")
        assert config.use_framework_identity_filter is False
        assert config.to_dict()["use_framework_identity_filter"] is False

    def test_neither_arm_means_no_filter_rather_than_an_empty_one(self) -> None:
        """None and frozenset() are not interchangeable downstream.

        run_single_fold records `stopwords is not None` in the fold provenance
        and skips the symmetry check on None. An empty set would claim an arm
        that filtered nothing.
        """
        assert filter_set(
            use_stopwords=False, use_framework_identity=False,
        ) is None

    def test_each_arm_selects_its_own_set(self, derived: frozenset[str]) -> None:
        stopwords = load_stopwords()
        assert filter_set(
            use_stopwords=False, use_framework_identity=True,
        ) == derived
        assert filter_set(
            use_stopwords=True, use_framework_identity=False,
        ) == stopwords
        both = filter_set(use_stopwords=True, use_framework_identity=True)
        assert both == derived | stopwords
        assert both is not None and len(both) == len(derived) + len(stopwords)

    def test_the_arm_label_says_which_arm_ran(self) -> None:
        """The results directory has to name the arm that produced it.

        Labelled from runpod_parallel, which reads the persisted config block
        and imports no training stack. run_fold._arm_label is the other half
        and tests/test_fold_aggregation.py holds the two in agreement.
        """
        from scripts.phase1b.runpod_parallel import _arm_from_config
        from tract.training.config import TrainingConfig

        assert _arm_from_config(TrainingConfig(name="p").to_dict()) == "prose"
        assert _arm_from_config(TrainingConfig(
            name="p", use_framework_identity_filter=True,
        ).to_dict()) == "prose-fwid"
        assert _arm_from_config(TrainingConfig(
            name="p", use_stopword_filter=True,
            use_framework_identity_filter=True,
        ).to_dict()) == "prose-stopwords-fwid"

    def test_the_arm_is_part_of_a_run_identity(self) -> None:
        """Two arms averaging into one number is silent and describes neither."""
        from tract.training.config import TrainingConfig

        flags = {
            key for key, value in TrainingConfig(name="p").to_dict().items()
            if key.startswith("use_") and isinstance(value, bool)
        }
        assert "use_framework_identity_filter" in flags
        # ARM_DEFINING_KEYS lives beside the training stack, so read it from
        # the source rather than importing `datasets` to get at a tuple.
        source = (
            Path(__file__).resolve().parents[1]
            / "tract" / "training" / "orchestrate.py"
        ).read_text(encoding="utf-8")
        block = source.split("ARM_DEFINING_KEYS", 1)[1].split(")", 1)[0]
        for flag in sorted(flags):
            assert f'"{flag}"' in block, (
                f"{flag} changes what a run is and does not separate folds."
            )


class TestFilteringRemovesEveryPublisherAndNoHubWord:
    """End to end, on the text an anchor is made of."""

    def test_every_framework_loses_its_own_name(
        self, corpora: list[FrameworkCorpus],
        counts: tuple[Counter[str], Counter[str]], derived: frozenset[str],
        hub_vocabulary: set[str],
    ) -> None:
        """Not the set restated: the acronyms come from self_acronym.

        Range: 11 frameworks are checked and each could survive or not, so the
        assertion spans 0 to 11 removals. It requires 11.
        """
        total, _ = counts
        checked = 0
        for corpus in corpora:
            acronym = self_acronym(corpus.framework_id, corpus.framework_name)
            if acronym in hub_vocabulary or total[acronym] == 0:
                continue
            sentence = f"The {acronym.upper()} control requires session rotation."
            filtered = filter_stopwords(sentence, derived)
            assert acronym.upper() not in filtered, (
                f"{corpus.framework_id} keeps its own name {acronym!r} while "
                f"other frameworks lose theirs."
            )
            assert "session rotation" in filtered
            checked += 1
        assert checked >= 11

    def test_a_control_keeps_its_security_content(
        self, derived: frozenset[str],
    ) -> None:
        text = (
            "OWASP ASVS V2.1.1 requires the NIST SP 800-63B verifier to "
            "reject a JWT whose CWE-347 signature check is absent."
        )
        filtered = filter_stopwords(text, derived)
        for gone in ("OWASP", "ASVS", "CWE"):
            assert gone not in filtered
        for kept in ("V2.1.1", "NIST", "SP", "800-63B", "JWT", "347", "signature"):
            assert kept in filtered, f"{kept} was removed"
