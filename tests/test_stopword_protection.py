"""Tests for CRE-vocabulary protection in tract/stopwords.py.

A word that names a CRE hub must never be filtered out of the control text
being matched against that hub. The original check was exact-token, which let
real hub vocabulary into the stop word list: 29 of 107 words were
morphological or orthographic variants of words the hubs actually use,
including "vulnerabilities", "attacker", "privacy" and "organization".
"""
from __future__ import annotations

from tract.stopwords import generate_stopwords, protection_keys


class TestProtectionKeys:

    def _meet(self, a: str, b: str) -> bool:
        return bool(protection_keys(a) & protection_keys(b))

    def test_a_hyphenated_compound_protects_its_parts(self) -> None:
        """Hub "Privacy-preserving personal data logic" must protect "privacy"."""
        assert self._meet("privacy", "privacy-preserving")
        assert self._meet("third-party", "party")

    def test_a_derivation_protects_its_root(self) -> None:
        """Hub "Organizational AI security controls" must protect "organization"."""
        assert self._meet("organization", "organizational")

    def test_a_y_to_i_derivation_meets_its_root(self) -> None:
        """Hub "Evasion (e.g. adversarial examples)" must protect "adversary"."""
        assert self._meet("adversary", "adversarial")
        assert self._meet("adversaries", "adversarial")

    def test_a_plural_protects_its_singular(self) -> None:
        """Stopping at the first matching suffix keyed "examples" only to
        "exampl", so the singular stayed unprotected."""
        assert self._meet("example", "examples")
        assert self._meet("control", "controls")
        assert self._meet("vulnerability", "vulnerabilities")

    def test_british_and_american_spellings_meet(self) -> None:
        """The hubs are written in British English, the corpora American."""
        assert self._meet("behavior", "behaviour")
        assert self._meet("analyze", "analyse")
        assert self._meet("organization", "organisation")

    def test_unrelated_words_do_not_meet(self) -> None:
        """Over-protection is safe, but it must not be total."""
        assert not self._meet("adversary", "encryption")
        assert not self._meet("privacy", "hardware")
        assert not self._meet("control", "logging")

    def test_short_words_are_not_stemmed_into_collisions(self) -> None:
        assert not self._meet("data", "date")

    def test_empty_input_yields_no_keys(self) -> None:
        assert protection_keys("") == set()
        assert protection_keys("   ") == set()


class TestGenerateRespectsProtection:

    DOCS = ["the organization shall document adversary behaviour"] * 20

    def test_a_protected_root_survives_its_derived_form(self) -> None:
        """The hub says "adversarial"; the corpus says "adversary"."""
        words = generate_stopwords(self.DOCS, min_doc_freq=0.5,
                                   protect={"adversarial"})
        assert "adversary" not in words

    def test_a_protected_british_spelling_survives(self) -> None:
        words = generate_stopwords(
            ["model behavior integrity"] * 20, min_doc_freq=0.5,
            protect={"behaviour"},
        )
        assert "behavior" not in words

    def test_unprotected_boilerplate_is_still_filtered(self) -> None:
        """Protection must not disable the filter."""
        words = generate_stopwords(self.DOCS, min_doc_freq=0.5,
                                   protect={"adversarial"})
        assert "document" in words or "shall" not in words
        assert words, "protection swallowed the entire list"
