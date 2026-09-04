"""The one document that leaves the project is the one nothing checks.

`claudedocs/curation-package.md` holds the recruiting persona and the annotator
handbook. It is what outside volunteers actually receive. It is also matched by
`.gitignore:207` (`claudedocs/*`), so it is untracked: it is not in CI, not in
any lint, and not in any repository-wide grep that respects gitignore.

That is not hypothetical harm. Premortem round 1 finding B6 corrected a
contamination warning that named `results/review/hub_reference.md`, a path that
does not resolve, across thirteen tracked files. The handbook carried the same
dead path and was missed by that sweep **because it is gitignored** -- so the
single copy of the warning that an annotator coordinator would actually read
was the one left pointing at nothing.

Three claims in it were also unsourced, and one was load-bearing for how the
round is staffed and paid. They are corrected; these tests keep them corrected.

Every test skips when the file is absent, which is the normal state on CI and
on any fresh clone. They fire on the machine where the packet is built and
sent, which is the only machine where it matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tract.config import PROJECT_ROOT

PACKAGE: Final[Path] = PROJECT_ROOT / "claudedocs" / "curation-package.md"

# The real file. 422 KB, 522 hubs, 400 of the descriptions LLM-written from the
# gold links -- sending it makes the whole round Tier 3.
FORBIDDEN_REFERENCE: Final[str] = "results/ceiling_study/hub_reference.md"
DEAD_PATH: Final[str] = "results/review/hub_reference.md"


@pytest.fixture(scope="module")
def package_text() -> str:
    if not PACKAGE.is_file():
        pytest.skip(f"{PACKAGE} absent (untracked; expected on CI)")
    return PACKAGE.read_text(encoding="utf-8")


class TestTheContaminationWarningResolves:
    def test_it_does_not_name_the_dead_path(self, package_text: str) -> None:
        """A guardrail spelled against a path that does not resolve is not one.

        Someone told to avoid a file, who then finds nothing there, reasonably
        concludes there is nothing to avoid.
        """
        assert DEAD_PATH not in package_text, (
            f"The handbook warns against {DEAD_PATH!r}, which does not exist. "
            f"The file that would actually contaminate the round is "
            f"{FORBIDDEN_REFERENCE!r}."
        )

    def test_it_names_the_file_that_actually_exists(self, package_text: str) -> None:
        assert FORBIDDEN_REFERENCE in package_text
        assert (PROJECT_ROOT / FORBIDDEN_REFERENCE).is_file(), (
            "The warning names a path that no longer resolves. Re-point it at "
            "the real reference file rather than deleting the warning."
        )


class TestNoUnsourcedAgreementClaims:
    """The retracted claims, pinned by the exact figures that were fabricated.

    `results/ceiling_study/panel_agreement.md` reports ONE human annotator --
    its own text says "the single human annotator" -- against five LLM judges,
    as raw agreement rates. The word 'kappa' appears nowhere in it.
    """

    def test_no_two_annotator_kappa_claim(self, package_text: str) -> None:
        """0.71-0.73 was LLM-versus-LLM, presented as annotator-versus-annotator."""
        for figure in ("0.71–0.73", "0.71-0.73"):
            assert f"κ ≈ {figure}" not in package_text, (
                "This figure is LLM-versus-LLM raw agreement from "
                "panel_agreement.md Section 3. There was no second human "
                "annotator, and it is not a chance-corrected kappa."
            )

    def test_no_fabricated_kappa_expectation_band(self, package_text: str) -> None:
        """A target band nobody measured invites managing people against it."""
        assert "0.35–0.65" not in package_text and "0.35-0.65" not in package_text

    def test_no_fabricated_spearman_correlation(self, package_text: str) -> None:
        """rho = -0.93 against links-per-hub appears nowhere in the repository."""
        assert "0.93" not in package_text

    def test_it_states_that_two_human_agreement_was_never_measured(
        self, package_text: str
    ) -> None:
        """The correction has to be positive, not just an absence.

        Deleting the false claim and leaving silence lets the next reader
        reinvent it. The document must say the number does not exist.
        """
        assert "never measured two-human agreement" in package_text


class TestTheSourceStillSaysWhatTheCorrectionClaims:
    """Guard the correction against its own source drifting.

    If panel_agreement.md is ever rewritten with a second human annotator or a
    real kappa, this correction becomes wrong in the other direction.
    """

    @pytest.fixture(scope="class")
    def panel(self) -> str:
        path = PROJECT_ROOT / FORBIDDEN_REFERENCE.replace(
            "hub_reference.md", "panel_agreement.md"
        )
        if not path.is_file():
            pytest.skip(f"{path} absent")
        return path.read_text(encoding="utf-8")

    def test_the_source_reports_a_single_human_annotator(self, panel: str) -> None:
        assert "single human annotator" in panel

    def test_the_source_contains_no_kappa(self, panel: str) -> None:
        lowered = panel.lower()
        assert "kappa" not in lowered
        assert "κ" not in panel

    def test_the_human_opencre_agreement_is_what_the_correction_quotes(
        self, panel: str
    ) -> None:
        assert "0.572 [0.510, 0.632]" in panel
        assert "0.660 [0.599, 0.716]" in panel
