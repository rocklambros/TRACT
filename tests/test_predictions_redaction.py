"""A tracked results file must not carry a restricted standard's control text.

`results/phase1b/**/*.json` is negated back into tracking by `.gitignore:43`,
and `run_single_fold` writes `{"control_text": item.control_text, ...}` into
`predictions.json`. For ENISA and BIML that is harmless. Gate 2 adds ETSI, which
is in `RESTRICTED_FRAMEWORK_IDS`, and committing ETSI control statements into a
CC0 repository is the rights claim `tract/licensing.py` exists to prevent -- for
every downstream fork and mirror.

The fix is not a loss. `predictions.json` needs the control text for exactly one
purpose: asserting that two arms scored the same items in the same order before
a paired interval is computed between them. A digest does that better, because
it is comparable without being readable.
"""

from __future__ import annotations

import pytest

from tract.config import OPENCRE_FRAMEWORK_ID_MAP, RESTRICTED_FRAMEWORK_IDS
from tract.training.orchestrate import prediction_record


class TestRestrictedFrameworksAreRedacted:
    def test_etsi_control_text_is_not_stored(self) -> None:
        record = prediction_record(
            control_text="Verbatim ETSI clause that may not be redistributed.",
            ground_truth_hub_id="010-108",
            predicted_top10=["010-108"],
            framework_name="ETSI",
        )
        assert "control_text" not in record
        assert record["control_text_redacted"] is True

    def test_iso_27001_is_redacted_too(self) -> None:
        """The rule is the restriction set, not a hardcoded ETSI check."""
        record = prediction_record(
            control_text="Verbatim ISO clause.",
            ground_truth_hub_id="010-108",
            predicted_top10=[],
            framework_name="ISO 27001",
        )
        assert "control_text" not in record

    @pytest.mark.parametrize("framework", ["ENISA", "BIML", "NIST 800-53 v5"])
    def test_unrestricted_frameworks_keep_their_text(
        self, framework: str
    ) -> None:
        """Redacting everything would be a silent loss of debugging signal."""
        record = prediction_record(
            control_text="Open text.",
            ground_truth_hub_id="010-108",
            predicted_top10=[],
            framework_name=framework,
        )
        assert record["control_text"] == "Open text."
        assert record["control_text_redacted"] is False


class TestTheDigestIsAlwaysPresent:
    """The digest is what the arm-vs-arm comparison aligns on."""

    @pytest.mark.parametrize("framework", ["ETSI", "ENISA", "BIML"])
    def test_digest_present_regardless_of_restriction(
        self, framework: str
    ) -> None:
        record = prediction_record(
            control_text="Some control.",
            ground_truth_hub_id="010-108",
            predicted_top10=[],
            framework_name=framework,
        )
        assert len(record["control_text_sha256"]) == 64

    def test_same_text_gives_the_same_digest_across_frameworks(self) -> None:
        """Alignment must work even where one arm redacts and another does not."""
        a = prediction_record(
            control_text="Identical control text.",
            ground_truth_hub_id="x", predicted_top10=[], framework_name="ETSI",
        )
        b = prediction_record(
            control_text="Identical control text.",
            ground_truth_hub_id="x", predicted_top10=[], framework_name="ENISA",
        )
        assert a["control_text_sha256"] == b["control_text_sha256"]

    def test_different_text_gives_different_digests(self) -> None:
        a = prediction_record(
            control_text="One.", ground_truth_hub_id="x",
            predicted_top10=[], framework_name="ENISA",
        )
        b = prediction_record(
            control_text="Two.", ground_truth_hub_id="x",
            predicted_top10=[], framework_name="ENISA",
        )
        assert a["control_text_sha256"] != b["control_text_sha256"]


class TestTheRestrictionSetIsResolvable:
    def test_every_restricted_id_is_reachable_from_a_display_name(self) -> None:
        """A restriction nothing maps onto protects nothing.

        If a restricted framework's display name is missing from the id map,
        `prediction_record` cannot recognise it and writes the text out.
        """
        reachable = set(OPENCRE_FRAMEWORK_ID_MAP.values())
        assert RESTRICTED_FRAMEWORK_IDS <= reachable, (
            f"{sorted(RESTRICTED_FRAMEWORK_IDS - reachable)} can never be "
            "matched from a framework display name, so their control text "
            "would be written to a tracked file."
        )

    def test_an_unknown_framework_name_is_treated_as_unrestricted(self) -> None:
        """Deliberate: an unknown name is not a restricted one.

        Erring the other way would silently redact every future framework and
        the loss would look like normal behaviour.
        """
        record = prediction_record(
            control_text="Text.", ground_truth_hub_id="x",
            predicted_top10=[], framework_name="Some New Framework",
        )
        assert record["control_text"] == "Text."
