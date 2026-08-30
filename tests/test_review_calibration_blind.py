"""Calibration items must not be identifiable in the reviewer-facing export.

F19. Calibration items exist to measure whether a reviewer is paying attention:
they are assignments whose correct hub is already known, mixed into the work so
that agreement with them estimates reviewer quality. That estimate is worth
nothing if the reviewer can tell which items they are.

Before this module they were identifiable three ways at once -- a negative `id`,
a `provenance` of `ground_truth_T1-AI` where every real item said
`active_learning_round_2` or `model_prediction`, and a position at the very end
of the array because they were appended after the main loop. Any one of those
sorts them out of the file in a single pass; `metadata.calibration_items` even
published the count to check against.

These tests are written against the reviewer-facing document only. The operator
sidecar is allowed to say everything -- that is what it is for.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tract.crosswalk.schema import create_database, get_connection
from tract.inference import HubPrediction
from tract.review.export import generate_review_export
from tract.review.guide import generate_hub_reference, generate_reviewer_guide

# Fields a reviewer sees. If any of these takes one value on calibration items
# and a different value on every real item, it is a tell.
REVIEWER_VISIBLE_FIELDS = (
    "provenance", "framework_id", "framework_name", "status",
    "decision", "reviewer_hub_id", "reviewer_notes",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def review_db(tmp_path: Path) -> Path:
    """A crosswalk DB with both reviewable predictions and calibration gold.

    Six reviewable rows and four `ground_truth_T1-AI` rows. The counts matter:
    the interleaving test needs enough real items that a trailing block of
    calibration items would be visibly a trailing block, and the separability
    test needs both strata to span more than one framework so that
    `framework_id` cannot separate them by accident.
    """
    db_path = tmp_path / "blind.db"
    create_database(db_path)
    conn = get_connection(db_path)
    try:
        conn.executemany(
            "INSERT INTO frameworks (id, name, version, fetch_date, control_count) "
            "VALUES (?, ?, ?, ?, ?)",
            [("fw_a", "Alpha", "1.0", "2026-05-01", 5),
             ("fw_b", "Beta", "1.0", "2026-05-01", 5)],
        )
        conn.executemany(
            "INSERT INTO hubs (id, name, path, parent_id) VALUES (?, ?, ?, ?)",
            [("100-200", "Hub Alpha", "/alpha", None),
             ("200-300", "Hub Beta", "/beta", None),
             ("300-400", "Hub Gamma", "/gamma", None)],
        )
        controls = [
            (f"fw_{s}:c{i}", f"fw_{s}", f"C-{s}{i}", f"Control {s}{i}",
             "D" * 150, None)
            for s in ("a", "b") for i in range(1, 6)
        ]
        conn.executemany(
            "INSERT INTO controls (id, framework_id, section_id, title, "
            "description, full_text) VALUES (?, ?, ?, ?, ?, ?)", controls,
        )
        rows = []
        # Reviewable work, spanning both frameworks and both provenances.
        for i in range(1, 4):
            rows.append((i, f"fw_a:c{i}", "100-200", 0.7, 1, 0,
                         "active_learning_round_2", "v1"))
        for i in range(1, 4):
            rows.append((10 + i, f"fw_b:c{i}", "200-300", 0.6, 1, 0,
                         "model_prediction", "v1"))
        # Calibration gold, also spanning both frameworks.
        rows.append((21, "fw_a:c4", "300-400", 1.0, 1, 0, "ground_truth_T1-AI", "v1"))
        rows.append((22, "fw_a:c5", "100-200", 1.0, 1, 0, "ground_truth_T1-AI", "v1"))
        rows.append((23, "fw_b:c4", "200-300", 1.0, 1, 0, "ground_truth_T1-AI", "v1"))
        rows.append((24, "fw_b:c5", "300-400", 1.0, 1, 0, "ground_truth_T1-AI", "v1"))
        conn.executemany(
            "INSERT INTO assignments (id, control_id, hub_id, confidence, "
            "in_conformal_set, is_ood, provenance, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows,
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture()
def blind_export(tmp_path: Path, review_db: Path) -> tuple[Path, Path]:
    """Run the export with a mocked predictor; return (reviewer file, sidecar)."""
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({"global_threshold": 0.5}), encoding="utf-8")

    def _preds(hub: str) -> list[HubPrediction]:
        return [
            HubPrediction(hub_id=hub, hub_name=f"Hub {hub}", hierarchy_path=f"/{hub}",
                          raw_similarity=0.7, calibrated_confidence=0.8,
                          in_conformal_set=True, is_ood=False),
            HubPrediction(hub_id="200-300", hub_name="Hub Beta", hierarchy_path="/beta",
                          raw_similarity=0.3, calibrated_confidence=0.15,
                          in_conformal_set=False, is_ood=False),
        ]

    predictor = MagicMock()
    predictor._artifacts = MagicMock()
    predictor._artifacts.model_adapter_hash = "aabbccddeeff001122334455"
    predictor.predict_batch.side_effect = (
        lambda texts, top_k=5: [_preds("100-200") for _ in texts]
    )

    output_dir = tmp_path / "out"
    operator_dir = tmp_path / "operator"
    with patch("tract.inference.TRACTPredictor", return_value=predictor):
        metadata = generate_review_export(
            review_db, tmp_path / "model", output_dir, calibration_path,
            operator_dir=operator_dir,
        )
    # The reviewer is sent the whole directory, so the guide and hub reference
    # are built here too -- they are the files most likely to render a metadata
    # value the predictions file no longer carries.
    generate_reviewer_guide(output_dir, metadata)
    generate_hub_reference(review_db, output_dir)
    return output_dir / "review_export.json", operator_dir / "review_export.calibration.json"


class TestReviewerFacingExportIsBlind:
    """The reviewer-facing JSON must not partition calibration from real work."""

    def test_no_negative_ids(self, blind_export: tuple[Path, Path]) -> None:
        """`id < 0` was the single cheapest way to find every calibration item."""
        export_path, _ = blind_export
        ids = [p["id"] for p in _load(export_path)["predictions"]]
        assert ids, "export has no predictions"
        negative = [i for i in ids if isinstance(i, int) and i < 0]
        assert not negative, (
            f"{len(negative)} predictions carry a negative id. Sorting by id "
            "separates every calibration item from the real work in one pass."
        )

    def test_every_id_is_a_real_assignment(
        self, blind_export: tuple[Path, Path], review_db: Path,
    ) -> None:
        """Calibration ids must be real rows, not synthetic ones.

        A synthetic id is a tell even when positive: it falls outside the range
        the real items occupy, or fails to resolve against the database.
        """
        from tract.crosswalk.schema import get_connection

        export_path, _ = blind_export
        ids = {p["id"] for p in _load(export_path)["predictions"]}
        conn = get_connection(review_db)
        try:
            known = {r["id"] for r in conn.execute("SELECT id FROM assignments")}
        finally:
            conn.close()
        assert ids <= known, f"ids not present in assignments: {sorted(ids - known)}"

    def test_no_visible_field_partitions_the_calibration_set(
        self, blind_export: tuple[Path, Path],
    ) -> None:
        """No reviewer-visible field may separate calibration from real items.

        The general form of the bug rather than the three known instances: if
        the value sets are disjoint on any visible field, a reviewer can filter
        on it without knowing what it means.
        """
        export_path, sidecar_path = blind_export
        predictions = _load(export_path)["predictions"]
        calibration_ids = set(_load(sidecar_path)["calibration_ids"])
        assert calibration_ids, "sidecar names no calibration items"

        calibration = [p for p in predictions if p["id"] in calibration_ids]
        real = [p for p in predictions if p["id"] not in calibration_ids]
        assert calibration and real, "need both strata to test separability"

        for field in REVIEWER_VISIBLE_FIELDS:
            cal_values = {json.dumps(p.get(field), sort_keys=True) for p in calibration}
            real_values = {json.dumps(p.get(field), sort_keys=True) for p in real}
            assert cal_values & real_values or not cal_values, (
                f"field {field!r} fully separates the two strata: calibration "
                f"takes {sorted(cal_values)} and real items take "
                f"{sorted(real_values)}. A reviewer can filter on it."
            )

    def test_calibration_items_are_not_all_at_the_end(
        self, blind_export: tuple[Path, Path],
    ) -> None:
        """Position is a tell too: appending them put every one in a trailing run."""
        export_path, sidecar_path = blind_export
        predictions = _load(export_path)["predictions"]
        calibration_ids = set(_load(sidecar_path)["calibration_ids"])
        positions = [
            i for i, p in enumerate(predictions) if p["id"] in calibration_ids
        ]
        assert positions, "no calibration items found by position"
        trailing_run = list(range(len(predictions) - len(positions), len(predictions)))
        assert positions != trailing_run, (
            "every calibration item sits in one contiguous block at the end of "
            "the array; taking the last N rows finds all of them."
        )

    def test_metadata_does_not_publish_the_count(
        self, blind_export: tuple[Path, Path],
    ) -> None:
        """The count is a check-figure for anyone guessing which items they are."""
        export_path, _ = blind_export
        metadata = _load(export_path)["metadata"]
        assert "calibration_items" not in metadata, (
            "metadata.calibration_items publishes how many there are, which "
            "turns a guess into something a reviewer can verify."
        )


class TestOperatorSidecar:
    """The sidecar must carry what the reviewer file no longer does."""

    def test_sidecar_names_the_calibration_ids_and_their_gold(
        self, blind_export: tuple[Path, Path],
    ) -> None:
        export_path, sidecar_path = blind_export
        sidecar = _load(sidecar_path)
        assert sidecar["calibration_ids"], "sidecar carries no ids"
        assert set(sidecar["gold_hub_ids"]) == {
            str(i) for i in sidecar["calibration_ids"]
        }, "every calibration id needs its known-correct hub recorded"

    def test_sidecar_is_not_written_beside_the_reviewer_file(
        self, blind_export: tuple[Path, Path],
    ) -> None:
        """Shipping the whole directory must not ship the answer key.

        The reviewer is sent `review_export.json`. If the sidecar sits in the
        same directory under a similar name, the most natural way to hand over
        the work -- copy the output directory -- discloses it.
        """
        export_path, sidecar_path = blind_export
        assert sidecar_path.parent != export_path.parent, (
            f"sidecar {sidecar_path.name} sits in the same directory as "
            f"{export_path.name}; sending the directory sends the answer key."
        )


class TestNothingElseInTheBundleLeaksIt:
    """The reviewer receives a directory, not just the predictions file."""

    def test_no_file_in_the_export_directory_names_a_calibration_id(
        self, blind_export: tuple[Path, Path],
    ) -> None:
        """Guide and hub reference are written beside the export and shipped.

        `generate_reviewer_guide` takes the metadata dict returned by
        `generate_review_export`, and that dict still carries
        `calibration_items` because the operator needs it. Nothing may render
        it into a file the reviewer receives.
        """
        export_path, _ = blind_export
        for path in sorted(export_path.parent.rglob("*")):
            if not path.is_file() or path == export_path:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "calibration" not in text.lower(), (
                f"{path.name} in the reviewer's directory mentions calibration"
            )

    def test_the_export_directory_does_not_contain_the_key(
        self, blind_export: tuple[Path, Path],
    ) -> None:
        export_path, _ = blind_export
        stray = [
            p.name for p in export_path.parent.rglob("*")
            if p.is_file() and "calibration" in p.name.lower()
        ]
        assert not stray, f"answer key present in the reviewer's directory: {stray}"
