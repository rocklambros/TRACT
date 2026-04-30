"""Tests for review JSON generation and ingestion."""
from __future__ import annotations

import json

import numpy as np
import pytest


class TestGenerateReviewJSON:
    def test_generates_valid_schema(self, tmp_path) -> None:
        from tract.active_learning.review import generate_review_json

        items = [
            {"control_id": "fw1:c1", "framework": "FW1", "control_text": "Control text 1"},
            {"control_id": "fw1:c2", "framework": "FW1", "control_text": "Control text 2"},
        ]
        hub_ids = ["h1", "h2", "h3"]
        sims = np.array([[0.9, 0.3, 0.1], [0.2, 0.8, 0.05]])
        probs = np.array([[0.7, 0.2, 0.1], [0.15, 0.75, 0.10]])
        conformal_sets = [{"h1"}, {"h2"}]
        ood_flags = [False, False]

        out = tmp_path / "review.json"
        generate_review_json(
            items=items,
            hub_ids=hub_ids,
            probs=probs,
            conformal_sets=conformal_sets,
            ood_flags=ood_flags,
            threshold=0.5,
            temperature=0.42,
            model_version="abc123",
            round_number=1,
            output_path=out,
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["round"] == 1
        assert data["temperature"] == 0.42
        assert len(data["items"]) == 2

        item = data["items"][0]
        assert "control_id" in item
        assert "predictions" in item
        assert "is_ood" in item
        assert "auto_accept_candidate" in item
        assert item["review"] is None

    def test_auto_accept_if_above_threshold_and_in_conformal(self, tmp_path) -> None:
        from tract.active_learning.review import generate_review_json

        items = [{"control_id": "c1", "framework": "F", "control_text": "T"}]
        hub_ids = ["h1"]
        probs = np.array([[0.9]])
        conformal_sets = [{"h1"}]
        ood_flags = [False]

        out = tmp_path / "r.json"
        generate_review_json(items, hub_ids, probs, conformal_sets, ood_flags,
                             threshold=0.5, temperature=1.0, model_version="v1",
                             round_number=1, output_path=out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["items"][0]["auto_accept_candidate"] is True


class TestIngestReviews:
    def test_ingest_accepted(self) -> None:
        from tract.active_learning.review import ingest_reviews
        from tract.training.data_quality import QualityTier

        review_data = {
            "round": 1,
            "items": [
                {
                    "control_id": "fw1:c1",
                    "framework": "FW1",
                    "control_text": "Control text",
                    "predictions": [{"hub_id": "h1", "confidence": 0.9, "in_conformal_set": True}],
                    "is_ood": False,
                    "review": {"status": "accepted", "corrected_hub_id": None, "notes": ""},
                },
                {
                    "control_id": "fw1:c2",
                    "framework": "FW1",
                    "control_text": "Control text 2",
                    "predictions": [{"hub_id": "h2", "confidence": 0.7, "in_conformal_set": False}],
                    "is_ood": False,
                    "review": {"status": "rejected", "corrected_hub_id": None, "notes": "wrong"},
                },
            ],
        }

        accepted = ingest_reviews(review_data)
        assert len(accepted) == 1
        assert accepted[0].tier == QualityTier.AL
        assert accepted[0].link["cre_id"] == "h1"
        assert accepted[0].link["link_type"] == "active_learning"

    def test_ingest_corrected(self) -> None:
        from tract.active_learning.review import ingest_reviews

        review_data = {
            "round": 1,
            "items": [
                {
                    "control_id": "fw1:c1",
                    "framework": "FW1",
                    "control_text": "Control text",
                    "predictions": [{"hub_id": "h1", "confidence": 0.9, "in_conformal_set": True}],
                    "is_ood": False,
                    "review": {"status": "corrected", "corrected_hub_id": "h5", "notes": ""},
                },
            ],
        }

        accepted = ingest_reviews(review_data)
        assert len(accepted) == 1
        assert accepted[0].link["cre_id"] == "h5"
