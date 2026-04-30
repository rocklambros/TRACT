"""Review JSON generation and ingestion for active learning."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tract.training.data_quality import QualityTier, TieredLink

logger = logging.getLogger(__name__)


def generate_review_json(
    items: list[dict],
    hub_ids: list[str],
    probs: NDArray[np.floating],
    conformal_sets: list[set[str]],
    ood_flags: list[bool],
    threshold: float,
    temperature: float,
    model_version: str,
    round_number: int,
    output_path: Path,
    top_k: int = 10,
) -> Path:
    """Generate review.json for expert review.

    Canary items should be pre-interleaved in the items list — this function
    does not distinguish canaries from real predictions.
    """
    review_items = []
    for i, item in enumerate(items):
        predictions = []
        ranked_indices = np.argsort(probs[i])[::-1][:top_k]
        for j in ranked_indices:
            hub_id = hub_ids[j]
            predictions.append({
                "hub_id": hub_id,
                "confidence": round(float(probs[i, j]), 4),
                "in_conformal_set": hub_id in conformal_sets[i],
            })

        top_pred = predictions[0] if predictions else None
        auto_accept = (
            not ood_flags[i]
            and top_pred is not None
            and top_pred["confidence"] >= threshold
            and top_pred["in_conformal_set"]
        )

        review_items.append({
            "control_id": item["control_id"],
            "framework": item["framework"],
            "control_text": item["control_text"],
            "predictions": predictions,
            "is_ood": ood_flags[i],
            "auto_accept_candidate": auto_accept,
            "review": None,
        })

    data = {
        "round": round_number,
        "model_version": model_version,
        "temperature": temperature,
        "threshold": threshold,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": review_items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)
            f.write("\n")
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Generated review.json: %d items, round %d", len(review_items), round_number)
    return output_path


def ingest_reviews(review_data: dict) -> list[TieredLink]:
    """Ingest reviewed predictions, returning accepted/corrected as TieredLinks.

    Accepted items use the model's top prediction hub.
    Corrected items use the expert's corrected_hub_id.
    Rejected items are excluded.
    """
    round_number = review_data["round"]
    accepted: list[TieredLink] = []

    for item in review_data["items"]:
        review = item.get("review")
        if review is None:
            continue

        status = review["status"]
        if status == "rejected":
            continue

        if status == "corrected":
            hub_id = review["corrected_hub_id"]
            if not hub_id:
                logger.warning("Corrected item %s has no corrected_hub_id, skipping", item["control_id"])
                continue
        elif status == "accepted":
            preds = item.get("predictions", [])
            if not preds:
                continue
            hub_id = preds[0]["hub_id"]
        else:
            logger.warning("Unknown review status: %s", status)
            continue

        link_dict = {
            "standard_name": item["framework"],
            "section_name": item["control_text"],
            "section_id": item["control_id"].split(":", 1)[-1] if ":" in item["control_id"] else item["control_id"],
            "cre_id": hub_id,
            "link_type": "active_learning",
            "al_round": str(round_number),
        }
        accepted.append(TieredLink(link=link_dict, tier=QualityTier.AL))

    logger.info(
        "Ingested round %d: %d accepted/corrected out of %d reviewed",
        round_number, len(accepted), len(review_data["items"]),
    )
    return accepted
