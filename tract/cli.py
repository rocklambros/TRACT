"""TRACT CLI — 8 subcommands for model inference, comparison, and hub proposals.

Usage:
    python -m tract.cli assign "control text"
    tract assign "control text"  (via console_scripts entry)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tract.config import (
    BRIDGE_OUTPUT_DIR,
    BRIDGE_TOP_K,
    HF_DEFAULT_REPO_ID,
    HF_STAGING_DIR,
    HUB_PROPOSALS_DIR,
    PHASE1C_CROSSWALK_DB_PATH,
    PHASE1D_ARTIFACTS_PATH,
    PHASE1D_CALIBRATION_PATH,
    PHASE1D_DEFAULT_TOP_K,
    PHASE1D_DEPLOYMENT_MODEL_DIR,
    PHASE1D_PROPOSAL_BUDGET_CAP,
    PHASE3_DATASET_REPO_ID,
    PHASE3_DATASET_STAGING_DIR,
    PHASE3_REVIEW_OUTPUT_DIR,
    PROCESSED_DIR,
    TRAINING_DIR,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tract",
        description="TRACT — Translating Requirements Across CRE Trees",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── assign ───────────────────────────────────────────────────
    p_assign = subparsers.add_parser(
        "assign",
        help="Assign control text to CRE hubs",
        epilog=(
            "Examples:\n"
            "  tract assign 'Ensure AI models are tested for bias'\n"
            "  tract assign --file controls.txt --output results.jsonl\n"
            "  tract assign 'Access control policy' --raw --top-k 10\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_assign.add_argument("text", nargs="?", help="Control text to assign")
    p_assign.add_argument("--file", help="Newline-delimited text file (one control per line)")
    p_assign.add_argument("--top-k", type=int, default=PHASE1D_DEFAULT_TOP_K, help="Number of top hub assignments")
    p_assign.add_argument("--output", help="Output path for batch mode (default: {input}_assignments.jsonl)")
    p_assign.add_argument("--raw", action="store_true", help="Show raw cosine similarity instead of calibrated confidence")
    p_assign.add_argument("--verbose", action="store_true", help="Show both metrics, conformal set, and OOD status")
    p_assign.add_argument("--json", action="store_true", help="Output as JSON")

    # ── compare ──────────────────────────────────────────────────
    p_compare = subparsers.add_parser(
        "compare",
        help="Compare frameworks via shared CRE hubs",
        epilog="Example:\n  tract compare --framework mitre_atlas --framework owasp_ai_exchange\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_compare.add_argument("--framework", action="append", required=True, help="Framework ID (use twice for comparison)")
    p_compare.add_argument("--json", action="store_true", help="Output as JSON")

    # ── ingest ───────────────────────────────────────────────────
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest a new framework and generate review file",
        epilog="Example:\n  tract ingest --file new_framework.json\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ingest.add_argument("--file", required=True, help="Framework JSON file (FrameworkOutput schema)")
    p_ingest.add_argument("--force", action="store_true", help="Overwrite if framework ID already exists")
    p_ingest.add_argument("--json", action="store_true", help="Output as JSON")

    # ── accept ───────────────────────────────────────────────────
    p_accept = subparsers.add_parser(
        "accept",
        help="Commit reviewed ingest predictions to crosswalk DB",
        epilog=(
            "Examples:\n"
            "  tract accept --review new_framework_review.json\n"
            "  tract accept --review new_framework_review.json --force\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_accept.add_argument("--review", required=True, help="Reviewed JSON file from 'tract ingest'")
    p_accept.add_argument("--force", action="store_true", help="Replace if framework already exists in DB")
    p_accept.add_argument("--json", action="store_true", help="Output summary as JSON")

    # ── export ───────────────────────────────────────────────────
    p_export = subparsers.add_parser(
        "export",
        help="Export crosswalk assignments",
        epilog="Example:\n  tract export --format csv --framework mitre_atlas\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_export.add_argument("--format", choices=["csv", "json", "jsonl"], default="json", help="Output format")
    p_export.add_argument("--framework", help="Filter to single framework")
    p_export.add_argument("--hub", help="Filter to single hub")
    p_export.add_argument("--min-confidence", type=float, help="Minimum confidence threshold")
    p_export.add_argument("--status", default="accepted", help="Filter by review status")
    p_export.add_argument("--output", help="Output file path")
    p_export.add_argument("--opencre", action="store_true",
                          help="Export in OpenCRE CSV format (one CSV per framework)")
    p_export.add_argument("--opencre-proposals", action="store_true",
                          help="Export hub proposals document for OpenCRE")
    p_export.add_argument("--output-dir",
                          help="Output directory for OpenCRE export (default: ./opencre_export/)")
    p_export.add_argument("--dry-run", action="store_true",
                          help="Show what would be exported without writing files")
    p_export.add_argument("--skip-staleness", action="store_true",
                          help="Skip pre-export staleness check (offline mode)")

    # ── export-canonical ────────────────────────────────────────
    p_export_canonical = subparsers.add_parser(
        "export-canonical",
        help="Export canonical JSON snapshots for OpenCRE RFC",
        epilog=(
            "Examples:\n"
            "  tract export-canonical --dry-run\n"
            "  tract export-canonical --framework csa_aicm --with-embeddings\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_export_canonical.add_argument(
        "--framework", help="Export single framework (default: all mapped frameworks)")
    p_export_canonical.add_argument(
        "--output-dir", help="Output directory (default: ./canonical_export)")
    p_export_canonical.add_argument(
        "--with-embeddings", action="store_true",
        help="Include .npz embedding files per framework")
    p_export_canonical.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be exported without writing files or updating history")

    # ── hierarchy ────────────────────────────────────────────────
    p_hierarchy = subparsers.add_parser(
        "hierarchy",
        help="Show hub hierarchy information",
        epilog="Example:\n  tract hierarchy --hub 646-285\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_hierarchy.add_argument("--hub", required=True, help="Hub ID to inspect")
    p_hierarchy.add_argument("--json", action="store_true", help="Output as JSON")

    # ── propose-hubs ─────────────────────────────────────────────
    p_propose = subparsers.add_parser(
        "propose-hubs",
        help="Generate hub proposals from OOD controls",
        epilog="Example:\n  tract propose-hubs --name-with-llm --budget 20\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_propose.add_argument("--name-with-llm", action="store_true",
                           help="Use Claude API to generate hub names (sends control texts to API)")
    p_propose.add_argument("--budget", type=int, default=PHASE1D_PROPOSAL_BUDGET_CAP, help="Max proposals to generate")
    p_propose.add_argument("--json", action="store_true", help="Output as JSON")

    # ── review-proposals ─────────────────────────────────────────
    p_review = subparsers.add_parser(
        "review-proposals",
        help="Interactive review of hub proposals",
        epilog="Example:\n  tract review-proposals --round 1 --dry-run\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_review.add_argument("--round", type=int, required=True, help="Proposal round number")
    p_review.add_argument("--dry-run", action="store_true", help="Show proposals without modifying anything")

    # ── tutorial ─────────────────────────────────────────────────
    subparsers.add_parser(
        "tutorial",
        help="Guided walkthrough of TRACT capabilities",
    )

    # ── validate ─────────────────────────────────────────────────────
    p_validate = subparsers.add_parser(
        "validate",
        help="Validate a prepared framework JSON file",
        epilog=(
            "Examples:\n"
            "  tract validate --file prepared.json\n"
            "  tract validate --file prepared.json --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_validate.add_argument("--file", required=True, help="Framework JSON file to validate")
    p_validate.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    # ── prepare ──────────────────────────────────────────────────────
    p_prepare = subparsers.add_parser(
        "prepare",
        help="Prepare a raw framework document for ingestion",
        epilog=(
            "Examples:\n"
            "  tract prepare --file controls.csv --framework-id new_fw \\\n"
            "    --name 'New Framework' --version '1.0' \\\n"
            "    --source-url 'https://example.com' --mapping-unit control\n"
            "\n"
            "  tract prepare --file doc.pdf --llm --framework-id new_fw \\\n"
            "    --name 'New Framework' --version '1.0' \\\n"
            "    --source-url 'https://example.com' --mapping-unit control\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_prepare.add_argument("--file", required=True, help="Input file path (CSV, markdown, JSON, or unstructured)")
    p_prepare.add_argument("--framework-id", required=True, help="Framework ID slug (lowercase, underscores)")
    p_prepare.add_argument("--name", required=True, help="Human-readable framework name")
    p_prepare.add_argument("--version", default="1.0", help="Framework version string (default: 1.0)")
    p_prepare.add_argument("--source-url", default="", help="Official framework URL (default: empty)")
    p_prepare.add_argument("--mapping-unit", default="control", help="What each control represents (default: control)")
    p_prepare.add_argument("--fetched-date", default=None, help="Fetch date in YYYY-MM-DD format (default: today)")
    p_prepare.add_argument("--expected-count", type=int, default=None, help="Expected number of controls (warns on mismatch)")
    p_prepare.add_argument("--id-column", default=None, help="CSV column name for control_id (overrides auto-detect)")
    p_prepare.add_argument("--title-column", default=None, help="CSV column name for title (overrides auto-detect)")
    p_prepare.add_argument("--description-column", default=None, help="CSV column name for description (overrides auto-detect)")
    p_prepare.add_argument("--fulltext-column", default=None, help="CSV column name for full_text (overrides auto-detect)")
    p_prepare.add_argument("--llm", action="store_true", help="Use Claude API for LLM-assisted extraction")
    p_prepare.add_argument("--format", choices=["csv", "markdown", "json", "unstructured"], help="Override auto-detected format")
    p_prepare.add_argument("--output", help="Output file path (default: <input_stem>_prepared.json)")
    p_prepare.add_argument("--heading-level", type=int, help="Markdown heading depth to split on (default: auto-detect)")
    p_prepare.add_argument("--json", action="store_true", help="Output summary as JSON")

    # ── bridge ──────────────────────────────────────────────────
    p_bridge = subparsers.add_parser(
        "bridge",
        help="Discover AI/traditional CRE hub bridges",
        epilog=(
            "Examples:\n"
            "  tract bridge --skip-descriptions\n"
            "  tract bridge --top-k 5\n"
            "  tract bridge --commit --candidates results/bridge/bridge_candidates.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_bridge.add_argument("--output-dir", default=str(BRIDGE_OUTPUT_DIR), help="Output directory")
    p_bridge.add_argument("--top-k", type=int, default=BRIDGE_TOP_K, help="Top-K matches per AI hub")
    p_bridge.add_argument("--skip-descriptions", action="store_true", help="Skip LLM descriptions")
    p_bridge.add_argument("--commit", action="store_true", help="Commit reviewed candidates")
    p_bridge.add_argument("--candidates", help="Path to reviewed bridge_candidates.json (for --commit)")

    # ── publish-hf ──────────────────────────────────────────────
    p_publish = subparsers.add_parser(
        "publish-hf",
        help="Publish model to HuggingFace Hub",
        epilog=(
            "Examples:\n"
            "  tract publish-hf --repo-id rockCO78/tract-cre-assignment --dry-run\n"
            "  tract publish-hf --repo-id rockCO78/tract-cre-assignment\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_publish.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    p_publish.add_argument("--staging-dir", default=str(HF_STAGING_DIR), help="Local build dir")
    p_publish.add_argument("--dry-run", action="store_true", help="Build + scan, no upload")
    p_publish.add_argument("--skip-upload", action="store_true", help="Build + scan only")
    p_publish.add_argument("--gpu-hours", type=float, default=0.0, help="GPU training hours for model card")

    # ── import-ground-truth ─────────────────────────────────────────
    p_import_gt = subparsers.add_parser(
        "import-ground-truth",
        help="Import OpenCRE ground truth links and run inference on uncovered frameworks",
        epilog=(
            "Examples:\n"
            "  tract import-ground-truth\n"
            "  tract import-ground-truth --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_import_gt.add_argument(
        "--dry-run", action="store_true", help="Report counts without modifying DB",
    )

    # ── review-export ──────────────────────────────────────────────
    p_review_export = subparsers.add_parser(
        "review-export",
        help="Generate review JSON for expert review",
        epilog=(
            "Examples:\n"
            "  tract review-export\n"
            "  tract review-export --output results/review\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_review_export.add_argument(
        "--output", default=str(PHASE3_REVIEW_OUTPUT_DIR),
        help="Output directory for review files",
    )
    p_review_export.add_argument(
        "--model-dir", default=str(PHASE1D_DEPLOYMENT_MODEL_DIR),
        help="Path to deployment model directory",
    )

    # ── review-validate ─────────────────────────────────────────────
    p_review_validate = subparsers.add_parser(
        "review-validate",
        help="Validate a reviewed predictions JSON file",
        epilog=(
            "Examples:\n"
            "  tract review-validate --input results/review/review_export.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_review_validate.add_argument(
        "--input", required=True, help="Path to reviewed JSON file",
    )

    # ── review-import ──────────────────────────────────────────────
    p_review_import = subparsers.add_parser(
        "review-import",
        help="Import expert review decisions into crosswalk.db",
        epilog=(
            "Examples:\n"
            "  tract review-import --input review.json --reviewer expert_1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_review_import.add_argument(
        "--input", required=True, help="Path to reviewed JSON file",
    )
    p_review_import.add_argument(
        "--reviewer", required=True, help="Reviewer name/identifier",
    )

    # ── publish-dataset ──────────────────────────────────────────────
    p_pub_dataset = subparsers.add_parser(
        "publish-dataset",
        help="Publish crosswalk dataset to HuggingFace Datasets",
        epilog=(
            "Examples:\n"
            "  tract publish-dataset --dry-run\n"
            "  tract publish-dataset\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_pub_dataset.add_argument(
        "--repo-id", default=PHASE3_DATASET_REPO_ID, help="HuggingFace repo ID",
    )
    p_pub_dataset.add_argument(
        "--staging-dir", default=str(PHASE3_DATASET_STAGING_DIR),
        help="Local build dir",
    )
    p_pub_dataset.add_argument(
        "--dry-run", action="store_true", help="Build without upload",
    )
    p_pub_dataset.add_argument(
        "--skip-upload", action="store_true", help="Build only, no upload",
    )

    return parser


# ── Output Formatting ──────────────────────────────────────────────


def format_predictions_table(
    preds: list,
    raw: bool = False,
    verbose: bool = False,
) -> str:
    """Format predictions as a human-readable table."""
    lines = []
    metric_label = "Hub Similarity" if raw else "Confidence*"
    header = f" {'#':>2}  {'Hub ID':<9}{'Hub Name':<30}{metric_label}"
    if verbose:
        header = f" {'#':>2}  {'Hub ID':<9}{'Hub Name':<30}{'Confidence*':<14}{'Cosine':<8}{'Conformal'}"

    lines.append("Hub Assignments (top {0}):".format(len(preds)))
    lines.append("─" * max(len(header) + 5, 60))
    lines.append(header)

    for i, pred in enumerate(preds, 1):
        if verbose:
            conformal = "✓" if pred.in_conformal_set else " "
            line = (
                f" {i:>2}  {pred.hub_id:<9}{pred.hub_name:<30}"
                f"{pred.calibrated_confidence:<14.3f}{pred.raw_similarity:<8.3f}{conformal}"
            )
        elif raw:
            line = f" {i:>2}  {pred.hub_id:<9}{pred.hub_name:<30}{pred.raw_similarity:.3f}"
        else:
            line = f" {i:>2}  {pred.hub_id:<9}{pred.hub_name:<30}{pred.calibrated_confidence:.3f}"
        lines.append(line)

    if not raw:
        lines.append("")
        lines.append(" * Calibrated on traditional framework holdout. AI framework accuracy may differ.")

    if verbose and preds:
        ood_status = "Out-of-distribution" if preds[0].is_ood else "In-distribution"
        conformal_count = sum(1 for p in preds if p.in_conformal_set)
        lines.append(f" OOD Status: {ood_status}")
        lines.append(f" Conformal set: {conformal_count} hubs (90% coverage guarantee)")

    return "\n".join(lines)


def format_predictions_json(preds: list) -> str:
    """Format predictions as JSON."""
    return json.dumps([p.to_dict() for p in preds], indent=2)


# ── Command Handlers ────────────────────────────────────────────────


def _cmd_assign(args: argparse.Namespace) -> None:
    from tract.inference import TRACTPredictor

    predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)

        texts = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        results = predictor.predict_batch(texts, top_k=args.top_k)

        output_path = Path(args.output) if args.output else file_path.with_suffix(".jsonl").with_stem(file_path.stem + "_assignments")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for text, preds in sorted(zip(texts, results), key=lambda tp: tp[1][0].raw_similarity if tp[1] else 0):
                line = {"text": text[:100], "predictions": [p.to_dict() for p in preds]}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        ood_count = sum(1 for r in results if r and r[0].is_ood)
        high_conf = sum(1 for r in results if r and r[0].calibrated_confidence > 0.5)
        print(f"Wrote {len(results)} assignments to {output_path}")
        print(f"{ood_count}/{len(results)} controls flagged OOD, {high_conf}/{len(results)} high confidence")
        return

    if not args.text:
        print("Error: Provide control text or --file", file=sys.stderr)
        sys.exit(1)

    preds = predictor.predict(args.text, top_k=args.top_k)

    if args.json:
        print(format_predictions_json(preds))
    else:
        print(format_predictions_table(preds, raw=args.raw, verbose=args.verbose))


def _cmd_compare(args: argparse.Namespace) -> None:
    from tract.compare import cross_framework_matrix
    from tract.hierarchy import CREHierarchy

    if len(args.framework) < 2:
        print("Error: Need at least 2 --framework flags", file=sys.stderr)
        sys.exit(1)

    hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")
    result = cross_framework_matrix(PHASE1C_CROSSWALK_DB_PATH, args.framework, hierarchy)

    if args.json:
        output = {
            "equivalences": [
                {"hub_id": e.hub_id, "hub_name": e.hub_name,
                 "frameworks": e.frameworks, "controls": e.controls}
                for e in result.equivalences
            ],
            "related": [
                {"hub_a": r.hub_a, "hub_b": r.hub_b, "parent_hub": r.parent_hub}
                for r in result.related
            ],
            "gap_controls": result.gap_controls,
            "framework_pair_overlap": {
                f"{a},{b}": n for (a, b), n in result.framework_pair_overlap.items()
            },
            "total_shared_hubs": result.total_shared_hubs,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Cross-Framework Comparison: {', '.join(args.framework)}")
        print("=" * 60)
        print(f"\nEquivalent mappings (same hub): {len(result.equivalences)}")
        for eq in result.equivalences[:20]:
            print(f"  Hub {eq.hub_id} ({eq.hub_name}): {', '.join(eq.frameworks)}")
        print(f"\nRelated mappings (sibling hubs): {len(result.related)}")
        if result.gap_controls:
            print("\nGap controls (no cross-framework match):")
            for fw, ctrls in result.gap_controls.items():
                print(f"  {fw}: {len(ctrls)} unmatched")
        print(f"\nTotal shared hubs: {result.total_shared_hubs}")


def _cmd_ingest(args: argparse.Namespace) -> None:
    from tract.config import PHASE1D_INGEST_MAX_FILE_SIZE
    from tract.inference import TRACTPredictor
    from tract.io import atomic_write_json, load_json
    from tract.schema import FrameworkOutput

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    file_size = file_path.stat().st_size
    if file_size > PHASE1D_INGEST_MAX_FILE_SIZE:
        print(f"Error: File too large ({file_size / 1024 / 1024:.1f}MB > 50MB limit)", file=sys.stderr)
        sys.exit(1)

    raw_data = load_json(file_path)

    from tract.validate import validate_framework

    validation_issues = validate_framework(raw_data)
    val_errors = [i for i in validation_issues if i.severity == "error"]
    val_warnings = [i for i in validation_issues if i.severity == "warning"]

    if val_errors:
        print(f"Validation failed ({len(val_errors)} error(s)):", file=sys.stderr)
        for issue in val_errors:
            prefix = f"  [{issue.control_id}] " if issue.control_id else "  "
            print(f"{prefix}{issue.message}", file=sys.stderr)
        if val_warnings:
            print(f"\n  ({len(val_warnings)} warning(s) also found)", file=sys.stderr)
        sys.exit(1)

    if val_warnings:
        print(f"Validation warnings ({len(val_warnings)}):", file=sys.stderr)
        for issue in val_warnings:
            prefix = f"  [{issue.control_id}] " if issue.control_id else "  "
            print(f"{prefix}{issue.message}", file=sys.stderr)
        print("", file=sys.stderr)

    from tract.sanitize import sanitize_control

    for i, ctrl in enumerate(raw_data["controls"]):
        raw_data["controls"][i] = sanitize_control(ctrl)

    try:
        fw = FrameworkOutput.model_validate(raw_data)
    except Exception as e:
        print(f"Error: Schema validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    from tract.crosswalk.schema import get_connection

    conn = get_connection(PHASE1C_CROSSWALK_DB_PATH)
    try:
        existing = conn.execute(
            "SELECT id FROM frameworks WHERE id = ?", (fw.framework_id,)
        ).fetchone()
    finally:
        conn.close()

    if existing and not args.force:
        print(f"Error: Framework '{fw.framework_id}' already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)

    texts = []
    for ctrl in fw.controls:
        parts = [ctrl.title, ctrl.description]
        if ctrl.full_text:
            parts.append(ctrl.full_text)
        texts.append(" ".join(p for p in parts if p))

    batch_preds = predictor.predict_batch(texts, top_k=PHASE1D_DEFAULT_TOP_K)

    controls_output = []
    ood_count = 0
    dup_count = 0
    sim_count = 0
    high_conf = 0
    low_conf = 0

    from datetime import datetime, timezone

    for ctrl, text, preds in zip(fw.controls, texts, batch_preds):
        duplicates, similar = predictor.find_duplicates(text)

        is_ood = preds[0].is_ood if preds else False
        if is_ood:
            ood_count += 1
        if duplicates:
            dup_count += 1
        if similar:
            sim_count += 1
        if preds and preds[0].calibrated_confidence > 0.5:
            high_conf += 1
        else:
            low_conf += 1

        controls_output.append({
            "control_id": ctrl.control_id,
            "title": ctrl.title,
            "description": ctrl.description,
            "full_text": ctrl.full_text,
            "predictions": [p.to_dict() for p in preds],
            "is_ood": is_ood,
            "duplicates": [d.to_dict() for d in duplicates],
            "similar": [s.to_dict() for s in similar],
            "review": {"status": "pending"},
        })

    review_data = {
        "framework_id": fw.framework_id,
        "framework_name": fw.framework_name,
        "version": fw.version,
        "fetched_date": fw.fetched_date,
        "source_url": fw.source_url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "deployment_round2",
        "context": "ingestion",
        "summary": {
            "total_controls": len(fw.controls),
            "ood_flagged": ood_count,
            "duplicate_flagged": dup_count,
            "similar_flagged": sim_count,
            "high_confidence": high_conf,
            "low_confidence": low_conf,
        },
        "controls": controls_output,
    }

    if raw_data.get("metadata", {}).get("source") == "tract_prepare":
        review_data["calibration_note"] = (
            "Confidence scores were calibrated on parser-extracted text. "
            "This file was prepared via tract prepare, which may produce "
            "different text surface forms. Treat confidence scores as "
            "approximate rankings, not calibrated probabilities."
        )

    max_sims = [
        ctrl["predictions"][0]["confidence"]
        for ctrl in controls_output
        if ctrl.get("predictions")
    ]
    n = len(controls_output)
    review_data["quality_summary"] = {
        "mean_max_cosine_sim": round(sum(max_sims) / len(max_sims), 3) if max_sims else 0.0,
        "ood_fraction": round(ood_count / n, 3) if n else 0.0,
        "below_confidence_floor_count": low_conf,
        "below_confidence_floor_fraction": round(low_conf / n, 3) if n else 0.0,
    }

    output_path = file_path.with_stem(file_path.stem + "_review").with_suffix(".json")
    atomic_write_json(review_data, output_path)

    if args.json:
        print(json.dumps(review_data["summary"], indent=2))
    else:
        print(f"Ingestion complete: {fw.framework_name} ({fw.framework_id})")
        print(f"  Controls: {len(fw.controls)}")
        print(f"  OOD flagged: {ood_count}")
        print(f"  Duplicates: {dup_count}, Similar: {sim_count}")
        print(f"  High confidence: {high_conf}, Low confidence: {low_conf}")
        print(f"  Review file: {output_path}")
        qs = review_data["quality_summary"]
        print(f"  Average match quality: {qs['mean_max_cosine_sim']:.3f}", file=sys.stderr)
        print(f"  Unusual controls (out-of-distribution): {qs['ood_fraction']:.0%}", file=sys.stderr)
        print(f"  Below confidence floor: {qs['below_confidence_floor_count']} controls ({qs['below_confidence_floor_fraction']:.0%})", file=sys.stderr)


def _cmd_accept(args: argparse.Namespace) -> None:
    from tract.accept import accept_review
    from tract.io import load_json

    review_path = Path(args.review)
    if not review_path.exists():
        print(f"Error: File not found: {review_path}", file=sys.stderr)
        sys.exit(1)

    review_data = load_json(review_path)

    required_fields = ["framework_id", "framework_name", "controls"]
    for field in required_fields:
        if field not in review_data:
            print(f"Error: Review file missing required field '{field}'", file=sys.stderr)
            sys.exit(1)

    pending = [c for c in review_data["controls"]
               if c.get("review", {}).get("status") == "pending"]
    if pending and not args.force:
        print(f"Warning: {len(pending)} controls still pending review:")
        for p in pending[:5]:
            print(f"  {p['control_id']}: {p.get('title', 'untitled')}")
        if len(pending) > 5:
            print(f"  ... and {len(pending) - 5} more")
        print("These will be skipped (no assignment created).")

    try:
        result = accept_review(
            PHASE1C_CROSSWALK_DB_PATH,
            review_data,
            force=args.force,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Accepted: {review_data['framework_name']} ({result['framework_id']})")
        print(f"  Controls inserted: {result['controls_inserted']}")
        print(f"  Assignments created: {result['assignments_created']}")
        print(f"    Accepted: {result['accepted']}")
        if result['corrected']:
            print(f"    Corrected: {result['corrected']}")
        if result['rejected']:
            print(f"    Rejected: {result['rejected']}")
        if result['pending']:
            print(f"    Pending (skipped): {result['pending']}")


def _cmd_validate(args: argparse.Namespace) -> None:
    from tract.io import load_json
    from tract.validate import validate_framework

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = load_json(file_path)
    except Exception as e:
        print(f"Error: Failed to load JSON: {e}", file=sys.stderr)
        sys.exit(1)

    issues = validate_framework(data)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if args.json:
        output = {
            "file": str(file_path),
            "errors": [
                {"control_id": i.control_id, "rule": i.rule, "message": i.message}
                for i in errors
            ],
            "warnings": [
                {"control_id": i.control_id, "rule": i.rule, "message": i.message}
                for i in warnings
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        if errors:
            print(f"ERRORS ({len(errors)}):", file=sys.stderr)
            for i in errors:
                prefix = f"  [{i.control_id}] " if i.control_id else "  "
                print(f"{prefix}{i.message}", file=sys.stderr)

        if warnings:
            print(f"\nWARNINGS ({len(warnings)}):", file=sys.stderr)
            for i in warnings:
                prefix = f"  [{i.control_id}] " if i.control_id else "  "
                print(f"{prefix}{i.message}", file=sys.stderr)

        if not errors and not warnings:
            print("Validation passed: no errors, no warnings.")
        elif not errors:
            print(f"\nValidation passed with {len(warnings)} warning(s).")

    if errors:
        sys.exit(1)


def _cmd_prepare(args: argparse.Namespace) -> None:
    from datetime import date

    from tract.prepare import prepare_framework

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else None
    fetched_date = args.fetched_date if args.fetched_date else date.today().isoformat()

    column_overrides: dict[str, str] | None = None
    override_keys = {
        "id_column": "control_id",
        "title_column": "title",
        "description_column": "description",
        "fulltext_column": "full_text",
    }
    for attr, canonical in override_keys.items():
        val = getattr(args, attr, None)
        if val is not None:
            if column_overrides is None:
                column_overrides = {}
            column_overrides[canonical] = val

    try:
        result_path = prepare_framework(
            file_path=file_path,
            framework_id=args.framework_id,
            name=args.name,
            version=args.version,
            source_url=args.source_url,
            mapping_unit=args.mapping_unit,
            fetched_date=fetched_date,
            output_path=output_path,
            format_override=args.format,
            use_llm=args.llm,
            heading_level=args.heading_level,
            expected_count=args.expected_count,
            column_overrides=column_overrides,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        from tract.io import load_json
        data = load_json(result_path)
        summary = {
            "output_path": str(result_path),
            "framework_id": data["framework_id"],
            "controls": len(data["controls"]),
        }
        print(json.dumps(summary, indent=2))
    else:
        from tract.io import load_json
        data = load_json(result_path)
        print(f"Prepared: {data['framework_name']} ({data['framework_id']})")
        print(f"  Controls: {len(data['controls'])}")
        print(f"  Output: {result_path}")


def _cmd_export(args: argparse.Namespace) -> None:
    if getattr(args, "opencre", False):
        _cmd_export_opencre(args)
        return
    if getattr(args, "opencre_proposals", False):
        _cmd_export_opencre_proposals(args)
        return

    from tract.crosswalk.export import export_crosswalk

    fmt = args.format
    if fmt == "jsonl":
        fmt = "json"

    output_path = Path(args.output) if args.output else Path(f"crosswalk_export.{args.format}")
    export_crosswalk(PHASE1C_CROSSWALK_DB_PATH, output_path, fmt=fmt)
    print(f"Exported to {output_path}")


def _cmd_export_opencre(args: argparse.Namespace) -> None:
    from tract.config import (
        PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR,
        PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES,
    )
    from tract.export.filters import compute_filter_stats, query_exportable_assignments
    from tract.export.gaps import query_coverage_gaps
    from tract.export.manifest import build_manifest
    from tract.export.opencre_csv import write_opencre_csv
    from tract.export.opencre_names import TRACT_TO_OPENCRE_NAME
    from tract.io import atomic_write_json

    output_dir = Path(args.output_dir) if args.output_dir else Path("./opencre_export")

    confidence_floor = PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR
    confidence_overrides = dict(PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES)

    staleness_result = {"status": "skipped", "upstream_hub_count": 0, "message": "skipped"}
    if not args.skip_staleness and not args.dry_run:
        from tract.export.staleness import check_staleness

        print("Running pre-export staleness check...")
        staleness_result = check_staleness(PHASE1C_CROSSWALK_DB_PATH)
        if staleness_result["status"] == "warn":
            print(f"WARNING: {staleness_result['message']}")
            print(f"  Stale IDs: {staleness_result['stale_ids']}")
        elif staleness_result["status"] == "error":
            print(f"ERROR: Staleness check failed: {staleness_result['message']}")
            print("  Use --skip-staleness to bypass (offline mode)")
            sys.exit(1)
        else:
            print(f"Staleness check passed ({staleness_result['upstream_hub_count']} upstream hubs)")

    if args.framework:
        if args.framework not in TRACT_TO_OPENCRE_NAME:
            print(f"Error: Framework '{args.framework}' has no OpenCRE name mapping", file=sys.stderr)
            print(f"  Available: {', '.join(sorted(TRACT_TO_OPENCRE_NAME.keys()))}", file=sys.stderr)
            sys.exit(1)
        frameworks = [args.framework]
    else:
        frameworks = sorted(TRACT_TO_OPENCRE_NAME.keys())

    all_rows: dict[str, list[dict]] = {}
    for fw_id in frameworks:
        rows = query_exportable_assignments(
            PHASE1C_CROSSWALK_DB_PATH,
            confidence_floor=confidence_floor,
            confidence_overrides=confidence_overrides,
            framework_filter=fw_id,
        )
        if rows:
            all_rows[fw_id] = rows

    if not all_rows:
        print("No assignments survived filters. Nothing to export.")
        return

    if args.dry_run:
        print("\nDry run — would export:\n")
        total = 0
        for fw_id, rows in sorted(all_rows.items()):
            print(f"  {fw_id}: {len(rows)} assignments")
            total += len(rows)
        print(f"\n  Total: {total} assignments")
        return

    written_files: list[Path] = []
    total_exported = 0
    for fw_id, rows in sorted(all_rows.items()):
        csv_path = write_opencre_csv(rows, fw_id, output_dir)
        written_files.append(csv_path)
        total_exported += len(rows)
        print(f"  {fw_id}: {len(rows)} assignments → {csv_path}")

    all_exported = []
    for rows in all_rows.values():
        all_exported.extend(rows)

    stats = compute_filter_stats(
        PHASE1C_CROSSWALK_DB_PATH, all_exported,
        confidence_floor, confidence_overrides,
    )

    model_hash = "unknown"
    from tract.config import PHASE1D_ARTIFACTS_PATH
    if PHASE1D_ARTIFACTS_PATH.exists():
        import hashlib
        model_hash = hashlib.sha256(PHASE1D_ARTIFACTS_PATH.read_bytes()).hexdigest()[:12]

    manifest = build_manifest(
        per_framework_stats=stats,
        confidence_floor=confidence_floor,
        confidence_overrides=confidence_overrides,
        staleness_result=staleness_result,
        model_adapter_hash=model_hash,
    )

    manifest_path = output_dir / "export_manifest.json"
    atomic_write_json(manifest, manifest_path)
    print(f"\n  Manifest: {manifest_path}")

    exported_keys = {(r["control_id"], r["hub_id"]) for r in all_exported}
    gaps = query_coverage_gaps(
        PHASE1C_CROSSWALK_DB_PATH,
        exported_keys=exported_keys,
        confidence_floor=confidence_floor,
        confidence_overrides=confidence_overrides,
        framework_ids=frameworks,
    )
    gaps_path = output_dir / "coverage_gaps.json"
    atomic_write_json(gaps, gaps_path)
    print(f"  Coverage gaps: {gaps_path}")

    total_missing = sum(len(g["missing_controls"]) for g in gaps.values())
    if total_missing:
        print(f"\n  Coverage gaps ({total_missing} controls not exported):")
        for fw_id in sorted(gaps.keys()):
            g = gaps[fw_id]
            missing = g["missing_controls"]
            if missing:
                print(f"    {fw_id}: {g['exported_controls']}/{g['total_controls']} "
                      f"({g['coverage_pct']}%) — {len(missing)} missing")

    print(f"\n  Total exported: {total_exported} assignments across {len(written_files)} frameworks")


def _cmd_export_opencre_proposals(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else Path("./opencre_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    from tract.io import load_json

    proposals_dir = HUB_PROPOSALS_DIR
    if not proposals_dir.exists():
        print("No hub proposals found. Run 'tract propose-hubs' first.")
        return

    rounds = sorted(proposals_dir.glob("round_*"))
    if not rounds:
        print("No proposal rounds found.")
        return

    latest_round = rounds[-1]
    summary_path = latest_round / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json in {latest_round}")
        return

    summary = load_json(summary_path)
    proposals_output = {
        "source": "TRACT hub proposal pipeline",
        "round": latest_round.name,
        "proposals": summary.get("proposals", []),
    }

    from tract.io import atomic_write_json

    out_path = output_dir / "hub_proposals_for_opencre.json"
    atomic_write_json(proposals_output, out_path)
    print(f"Hub proposals written to {out_path}")


def _cmd_export_canonical(args: argparse.Namespace) -> None:
    import hashlib

    from tract.config import (
        PHASE5_CANONICAL_EXPORT_DIR,
        PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR,
        PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES,
    )
    from tract.export.canonical import export_canonical
    from tract.export.opencre_names import (
        TRACT_TO_OPENCRE_NAME,
        build_hyperlink,
    )

    output_dir = Path(args.output_dir) if args.output_dir else PHASE5_CANONICAL_EXPORT_DIR

    if args.framework:
        if args.framework not in TRACT_TO_OPENCRE_NAME:
            print(
                f"Error: Framework '{args.framework}' has no OpenCRE name mapping",
                file=sys.stderr,
            )
            print(
                f"  Available: {', '.join(sorted(TRACT_TO_OPENCRE_NAME.keys()))}",
                file=sys.stderr,
            )
            sys.exit(1)
        framework_ids = [args.framework]
    else:
        framework_ids = sorted(TRACT_TO_OPENCRE_NAME.keys())

    model_hash = "unknown"
    if PHASE1D_ARTIFACTS_PATH.exists():
        model_hash = hashlib.sha256(
            PHASE1D_ARTIFACTS_PATH.read_bytes()
        ).hexdigest()[:12]

    try:
        tract_version = (
            __import__("subprocess")
            .check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        tract_version = "unknown"

    result = export_canonical(
        db_path=PHASE1C_CROSSWALK_DB_PATH,
        framework_ids=framework_ids,
        output_dir=output_dir,
        confidence_floor=PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR,
        confidence_overrides=dict(PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES),
        model_adapter_hash=model_hash,
        tract_version=tract_version,
        hyperlink_fn=build_hyperlink,
        framework_names=dict(TRACT_TO_OPENCRE_NAME),
        artifacts_path=PHASE1D_ARTIFACTS_PATH if args.with_embeddings else None,
        with_embeddings=args.with_embeddings,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run — would export:\n")

    total_controls = 0
    total_mappings = 0
    for fw_id, info in sorted(result.items()):
        summary = info["changeset_summary"]
        total_controls += info["controls"]
        total_mappings += info["mappings"]
        change_parts = []
        for k, v in summary.items():
            if v > 0:
                change_parts.append(f"{k}={v}")
        changes_str = ", ".join(change_parts) if change_parts else "no changes"
        print(f"  {fw_id}: {info['controls']} controls, {info['mappings']} mappings "
              f"[{info['impact_scope']}] ({changes_str})")

    print(f"\n  Total: {total_controls} controls, {total_mappings} mappings "
          f"across {len(result)} frameworks")

    if not args.dry_run:
        print(f"  Output: {output_dir}")


def _cmd_hierarchy(args: argparse.Namespace) -> None:
    from tract.crosswalk.schema import get_connection
    from tract.hierarchy import CREHierarchy

    hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")

    if args.hub not in hierarchy.hubs:
        print(f"Error: Unknown hub ID: {args.hub}", file=sys.stderr)
        sys.exit(1)

    node = hierarchy.hubs[args.hub]
    parent = hierarchy.get_parent(args.hub)
    children = hierarchy.get_children(args.hub)
    siblings = hierarchy.get_siblings(args.hub)

    conn = get_connection(PHASE1C_CROSSWALK_DB_PATH)
    try:
        assigned_controls = conn.execute(
            "SELECT a.control_id, c.title, c.framework_id, a.confidence "
            "FROM assignments a JOIN controls c ON a.control_id = c.id "
            "WHERE a.hub_id = ? AND a.review_status IN ('accepted', 'ground_truth') "
            "ORDER BY a.confidence DESC",
            (args.hub,),
        ).fetchall()
    finally:
        conn.close()

    if args.json:
        output = {
            "hub_id": node.hub_id,
            "name": node.name,
            "hierarchy_path": node.hierarchy_path,
            "depth": node.depth,
            "is_leaf": node.is_leaf,
            "parent": {"hub_id": parent.hub_id, "name": parent.name} if parent else None,
            "children": [{"hub_id": c.hub_id, "name": c.name} for c in children],
            "siblings": [{"hub_id": s.hub_id, "name": s.name} for s in siblings],
            "assigned_controls": [dict(r) for r in assigned_controls],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Hub: {node.hub_id} — {node.name}")
        print(f"Path: {node.hierarchy_path}")
        print(f"Depth: {node.depth}, Leaf: {node.is_leaf}")
        if parent:
            print(f"Parent: {parent.hub_id} ({parent.name})")
        if children:
            print(f"Children ({len(children)}):")
            for c in children:
                print(f"  {c.hub_id} — {c.name}")
        if siblings:
            print(f"Siblings ({len(siblings)}):")
            for s in siblings[:10]:
                print(f"  {s.hub_id} — {s.name}")
        if assigned_controls:
            print(f"\nAssigned controls ({len(assigned_controls)}):")
            for r in assigned_controls[:20]:
                conf = f" ({r['confidence']:.3f})" if r['confidence'] else ""
                print(f"  [{r['framework_id']}] {r['title']}{conf}")


def _cmd_propose_hubs(args: argparse.Namespace) -> None:
    import numpy as np

    from tract.config import PHASE1D_ARTIFACTS_PATH, PHASE1D_CALIBRATION_PATH
    from tract.hierarchy import CREHierarchy
    from tract.inference import load_deployment_artifacts
    from tract.io import load_json
    from tract.proposals.cluster import cluster_ood_controls
    from tract.proposals.guardrails import apply_guardrails
    from tract.proposals.naming import generate_hub_names, generate_placeholder_names
    from tract.proposals.review import write_proposal_round

    artifacts = load_deployment_artifacts(PHASE1D_ARTIFACTS_PATH)
    calibration = load_json(PHASE1D_CALIBRATION_PATH)
    hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")

    ood_threshold = calibration["ood_threshold"]
    sims = artifacts.control_embeddings @ artifacts.hub_embeddings.T
    max_sims = sims.max(axis=1)
    ood_mask = max_sims < ood_threshold

    ood_embs = artifacts.control_embeddings[ood_mask]
    ood_ids = [artifacts.control_ids[i] for i in range(len(artifacts.control_ids)) if ood_mask[i]]

    print(f"OOD controls: {len(ood_ids)}/{len(artifacts.control_ids)}")

    clusters = cluster_ood_controls(
        ood_embs, ood_ids,
        hub_embeddings=artifacts.hub_embeddings,
        hub_ids=artifacts.hub_ids,
    )

    if not clusters:
        msg = "No proposals generated (insufficient OOD controls for clustering)."
        if args.json:
            print(json.dumps({"message": msg, "ood_count": len(ood_ids)}))
        else:
            print(msg)
        return

    results = apply_guardrails(
        clusters, hierarchy,
        artifacts.hub_embeddings, artifacts.hub_ids,
        {}, budget_cap=args.budget,
    )

    hub_names_map = {hid: hierarchy.hubs[hid].name for hid in artifacts.hub_ids if hid in hierarchy.hubs}

    if args.name_with_llm:
        names = generate_hub_names(results, hierarchy, {})
    else:
        names = generate_placeholder_names(results, hub_names_map)

    existing_rounds = sorted(HUB_PROPOSALS_DIR.glob("round_*")) if HUB_PROPOSALS_DIR.exists() else []
    round_num = len(existing_rounds) + 1

    round_dir = write_proposal_round(results, names, HUB_PROPOSALS_DIR, round_num)

    passing = [r for r in results if r.passed]
    if args.json:
        output = {
            "round": round_num,
            "ood_count": len(ood_ids),
            "clusters_found": len(clusters),
            "proposals_passing": len(passing),
            "round_dir": str(round_dir),
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nProposal round {round_num}:")
        print(f"  Clusters found: {len(clusters)}")
        print(f"  Proposals passing guardrails: {len(passing)}")
        print(f"  Written to: {round_dir}")
        for r in passing:
            name = names.get(r.cluster.cluster_id, "unnamed")
            print(f"    [{r.cluster.cluster_id}] {name} ({len(r.cluster.control_ids)} controls)")


def _cmd_review_proposals(args: argparse.Namespace) -> None:
    from tract.hierarchy import CREHierarchy
    from tract.proposals.review import run_review_session

    round_dir = HUB_PROPOSALS_DIR / f"round_{args.round}"
    if not round_dir.exists():
        print(f"Error: Round directory not found: {round_dir}", file=sys.stderr)
        sys.exit(1)

    hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")
    summary = run_review_session(round_dir, hierarchy, PHASE1C_CROSSWALK_DB_PATH, dry_run=args.dry_run)
    print(f"\nReview summary: {summary}")


def _cmd_tutorial(args: argparse.Namespace) -> None:
    print("TRACT Tutorial — Guided Walkthrough")
    print("=" * 40)

    model_exists = PHASE1D_DEPLOYMENT_MODEL_DIR.exists()
    db_exists = PHASE1C_CROSSWALK_DB_PATH.exists()
    hierarchy_exists = (PROCESSED_DIR / "cre_hierarchy.json").exists()

    if not all([model_exists, db_exists, hierarchy_exists]):
        print("\nPrerequisites missing:")
        if not model_exists:
            print(f"  - Deployment model: {PHASE1D_DEPLOYMENT_MODEL_DIR}")
        if not db_exists:
            print(f"  - Crosswalk database: {PHASE1C_CROSSWALK_DB_PATH}")
        if not hierarchy_exists:
            print(f"  - CRE hierarchy: {PROCESSED_DIR / 'cre_hierarchy.json'}")
        print("\nRun Phase 1C pipeline first to generate these artifacts.")
        return

    print("\nTRACT maps security framework controls to CRE (Common Requirements Enumeration)")
    print("hubs — a shared coordinate system for cross-framework comparison.\n")

    print("Step 1: Assign a control to CRE hubs")
    print("  Try: tract assign 'Ensure AI models are tested for bias before deployment'")
    print("  This encodes your text, compares against 522 hubs, and returns top-5 matches.\n")

    print("Step 2: Compare frameworks")
    print("  Try: tract compare --framework mitre_atlas --framework owasp_ai_exchange")
    print("  Shows which controls map to the same CRE hubs (equivalences) and gaps.\n")

    print("Step 3: Explore the hierarchy")
    print("  Try: tract hierarchy --hub 646-285")
    print("  See the full path, parent, children, and assigned controls for any hub.\n")

    print("Step 4: Prepare a new framework for ingestion")
    print("  Convert a CSV, Markdown, or JSON document into FrameworkOutput JSON:")
    print("  Try: tract prepare --file controls.csv --framework-id new_fw --name 'New Framework'")
    print("  Supports --llm for unstructured documents (PDF, plain text).\n")

    print("Step 5: Validate the prepared output")
    print("  Try: tract validate --file new_fw_prepared.json")
    print("  Checks for errors (block ingest) and warnings (informational).\n")

    print("Step 6: Ingest the framework")
    print("  Try: tract ingest --file new_fw_prepared.json")
    print("  Runs model inference and generates a _review.json for human review.\n")

    print("Step 7: Accept reviewed predictions into the crosswalk DB")
    print("  Edit the _review.json to set review status (accepted/rejected/corrected)")
    print("  Try: tract accept --review new_fw_prepared_review.json\n")

    print("Step 8: Export the crosswalk")
    print("  Try: tract export --format csv")
    print("  Exports all accepted assignments as CSV or JSON.\n")

    print("Step 9: Propose new hubs (advanced)")
    print("  Try: tract propose-hubs")
    print("  Clusters OOD controls to suggest new taxonomy extensions.\n")

    print("For more: https://github.com/rockcyber/TRACT")


def _cmd_publish_hf(args: argparse.Namespace) -> None:
    from tract.config import (
        PHASE1B_CORRECTED_METRICS_PATH,
        PHASE1B_TEXTAWARE_RESULTS_DIR,
        PHASE1C_ECE_GATE_PATH,
        PHASE1D_ARTIFACTS_PATH,
        PHASE1D_CALIBRATION_PATH,
    )
    from tract.publish import publish_to_huggingface

    model_dir = PHASE1D_DEPLOYMENT_MODEL_DIR / "model" / "model"
    bridge_report = BRIDGE_OUTPUT_DIR / "bridge_report.json"

    fold_results = _load_fold_results(
        PHASE1B_TEXTAWARE_RESULTS_DIR, PHASE1B_CORRECTED_METRICS_PATH,
    )

    publish_to_huggingface(
        repo_id=args.repo_id,
        staging_dir=Path(args.staging_dir),
        model_dir=model_dir,
        artifacts_path=PHASE1D_ARTIFACTS_PATH,
        hierarchy_path=PROCESSED_DIR / "cre_hierarchy.json",
        hub_descriptions_path=PROCESSED_DIR / "hub_descriptions_reviewed.json",
        calibration_path=PHASE1D_CALIBRATION_PATH,
        ece_gate_path=PHASE1C_ECE_GATE_PATH,
        bridge_report_path=bridge_report,
        fold_results=fold_results,
        gpu_hours=args.gpu_hours,
        dry_run=args.dry_run,
        skip_upload=args.skip_upload,
    )


def _load_fold_results(
    textaware_dir: Path,
    corrected_path: Path,
    zero_shot_path: Path | None = None,
) -> list[dict]:
    """Load LOFO fold results from Phase 1B artifacts.

    eval count comes from len(predictions.json), NOT n_pairs.
    corrected_metrics must come from the TEXTAWARE experiment.
    """
    from tract.io import load_json

    if zero_shot_path and zero_shot_path.exists():
        zs_data = load_json(zero_shot_path)
        zs_baselines = {
            fw: m.get("hit_at_1", 0)
            for fw, m in zs_data.get("per_framework", {}).items()
        }
    else:
        zs_baselines = {
            "MITRE ATLAS": 0.273,
            "NIST AI 100-2": 0.107,
            "OWASP AI Exchange": 0.619,
            "OWASP Top10 for LLM": 0.333,
            "OWASP Top10 for ML": 0.429,
        }
        logger.warning("Zero-shot baselines loaded from hardcoded fallback")

    fold_names = {
        "MITRE_ATLAS": "MITRE ATLAS",
        "NIST_AI_100-2": "NIST AI 100-2",
        "OWASP_AI_Exchange": "OWASP AI Exchange",
        "OWASP_Top10_for_LLM": "OWASP Top10 for LLM",
        "OWASP_Top10_for_ML": "OWASP Top10 for ML",
    }

    if not corrected_path.exists():
        raise FileNotFoundError(
            f"Corrected metrics not found: {corrected_path}. "
            "Run 'python scripts/phase1b/rescore_predictions.py' first."
        )
    corrected = load_json(corrected_path)
    corrected_folds = corrected.get("per_fold", {})

    results = []
    for file_key, display_name in fold_names.items():
        fold_dir = textaware_dir / f"fold_{file_key}"
        predictions_path = fold_dir / "predictions.json"
        summary_path = textaware_dir / f"fold_{file_key}_summary.json"

        if not predictions_path.exists():
            logger.warning("Predictions not found: %s", predictions_path)
            continue

        predictions = load_json(predictions_path)
        n_eval = len(predictions)

        summary = load_json(summary_path) if summary_path.exists() else {}
        metrics = summary.get("metrics", {})

        corrected_fold = corrected_folds.get(display_name, {})
        if not corrected_fold:
            raise ValueError(
                f"No corrected metrics for fold '{display_name}'. "
                f"Available keys: {list(corrected_folds.keys())}"
            )
        hit_any = corrected_fold.get("hit_at_1", metrics.get("hit_at_1", 0))

        results.append({
            "fold": display_name,
            "hit1": metrics.get("hit_at_1", 0),
            "zs_hit1": zs_baselines.get(display_name, 0),
            "n": n_eval,
            "hit_any": hit_any,
        })

    return results


def _cmd_bridge(args: argparse.Namespace) -> None:
    if args.commit:
        if not args.candidates:
            print("Error: --commit requires --candidates <path>", file=sys.stderr)
            sys.exit(1)
        from tract.bridge.review import commit_bridges
        from tract.io import load_json

        candidates_path = Path(args.candidates)
        candidates_data = load_json(candidates_path)
        hierarchy_path = PROCESSED_DIR / "cre_hierarchy.json"
        report_path = Path(args.output_dir) / "bridge_report.json"

        report = commit_bridges(candidates_data, hierarchy_path, report_path)
        print(f"Accepted: {report['counts']['accepted']}")
        print(f"Rejected: {report['counts']['rejected']}")
        print(f"Hierarchy updated: {hierarchy_path}")
        print(f"Report: {report_path}")
    else:
        from tract.bridge import run_bridge_analysis
        from tract.config import PHASE1D_ARTIFACTS_PATH, TRAINING_DIR

        run_bridge_analysis(
            artifacts_path=PHASE1D_ARTIFACTS_PATH,
            hub_links_path=TRAINING_DIR / "hub_links_by_framework.json",
            hierarchy_path=PROCESSED_DIR / "cre_hierarchy.json",
            output_dir=Path(args.output_dir),
            top_k=args.top_k,
            skip_descriptions=args.skip_descriptions,
        )


def _cmd_import_ground_truth(args: argparse.Namespace) -> None:
    from tract.crosswalk.ground_truth import import_ground_truth, run_uncovered_inference

    summary = import_ground_truth(
        PHASE1C_CROSSWALK_DB_PATH,
        TRAINING_DIR / "hub_links_by_framework.json",
        dry_run=args.dry_run,
    )
    logger.info("Ground truth import: %s", summary)
    print(
        f"Ground truth import: imported={summary['imported']}, "
        f"skipped_duplicate={summary['skipped_duplicate']}, "
        f"unresolved={summary['unresolved']}"
    )

    if not args.dry_run:
        inf_summary = run_uncovered_inference(
            PHASE1C_CROSSWALK_DB_PATH,
            PHASE1D_DEPLOYMENT_MODEL_DIR,
            dry_run=args.dry_run,
        )
        logger.info("Uncovered inference: %s", inf_summary)
        print(
            f"Uncovered inference: inserted={inf_summary['total_inserted']}, "
            f"skipped={inf_summary['skipped_duplicate']}"
        )


def _cmd_review_export(args: argparse.Namespace) -> None:
    from tract.review.export import generate_review_export
    from tract.review.guide import generate_hub_reference, generate_reviewer_guide

    output_dir = Path(args.output)
    model_dir = Path(args.model_dir)

    metadata = generate_review_export(
        PHASE1C_CROSSWALK_DB_PATH,
        model_dir,
        output_dir,
        PHASE1D_CALIBRATION_PATH,
    )
    generate_reviewer_guide(output_dir, metadata)
    generate_hub_reference(PHASE1C_CROSSWALK_DB_PATH, output_dir)

    print(
        f"Review export: {metadata['total_predictions']} predictions "
        f"({metadata['calibration_items']} calibration items)"
    )
    print(f"Priority breakdown: {metadata['priority_breakdown']}")
    print(f"Files written to {output_dir}")


def _cmd_review_validate(args: argparse.Namespace) -> None:
    from tract.review.validate import validate_review_json

    result = validate_review_json(Path(args.input), PHASE1C_CROSSWALK_DB_PATH)

    if result.errors:
        print(f"VALIDATION FAILED — {len(result.errors)} error(s):")
        for e in result.errors:
            print(f"  ERROR: {e}")
    if result.warnings:
        for w in result.warnings:
            print(f"  WARNING: {w}")
    if result.valid:
        msg = "Validation PASSED"
        if result.warnings:
            msg += f" ({len(result.warnings)} warning(s))"
        print(msg)
    sys.exit(0 if result.valid else 1)


def _cmd_review_import(args: argparse.Namespace) -> None:
    from tract.review.import_review import apply_review_decisions
    from tract.review.metrics import compute_review_metrics

    summary = apply_review_decisions(
        PHASE1C_CROSSWALK_DB_PATH,
        Path(args.input),
        args.reviewer,
    )
    print(
        f"Review import: accepted={summary['accepted']}, "
        f"rejected={summary['rejected']}, reassigned={summary['reassigned']}, "
        f"skipped_pending={summary['skipped_pending']}, "
        f"skipped_calibration={summary['skipped_calibration']}"
    )

    review_data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    metrics = compute_review_metrics(
        PHASE1C_CROSSWALK_DB_PATH,
        review_data,
        PHASE3_REVIEW_OUTPUT_DIR / "review_metrics.json",
    )
    print(
        f"Metrics: completion={metrics['coverage']['completion_pct']:.1f}%, "
        f"quality_score={metrics.get('calibration', {}).get('quality_score', 'N/A')}"
    )


def _cmd_publish_dataset(args: argparse.Namespace) -> None:
    from tract.dataset.bundle import bundle_dataset
    from tract.dataset.card import generate_dataset_card
    from tract.dataset.publish import publish_dataset

    staging_dir = Path(args.staging_dir)

    stats = bundle_dataset(
        PHASE1C_CROSSWALK_DB_PATH,
        staging_dir,
        hierarchy_path=TRAINING_DIR / "cre_hierarchy.json",
        hub_descriptions_path=TRAINING_DIR / "hub_descriptions.json",
        bridge_report_path=PHASE3_REVIEW_OUTPUT_DIR / "bridge_report.json",
        review_metrics_path=PHASE3_REVIEW_OUTPUT_DIR / "review_metrics.json",
    )
    print(f"Dataset bundled: {stats['total_rows']} rows, {stats['frameworks']} frameworks")

    fm_path = staging_dir / "framework_metadata.json"
    rm_path = staging_dir / "review_metrics.json"
    framework_metadata = json.loads(fm_path.read_text(encoding="utf-8")) if fm_path.exists() else []
    review_metrics = json.loads(rm_path.read_text(encoding="utf-8")) if rm_path.exists() else {}

    generate_dataset_card(staging_dir, framework_metadata, review_metrics, stats)
    print(f"Dataset card generated at {staging_dir / 'README.md'}")

    publish_dataset(args.repo_id, staging_dir, dry_run=args.dry_run, skip_upload=args.skip_upload)
    if args.dry_run:
        print("Dry run complete — no upload")
    elif args.skip_upload:
        print(f"Staging dir ready at {staging_dir}")
    else:
        print(f"Published to {args.repo_id}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    handlers = {
        "assign": _cmd_assign,
        "bridge": _cmd_bridge,
        "compare": _cmd_compare,
        "ingest": _cmd_ingest,
        "accept": _cmd_accept,
        "export": _cmd_export,
        "export-canonical": _cmd_export_canonical,
        "hierarchy": _cmd_hierarchy,
        "import-ground-truth": _cmd_import_ground_truth,
        "propose-hubs": _cmd_propose_hubs,
        "publish-dataset": _cmd_publish_dataset,
        "publish-hf": _cmd_publish_hf,
        "review-export": _cmd_review_export,
        "review-import": _cmd_review_import,
        "review-proposals": _cmd_review_proposals,
        "review-validate": _cmd_review_validate,
        "tutorial": _cmd_tutorial,
        "validate": _cmd_validate,
        "prepare": _cmd_prepare,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
