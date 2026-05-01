#!/usr/bin/env python3
"""Direct CSV import into OpenCRE fork DB, bypassing Flask and graph loading.

Parses TRACT-generated OpenCRE CSV and writes standard nodes + CRE links
directly via SQLAlchemy. Much faster than the REST API for small imports.

Usage:
    cd ~/github_projects/OpenCRE
    .venv/bin/python ~/github_projects/TRACT/scripts/direct_opencre_import.py \
        --csv ~/github_projects/TRACT/opencre_export/NIST_AI_600-1.csv \
        --db ./cre.db
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPENCRE_DIR = Path.home() / "github_projects" / "OpenCRE"
sys.path.insert(0, str(OPENCRE_DIR))

os.environ.setdefault("INSECURE_REQUESTS", "1")
os.environ.setdefault("NO_LOGIN", "1")


def parse_csv(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def import_rows(rows: list[dict], db_path: str) -> dict:
    db_uri = f"sqlite:///{os.path.abspath(db_path)}"
    os.environ["SQLALCHEMY_DATABASE_URI"] = db_uri

    from application.defs import cre_defs as defs
    from application.utils.external_project_parsers.parsers import export_format_parser

    logger.info("Parsing %d CSV rows...", len(rows))
    documents = export_format_parser.parse_export_format(rows)

    cre_key = defs.Credoctypes.CRE
    cres = documents.get(cre_key, [])
    standards_by_name = {k: v for k, v in documents.items() if k != cre_key}

    logger.info("Parsed %d CREs and %d standard groups", len(cres), len(standards_by_name))
    for name, entries in standards_by_name.items():
        logger.info("  Standard '%s': %d entries", name, len(entries))

    from application import create_app
    app = create_app(conf=type("Conf", (), {
        "SQLALCHEMY_DATABASE_URI": db_uri,
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SQLALCHEMY_RECORD_QUERIES": False,
        "ITEMS_PER_PAGE": 20,
        "SLOW_DB_QUERY_TIME": 0.5,
        "GAP_ANALYSIS_OPTIMIZED": False,
        "ENVIRONMENT": "CLI",
    })())

    with app.app_context():
        from application.database import db as db_mod

        collection = db_mod.Node_collection()

        nodes_created = 0
        links_created = 0

        for cre_obj in cres:
            existing = collection.get_CREs(external_id=cre_obj.id)
            if not existing:
                logger.warning("CRE %s (%s) not found in DB — skipping", cre_obj.id, cre_obj.name)
                continue

            db_cre = db_mod.dbCREfromCRE(existing[0])

            for link in cre_obj.links:
                if not isinstance(link.document, defs.Standard):
                    continue

                std = link.document
                db_node = db_mod.Node.query.filter(
                    db_mod.Node.name == std.name,
                    db_mod.Node.section == std.section,
                    db_mod.Node.section_id == std.sectionID,
                ).first()

                if db_node:
                    logger.info("  Node exists: %s %s (id=%s)", std.name, std.sectionID, db_node.id)
                else:
                    db_node = collection.add_node(std)
                    nodes_created += 1
                    logger.info("  Created node: %s %s (%s)", std.name, std.sectionID, std.section)

                collection.add_link(
                    cre=db_cre,
                    node=db_node,
                    ltype=link.ltype,
                )
                links_created += 1
                logger.info("  Linked: CRE %s -> %s %s", cre_obj.id, std.name, std.sectionID)

        return {
            "cres_referenced": len(cres),
            "standard_groups": len(standards_by_name),
            "nodes_created": nodes_created,
            "links_created": links_created,
        }


def main():
    parser = argparse.ArgumentParser(description="Direct OpenCRE CSV import")
    parser.add_argument("--csv", required=True, help="Path to TRACT-generated CSV")
    parser.add_argument("--db", required=True, help="Path to OpenCRE SQLite database")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        logger.error("CSV not found: %s", args.csv)
        sys.exit(1)
    if not Path(args.db).exists():
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    rows = parse_csv(args.csv)
    result = import_rows(rows, args.db)

    print("\n=== Import Summary ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
