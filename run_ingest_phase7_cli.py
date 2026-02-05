from __future__ import annotations
import argparse
from pathlib import Path
import json

from research_assistant_ai.assistant.orchestrator_phase7 import run_phase7
from research_assistant_ai.data.pipelines.harmonize_and_audit import Phase7Schema

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 7: Real-data ingestion + audit + Russia causal run.")
    ap.add_argument("--icews", required=True, type=Path)
    ap.add_argument("--eurepoc", required=True, type=Path)
    ap.add_argument("--sanctions", required=False, type=Path, default=None)
    ap.add_argument("--iso3_map", required=False, type=Path, default=None)
    ap.add_argument("--export_dir", required=True, type=Path)
    ap.add_argument("--treated_country", required=False, type=str, default="RUS")
    ap.add_argument("--intervention_month", required=False, type=str, default=None)

    # Schema args
    ap.add_argument("--icews_date_col", required=True, type=str)
    ap.add_argument("--icews_country_col", required=True, type=str)
    ap.add_argument("--icews_intensity_col", required=False, type=str, default=None)

    ap.add_argument("--eurepoc_date_col", required=True, type=str)
    ap.add_argument("--eurepoc_country_col", required=True, type=str)
    ap.add_argument("--eurepoc_severity_col", required=False, type=str, default=None)

    ap.add_argument("--sanctions_date_col", required=False, type=str, default="sanction_date")
    ap.add_argument("--sanctions_country_col", required=False, type=str, default="country_iso3")
    ap.add_argument("--sanctions_intensity_col", required=False, type=str, default=None)

    args = ap.parse_args()

    schema = Phase7Schema(
        icews_date_col=args.icews_date_col,
        icews_country_col=args.icews_country_col,
        icews_intensity_col=args.icews_intensity_col,
        eurepoc_date_col=args.eurepoc_date_col,
        eurepoc_country_col=args.eurepoc_country_col,
        eurepoc_severity_col=args.eurepoc_severity_col,
        sanctions_date_col=args.sanctions_date_col,
        sanctions_country_col=args.sanctions_country_col,
        sanctions_intensity_col=args.sanctions_intensity_col,
    )

    out = run_phase7(
        icews_csv=args.icews,
        eurepoc_csv=args.eurepoc,
        sanctions_csv=args.sanctions,
        iso3_mapping_csv=args.iso3_map,
        schema=schema,
        export_dir=args.export_dir,
        treated_country=args.treated_country,
        intervention_month=args.intervention_month,
        x_cols=[],
    )

    print(json.dumps(out.ingest_audit, indent=2))
    print(json.dumps(out.phase6_outputs.russia_results, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
