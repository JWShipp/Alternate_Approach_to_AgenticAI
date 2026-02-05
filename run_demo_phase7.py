from pathlib import Path
import json

from research_assistant_ai.assistant.orchestrator_phase7 import run_phase7
from research_assistant_ai.data.pipelines.harmonize_and_audit import Phase7Schema

HERE = Path(__file__).resolve().parent
data = HERE / "sample_data"
exports = HERE / "exports_phase7"

schema = Phase7Schema(
    icews_date_col="event_date",
    icews_country_col="country_iso3",
    icews_intensity_col="intensity",
    eurepoc_date_col="incident_date",
    eurepoc_country_col="target_country_iso3",
    eurepoc_severity_col="severity",
    sanctions_date_col="sanction_date",
    sanctions_country_col="country_iso3",
    sanctions_intensity_col="intensity",
)

out = run_phase7(
    icews_csv=data/"icews_like_events.csv",
    eurepoc_csv=data/"eurepoc_like_incidents.csv",
    sanctions_csv=data/"sanctions.csv",
    iso3_mapping_csv=None,
    schema=schema,
    export_dir=exports,
    treated_country="RUS",
    intervention_month=None,
    x_cols=[],
)

print("\n=== Phase 7: Ingest audit summary ===")
print(json.dumps(out.ingest_audit, indent=2))

print("\n=== Phase 7: Russia causal summary (from Phase 6 engine) ===")
print(json.dumps({
    "treated_country": out.phase6_outputs.russia_results["treated_country"],
    "intervention_month": out.phase6_outputs.russia_results["intervention_month"],
    "did": out.phase6_outputs.russia_results["did"],
    "synthetic_control": {k: out.phase6_outputs.russia_results["synthetic_control"][k] for k in ["pre_rmse","post_gap_mean"]},
    "its": out.phase6_outputs.russia_results["its"],
    "rdd": out.phase6_outputs.russia_results["rdd"],
    "exports": out.phase6_outputs.exports
}, indent=2))
