from pathlib import Path
import json

from research_assistant_ai.assistant.orchestrator_phase7 import run_phase7
from research_assistant_ai.assistant.orchestrator_phase8 import run_phase8_rigor
from research_assistant_ai.assistant.orchestrator_phase9 import run_phase9_from_phase8_exports
from research_assistant_ai.data.pipelines.harmonize_and_audit import Phase7Schema

HERE = Path(__file__).resolve().parent
data = HERE / "sample_data"
exports = HERE / "exports_phase9"

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

p7 = run_phase7(
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

intervention_month = p7.phase6_outputs.russia_results["intervention_month"]

p8 = run_phase8_rigor(
    p7.panel,
    treated_country="RUS",
    intervention_month=intervention_month,
    treatment_col="sanctions_count",
    covariates=[],
    export_dir=exports,
)

# Build Phase 9 draft
p9 = run_phase9_from_phase8_exports(
    phase8_export_dir=exports,
    phase8_report={
        "treated_country": p8.treated_country,
        "intervention_month": p8.intervention_month,
        "did": p8.did,
        "its": p8.its,
        "diagnostics": p8.diagnostics,
        "synthetic_control": p8.synthetic_control,
    },
    out_dir=exports,
)

print("\n=== Phase 9: Dissertation draft ===")
print(json.dumps({"results_docx": p9.results_docx, "inputs_used": p9.inputs_used}, indent=2))
