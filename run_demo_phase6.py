from pathlib import Path
import json
import pandas as pd

from research_assistant_ai.data.adapters.icews_adapter import ICEWSConfig, load_icews_events
from research_assistant_ai.data.adapters.eurepoc_adapter import EuRepoCConfig, load_eurepoc_incidents
from research_assistant_ai.data.adapters.sanctions_adapter import SanctionsConfig, load_sanctions
from research_assistant_ai.data.panel_builder import build_country_month_panel
from research_assistant_ai.assistant.orchestrator_phase6 import run_phase6_russia_causal

HERE = Path(__file__).resolve().parent
data = HERE / "sample_data"
exports = HERE / "exports"

# For MVP reproducibility, we generate a sanctions CSV aligned to sample months.
# Replace this file with your real sanctions timeline CSV when available.
sanctions_path = data / "sanctions.csv"
if not sanctions_path.exists():
    # basic synthetic timeline: sanctions begin mid-sample for RUS
    # NOTE: This is only a demo file. For dissertation runs, replace with real data.
    months = pd.read_csv(data / "gdelt_events.csv")["month"].astype(str).unique().tolist()
    months = sorted(months)
    start = months[len(months)//2]
    rows = []
    for m in months:
        if m >= start:
            rows.append({"sanction_date": f"{m}-01", "country_iso3": "RUS", "label": "US-led sanction package", "intensity": 1})
    pd.DataFrame(rows).to_csv(sanctions_path, index=False)

icews = load_icews_events(ICEWSConfig(csv_path=data/"icews_like_events.csv"))
eurepoc = load_eurepoc_incidents(EuRepoCConfig(csv_path=data/"eurepoc_like_incidents.csv"))
sanctions = load_sanctions(SanctionsConfig(csv_path=sanctions_path))

panel = build_country_month_panel(
    icews_events=icews,
    eurepoc_incidents=eurepoc,
    sanctions=sanctions,
)

out = run_phase6_russia_causal(
    panel,
    treated_country="RUS",
    intervention_month=None,
    x_cols=[],
    export_dir=exports,
)

print("\n=== Phase 6: Russia-focused causal summary ===")
print(json.dumps({
    "treated_country": out.russia_results["treated_country"],
    "intervention_month": out.russia_results["intervention_month"],
    "did": out.russia_results["did"],
    "synthetic_control": {k: out.russia_results["synthetic_control"][k] for k in ["pre_rmse","post_gap_mean"]},
    "its": out.russia_results["its"],
    "rdd": out.russia_results["rdd"],
    "exports": out.exports
}, indent=2))
