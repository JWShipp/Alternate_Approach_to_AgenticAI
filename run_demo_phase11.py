from pathlib import Path
import json

from research_assistant_ai.assistant.orchestrator_phase7 import run_phase7
from research_assistant_ai.data.pipelines.harmonize_and_audit import Phase7Schema
from research_assistant_ai.agent.research_agent import ResearchAgent

HERE = Path(__file__).resolve().parent
data = HERE / "sample_data"
exports = HERE / "exports_phase11"

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

agent = ResearchAgent(workdir=exports / "agent_state")

h = agent.propose_hypothesis(
    treated_country="RUS",
    outcome="cyber_incidents",
    treatment="sanctions_count",
    expected_direction="decrease",
    statement="In the post-intervention period, sanctions are associated with a decrease in cyber incident volume for the treated unit, conditional on observed covariates.",
    priors={"alpha": 0.05},
    notes="MVP hypothesis for Phase 11 agent demonstration.",
)

out = agent.test_hypothesis(
    panel=p7.panel,
    hypothesis_id=h.hypothesis_id,
    intervention_month=intervention_month,
    export_dir=exports / "agent_exports",
)

print("\n=== Phase 11 Agent Outputs ===")
print(json.dumps({
    "hypothesis": out.hypothesis,
    "top_model": out.model_comparison[0] if out.model_comparison else None,
    "registry_run_id": out.registry_run_id,
    "kg_path": out.kg_path,
    "memory_path": out.memory_path,
    "exports": out.phase8["exports"],
}, indent=2))
