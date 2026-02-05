from pathlib import Path
import json

from research_assistant_ai.data.ingest import IngestConfig
from research_assistant_ai.assistant.orchestrator import ResearchAssistantAI

HERE = Path(__file__).resolve().parent
data = HERE / "sample_data"

dissertation_docs = [
    Path("/mnt/data/Dissertation_Concept_3_Template_Matched.docx"),
    Path("/mnt/data/ShippJTIM7211_8.docx"),
]

cfg = IngestConfig(
    gdelt_events_csv=data/"gdelt_events.csv",
    eurepoc_csv=data/"eurepoc.csv",
    worldbank_covariates_csv=data/"worldbank.csv",
    nvd_cve_csv=data/"nvd_cve.csv",
    dependency_updates_csv=data/"dependency_updates.csv",
    telemetry_signals_csv=data/"telemetry_signals.csv",
)

ai = ResearchAssistantAI(cfg, memory_dir=HERE/"research_memory", dissertation_paths=dissertation_docs, export_dir=HERE/"exports")
outputs = ai.run()

print("\n=== Phase 5: Causal suite summary ===")
print(json.dumps({k: outputs.causal_suite[k] for k in ['treated_unit','cutoff_month','did','synthetic_control','its','rdd']}, indent=2))

print("\n=== Robustness counts ===")
print(json.dumps({k: len(v) for k,v in outputs.robustness.items()}, indent=2))

print("\n=== Exports ===")
print(json.dumps(outputs.exports, indent=2))
