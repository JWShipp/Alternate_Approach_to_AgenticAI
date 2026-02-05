from pathlib import Path
import json

from research_assistant_ai.data.ingest import IngestConfig
from research_assistant_ai.assistant.orchestrator import ResearchAssistantAI

HERE = Path(__file__).resolve().parent
data = HERE / "sample_data"

# Point these at your dissertation documents if available
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

ai = ResearchAssistantAI(cfg, memory_dir=HERE/"research_memory", dissertation_paths=dissertation_docs)
outputs = ai.run()

print("\n=== Dissertation signals ===")
print(json.dumps(outputs.dissertation_signals, indent=2))

print("\n=== Geopolitics model summary ===")
print(json.dumps(outputs.geopolitics_model_summary, indent=2))

print("\n=== Vulnerability model summary ===")
print(json.dumps(outputs.vuln_model_summary, indent=2))

print("\n=== Experiment ranking (top 15) ===")
print(json.dumps(outputs.experiment_ranking, indent=2))

print("\n=== Next actions ===")
print(json.dumps(outputs.next_actions, indent=2))

print("\n=== Verification spine ===")
print(json.dumps(outputs.verification, indent=2))
