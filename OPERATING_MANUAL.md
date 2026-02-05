# Dissertation Research Assistant AI Operating Manual (Final Integrated System)

## 1. System purpose
This system is a modular research assistant designed to support quantitative dissertation workflows that integrate geopolitical events (e.g., ICEWS-style event streams), cyber incident data (e.g., EuRepoC-style incidents), and sanctions timelines, with an emphasis on causal inference and reproducibility.

The system implements:
- Phase 7: Real-data ingestion, harmonization, audits, and dataset versioning.
- Phase 8: Methodological rigor diagnostics and sensitivity checks.
- Phase 9: Automated Results drafting (Word document) with APA-style table structure.
- Phase 10: Streamlit user interface for interactive daily use.
- Phase 11: Hypothesis agent with structured memory, experiment registry, and knowledge graph.

## 2. Repository structure
- `research_assistant_ai/`: Python package containing ingestion, modeling, reporting, UI orchestration, and agent modules.
- `sample_data/`: Small demonstration CSVs that prove the pipeline runs end to end.
- `app_phase10_streamlit.py`: Streamlit “research cockpit” entrypoint (Phase 10, with Phase 11 agent section).
- `run_demo_phase7.py` … `run_demo_phase11.py`: End-to-end demo scripts for each phase.
- `run_ingest_phase7_cli.py`: CLI ingestion runner for real datasets.
- `exports_*`: Output folders created when you run demos.

## 3. Install and setup

### 3.1 Prerequisites
- Python 3.10+ recommended.
- Git optional but strongly recommended for reproducibility.
- Disk space: large for real ICEWS/incident datasets.

### 3.2 Create a virtual environment (recommended)
Windows (PowerShell):
- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`

macOS/Linux (bash/zsh):
- `python3 -m venv .venv`
- `source .venv/bin/activate`

### 3.3 Install dependencies
- `pip install -r requirements.txt`

## 4. Quick verification (end-to-end)
Run the Phase 11 demo, which runs Phase 7 ingestion → Phase 8 rigor → Phase 11 agent test.

- `python run_demo_phase11.py`

Expected outputs:
- A folder `exports_phase11/` (or similar) containing:
  - `phase8_rigor_results.xlsx`
  - `phase8_rigor_summary.docx`
  - `phase8_event_study.png`
  - `phase9_dissertation_results_draft.docx`
  - `agent_state/` including `research_memory.json`, `knowledge_graph.json`, and `experiment_registry/runs.jsonl`

## 5. Daily use via the Streamlit cockpit (Phase 10 + Phase 11)
Start the application:
- `streamlit run app_phase10_streamlit.py`

Workflow:
1) Choose “Upload CSVs” for small files or “Local file paths” for large datasets.
2) Map dataset columns in the Schema panel.
3) Run the pipeline (Phase 7 → 8 → 9).
4) Review diagnostics, preview tables, and open exported Word/Excel outputs.
5) Optionally open the Phase 11 agent expander to propose and test a hypothesis.

## 6. Using real dissertation datasets

### 6.1 Prepare inputs
You need three inputs (sanctions optional but recommended):
- ICEWS-like events CSV
- EuRepoC-like incident CSV
- Sanctions timeline CSV

Optional:
- ISO3 alias mapping CSV with columns: `alias, iso3`

### 6.2 Run ingestion + causal suite via CLI (recommended for large datasets)
Example (replace column names to match your files):

`python run_ingest_phase7_cli.py \
  --icews /path/to/icews.csv \
  --eurepoc /path/to/eurepoc.csv \
  --sanctions /path/to/sanctions.csv \
  --export_dir exports_real_run \
  --treated_country RUS \
  --icews_date_col <YOUR_ICEWS_DATE_COL> \
  --icews_country_col <YOUR_ICEWS_COUNTRY_COL> \
  --eurepoc_date_col <YOUR_EUREPOC_DATE_COL> \
  --eurepoc_country_col <YOUR_EUREPOC_COUNTRY_COL>`

Outputs:
- `exports_real_run/data_audit/` (missingness and checks)
- `exports_real_run/datasets/<UTCSTAMP>/panel_country_month.csv` (versioned panel)
- Causal suite artifacts and Phase 9 Word draft in the export directory.

## 7. “Training” in this system
This system does not train a large language model. “Training” means:
- Selecting datasets and unit of analysis (country-month by default).
- Choosing interventions (e.g., EO 14024 month).
- Running sensitivity and diagnostics until specifications are defensible.
- Logging runs and decisions in the registry and research memory.
- Iterating on hypotheses using the Phase 11 agent.

## 8. Reproducibility and committee defense practices
Recommended practices:
- Treat `exports_*` as generated artifacts and keep them under version control as needed.
- Keep a stable “inputs” folder with raw dataset hashes.
- Use the Phase 7 dataset fingerprinting output to document exact inputs.
- Preserve `agent_state/experiment_registry/runs.jsonl` to show a timestamped audit trail.

## 9. Extending to other countries and interventions
- Change `treated_country` (ISO3).
- Provide `intervention_month` (YYYY-MM).
- Re-run Phase 8/9 via Streamlit or CLI.
- Use the Phase 11 agent to track hypotheses per country and intervention.

## 10. Troubleshooting
- If CSV ingestion fails: check column mapping, date formats, and whether separators/quotes are nonstandard.
- If pre-period is short: donor optimization will safely fall back to baseline synthetic control.
- If missingness is high: check panel builder logic and your join keys (country and month).

## 11. What to do next
Your next real step is to replace `sample_data/` with your dissertation datasets and run Phase 7 ingestion at scale, then use Phase 8 diagnostics to settle on defensible specifications for your primary case study (Russia) before extending to other countries.

