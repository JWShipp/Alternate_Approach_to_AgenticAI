# ResearchAssistantAI MVP (Phase 2)

This codebase implements a modular, non-LLM research assistant architecture:
- Explicit probabilistic beliefs and action selection (ActiveInferencePlanner)
- Causal estimator scaffolding (Interrupted Time Series and Regression Discontinuity)
- ExperimentRunner for automated specification search (lag search MVP)
- ResearchMemory for auditable JSONL logging of experiments
- Dissertation doc parser (optional) to extract RQs and keywords for pipeline alignment

## Quickstart

```bash
pip install -r requirements.txt
python run_demo_phase2.py
```

## Replace sample data

Edit `IngestConfig` in `run_demo_phase2.py` to point to your real CSV extracts.

Expected columns are described in `research_assistant_ai/data/ingest.py`.

## Outputs

- Model summaries
- Experiment ranking table (top 15)
- Verification checks
- Research logs in `research_memory/`


## Phase 3

Run the specification search layer:

```bash
python run_demo_phase3.py
```

Exports CSV tables to `exports/`:
- `phase2_lag_experiments.csv`
- `phase3_spec_search_results.csv`


## Phase 4

Phase 4 adds:
- A real data contract validator for the core panel
- Hypothesis registry + experiment protocol logging (JSONL)
- Multiple testing correction (Benjamini-Hochberg)
- Automatic exports: Excel, Word, and a figure of top specifications

Run:

```bash
python run_demo_phase4.py
```

Outputs saved to `exports/`.


## Phase 5

Adds a causal suite + robustness checks:
- Difference-in-Differences (DiD)
- Event study
- Synthetic control
- Robustness: RDD bandwidth sweep, placebo ITS cutoffs, synthetic control placebos

Run:

```bash
python run_demo_phase5.py
```


## Phase 6

Real-data dissertation execution layer (CSV adapters):
- ICEWS adapter (CSV)
- EuRepoC adapter (CSV)
- Sanctions timeline adapter (CSV)
- Integrated country-month panel builder
- Russia-focused causal run (DiD, event study, synthetic control, ITS, RDD) + robustness

Run:

```bash
python run_demo_phase6.py
```

Outputs:
- exports/phase6_russia_results.xlsx
- exports/phase6_russia_summary.docx
- exports/phase6_russia_event_study.png


## Phase 7

Real-data integration layer (scale ingestion + harmonization + audits + dataset versioning).

- Chunked CSV adapters with configurable schemas for ICEWS / EuRepoC / sanctions.
- ISO3 normalization with optional user-supplied alias mapping CSV.
- Missingness audits and sanity checks.
- Dataset fingerprinting (SHA-256) + versioned panel outputs.
- Runs the Phase 6 Russia causal engine on the resulting panel.

Demo:

```bash
python run_demo_phase7.py
```

CLI:

```bash
python run_ingest_phase7_cli.py --icews ... --eurepoc ... --export_dir ... --icews_date_col ... --icews_country_col ... --eurepoc_date_col ... --eurepoc_country_col ...
```


## Phase 8

Methodological rigor layer.

Adds:
- Parallel trends pre-test (treated × trend in pre-period)
- Event study pre-period joint test (Fisher's method MVP)
- Synthetic control donor pool optimization (correlation screening + greedy RMSE selection)
- Covariate-set sensitivity for DiD (subsets of covariates)

Run:

```bash
python run_demo_phase8.py
```

Outputs:
- exports_phase8/phase8_rigor_results.xlsx
- exports_phase8/phase8_rigor_summary.docx
- exports_phase8/phase8_event_study.png


## Phase 9

Dissertation writing automation layer (APA-style tables + narrative draft).

Adds:
- Converts Phase 8 outputs into a dissertation-ready Results draft in Word (.docx)
- Inserts APA-style table number and title lines plus table notes
- Generates narrative paragraphs with method placeholders requiring citation insertion

Run:

```bash
python run_demo_phase9.py
```

Outputs:
- exports_phase9/phase9_dissertation_results_draft.docx


## Phase 10

Interactive research cockpit (Streamlit).

Adds:
- GUI to upload datasets or point to local paths
- Schema mapping UI for column names
- One-click execution of Phase 7 → Phase 8 → Phase 9
- Output preview for key tables

Run:

```bash
pip install -r requirements.txt
streamlit run app_phase10_streamlit.py
```


## Phase 11

Advanced research agent layer (beyond LLM/RAG).

Adds:
- Structured research memory for hypotheses (not text retrieval)
- Experiment registry (JSONL run tracking)
- Knowledge graph of datasets, variables, hypotheses, and results
- Bayesian-flavored model comparison via BIC weights over candidate DiD specifications
- Agent that proposes hypotheses, runs Phase 8 rigor tests, updates hypothesis status, logs a run, and updates the graph

Run:

```bash
python run_demo_phase11.py
```


## Final Integration Deliverables

- `OPERATING_MANUAL.md` and `OPERATING_MANUAL.docx`: step-by-step setup and usage instructions.
- `QUICKSTART.txt`: minimal commands to verify and run the system.
Repo push verified on 2026-02-05 09:59:22
