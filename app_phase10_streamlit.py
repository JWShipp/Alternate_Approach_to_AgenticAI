from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import streamlit as st

from research_assistant_ai.assistant.orchestrator_phase7 import run_phase7
from research_assistant_ai.assistant.orchestrator_phase8 import run_phase8_rigor
from research_assistant_ai.assistant.orchestrator_phase9 import run_phase9_from_phase8_exports
from research_assistant_ai.agent.research_agent import ResearchAgent
from research_assistant_ai.data.pipelines.harmonize_and_audit import Phase7Schema

st.set_page_config(page_title="Dissertation Research Assistant (Phase 10)", layout="wide")

st.title("Dissertation Research Assistant AI")
st.caption("Phase 10: Interactive research cockpit for ingestion, causal runs, diagnostics, and dissertation outputs.")

with st.sidebar:
    st.header("Inputs")
    st.write("Upload CSVs or point to local paths. For large files, local paths are recommended.")
    mode = st.radio("Input mode", ["Upload CSVs", "Local file paths"], index=0)

    icews_file = eurepoc_file = sanctions_file = iso3_map_file = None
    icews_path = eurepoc_path = sanctions_path = iso3_map_path = ""

    if mode == "Upload CSVs":
        icews_file = st.file_uploader("ICEWS CSV", type=["csv"])
        eurepoc_file = st.file_uploader("EuRepoC CSV", type=["csv"])
        sanctions_file = st.file_uploader("Sanctions CSV (optional)", type=["csv"])
        iso3_map_file = st.file_uploader("ISO3 alias mapping CSV (optional)", type=["csv"])
    else:
        icews_path = st.text_input("ICEWS CSV path", value=str(Path("sample_data/icews_like_events.csv")))
        eurepoc_path = st.text_input("EuRepoC CSV path", value=str(Path("sample_data/eurepoc_like_incidents.csv")))
        sanctions_path = st.text_input("Sanctions CSV path (optional)", value=str(Path("sample_data/sanctions.csv")))
        iso3_map_path = st.text_input("ISO3 mapping CSV path (optional)", value="")

    st.header("Schema mapping")
    st.write("Set the column names in your source datasets.")
    icews_date_col = st.text_input("ICEWS date column", value="event_date")
    icews_country_col = st.text_input("ICEWS country column", value="country_iso3")
    icews_intensity_col = st.text_input("ICEWS intensity column (optional)", value="intensity")

    eurepoc_date_col = st.text_input("EuRepoC date column", value="incident_date")
    eurepoc_country_col = st.text_input("EuRepoC country column", value="target_country_iso3")
    eurepoc_severity_col = st.text_input("EuRepoC severity column (optional)", value="severity")

    sanctions_date_col = st.text_input("Sanctions date column", value="sanction_date")
    sanctions_country_col = st.text_input("Sanctions country column", value="country_iso3")
    sanctions_intensity_col = st.text_input("Sanctions intensity column (optional)", value="intensity")

    st.header("Run configuration")
    treated_country = st.text_input("Treated country ISO3", value="RUS")
    intervention_month = st.text_input("Intervention month (YYYY-MM) optional", value="")
    covariates = st.text_input("Additional covariates (comma-separated) optional", value="")
    export_dir = st.text_input("Export directory", value="exports_phase10")
    run_btn = st.button("Run pipeline (Phase 7 → 8 → 9)")

def _persist_upload(upload, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(upload.getbuffer())
    return target

def _get_paths() -> tuple[Path, Path, Path|None, Path|None]:
    if mode == "Upload CSVs":
        if icews_file is None or eurepoc_file is None:
            st.error("ICEWS and EuRepoC CSVs are required.")
            st.stop()
        data_dir = Path("uploads")
        icews = _persist_upload(icews_file, data_dir / "icews.csv")
        eurepoc = _persist_upload(eurepoc_file, data_dir / "eurepoc.csv")
        sanctions = _persist_upload(sanctions_file, data_dir / "sanctions.csv") if sanctions_file else None
        iso3 = _persist_upload(iso3_map_file, data_dir / "iso3_map.csv") if iso3_map_file else None
        return icews, eurepoc, sanctions, iso3
    else:
        icews = Path(icews_path)
        eurepoc = Path(eurepoc_path)
        sanctions = Path(sanctions_path) if sanctions_path.strip() else None
        iso3 = Path(iso3_map_path) if iso3_map_path.strip() else None
        return icews, eurepoc, sanctions, iso3

if run_btn:
    icews_p, eurepoc_p, sanctions_p, iso3_p = _get_paths()

    schema = Phase7Schema(
        icews_date_col=icews_date_col,
        icews_country_col=icews_country_col,
        icews_intensity_col=icews_intensity_col if icews_intensity_col.strip() else None,
        eurepoc_date_col=eurepoc_date_col,
        eurepoc_country_col=eurepoc_country_col,
        eurepoc_severity_col=eurepoc_severity_col if eurepoc_severity_col.strip() else None,
        sanctions_date_col=sanctions_date_col,
        sanctions_country_col=sanctions_country_col,
        sanctions_intensity_col=sanctions_intensity_col if sanctions_intensity_col.strip() else None,
    )

    exp = Path(export_dir)
    exp.mkdir(parents=True, exist_ok=True)

    # Phase 7
    st.info("Running Phase 7 (ingest, harmonize, audit, version)...")
    p7 = run_phase7(
        icews_csv=icews_p,
        eurepoc_csv=eurepoc_p,
        sanctions_csv=sanctions_p,
        iso3_mapping_csv=iso3_p,
        schema=schema,
        export_dir=exp,
        treated_country=treated_country.strip().upper(),
        intervention_month=intervention_month.strip() or None,
        x_cols=[],
    )
    st.success("Phase 7 complete.")
    st.subheader("Phase 7 Audit Summary")
    st.json(p7.ingest_audit)

    # Phase 8
    st.info("Running Phase 8 (rigor diagnostics, donor optimization, sensitivity)...")
    interv = p7.phase6_outputs.russia_results["intervention_month"]
    p8 = run_phase8_rigor(
        p7.panel,
        treated_country=treated_country.strip().upper(),
        intervention_month=intervention_month.strip() or interv,
        treatment_col="sanctions_count",
        covariates=[c.strip() for c in covariates.split(",") if c.strip()],
        export_dir=exp,
    )
    st.success("Phase 8 complete.")
    st.subheader("Phase 8 Diagnostics")
    st.json(p8.diagnostics)

    # Phase 9
    st.info("Running Phase 9 (Word Results draft)...")
    p9 = run_phase9_from_phase8_exports(
        phase8_export_dir=exp,
        phase8_report={
            "treated_country": p8.treated_country,
            "intervention_month": p8.intervention_month,
            "did": p8.did,
            "its": p8.its,
            "diagnostics": p8.diagnostics,
            "synthetic_control": p8.synthetic_control,
        },
        out_dir=exp,
    )
    st.success("Phase 9 complete.")

    st.subheader("Outputs")
    outputs = {
        "phase7_version_dir": p7.ingest_audit.get("version_dir"),
        "phase7_audit_dir": p7.ingest_audit.get("audit_dir"),
        "phase8_results_xlsx": str(exp / "phase8_rigor_results.xlsx"),
        "phase8_summary_docx": str(exp / "phase8_rigor_summary.docx"),
        "phase8_event_study_png": str(exp / "phase8_event_study.png"),
        "phase9_results_draft_docx": p9.results_docx,
    }
    st.json(outputs)

    # Preview tables
    st.subheader("Event Study Table Preview")
    try:
        es = pd.read_excel(exp / "phase8_rigor_results.xlsx", sheet_name="EventStudy")
        st.dataframe(es)
    except Exception as e:
        st.warning(f"Could not preview EventStudy sheet: {e}")

    st.subheader("DiD Sensitivity Preview")
    try:
        ds = pd.read_excel(exp / "phase8_rigor_results.xlsx", sheet_name="DiD_Sensitivity")
        st.dataframe(ds)
    except Exception as e:
        st.warning(f"Could not preview DiD_Sensitivity sheet: {e}")


st.markdown("## Phase 11: Hypothesis Agent")
st.caption("Non-LLM agent that tracks hypotheses, runs tests, updates status, and builds a knowledge graph.")
with st.expander("Run Phase 11 agent (optional)", expanded=False):
    hyp_statement = st.text_area(
        "Hypothesis statement",
        value="Sanctions are associated with a decrease in cyber incidents for the treated unit in the post-intervention period.",
        height=100
    )
    expected_direction = st.selectbox("Expected direction", ["decrease", "increase", "ambiguous"], index=0)
    alpha = st.number_input("Alpha (significance threshold)", min_value=0.001, max_value=0.25, value=0.05, step=0.01)
    run_agent = st.button("Run Phase 11 hypothesis test")
    if run_agent:
        agent = ResearchAgent(workdir=Path(export_dir) / "agent_state")
        h = agent.propose_hypothesis(
            treated_country=treated_country.strip().upper(),
            outcome="cyber_incidents",
            treatment="sanctions_count",
            expected_direction=expected_direction,
            statement=hyp_statement,
            priors={"alpha": float(alpha)},
            notes="Created via Phase 10 UI (Phase 11 agent).",
        )
        out11 = agent.test_hypothesis(
            panel=p7.panel,
            hypothesis_id=h.hypothesis_id,
            intervention_month=intervention_month.strip() or interv,
            export_dir=Path(export_dir) / "agent_exports",
            alpha=float(alpha),
        )
        st.subheader("Agent decision")
        st.json(out11.hypothesis)
        st.subheader("Model comparison (top)")
        st.json(out11.model_comparison[:5])
        st.subheader("Agent artifacts")
        st.json({"registry_run_id": out11.registry_run_id, "kg_path": out11.kg_path, "memory_path": out11.memory_path})

st.markdown("---")
st.caption("Note: Phase 10 focuses on orchestration and usability. For dissertation runs, use local paths for large datasets, and keep exports under version control.")
