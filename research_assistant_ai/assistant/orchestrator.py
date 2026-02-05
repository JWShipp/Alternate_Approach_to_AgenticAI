from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger
from ..data.ingest import Ingestor, IngestConfig
from ..data.contracts import CountryTimePanelContract, validate_panel
from ..models.causal_estimators import interrupted_time_series, regression_discontinuity
from ..models.causal_plus import difference_in_differences, event_study, synthetic_control
from ..models.vuln_predict import train_exploit_predictor
from ..verify.checks import Verifier, prob_bounds_check
from ..core.active_inference import ActiveInferencePlanner
from .research_memory import ResearchMemory
from .dissertation_parser import parse_docx
from .spec_search import Spec, run_spec_search
from .registry import HypothesisRegistry
from .protocol import ExperimentProtocol, ProtocolLogger
from .reporting import export_excel, export_word, plot_top_specs
from .reporting_plus import plot_event_study
from .robustness import rdd_bandwidth_sensitivity, placebo_its_cutoffs, synthetic_control_placebos
from ..utils.stats_utils import benjamini_hochberg

logger = get_logger(__name__)

@dataclass
class ResearchAssistantOutputs:
    dissertation_signals: Dict[str, Any]
    panel_contract_validation: Dict[str, Any]
    spec_search_ranking: List[Dict[str, Any]]
    causal_suite: Dict[str, Any]
    robustness: Dict[str, Any]
    vuln_model_summary: Dict[str, Any]
    next_actions: Dict[str, Any]
    verification: Dict[str, Any]
    exports: Dict[str, str]

class ResearchAssistantAI:
    def __init__(
        self,
        ingest_config: IngestConfig,
        *,
        memory_dir: Optional[Path] = None,
        dissertation_paths: Optional[List[Path]] = None,
        export_dir: Optional[Path] = None,
        registry_dir: Optional[Path] = None
    ):
        self.ingestor = Ingestor(ingest_config)
        self.verifier = Verifier()
        self.verifier.add_check(prob_bounds_check("exploit_probability"))
        self.verifier.add_check(prob_bounds_check("enterprise_exposure_probability"))
        self.memory = ResearchMemory(Path(memory_dir) if memory_dir else Path("research_memory"))
        self.dissertation_paths = dissertation_paths or []
        self.export_dir = Path(export_dir) if export_dir else Path("exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.registry_dir = Path(registry_dir) if registry_dir else (self.export_dir / "registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.hyp_registry = HypothesisRegistry(self.registry_dir)
        self.protocol_logger = ProtocolLogger(self.registry_dir)

        self.panel_contract = CountryTimePanelContract()

    def _parse_dissertation_signals(self) -> Dict[str, Any]:
        docs: List[Dict[str, Any]] = []
        for p in self.dissertation_paths:
            try:
                s = parse_docx(Path(p))
                docs.append({"path": str(p), "title": s.title, "research_questions": s.research_questions, "keywords": s.keywords})
            except Exception as e:
                docs.append({"path": str(p), "error": str(e)})
        return {"documents": docs}

    def _build_contract_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        out = out.rename(columns={"month":"period", "cyber_incidents":"outcome", "unrest_count":"treatment"})
        keep = ["country_iso3","period","outcome","treatment","unrest_intensity","internet_users_pct","gdp_current_usd"]
        return out[keep].copy()

    def _run_spec_search(self, panel: pd.DataFrame) -> pd.DataFrame:
        specs = [
            Spec(name="CM_Pois_FE", unit_of_analysis="country-month", model_family="poisson", max_lag=12,
                 covariates=["unrest_intensity","internet_users_pct","gdp_current_usd"], fixed_effects=True),
            Spec(name="CM_NB_FE", unit_of_analysis="country-month", model_family="negbin", max_lag=12,
                 covariates=["unrest_intensity","internet_users_pct","gdp_current_usd"], fixed_effects=True),
            Spec(name="CM_ZIP_FE", unit_of_analysis="country-month", model_family="zip", max_lag=6,
                 covariates=["unrest_intensity","internet_users_pct","gdp_current_usd"], fixed_effects=True),
            Spec(name="GLOB_Pois", unit_of_analysis="global-month", model_family="poisson", max_lag=12,
                 covariates=["unrest_intensity"], fixed_effects=False),
            Spec(name="CQ_Pois_FE", unit_of_analysis="country-quarter", model_family="poisson", max_lag=4,
                 covariates=["unrest_intensity","internet_users_pct","gdp_current_usd"], fixed_effects=True),
        ]
        df = run_spec_search(panel, specs)
        if not df.empty:
            df["q_value_bh"] = benjamini_hochberg(df["p_value"].astype(float).tolist())
            df.to_csv(self.export_dir / "phase5_spec_search_results.csv", index=False)
        return df

    def _build_did_dataset(self, panel: pd.DataFrame):
        d = panel.copy().sort_values(["country_iso3","month"]).reset_index(drop=True)
        months = sorted(d["month"].unique().tolist())
        mid = len(months) // 2
        cutoff_month = months[mid]
        d["post"] = (d["month"].astype(str) >= str(cutoff_month)).astype(int)

        treated = str(d.groupby("country_iso3")["unrest_count"].mean().sort_values(ascending=False).index[0])
        d["treated"] = (d["country_iso3"].astype(str) == treated).astype(int)

        idx = {m:i for i,m in enumerate(months)}
        d["event_k"] = d["month"].map(idx).astype(int) - int(idx[cutoff_month])
        return d, treated, cutoff_month

    def run(self) -> ResearchAssistantOutputs:
        dissertation_signals = self._parse_dissertation_signals()

        panel = self.ingestor.load_country_month_panel()
        contract_panel = self._build_contract_panel(panel)
        contract_validation = validate_panel(contract_panel, self.panel_contract)

        hyp = self.hyp_registry.create(
            statement="Geopolitical crises cause measurable discontinuities and post-event increases in cyber incident counts in treated targets relative to controls.",
            unit_of_analysis="country-month",
            outcome="cyber_incidents",
            treatment="unrest_count",
            tags=["MVP","causal","DiD","SyntheticControl","EventStudy"],
        )
        protocol = ExperimentProtocol(
            hypothesis_id=hyp.hypothesis_id,
            dataset_name="country_month_panel",
            unit_of_analysis="country-month",
            model_family="spec_search + causal_suite",
            time_window=f"{panel['month'].min()} to {panel['month'].max()}",
            covariates=["unrest_intensity","internet_users_pct","gdp_current_usd"],
            max_lag=12,
            correction="BH",
        )
        self.protocol_logger.log(protocol)

        spec_rank = self._run_spec_search(panel)
        spec_search_ranking = spec_rank.head(15).to_dict(orient="records") if not spec_rank.empty else []

        did_df, treated_unit, cutoff_month = self._build_did_dataset(panel)

        did_res = difference_in_differences(
            did_df,
            unit_col="country_iso3",
            time_col="month",
            y_col="cyber_incidents",
            treated_col="treated",
            post_col="post",
            x_cols=["unrest_intensity","internet_users_pct","gdp_current_usd"],
            cluster_col="country_iso3",
        )

        es_res = event_study(
            did_df,
            unit_col="country_iso3",
            time_col="month",
            y_col="cyber_incidents",
            treated_col="treated",
            event_time_col="event_k",
            k_min=-6,
            k_max=12,
            omit_k=-1,
            x_cols=["unrest_intensity","internet_users_pct","gdp_current_usd"],
            cluster_col="country_iso3",
        )

        sc_res = synthetic_control(
            panel,
            unit_col="country_iso3",
            time_col="month",
            y_col="cyber_incidents",
            treated_unit=treated_unit,
            intervention_time=str(cutoff_month),
        )

        ts = panel.groupby("month", as_index=False)[["cyber_incidents","unrest_count","unrest_intensity"]].sum()
        its = interrupted_time_series(ts, time_col="month", y_col="cyber_incidents", intervention_time=str(cutoff_month), x_cols=["unrest_count","unrest_intensity"], hac_lags=3)

        try:
            ts["_t"] = np.arange(len(ts))
            cutoff = float(ts["_t"].iloc[len(ts)//2])
            rdd = regression_discontinuity(ts, running_col="_t", y_col="cyber_incidents", cutoff=cutoff, bandwidth=6.0, x_cols=["unrest_count","unrest_intensity"])
            rdd_block = {"cutoff_index": cutoff, "bandwidth": rdd.bandwidth, "discontinuity": rdd.discontinuity, "p_value": rdd.p_value, "summary": rdd.model_summary}
        except Exception as e:
            rdd_block = {"error": str(e)}

        causal_suite = {
            "treated_unit": treated_unit,
            "cutoff_month": str(cutoff_month),
            "did": {"att": did_res.att, "p_value": did_res.p_value, "summary": did_res.model_summary},
            "event_study": {"coef_by_k": es_res.coef_by_k, "p_by_k": es_res.p_by_k, "summary": es_res.model_summary},
            "synthetic_control": {"pre_rmse": sc_res.pre_rmse, "post_gap_mean": sc_res.post_gap_mean, "weights": sc_res.weights, "summary": sc_res.model_summary},
            "its": {"level_change": its.level_change, "p_level": its.p_level, "slope_change": its.slope_change, "p_slope": its.p_slope, "summary": its.model_summary},
            "rdd": rdd_block,
        }

        rob = {}
        rob["rdd_bandwidth_sweep"] = rdd_bandwidth_sensitivity(ts.assign(_t=np.arange(len(ts))), running_col="_t", y_col="cyber_incidents", cutoff=float(len(ts)//2), bandwidths=[3.0,6.0,9.0,12.0], x_cols=["unrest_count","unrest_intensity"])
        cut_idx = len(ts)//2
        placebo_months = []
        for j in [-4,-2,2,4]:
            k = max(0, min(len(ts)-1, cut_idx+j))
            placebo_months.append(str(ts["month"].iloc[k]))
        rob["placebo_its_cutoffs"] = placebo_its_cutoffs(ts, time_col="month", y_col="cyber_incidents", cutoffs=placebo_months)
        rob["synth_placebos"] = synthetic_control_placebos(panel, unit_col="country_iso3", time_col="month", y_col="cyber_incidents", intervention_time=str(cutoff_month), treated_unit=treated_unit)

        robustness = rob

        vuln = self.ingestor.load_vuln_instance_month()
        _, vuln_res = train_exploit_predictor(vuln)
        vuln_summary = {"auc": vuln_res.auc, "top_decile_precision": vuln_res.top_decile_precision, "model_info": vuln_res.model_info}

        exploit_prob = float(max(0.0, min(1.0, vuln_res.top_decile_precision)))
        belief = np.array([1.0 - exploit_prob, exploit_prob])

        def transition(action: str, b: np.ndarray) -> np.ndarray:
            low, high = b
            if action == "PatchNow":
                return np.array([min(1.0, low + 0.35 * high), max(0.0, high - 0.35 * high)])
            if action == "Investigate":
                return np.array([low + 0.05 * high, high - 0.05 * high])
            if action == "Defer":
                return np.array([max(0.0, low - 0.10 * low), min(1.0, high + 0.10 * low)])
            return b

        planner = ActiveInferencePlanner(actions=["PatchNow","Investigate","Defer"], transition_fn=transition, preference=np.array([1.0, 0.0]), uncertainty_weight=0.4)
        chosen, scored = planner.choose(belief)
        next_actions = {"chosen_action": chosen, "scored_actions": [s.__dict__ for s in scored]}

        artifact = {"exploit_probability": exploit_prob, "enterprise_exposure_probability": float(min(1.0, max(0.0, 0.5 * exploit_prob))), "recommended_action": chosen}
        checks = self.verifier.run(artifact)
        verification = {"artifact": artifact, "checks": [c.__dict__ for c in checks], "passed_all": bool(all(c.passed for c in checks))}

        # Exports
        dfs = {}
        if not spec_rank.empty:
            dfs["SpecSearchResults"] = spec_rank.copy()
        dfs["ContractPanelHead"] = contract_panel.head(50).copy()

        es_tbl = pd.DataFrame({
            "k": list(es_res.coef_by_k.keys()),
            "coef": list(es_res.coef_by_k.values()),
            "p_value": [es_res.p_by_k[k] for k in es_res.coef_by_k.keys()],
        }).sort_values("k")
        dfs["EventStudy"] = es_tbl
        dfs["Robust_RDD_BW"] = pd.DataFrame(robustness["rdd_bandwidth_sweep"])
        dfs["Robust_ITS_Placebos"] = pd.DataFrame(robustness["placebo_its_cutoffs"])
        dfs["Robust_Synth_Placebos"] = pd.DataFrame(robustness["synth_placebos"])

        excel_path = export_excel(dfs, self.export_dir / "phase5_results.xlsx")
        fig_specs = plot_top_specs(spec_rank, self.export_dir / "phase5_top_specs.png", top_k=10)
        fig_es = plot_event_study(es_res.coef_by_k, es_res.p_by_k, self.export_dir / "phase5_event_study.png")

        word_summary = {
            "Protocol": protocol.to_dict(),
            "PanelContractValidation": contract_validation,
            "TopSpecSearch": (spec_rank.head(10).to_dict(orient="records") if not spec_rank.empty else []),
            "CausalSuite": {
                "treated_unit": treated_unit,
                "cutoff_month": str(cutoff_month),
                "did_att": did_res.att,
                "did_p_value": did_res.p_value,
                "synth_pre_rmse": sc_res.pre_rmse,
                "synth_post_gap_mean": sc_res.post_gap_mean,
                "its_level_change": its.level_change,
                "its_p_level": its.p_level,
            },
            "Robustness": {
                "rdd_bandwidth_sweep_rows": len(robustness["rdd_bandwidth_sweep"]),
                "placebo_its_cutoffs_rows": len(robustness["placebo_its_cutoffs"]),
                "synth_placebos_rows": len(robustness["synth_placebos"]),
            },
            "VulnerabilityModel": vuln_summary,
        }
        word_path = export_word(word_summary, self.export_dir / "phase5_summary.docx")

        exports = {
            "excel": str(excel_path),
            "word": str(word_path),
            "fig_specs": str(fig_specs) if fig_specs else "",
            "fig_event_study": str(fig_es) if fig_es else "",
            "spec_results_csv": str(self.export_dir / "phase5_spec_search_results.csv") if (self.export_dir / "phase5_spec_search_results.csv").exists() else "",
            "hypotheses_log": str(self.registry_dir / "hypotheses.jsonl"),
            "protocols_log": str(self.registry_dir / "protocols.jsonl"),
        }

        return ResearchAssistantOutputs(
            dissertation_signals=dissertation_signals,
            panel_contract_validation=contract_validation,
            spec_search_ranking=spec_search_ranking,
            causal_suite=causal_suite,
            robustness=robustness,
            vuln_model_summary=vuln_summary,
            next_actions=next_actions,
            verification=verification,
            exports=exports,
        )
