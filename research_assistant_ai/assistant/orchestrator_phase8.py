from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from ..models.did_diagnostics import parallel_trends_pretest, event_study_preperiod_joint_test
from ..models.synth_opt import donor_pool_search
from ..models.sensitivity import covariate_set_sensitivity
from ..models.causal_plus import event_study, difference_in_differences
from ..models.causal_plus import synthetic_control
from ..models.causal_estimators import interrupted_time_series
from .reporting import export_excel, export_word
from .reporting_plus import plot_event_study
from .robustness import placebo_its_cutoffs, rdd_bandwidth_sensitivity
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class Phase8Outputs:
    treated_country: str
    intervention_month: str
    did: Dict[str, Any]
    event_study: Dict[str, Any]
    synthetic_control: Dict[str, Any]
    its: Dict[str, Any]
    diagnostics: Dict[str, Any]
    sensitivity: Dict[str, Any]
    exports: Dict[str, str]

def run_phase8_rigor(
    panel: pd.DataFrame,
    *,
    treated_country: str,
    intervention_month: str,
    country_col: str = "country_iso3",
    month_col: str = "month",
    outcome_col: str = "cyber_incidents",
    treatment_col: str = "sanctions_count",
    covariates: Optional[List[str]] = None,
    export_dir: Optional[Path] = None,
) -> Phase8Outputs:
    covariates = covariates or []
    export_dir = Path(export_dir) if export_dir else Path("exports_phase8")
    export_dir.mkdir(parents=True, exist_ok=True)

    d = panel.copy()
    d[country_col] = d[country_col].astype(str)
    d[month_col] = d[month_col].astype(str)

    # Build DiD frame
    did_df = d.copy()
    did_df["treated"] = (did_df[country_col] == str(treated_country)).astype(int)
    did_df["post"] = (did_df[month_col] >= str(intervention_month)).astype(int)
    months = sorted(did_df[month_col].unique().tolist())
    idx = {m:i for i,m in enumerate(months)}
    did_df["event_k"] = did_df[month_col].map(idx).astype(int) - int(idx[str(intervention_month)])

    x_cols = list(dict.fromkeys(covariates + [treatment_col]))

    did_res = difference_in_differences(
        did_df,
        unit_col=country_col,
        time_col=month_col,
        y_col=outcome_col,
        treated_col="treated",
        post_col="post",
        x_cols=x_cols,
        cluster_col=country_col,
    )

    es_res = event_study(
        did_df,
        unit_col=country_col,
        time_col=month_col,
        y_col=outcome_col,
        treated_col="treated",
        event_time_col="event_k",
        k_min=-6,
        k_max=12,
        omit_k=-1,
        x_cols=x_cols,
        cluster_col=country_col,
    )

    # Synthetic control baseline + optimized donor pool
    sc_base = synthetic_control(d, unit_col=country_col, time_col=month_col, y_col=outcome_col,
                                treated_unit=str(treated_country), intervention_time=str(intervention_month))
    sc_opt = donor_pool_search(d, unit_col=country_col, time_col=month_col, y_col=outcome_col,
                               treated_unit=str(treated_country), intervention_time=str(intervention_month),
                               max_donors=15, top_k_candidates=30)

    # ITS on treated country series
    rus_ts = d[d[country_col]==str(treated_country)].groupby(month_col, as_index=False)[[outcome_col, treatment_col]].sum()
    its = interrupted_time_series(
        rus_ts,
        time_col=month_col,
        y_col=outcome_col,
        intervention_time=str(intervention_month),
        x_cols=[treatment_col],
        hac_lags=3,
    )

    # Diagnostics
    pt = parallel_trends_pretest(
        did_df,
        unit_col=country_col,
        time_col=month_col,
        y_col=outcome_col,
        treated_col="treated",
        intervention_time=str(intervention_month),
        x_cols=x_cols,
        cluster_col=country_col,
    )
    pre_joint = event_study_preperiod_joint_test(es_res.coef_by_k, es_res.p_by_k, k_pre_max=-2)

    diagnostics = {
        "parallel_trends_pretest": {"coef_treated_x_trend": pt.coef, "p_value": pt.p_value, "n": pt.n, "summary": pt.model_summary},
        "event_study_preperiod_joint_test_fisher": {"k_values": pre_joint.k_values, "stat": pre_joint.f_stat, "p_value": pre_joint.p_value, "df_num": pre_joint.df_num},
        "synthetic_control_donor_search": {"best_pre_rmse": sc_opt.best.pre_rmse, "best_post_gap_mean": sc_opt.best.post_gap_mean, "best_donors_n": len(sc_opt.best.donor_units), "tried_rows": len(sc_opt.tried)},
    }

    # Sensitivity: covariate subsets for DiD
    sens = covariate_set_sensitivity(
        did_df,
        unit_col=country_col,
        time_col=month_col,
        y_col=outcome_col,
        treated_col="treated",
        post_col="post",
        base_covariates=x_cols,
        cluster_col=country_col,
        max_models=25,
    )
    sensitivity = {"covariate_set_runs": sens.runs, "base_covariates": sens.base_covariates}

    # Robustness small add-on: placebo ITS cutoffs (treated)
    cut_idx = max(0, min(len(rus_ts)-1, int(len(rus_ts)//2)))
    placebo_months = []
    for j in [-4,-2,2,4]:
        k = max(0, min(len(rus_ts)-1, cut_idx+j))
        placebo_months.append(str(rus_ts[month_col].iloc[k]))
    placebo_its = placebo_its_cutoffs(rus_ts, time_col=month_col, y_col=outcome_col, cutoffs=placebo_months)

    # Exports
    es_tbl = pd.DataFrame({
        "k": list(es_res.coef_by_k.keys()),
        "coef": list(es_res.coef_by_k.values()),
        "p_value": [es_res.p_by_k[k] for k in es_res.coef_by_k.keys()],
    }).sort_values("k")

    did_sens_tbl = pd.DataFrame([{"covariates": ",".join(r["covariates"]), "att": r["att"], "p_value": r["p_value"]} for r in sens.runs])

    donor_trials_tbl = pd.DataFrame(sc_opt.tried) if sc_opt.tried else pd.DataFrame(columns=["step","donors_n","candidate_added","pre_rmse","post_gap_mean"])
    placebo_tbl = pd.DataFrame(placebo_its)

    dfs = {
        "PanelHead": d.head(200),
        "Treated_TS": rus_ts,
        "EventStudy": es_tbl,
        "DiD_Sensitivity": did_sens_tbl,
        "Synth_DonorSearch": donor_trials_tbl,
        "Placebo_ITS_Treated": placebo_tbl,
    }

    excel_path = export_excel(dfs, export_dir / "phase8_rigor_results.xlsx")
    fig_es = plot_event_study(es_res.coef_by_k, es_res.p_by_k, export_dir / "phase8_event_study.png")

    word_payload = {
        "FigureCaptions_APA": {
            "Figure 1": "Event study estimates of post-intervention changes in cyber incidents for the treated unit relative to the omitted baseline month (k = -1).",
        },
        "TableTitles_APA": {
            "Table 1": "Causal Estimates and Diagnostics Summary (DiD, Event Study, ITS, Synthetic Control, and Sensitivity Checks).",
        },
        "KeyResults": {
            "treated_country": str(treated_country),
            "intervention_month": str(intervention_month),
            "did_att": did_res.att,
            "did_p_value": did_res.p_value,
            "parallel_trends_coef": pt.coef,
            "parallel_trends_p": pt.p_value,
            "preperiod_joint_p": pre_joint.p_value,
            "synth_base_pre_rmse": sc_base.pre_rmse,
            "synth_opt_pre_rmse": sc_opt.best.pre_rmse,
        },
        "Diagnostics": diagnostics,
        "Sensitivity": {
            "models_n": len(sens.runs),
            "att_min": float(min(r["att"] for r in sens.runs)) if sens.runs else None,
            "att_max": float(max(r["att"] for r in sens.runs)) if sens.runs else None,
        },
        "PlaceboITS": {"rows": len(placebo_its)},
    }
    word_path = export_word(word_payload, export_dir / "phase8_rigor_summary.docx")

    exports = {"excel": str(excel_path), "word": str(word_path), "fig_event_study": str(fig_es)}

    return Phase8Outputs(
        treated_country=str(treated_country),
        intervention_month=str(intervention_month),
        did={"att": did_res.att, "p_value": did_res.p_value, "summary": did_res.model_summary},
        event_study={"coef_by_k": es_res.coef_by_k, "p_by_k": es_res.p_by_k, "summary": es_res.model_summary},
        synthetic_control={
            "base": {"pre_rmse": sc_base.pre_rmse, "post_gap_mean": sc_base.post_gap_mean, "weights": sc_base.weights},
            "optimized": {"pre_rmse": sc_opt.best.pre_rmse, "post_gap_mean": sc_opt.best.post_gap_mean, "weights": sc_opt.best.weights, "donors": sc_opt.best.donor_units},
        },
        its={"level_change": its.level_change, "p_level": its.p_level, "slope_change": its.slope_change, "p_slope": its.p_slope, "summary": its.model_summary},
        diagnostics=diagnostics,
        sensitivity=sensitivity,
        exports=exports,
    )
