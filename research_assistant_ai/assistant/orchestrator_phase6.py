from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger
from ..models.causal_estimators import interrupted_time_series, regression_discontinuity
from ..models.causal_plus import difference_in_differences, event_study, synthetic_control
from ..utils.stats_utils import benjamini_hochberg
from .reporting import export_excel, export_word
from .reporting_plus import plot_event_study
from .robustness import rdd_bandwidth_sensitivity, placebo_its_cutoffs, synthetic_control_placebos

logger = get_logger(__name__)

@dataclass
class Phase6Outputs:
    panel: pd.DataFrame
    russia_results: Dict[str, Any]
    robustness: Dict[str, Any]
    exports: Dict[str, str]

def _month_index(months: List[str]) -> Dict[str, int]:
    return {m:i for i,m in enumerate(sorted(months))}

def run_phase6_russia_causal(
    panel: pd.DataFrame,
    *,
    country_col: str = "country_iso3",
    month_col: str = "month",
    outcome_col: str = "cyber_incidents",
    treatment_col: str = "sanctions_count",
    treated_country: str = "RUS",
    intervention_month: Optional[str] = None,
    x_cols: Optional[List[str]] = None,
    export_dir: Optional[Path] = None
) -> Phase6Outputs:
    """Phase 6: Russia-focused causal engine using sanctions timeline as intervention.

    - ITS on Russia monthly outcomes with sanctions covariate.
    - RDD on global monthly totals around intervention index.
    - Synthetic control for Russia using other countries as donors.
    - DiD and event study using treated=Russia, post=intervention_month (two-way FE).
    """
    x_cols = x_cols or []
    export_dir = Path(export_dir) if export_dir else Path("exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    d = panel.copy()
    d[country_col] = d[country_col].astype(str)
    d[month_col] = d[month_col].astype(str)

    months = sorted(d[month_col].unique().tolist())
    if intervention_month is None:
        # pick first month where sanctions for RUS are non-zero; else midpoint
        rus = d[d[country_col]==treated_country].copy()
        nonzero = rus[rus[treatment_col] > 0]
        intervention_month = str(nonzero[month_col].iloc[0]) if len(nonzero) else str(months[len(months)//2])

    # Build DiD frame
    did_df = d.copy()
    did_df["treated"] = (did_df[country_col] == treated_country).astype(int)
    did_df["post"] = (did_df[month_col] >= str(intervention_month)).astype(int)
    idx = _month_index(months)
    did_df["event_k"] = did_df[month_col].map(idx).astype(int) - int(idx[intervention_month])

    did_res = difference_in_differences(
        did_df,
        unit_col=country_col,
        time_col=month_col,
        y_col=outcome_col,
        treated_col="treated",
        post_col="post",
        x_cols=list(set(x_cols + [treatment_col])),
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
        x_cols=list(set(x_cols + [treatment_col])),
        cluster_col=country_col,
    )

    sc_res = synthetic_control(
        d,
        unit_col=country_col,
        time_col=month_col,
        y_col=outcome_col,
        treated_unit=treated_country,
        intervention_time=str(intervention_month),
    )

    # Russia ITS
    rus_ts = d[d[country_col]==treated_country].groupby(month_col, as_index=False)[[outcome_col, treatment_col]].sum()
    its_res = interrupted_time_series(
        rus_ts,
        time_col=month_col,
        y_col=outcome_col,
        intervention_time=str(intervention_month),
        x_cols=[treatment_col],
        hac_lags=3,
    )

    # Global RDD (index)
    glob_ts = d.groupby(month_col, as_index=False)[[outcome_col, treatment_col]].sum()
    glob_ts["_t"] = np.arange(len(glob_ts))
    cutoff = float(glob_ts.loc[glob_ts[month_col].astype(str) == str(intervention_month), "_t"].iloc[0]) if str(intervention_month) in set(glob_ts[month_col].astype(str)) else float(len(glob_ts)//2)
    try:
        rdd_res = regression_discontinuity(glob_ts, running_col="_t", y_col=outcome_col, cutoff=cutoff, bandwidth=6.0, x_cols=[treatment_col])
        rdd_block = {"cutoff_index": cutoff, "bandwidth": rdd_res.bandwidth, "discontinuity": rdd_res.discontinuity, "p_value": rdd_res.p_value, "summary": rdd_res.model_summary}
    except Exception as e:
        rdd_block = {"error": str(e)}

    # Robustness
    cut_idx = int(cutoff)
    placebo_months = []
    for j in [-4,-2,2,4]:
        k = max(0, min(len(glob_ts)-1, cut_idx+j))
        placebo_months.append(str(glob_ts[month_col].iloc[k]))
    robustness = {
        "rdd_bandwidth_sweep": rdd_bandwidth_sensitivity(glob_ts, running_col="_t", y_col=outcome_col, cutoff=cutoff, bandwidths=[3.0,6.0,9.0,12.0], x_cols=[treatment_col]),
        "placebo_its_cutoffs_rus": placebo_its_cutoffs(rus_ts, time_col=month_col, y_col=outcome_col, cutoffs=placebo_months),
        "synth_placebos": synthetic_control_placebos(d, unit_col=country_col, time_col=month_col, y_col=outcome_col, intervention_time=str(intervention_month), treated_unit=treated_country),
    }

    # Exports (Excel + Word + Figure)
    es_tbl = pd.DataFrame({
        "k": list(es_res.coef_by_k.keys()),
        "coef": list(es_res.coef_by_k.values()),
        "p_value": [es_res.p_by_k[k] for k in es_res.coef_by_k.keys()],
    }).sort_values("k")

    dfs = {
        "PanelHead": d.head(200),
        "Russia_TS": rus_ts,
        "Global_TS": glob_ts,
        "EventStudy": es_tbl,
        "Robust_RDD_BW": pd.DataFrame(robustness["rdd_bandwidth_sweep"]),
        "Robust_ITS_Placebos_RUS": pd.DataFrame(robustness["placebo_its_cutoffs_rus"]),
        "Robust_Synth_Placebos": pd.DataFrame(robustness["synth_placebos"]),
    }

    excel_path = export_excel(dfs, export_dir / "phase6_russia_results.xlsx")
    fig_es = plot_event_study(es_res.coef_by_k, es_res.p_by_k, export_dir / "phase6_russia_event_study.png")

    # Minimal APA-ready text placeholders (captions ready to paste into dissertation)
    word_payload = {
        "FigureCaptions_APA": {
            "Figure 1": "Russia event study estimates of the post-sanctions change in cyber incidents relative to the omitted baseline month (k = -1).",
        },
        "TableTitles_APA": {
            "Table 1": "Russia Causal Estimates Summary (DiD, ITS, Synthetic Control, and RDD).",
        },
        "RussiaResults": {
            "treated_country": treated_country,
            "intervention_month": str(intervention_month),
            "did": {"att": did_res.att, "p_value": did_res.p_value, "summary": did_res.model_summary},
            "its": {"level_change": its_res.level_change, "p_level": its_res.p_level, "slope_change": its_res.slope_change, "p_slope": its_res.p_slope, "summary": its_res.model_summary},
            "synthetic_control": {"pre_rmse": sc_res.pre_rmse, "post_gap_mean": sc_res.post_gap_mean},
            "rdd": rdd_block,
        },
    }
    word_path = export_word(word_payload, export_dir / "phase6_russia_summary.docx")

    exports = {"excel": str(excel_path), "word": str(word_path), "fig_event_study": str(fig_es)}

    russia_results = {
        "treated_country": treated_country,
        "intervention_month": str(intervention_month),
        "did": {"att": did_res.att, "p_value": did_res.p_value, "summary": did_res.model_summary},
        "event_study": {"coef_by_k": es_res.coef_by_k, "p_by_k": es_res.p_by_k, "summary": es_res.model_summary},
        "synthetic_control": {"pre_rmse": sc_res.pre_rmse, "post_gap_mean": sc_res.post_gap_mean, "weights": sc_res.weights, "summary": sc_res.model_summary},
        "its": {"level_change": its_res.level_change, "p_level": its_res.p_level, "slope_change": its_res.slope_change, "p_slope": its_res.p_slope, "summary": its_res.model_summary},
        "rdd": rdd_block,
    }

    return Phase6Outputs(panel=d, russia_results=russia_results, robustness=robustness, exports=exports)
