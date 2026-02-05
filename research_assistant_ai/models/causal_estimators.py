from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

@dataclass
class ITSResult:
    level_change: float
    slope_change: float
    p_level: float
    p_slope: float
    model_summary: Dict[str, Any]

def interrupted_time_series(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    intervention_time: str,
    x_cols: Optional[list[str]] = None,
    hac_lags: Optional[int] = 3
) -> ITSResult:
    """Interrupted Time Series (ITS) with optional Newey-West (HAC) standard errors.

    Model: y ~ time + post + time_post + covariates

    This MVP supports OLS with HAC SE to reduce autocorrelation-induced false positives.
    """
    import statsmodels.api as sm

    d = df.copy().sort_values(time_col).reset_index(drop=True)
    d["_t"] = np.arange(len(d))

    tmask = d[time_col].astype(str) >= str(intervention_time)
    if not tmask.any():
        raise ValueError("intervention_time is after all observations.")
    t0 = int(np.where(tmask.values)[0][0])

    d["_post"] = (d["_t"] >= t0).astype(int)
    d["_t_post"] = (d["_t"] - t0) * d["_post"]

    X_cols = ["_t", "_post", "_t_post"] + list(x_cols or [])
    X = sm.add_constant(d[X_cols], has_constant="add")
    y = d[y_col].astype(float)

    model = sm.OLS(y, X).fit()
    if hac_lags is not None and hac_lags > 0:
        model = model.get_robustcov_results(cov_type="HAC", maxlags=int(hac_lags))

    return ITSResult(
        level_change=float(model.params[X.columns.get_loc("_post")]) if hasattr(model.params, "__len__") else float(model.params["_post"]),
        slope_change=float(model.params[X.columns.get_loc("_t_post")]) if hasattr(model.params, "__len__") else float(model.params["_t_post"]),
        p_level=float(model.pvalues[X.columns.get_loc("_post")]) if hasattr(model.pvalues, "__len__") else float(model.pvalues["_post"]),
        p_slope=float(model.pvalues[X.columns.get_loc("_t_post")]) if hasattr(model.pvalues, "__len__") else float(model.pvalues["_t_post"]),
        model_summary={"n": int(model.nobs), "r2": float(model.rsquared), "aic": float(model.aic), "bic": float(model.bic)},
    )

@dataclass
class RDDResult:
    discontinuity: float
    p_value: float
    bandwidth: float
    model_summary: Dict[str, Any]

def regression_discontinuity(
    df: pd.DataFrame,
    *,
    running_col: str,
    y_col: str,
    cutoff: float,
    bandwidth: float,
    x_cols: Optional[list[str]] = None
) -> RDDResult:
    import statsmodels.api as sm

    d = df.copy()
    d["_r"] = d[running_col].astype(float) - float(cutoff)
    d = d[np.abs(d["_r"]) <= float(bandwidth)].copy()
    if len(d) < 20:
        raise ValueError("Not enough observations in bandwidth to fit RDD.")
    d["_treat"] = (d["_r"] >= 0).astype(int)
    d["_r_treat"] = d["_r"] * d["_treat"]

    X_cols = ["_r", "_treat", "_r_treat"] + list(x_cols or [])
    X = sm.add_constant(d[X_cols], has_constant="add")
    y = d[y_col].astype(float)

    model = sm.OLS(y, X).fit()

    return RDDResult(
        discontinuity=float(model.params["_treat"]),
        p_value=float(model.pvalues["_treat"]),
        bandwidth=float(bandwidth),
        model_summary={"n": int(model.nobs), "r2": float(model.rsquared), "aic": float(model.aic), "bic": float(model.bic)},
    )
