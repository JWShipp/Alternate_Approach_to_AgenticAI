from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class DiDResult:
    att: float
    p_value: float
    model_summary: Dict[str, Any]

def difference_in_differences(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    y_col: str,
    treated_col: str,
    post_col: str,
    x_cols: Optional[List[str]] = None,
    cluster_col: Optional[str] = None
) -> DiDResult:
    """Two-way DiD: y ~ treated + post + treated*post + covariates + unit FE + time FE."""
    import statsmodels.formula.api as smf

    d = df.copy()
    d[unit_col] = d[unit_col].astype("category")
    d[time_col] = d[time_col].astype("category")
    d["_int"] = d[treated_col].astype(int) * d[post_col].astype(int)

    cov = " + ".join(x_cols or [])
    rhs = f"{treated_col} + {post_col} + _int"
    if cov:
        rhs += " + " + cov
    rhs += f" + C({unit_col}) + C({time_col})"

    m = smf.ols(formula=f"{y_col} ~ {rhs}", data=d).fit()
    if cluster_col:
        m = m.get_robustcov_results(cov_type="cluster", groups=d[cluster_col])

    idx = m.model.exog_names.index("_int")
    return DiDResult(
        att=float(m.params[idx]),
        p_value=float(m.pvalues[idx]),
        model_summary={"n": int(m.nobs), "r2": float(m.rsquared)},
    )

@dataclass
class EventStudyResult:
    coef_by_k: Dict[int, float]
    p_by_k: Dict[int, float]
    model_summary: Dict[str, Any]

def _kname(k: int) -> str:
    if k < 0:
        return f"k_m{abs(k)}"
    if k > 0:
        return f"k_p{k}"
    return "k_0"

def event_study(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    y_col: str,
    treated_col: str,
    event_time_col: str,
    k_min: int = -6,
    k_max: int = 12,
    omit_k: int = -1,
    x_cols: Optional[List[str]] = None,
    cluster_col: Optional[str] = None
) -> EventStudyResult:
    """Event study with treated x relative-time dummies. Baseline period omit_k excluded."""
    import statsmodels.formula.api as smf

    d = df.copy()
    d[unit_col] = d[unit_col].astype("category")
    d[time_col] = d[time_col].astype("category")

    ks = [k for k in range(k_min, k_max + 1) if k != omit_k]
    name_by_k = {k: _kname(k) for k in ks}

    for k in ks:
        d[name_by_k[k]] = ((d[event_time_col] == k).astype(int) * d[treated_col].astype(int))

    cov = " + ".join(x_cols or [])
    k_terms = " + ".join([name_by_k[k] for k in ks])
    rhs = f"{treated_col} + {k_terms}"
    if cov:
        rhs += " + " + cov
    rhs += f" + C({unit_col}) + C({time_col})"

    m = smf.ols(formula=f"{y_col} ~ {rhs}", data=d).fit()
    if cluster_col:
        m = m.get_robustcov_results(cov_type="cluster", groups=d[cluster_col])

    coef_by_k: Dict[int, float] = {}
    p_by_k: Dict[int, float] = {}
    for k in ks:
        name = name_by_k[k]
        idx = m.model.exog_names.index(name)
        coef_by_k[k] = float(m.params[idx])
        p_by_k[k] = float(m.pvalues[idx])

    return EventStudyResult(
        coef_by_k=coef_by_k,
        p_by_k=p_by_k,
        model_summary={"n": int(m.nobs), "r2": float(m.rsquared), "k_min": k_min, "k_max": k_max, "omit_k": omit_k},
    )

@dataclass
class SyntheticControlResult:
    treated_unit: str
    donor_units: List[str]
    weights: Dict[str, float]
    pre_rmse: float
    post_gap_mean: float
    model_summary: Dict[str, Any]

def synthetic_control(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    y_col: str,
    treated_unit: str,
    intervention_time: str,
    donor_units: Optional[List[str]] = None
) -> SyntheticControlResult:
    """Simple synthetic control with non-negative weights summing to 1.

    Aligns all series on the global time index; within-unit missing values are forward-filled then set to 0.
    """
    d = df.copy()
    d[time_col] = d[time_col].astype(str)

    donors = donor_units or sorted([u for u in d[unit_col].unique().tolist() if u != treated_unit])

    all_times = sorted(d[time_col].unique().tolist())
    pre_times = [t for t in all_times if t < str(intervention_time)]
    post_times = [t for t in all_times if t >= str(intervention_time)]

    def _series(u: str, times: List[str]) -> np.ndarray:
        s = d[d[unit_col] == u].set_index(time_col)[y_col].astype(float)
        s = s.reindex(times).ffill().fillna(0.0)
        return s.values

    Yt_pre = _series(treated_unit, pre_times)
    Xpre = np.vstack([_series(u, pre_times) for u in donors]).T  # T x donors

    w = np.ones(Xpre.shape[1], dtype=float) / Xpre.shape[1]
    lr = 0.05
    for _ in range(4000):
        grad = -2 * Xpre.T @ (Yt_pre - Xpre @ w) / max(1, len(Yt_pre))
        w = w - lr * grad
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s > 0:
            w = w / s

    yhat_pre = Xpre @ w
    pre_rmse = float(np.sqrt(np.mean((Yt_pre - yhat_pre) ** 2))) if len(Yt_pre) else float("nan")

    if post_times:
        Yt_post = _series(treated_unit, post_times)
        Xpost = np.vstack([_series(u, post_times) for u in donors]).T
        yhat_post = Xpost @ w
        post_gap_mean = float(np.mean(Yt_post - yhat_post))
    else:
        post_gap_mean = float("nan")

    weights = {str(donors[i]): float(w[i]) for i in range(len(donors))}
    return SyntheticControlResult(
        treated_unit=str(treated_unit),
        donor_units=[str(u) for u in donors],
        weights=weights,
        pre_rmse=pre_rmse,
        post_gap_mean=post_gap_mean,
        model_summary={"n_pre": int(len(pre_times)), "n_post": int(len(post_times))},
    )
