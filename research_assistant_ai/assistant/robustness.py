from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd

from ..models.causal_estimators import regression_discontinuity, interrupted_time_series
from ..models.causal_plus import synthetic_control

def rdd_bandwidth_sensitivity(
    df: pd.DataFrame,
    *,
    running_col: str,
    y_col: str,
    cutoff: float,
    bandwidths: List[float],
    x_cols: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bw in bandwidths:
        try:
            r = regression_discontinuity(df, running_col=running_col, y_col=y_col, cutoff=cutoff, bandwidth=bw, x_cols=x_cols)
            out.append({"bandwidth": float(bw), "discontinuity": float(r.discontinuity), "p_value": float(r.p_value), "n": int(r.model_summary["n"])})
        except Exception as e:
            out.append({"bandwidth": float(bw), "error": str(e)})
    return out

def placebo_its_cutoffs(ts: pd.DataFrame, *, time_col: str, y_col: str, cutoffs: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in cutoffs:
        try:
            r = interrupted_time_series(ts, time_col=time_col, y_col=y_col, intervention_time=str(c), x_cols=None, hac_lags=3)
            out.append({"cutoff": str(c), "level_change": float(r.level_change), "p_level": float(r.p_level), "slope_change": float(r.slope_change), "p_slope": float(r.p_slope)})
        except Exception as e:
            out.append({"cutoff": str(c), "error": str(e)})
    return out

def synthetic_control_placebos(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    y_col: str,
    intervention_time: str,
    treated_unit: str
) -> List[Dict[str, Any]]:
    units = sorted(df[unit_col].unique().tolist())
    out: List[Dict[str, Any]] = []
    for u in units:
        try:
            r = synthetic_control(df, unit_col=unit_col, time_col=time_col, y_col=y_col, treated_unit=str(u), intervention_time=str(intervention_time))
            out.append({"placebo_treated": str(u), "pre_rmse": float(r.pre_rmse), "post_gap_mean": float(r.post_gap_mean)})
        except Exception as e:
            out.append({"placebo_treated": str(u), "error": str(e)})
    return out
