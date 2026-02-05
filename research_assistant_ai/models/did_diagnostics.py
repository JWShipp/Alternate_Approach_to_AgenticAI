from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class ParallelTrendsResult:
    coef: float
    p_value: float
    n: int
    model_summary: Dict[str, Any]

def parallel_trends_pretest(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    y_col: str,
    treated_col: str,
    intervention_time: str,
    trend_col_name: str = "_trend",
    x_cols: Optional[List[str]] = None,
    cluster_col: Optional[str] = None,
) -> ParallelTrendsResult:
    """Pre-trends test using pre-period data only.

    Model (pre-period): y ~ treated * trend + covariates + unit FE + time FE
    Null: coefficient on treated:trend == 0
    """
    import statsmodels.formula.api as smf

    d = df.copy()
    d[time_col] = d[time_col].astype(str)
    pre = d[d[time_col] < str(intervention_time)].copy()
    if len(pre) < 10:
        raise ValueError("Not enough pre-period observations for parallel trends test.")

    # Create a numeric trend index on pre-period months
    months = sorted(pre[time_col].unique().tolist())
    idx = {m:i for i,m in enumerate(months)}
    pre[trend_col_name] = pre[time_col].map(idx).astype(float)

    pre[unit_col] = pre[unit_col].astype("category")
    pre[time_col] = pre[time_col].astype("category")
    pre["_int"] = pre[treated_col].astype(int) * pre[trend_col_name]

    cov = " + ".join(x_cols or [])
    rhs = f"{treated_col} + {trend_col_name} + _int"
    if cov:
        rhs += " + " + cov
    rhs += f" + C({unit_col}) + C({time_col})"
    m = smf.ols(formula=f"{y_col} ~ {rhs}", data=pre).fit()
    if cluster_col:
        m = m.get_robustcov_results(cov_type="cluster", groups=pre[cluster_col])

    idx_int = m.model.exog_names.index("_int")
    return ParallelTrendsResult(
        coef=float(m.params[idx_int]),
        p_value=float(m.pvalues[idx_int]),
        n=int(m.nobs),
        model_summary={"r2": float(m.rsquared)},
    )

@dataclass
class PrePeriodJointTestResult:
    k_values: List[int]
    f_stat: float
    p_value: float
    df_denom: float
    df_num: float

def event_study_preperiod_joint_test(
    coef_by_k: Dict[int, float],
    p_by_k: Dict[int, float],
    *,
    k_pre_max: int = -2
) -> PrePeriodJointTestResult:
    """MVP joint test proxy for pre-period stability.

    In a full econometric implementation, we would use an F-test on the regression restriction that all pre-period
    coefficients equal zero. Here, as an MVP, we aggregate p-values using Fisher's method for k <= k_pre_max.
    """
    from scipy.stats import chi2

    ks = sorted([k for k in coef_by_k.keys() if k <= k_pre_max])
    if not ks:
        raise ValueError("No pre-period coefficients available for joint test.")

    ps = [max(1e-12, min(1.0, float(p_by_k[k]))) for k in ks]
    stat = -2.0 * float(np.sum(np.log(ps)))
    df_num = 2.0 * len(ps)
    p = float(1.0 - chi2.cdf(stat, df=df_num))
    return PrePeriodJointTestResult(k_values=ks, f_stat=stat, p_value=p, df_denom=float("nan"), df_num=df_num)
