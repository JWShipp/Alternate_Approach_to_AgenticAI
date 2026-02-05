from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import math
import numpy as np
import pandas as pd

@dataclass
class ModelScore:
    name: str
    bic: float
    n: int
    k: int  # parameters
    weight: float
    meta: Dict[str, Any]

def bic_weights(bics: List[float]) -> List[float]:
    bmin = min(bics)
    deltas = [b - bmin for b in bics]
    w = [math.exp(-0.5 * d) for d in deltas]
    s = sum(w) or 1.0
    return [wi / s for wi in w]

def compare_did_models(
    df: pd.DataFrame,
    *,
    unit_fe: str,
    time_fe: str,
    y_col: str,
    treated_col: str,
    post_col: str,
    candidate_covariates: List[List[str]],
    cluster_col: Optional[str] = None,
) -> List[ModelScore]:
    """Compare DiD specifications via BIC-based weights (MVP Bayesian flavor).

    Uses the same DiD estimator used elsewhere (statsmodels OLS with FE dummies).
    """
    from .causal_plus import difference_in_differences

    scores: List[ModelScore] = []
    bics = []
    tmp = []
    for i, covs in enumerate(candidate_covariates):
        res = difference_in_differences(
            df,
            unit_col=unit_fe,
            time_col=time_fe,
            y_col=y_col,
            treated_col=treated_col,
            post_col=post_col,
            x_cols=covs,
            cluster_col=cluster_col,
        )
        bic = float(res.model_summary.get("bic", np.nan))
        if np.isnan(bic):
            # crude fallback: use AIC if BIC unavailable
            bic = float(res.model_summary.get("aic", 0.0))
        bics.append(bic)
        tmp.append((f"did_spec_{i+1}", bic, res))

    ws = bic_weights(bics)
    for (name, bic, res), w in zip(tmp, ws):
        scores.append(ModelScore(
            name=name,
            bic=float(bic),
            n=int(res.model_summary.get("n", 0)),
            k=int(res.model_summary.get("k", 0)),
            weight=float(w),
            meta={"att": float(res.att), "p_value": float(res.p_value), "covariates": res.model_summary.get("covariates", [])},
        ))
    return sorted(scores, key=lambda s: s.weight, reverse=True)
