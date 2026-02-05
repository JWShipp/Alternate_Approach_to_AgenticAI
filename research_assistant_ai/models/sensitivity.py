from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import itertools
import pandas as pd

from .causal_plus import difference_in_differences, DiDResult

@dataclass
class CovariateSensitivityResult:
    base_covariates: List[str]
    runs: List[Dict[str, Any]]

def covariate_set_sensitivity(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    y_col: str,
    treated_col: str,
    post_col: str,
    base_covariates: List[str],
    cluster_col: Optional[str] = None,
    max_models: int = 25,
) -> CovariateSensitivityResult:
    """Re-run DiD across subsets of covariates (MVP).

    Returns a list of models with ATT and p-values across covariate subsets.
    """
    runs: List[Dict[str, Any]] = []
    base = list(dict.fromkeys(base_covariates))  # dedupe stable

    # Always include empty and full set; add single-drop variants; then sample combinations if needed.
    cov_sets = [[], base] + [[c for c in base if c != drop] for drop in base]
    # Add some pairwise combos if budget remains
    if len(cov_sets) < max_models:
        for r in range(1, min(3, len(base)) + 1):
            for comb in itertools.combinations(base, r):
                cov_sets.append(list(comb))
                if len(cov_sets) >= max_models:
                    break
            if len(cov_sets) >= max_models:
                break

    seen = set()
    uniq_cov_sets = []
    for cs in cov_sets:
        key = tuple(sorted(cs))
        if key in seen:
            continue
        seen.add(key)
        uniq_cov_sets.append(cs)

    for cs in uniq_cov_sets[:max_models]:
        res: DiDResult = difference_in_differences(
            df,
            unit_col=unit_col,
            time_col=time_col,
            y_col=y_col,
            treated_col=treated_col,
            post_col=post_col,
            x_cols=cs,
            cluster_col=cluster_col,
        )
        runs.append({"covariates": cs, "att": float(res.att), "p_value": float(res.p_value), "summary": res.model_summary})
    return CovariateSensitivityResult(base_covariates=base, runs=runs)
