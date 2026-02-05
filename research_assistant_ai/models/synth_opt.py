from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .causal_plus import synthetic_control, SyntheticControlResult

@dataclass
class DonorSearchResult:
    best: SyntheticControlResult
    tried: List[Dict[str, Any]]

def donor_pool_search(
    df: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    y_col: str,
    treated_unit: str,
    intervention_time: str,
    max_donors: int = 15,
    top_k_candidates: int = 30,
) -> DonorSearchResult:
    """Heuristic donor pool optimization (MVP).

    Steps:
      1) Compute pre-period correlations between each donor and treated.
      2) Keep top_k_candidates donors by correlation.
      3) Greedy forward selection (up to max_donors) minimizing pre-period RMSE.
    """
    d = df.copy()
    d[time_col] = d[time_col].astype(str)

    all_donors = sorted([u for u in d[unit_col].unique().tolist() if str(u) != str(treated_unit)])

    times = sorted(d[time_col].unique().tolist())
    pre_times = [t for t in times if t < str(intervention_time)]
    if len(pre_times) < 6:
        # MVP fallback: skip donor search when pre-period is too short
        base = synthetic_control(d, unit_col=unit_col, time_col=time_col, y_col=y_col,
                                treated_unit=str(treated_unit), intervention_time=str(intervention_time))
        return DonorSearchResult(best=base, tried=[{"note":"fallback_small_pre_period","pre_times_n":len(pre_times)}])

    def series(u: str) -> np.ndarray:
        s = d[d[unit_col] == u].set_index(time_col)[y_col].astype(float)
        return s.reindex(pre_times).ffill().fillna(0.0).values

    yt = series(str(treated_unit))
    scored: List[Tuple[str, float]] = []
    for u in all_donors:
        xu = series(str(u))
        if np.std(xu) < 1e-9 or np.std(yt) < 1e-9:
            c = 0.0
        else:
            c = float(np.corrcoef(yt, xu)[0,1])
        scored.append((str(u), c))

    scored.sort(key=lambda x: x[1], reverse=True)
    candidates = [u for u,_ in scored[:max(1, min(top_k_candidates, len(scored)))]]

    tried: List[Dict[str, Any]] = []

    selected: List[str] = []
    best_res: Optional[SyntheticControlResult] = None
    best_rmse = float("inf")

    # Greedy forward selection
    for step in range(min(max_donors, len(candidates))):
        best_step = None
        best_step_rmse = float("inf")
        best_step_res = None
        for u in candidates:
            if u in selected:
                continue
            donors = selected + [u]
            res = synthetic_control(d, unit_col=unit_col, time_col=time_col, y_col=y_col,
                                    treated_unit=str(treated_unit), intervention_time=str(intervention_time),
                                    donor_units=donors)
            tried.append({"step": step+1, "donors_n": len(donors), "candidate_added": u, "pre_rmse": float(res.pre_rmse), "post_gap_mean": float(res.post_gap_mean)})
            if res.pre_rmse < best_step_rmse:
                best_step_rmse = float(res.pre_rmse)
                best_step = u
                best_step_res = res
        if best_step is None:
            break
        selected.append(best_step)
        if best_step_rmse < best_rmse:
            best_rmse = best_step_rmse
            best_res = best_step_res
        # Early stop if improvement is tiny
        if step >= 2 and abs(best_step_rmse - best_rmse) < 1e-6:
            break

    if best_res is None:
        best_res = synthetic_control(d, unit_col=unit_col, time_col=time_col, y_col=y_col,
                                     treated_unit=str(treated_unit), intervention_time=str(intervention_time),
                                     donor_units=candidates[:max(1, min(5, len(candidates)))])
    return DonorSearchResult(best=best_res, tried=tried)
