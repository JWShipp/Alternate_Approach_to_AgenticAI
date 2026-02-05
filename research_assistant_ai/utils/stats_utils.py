from __future__ import annotations
from typing import Iterable, List
import numpy as np

def benjamini_hochberg(pvals: Iterable[float]) -> List[float]:
    """Return BH-adjusted q-values (FDR control)."""
    p = np.asarray(list(pvals), dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out.tolist()
