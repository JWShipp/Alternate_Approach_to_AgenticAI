from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable, Any, Optional
import numpy as np

@dataclass
class StructuralCausalModel:
    assignments: Dict[str, Callable[[Dict[str, Any], np.random.Generator], Any]]

    def sample(self, exogenous: Optional[Dict[str, Any]] = None, do: Optional[Dict[str, Any]] = None,
               rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        exogenous = dict(exogenous or {})
        do = dict(do or {})
        rng = rng or np.random.default_rng()
        values: Dict[str, Any] = {}
        for var, fn in self.assignments.items():
            if var in do:
                values[var] = do[var]
            else:
                ctx = {**exogenous, **values}
                values[var] = fn(ctx, rng)
        return values

    def counterfactual(self, evidence: Dict[str, Any], do: Dict[str, Any], n: int = 1000,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        rng = rng or np.random.default_rng()
        samples = [self.sample(exogenous=evidence, do=do, rng=rng) for _ in range(n)]
        out: Dict[str, float] = {}
        for k in samples[0].keys():
            vals = [x[k] for x in samples if isinstance(x[k], (int, float, np.number))]
            if vals:
                out[k] = float(np.mean(vals))
        return out
