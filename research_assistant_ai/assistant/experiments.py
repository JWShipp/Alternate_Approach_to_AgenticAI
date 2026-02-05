from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable, Optional
import pandas as pd

@dataclass
class ExperimentResult:
    name: str
    metric: float
    p_value: float
    model_type: str
    notes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ExperimentRunner:
    def __init__(self):
        self.results: List[ExperimentResult] = []

    def run_experiment(self, name: str, model_fn: Callable[[pd.DataFrame], Dict[str, Any]], data: pd.DataFrame,
                       *, fail_silently: bool = False) -> Optional[ExperimentResult]:
        try:
            out = model_fn(data)
            res = ExperimentResult(
                name=name,
                metric=float(out["metric"]),
                p_value=float(out["p_value"]),
                model_type=str(out.get("model_type", "unspecified")),
                notes=dict(out.get("notes", {})),
            )
            self.results.append(res)
            return res
        except Exception:
            if fail_silently:
                return None
            raise

    def rank_results(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=["name","metric","p_value","model_type","notes"])
        df = pd.DataFrame([r.to_dict() for r in self.results])
        df = df.sort_values(["metric","p_value"], ascending=[False, True])
        return df.reset_index(drop=True)
