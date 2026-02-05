from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

@dataclass(frozen=True)
class CountryTimePanelContract:
    """Contract for the core panel used in causal/predictive experiments.

    Required columns:
      - country_iso3: ISO3 string (or 'GLOBAL' for global series)
      - period: standardized time period label (default: YYYY-MM)
      - outcome: cyber incident count (int)
      - treatment: geopolitical/crisis intensity proxy (float or int)

    Optional covariates: any numeric columns.
    """
    country_col: str = "country_iso3"
    period_col: str = "period"
    outcome_col: str = "outcome"
    treatment_col: str = "treatment"
    min_rows: int = 50

    def required(self) -> List[str]:
        return [self.country_col, self.period_col, self.outcome_col, self.treatment_col]

def validate_panel(df: pd.DataFrame, contract: CountryTimePanelContract) -> Dict[str, Any]:
    missing = [c for c in contract.required() if c not in df.columns]
    ok = len(missing) == 0

    issues: List[str] = []
    if missing:
        issues.append(f"Missing required columns: {missing}")

    if ok:
        if len(df) < contract.min_rows:
            issues.append(f"Panel has {len(df)} rows; expected at least {contract.min_rows} for stable estimation.")
        # period parse check
        try:
            pd.to_datetime(df[contract.period_col].astype(str) + "-01")
        except Exception:
            issues.append(f"Could not parse {contract.period_col} into YYYY-MM format.")
        # outcome int-like
        if not pd.api.types.is_numeric_dtype(df[contract.outcome_col]):
            issues.append(f"{contract.outcome_col} must be numeric.")
        # treatment numeric
        if not pd.api.types.is_numeric_dtype(df[contract.treatment_col]):
            issues.append(f"{contract.treatment_col} must be numeric.")

    return {"passed": ok and len(issues) == 0, "missing": missing, "issues": issues, "n_rows": int(len(df))}
