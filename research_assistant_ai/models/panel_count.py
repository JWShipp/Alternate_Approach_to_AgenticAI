from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class PanelCountResult:
    model_name: str
    params: pd.Series
    standard_errors: pd.Series

def fit_poisson_glm(panel: pd.DataFrame) -> PanelCountResult:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = panel.copy()
    df["month_fe"] = df["month"].astype("category")
    df["country_fe"] = df["country_iso3"].astype("category")

    formula = "cyber_incidents ~ unrest_count + unrest_intensity + internet_users_pct + gdp_current_usd + C(month_fe) + C(country_fe)"
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
    return PanelCountResult(model_name="PoissonGLM", params=model.params, standard_errors=model.bse)
