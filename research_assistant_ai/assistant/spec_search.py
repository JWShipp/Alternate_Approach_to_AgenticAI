from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Callable, Tuple
import pandas as pd
import numpy as np

@dataclass
class Spec:
    name: str
    unit_of_analysis: str
    model_family: str
    max_lag: int
    covariates: List[str]
    fixed_effects: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SpecResult:
    spec: Spec
    metric: float
    p_value: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = {"metric": self.metric, "p_value": self.p_value, **self.details}
        d.update({f"spec_{k}": v for k, v in self.spec.to_dict().items()})
        return d

def _make_unit(panel: pd.DataFrame, unit: str) -> pd.DataFrame:
    """MVP unit-of-analysis transformer.

    - country-month: as-is
    - global-month: aggregates across countries
    - country-quarter: aggregates months into quarters
    """
    df = panel.copy()
    if unit == "country-month":
        return df
    if unit == "global-month":
        out = df.groupby("month", as_index=False)[["cyber_incidents","unrest_count","unrest_intensity","internet_users_pct","gdp_current_usd"]].sum()
        out["country_iso3"] = "GLOBAL"
        return out[["country_iso3","month","unrest_count","unrest_intensity","cyber_incidents","internet_users_pct","gdp_current_usd"]]
    if unit == "country-quarter":
        # month -> quarter label YYYY-Q#
        m = pd.to_datetime(df["month"] + "-01")
        q = m.dt.to_period("Q").astype(str)
        df["quarter"] = q
        out = df.groupby(["country_iso3","quarter"], as_index=False)[["cyber_incidents","unrest_count","unrest_intensity"]].sum()
        # carry forward covariates by mean (MVP)
        cov = df.groupby(["country_iso3","quarter"], as_index=False)[["internet_users_pct","gdp_current_usd"]].mean()
        out = pd.merge(out, cov, on=["country_iso3","quarter"], how="left")
        out = out.rename(columns={"quarter":"month"})
        return out
    raise ValueError(f"Unsupported unit_of_analysis: {unit}")

def fit_count_model(panel: pd.DataFrame, *, model_family: str, x: str, covariates: List[str], fixed_effects: bool) -> Dict[str, Any]:
    """Fit a count model and return coefficient magnitude and p-value for x.

    model_family:
      - poisson
      - negbin
      - zip (zero-inflated poisson)
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.discrete.count_model import ZeroInflatedPoisson

    d = panel.copy()
    d = d.replace([np.inf, -np.inf], np.nan)

    fe_terms = ""
    if fixed_effects:
        d["month_fe"] = d["month"].astype("category")
        d["country_fe"] = d["country_iso3"].astype("category")
        fe_terms = " + C(month_fe) + C(country_fe)"

    rhs = " + ".join([x] + covariates) + fe_terms
    formula = f"cyber_incidents ~ {rhs}"

    if model_family == "poisson":
        m = smf.glm(formula=formula, data=d, family=sm.families.Poisson()).fit()
        coef = float(m.params[x])
        pval = float(m.pvalues[x])
        return {"coef": coef, "p_value": pval, "aic": float(m.aic), "bic": float(m.bic), "n": int(m.nobs)}
    if model_family == "negbin":
        m = smf.glm(formula=formula, data=d, family=sm.families.NegativeBinomial()).fit()
        coef = float(m.params[x])
        pval = float(m.pvalues[x])
        return {"coef": coef, "p_value": pval, "aic": float(m.aic), "bic": float(m.bic), "n": int(m.nobs)}
    if model_family == "zip":
        # build design matrices
        y = d["cyber_incidents"].astype(float).values
        # use patsy via formula for exog
        import patsy
        ymat, X = patsy.dmatrices(formula, data=d, return_type="dataframe")
        # inflation model: intercept only (MVP)
        exog = X
        exog_infl = np.ones((len(d), 1))
        m = ZeroInflatedPoisson(endog=y, exog=exog, exog_infl=exog_infl, inflation="logit").fit(disp=0)
        # column name for x in patsy matrix
        col = [c for c in exog.columns if c == x or c.endswith(f":{x}") or c.endswith(f"{x}")][0]
        coef = float(m.params[col])
        pval = float(m.pvalues[col])
        return {"coef": coef, "p_value": pval, "aic": float(m.aic), "bic": float(m.bic), "n": int(len(d))}
    raise ValueError(f"Unsupported model_family: {model_family}")

def run_spec_search(panel: pd.DataFrame, specs: List[Spec]) -> pd.DataFrame:
    results: List[SpecResult] = []
    for spec in specs:
        df = _make_unit(panel, spec.unit_of_analysis)
        df = df.sort_values(["country_iso3","month"]).reset_index(drop=True)

        # lag sweep for the focal predictor unrest_count
        best: Optional[SpecResult] = None
        for lag in range(0, spec.max_lag + 1):
            d = df.copy()
            if lag > 0:
                d["x_lag"] = d.groupby("country_iso3")["unrest_count"].shift(lag)
                d = d.dropna(subset=["x_lag"])
                x = "x_lag"
            else:
                x = "unrest_count"

            try:
                out = fit_count_model(d, model_family=spec.model_family, x=x, covariates=spec.covariates, fixed_effects=spec.fixed_effects)
                metric = abs(out["coef"])
                pval = out["p_value"]
                details = {"lag": lag, "coef": out["coef"], "aic": out["aic"], "bic": out["bic"], "n": out["n"]}
                cand = SpecResult(spec=spec, metric=metric, p_value=pval, details=details)
                if best is None or (cand.metric > best.metric) or (cand.metric == best.metric and cand.p_value < best.p_value):
                    best = cand
            except Exception as e:
                continue

        if best is not None:
            results.append(best)

    if not results:
        return pd.DataFrame()
    df_out = pd.DataFrame([r.to_dict() for r in results])
    df_out = df_out.sort_values(["metric","p_value"], ascending=[False, True]).reset_index(drop=True)
    return df_out
