from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd

def to_month(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt).dt.to_period("M").astype(str)

@dataclass
class PanelBuildOptions:
    country_col: str = "country_iso3"
    month_col: str = "month"
    outcome_col: str = "cyber_incidents"
    treatment_col: str = "unrest_count"
    treatment_intensity_col: str = "unrest_intensity"
    sanctions_col: str = "sanctions_count"
    sanctions_intensity_col: str = "sanctions_intensity"

def build_country_month_panel(
    *,
    icews_events: pd.DataFrame,
    eurepoc_incidents: pd.DataFrame,
    sanctions: Optional[pd.DataFrame] = None,
    icews_date_col: str = "event_date",
    icews_country_col: str = "country_iso3",
    icews_intensity_col: str = "intensity",
    eurepoc_date_col: str = "incident_date",
    eurepoc_country_col: str = "target_country_iso3",
    eurepoc_severity_col: str = "severity",
    sanctions_date_col: str = "sanction_date",
    sanctions_country_col: str = "country_iso3",
    sanctions_intensity_col: str = "intensity",
    options: Optional[PanelBuildOptions] = None,
) -> pd.DataFrame:
    """Build a country-month panel suitable for spec search and causal estimators.

    The resulting panel includes:
      - cyber_incidents: count of incidents per country-month
      - unrest_count: count of ICEWS events per country-month
      - unrest_intensity: sum of ICEWS intensity per country-month
      - sanctions_count: count of sanctions per country-month (optional)
      - sanctions_intensity: sum intensity per country-month (optional)

    Covariates can be merged later (e.g., World Bank).
    """
    opt = options or PanelBuildOptions()

    ice = icews_events.copy()
    ice["month"] = to_month(ice[icews_date_col])
    ice["one"] = 1
    ice_agg = ice.groupby([icews_country_col, "month"], as_index=False).agg(
        unrest_count=("one","sum"),
        unrest_intensity=(icews_intensity_col,"sum"),
    ).rename(columns={icews_country_col: opt.country_col})

    eu = eurepoc_incidents.copy()
    eu["month"] = to_month(eu[eurepoc_date_col])
    eu["one"] = 1
    eu_agg = eu.groupby([eurepoc_country_col, "month"], as_index=False).agg(
        cyber_incidents=("one","sum"),
        cyber_severity=(eurepoc_severity_col,"sum"),
    ).rename(columns={eurepoc_country_col: opt.country_col})

    panel = pd.merge(eu_agg, ice_agg, on=[opt.country_col, "month"], how="outer")
    panel["cyber_incidents"] = panel["cyber_incidents"].fillna(0).astype(int)
    panel["unrest_count"] = panel["unrest_count"].fillna(0).astype(int)
    panel["unrest_intensity"] = panel["unrest_intensity"].fillna(0.0).astype(float)
    panel["cyber_severity"] = panel["cyber_severity"].fillna(0.0).astype(float)

    if sanctions is not None and len(sanctions) > 0:
        s = sanctions.copy()
        s["month"] = to_month(s[sanctions_date_col])
        s["one"] = 1
        s_agg = s.groupby([sanctions_country_col, "month"], as_index=False).agg(
            sanctions_count=("one","sum"),
            sanctions_intensity=(sanctions_intensity_col,"sum"),
        ).rename(columns={sanctions_country_col: opt.country_col})
        panel = pd.merge(panel, s_agg, on=[opt.country_col, "month"], how="left")
        panel["sanctions_count"] = panel["sanctions_count"].fillna(0).astype(int)
        panel["sanctions_intensity"] = panel["sanctions_intensity"].fillna(0.0).astype(float)
    else:
        panel["sanctions_count"] = 0
        panel["sanctions_intensity"] = 0.0

    panel = panel.rename(columns={"month": opt.month_col})
    panel = panel.sort_values([opt.country_col, opt.month_col]).reset_index(drop=True)
    return panel
