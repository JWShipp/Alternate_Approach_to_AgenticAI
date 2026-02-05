from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass(frozen=True)
class CountryMonthPanelSchema:
    country_iso3: str
    month: str
    unrest_count: float
    unrest_intensity: float
    cyber_incidents: int
    internet_users_pct: Optional[float] = None
    gdp_current_usd: Optional[float] = None

@dataclass(frozen=True)
class VulnerabilityInstanceMonthSchema:
    cve_id: str
    month: str
    cvss_base_score: Optional[float]
    description: str
    exploited_observed: int
    dep_update_delay_days: Optional[float] = None
    scanner_confirmed_present: Optional[int] = None
    telemetry_exploit_signal: Optional[int] = None
    sbom_present: Optional[int] = None
    time_to_remediate_days: Optional[float] = None

def ensure_month_str(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series)
    return s.dt.strftime("%Y-%m")
