from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
from .schemas import ensure_month_str

@dataclass
class IngestConfig:
    gdelt_events_csv: Optional[Path] = None
    eurepoc_csv: Optional[Path] = None
    worldbank_covariates_csv: Optional[Path] = None

    nvd_cve_csv: Optional[Path] = None
    dependency_updates_csv: Optional[Path] = None
    telemetry_signals_csv: Optional[Path] = None

class Ingestor:
    def __init__(self, config: IngestConfig):
        self.config = config

    def load_country_month_panel(self) -> pd.DataFrame:
        if not (self.config.gdelt_events_csv and self.config.eurepoc_csv):
            raise FileNotFoundError("Provide gdelt_events_csv and eurepoc_csv in IngestConfig.")

        gdelt = pd.read_csv(self.config.gdelt_events_csv)
        eurepoc = pd.read_csv(self.config.eurepoc_csv)

        gdelt["month"] = ensure_month_str(gdelt["date"])
        gdelt_agg = gdelt.groupby(["country_iso3","month"], as_index=False)[["unrest_count","unrest_intensity"]].sum()

        eurepoc["month"] = ensure_month_str(eurepoc["incident_date"])
        eurepoc["country_iso3"] = eurepoc["target_country_iso3"]
        cyber = eurepoc.groupby(["country_iso3","month"], as_index=False)["incident_id"].nunique().rename(columns={"incident_id":"cyber_incidents"})

        panel = pd.merge(gdelt_agg, cyber, on=["country_iso3","month"], how="outer").fillna({"unrest_count":0,"unrest_intensity":0,"cyber_incidents":0})

        if self.config.worldbank_covariates_csv:
            wb = pd.read_csv(self.config.worldbank_covariates_csv)
            rows = []
            for _, r in wb.iterrows():
                for m in range(1, 13):
                    rows.append({
                        "country_iso3": r["country_iso3"],
                        "month": f"{int(r['year']):04d}-{m:02d}",
                        "internet_users_pct": r.get("internet_users_pct"),
                        "gdp_current_usd": r.get("gdp_current_usd"),
                    })
            wb_monthly = pd.DataFrame(rows)
            panel = pd.merge(panel, wb_monthly, on=["country_iso3","month"], how="left")

        panel["cyber_incidents"] = panel["cyber_incidents"].astype(int)
        return panel.sort_values(["country_iso3","month"]).reset_index(drop=True)

    def load_vuln_instance_month(self) -> pd.DataFrame:
        if not self.config.nvd_cve_csv:
            raise FileNotFoundError("Provide nvd_cve_csv in IngestConfig.")
        cve = pd.read_csv(self.config.nvd_cve_csv)
        cve["month"] = ensure_month_str(cve["published_date"])
        out = cve[["cve_id","month","cvss_base_score","description","exploited_observed"]].copy()

        if self.config.dependency_updates_csv:
            deps = pd.read_csv(self.config.dependency_updates_csv)
            out = pd.merge(out, deps, on="cve_id", how="left")

        if self.config.telemetry_signals_csv:
            tel = pd.read_csv(self.config.telemetry_signals_csv)
            out = pd.merge(out, tel, on="cve_id", how="left")

        out["exploited_observed"] = out["exploited_observed"].astype(int)
        return out.sort_values(["cve_id","month"]).reset_index(drop=True)
