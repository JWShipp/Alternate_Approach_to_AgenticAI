from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class EuRepoCConfig:
    csv_path: Path
    date_col: str = "incident_date"
    country_col: str = "target_country_iso3"
    incident_type_col: str = "incident_type"
    severity_col: str = "severity"

def load_eurepoc_incidents(cfg: EuRepoCConfig) -> pd.DataFrame:
    """Load EuRepoC-like cyber incidents from a local CSV.

    Expected minimum columns (defaults):
      - incident_date
      - target_country_iso3
    Optional:
      - incident_type
      - severity (numeric)
    """
    df = pd.read_csv(cfg.csv_path)
    if cfg.date_col not in df.columns or cfg.country_col not in df.columns:
        raise ValueError(f"EuRepoC CSV must include columns {cfg.date_col} and {cfg.country_col}")
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.country_col])
    df[cfg.country_col] = df[cfg.country_col].astype(str)
    if cfg.severity_col in df.columns:
        df[cfg.severity_col] = pd.to_numeric(df[cfg.severity_col], errors="coerce").fillna(1.0)
    else:
        df[cfg.severity_col] = 1.0
    if cfg.incident_type_col not in df.columns:
        df[cfg.incident_type_col] = "UNKNOWN"
    return df[[cfg.date_col, cfg.country_col, cfg.incident_type_col, cfg.severity_col]].copy()
