from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class SanctionsConfig:
    csv_path: Path
    date_col: str = "sanction_date"
    country_col: str = "country_iso3"
    label_col: str = "label"
    intensity_col: str = "intensity"

def load_sanctions(cfg: SanctionsConfig) -> pd.DataFrame:
    """Load sanctions timeline from CSV.

    Minimum columns:
      - sanction_date (date/datetime)
      - country_iso3
    Optional:
      - label
      - intensity (numeric)
    """
    df = pd.read_csv(cfg.csv_path)
    if cfg.date_col not in df.columns or cfg.country_col not in df.columns:
        raise ValueError(f"Sanctions CSV must include columns {cfg.date_col} and {cfg.country_col}")
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.country_col])
    df[cfg.country_col] = df[cfg.country_col].astype(str)
    if cfg.label_col not in df.columns:
        df[cfg.label_col] = "SANCTION"
    if cfg.intensity_col in df.columns:
        df[cfg.intensity_col] = pd.to_numeric(df[cfg.intensity_col], errors="coerce").fillna(1.0)
    else:
        df[cfg.intensity_col] = 1.0
    return df[[cfg.date_col, cfg.country_col, cfg.label_col, cfg.intensity_col]].copy()
