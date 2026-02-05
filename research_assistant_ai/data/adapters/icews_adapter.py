from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd

@dataclass
class ICEWSConfig:
    csv_path: Path
    date_col: str = "event_date"
    country_col: str = "country_iso3"
    event_type_col: str = "event_type"
    intensity_col: str = "intensity"

def load_icews_events(cfg: ICEWSConfig) -> pd.DataFrame:
    """Load ICEWS-like events from a local CSV.

    Expected minimum columns (defaults):
      - event_date: date or datetime
      - country_iso3: ISO3 country code
      - event_type: string/categorical
      - intensity: numeric (optional but recommended)

    This adapter is intentionally CSV-first for dissertation reproducibility.
    """
    df = pd.read_csv(cfg.csv_path)
    if cfg.date_col not in df.columns or cfg.country_col not in df.columns:
        raise ValueError(f"ICEWS CSV must include columns {cfg.date_col} and {cfg.country_col}")
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.country_col])
    df[cfg.country_col] = df[cfg.country_col].astype(str)
    if cfg.intensity_col in df.columns:
        df[cfg.intensity_col] = pd.to_numeric(df[cfg.intensity_col], errors="coerce").fillna(1.0)
    else:
        df[cfg.intensity_col] = 1.0
    if cfg.event_type_col not in df.columns:
        df[cfg.event_type_col] = "UNKNOWN"
    return df[[cfg.date_col, cfg.country_col, cfg.event_type_col, cfg.intensity_col]].copy()
