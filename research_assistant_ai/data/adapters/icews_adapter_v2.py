from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Iterable, List
import pandas as pd

from ...utils.iso3 import ISO3Mapper

@dataclass
class ICEWSV2Config:
    path: Path
    date_col: str
    country_col: str
    intensity_col: Optional[str] = None
    event_code_col: Optional[str] = None
    actor1_col: Optional[str] = None
    actor2_col: Optional[str] = None
    chunksize: int = 250_000
    iso3_mapping_csv: Optional[Path] = None

def _iter_csv(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    # engine='python' handles odd quoting better in messy CSVs
    return pd.read_csv(path, chunksize=chunksize, low_memory=False, engine="c", on_bad_lines="skip")

def load_icews_v2(cfg: ICEWSV2Config) -> pd.DataFrame:
    """Load ICEWS (or ICEWS-like) events at scale from CSV, returning a standardized frame.

    Output columns:
      - event_date (datetime64)
      - country_iso3 (str)
      - intensity (float)
      - event_code (str, optional)
      - actor1 (str, optional)
      - actor2 (str, optional)
    """
    path = Path(cfg.path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    mapper = ISO3Mapper.from_optional_csv(cfg.iso3_mapping_csv)

    out_parts: List[pd.DataFrame] = []
    required = [cfg.date_col, cfg.country_col]
    for chunk in _iter_csv(path, cfg.chunksize):
        for c in required:
            if c not in chunk.columns:
                raise ValueError(f"ICEWS file missing required column: {c}")

        dt = pd.to_datetime(chunk[cfg.date_col], errors="coerce")
        country = chunk[cfg.country_col].astype(str).map(mapper.normalize)

        intensity = pd.Series(1.0, index=chunk.index)
        if cfg.intensity_col and cfg.intensity_col in chunk.columns:
            intensity = pd.to_numeric(chunk[cfg.intensity_col], errors="coerce").fillna(1.0).astype(float)

        df = pd.DataFrame({
            "event_date": dt,
            "country_iso3": country,
            "intensity": intensity,
        })
        if cfg.event_code_col and cfg.event_code_col in chunk.columns:
            df["event_code"] = chunk[cfg.event_code_col].astype(str)
        if cfg.actor1_col and cfg.actor1_col in chunk.columns:
            df["actor1"] = chunk[cfg.actor1_col].astype(str)
        if cfg.actor2_col and cfg.actor2_col in chunk.columns:
            df["actor2"] = chunk[cfg.actor2_col].astype(str)

        df = df.dropna(subset=["event_date","country_iso3"])
        df = df[df["country_iso3"].str.len() >= 3]
        out_parts.append(df)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=["event_date","country_iso3","intensity"])
    return out
