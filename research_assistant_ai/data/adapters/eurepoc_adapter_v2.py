from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List
import pandas as pd

from ...utils.iso3 import ISO3Mapper

@dataclass
class EuRepoCV2Config:
    path: Path
    date_col: str
    country_col: str
    severity_col: Optional[str] = None
    incident_type_col: Optional[str] = None
    chunksize: int = 250_000
    iso3_mapping_csv: Optional[Path] = None

def _iter_csv(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(path, chunksize=chunksize, low_memory=False, engine="c", on_bad_lines="skip")

def load_eurepoc_v2(cfg: EuRepoCV2Config) -> pd.DataFrame:
    """Load EuRepoC (or EuRepoC-like) incidents at scale from CSV, returning a standardized frame.

    Output columns:
      - incident_date (datetime64)
      - target_country_iso3 (str)
      - severity (float)
      - incident_type (str, optional)
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
                raise ValueError(f"EuRepoC file missing required column: {c}")

        dt = pd.to_datetime(chunk[cfg.date_col], errors="coerce")
        country = chunk[cfg.country_col].astype(str).map(mapper.normalize)

        sev = pd.Series(1.0, index=chunk.index)
        if cfg.severity_col and cfg.severity_col in chunk.columns:
            sev = pd.to_numeric(chunk[cfg.severity_col], errors="coerce").fillna(1.0).astype(float)

        df = pd.DataFrame({
            "incident_date": dt,
            "target_country_iso3": country,
            "severity": sev,
        })
        if cfg.incident_type_col and cfg.incident_type_col in chunk.columns:
            df["incident_type"] = chunk[cfg.incident_type_col].astype(str)

        df = df.dropna(subset=["incident_date","target_country_iso3"])
        df = df[df["target_country_iso3"].str.len() >= 3]
        out_parts.append(df)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=["incident_date","target_country_iso3","severity"])
    return out
