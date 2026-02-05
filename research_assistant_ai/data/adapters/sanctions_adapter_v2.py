from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List
import pandas as pd

from ...utils.iso3 import ISO3Mapper

@dataclass
class SanctionsV2Config:
    path: Path
    date_col: str
    country_col: str
    label_col: Optional[str] = None
    intensity_col: Optional[str] = None
    chunksize: int = 250_000
    iso3_mapping_csv: Optional[Path] = None

def _iter_csv(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(path, chunksize=chunksize, low_memory=False, engine="c", on_bad_lines="skip")

def load_sanctions_v2(cfg: SanctionsV2Config) -> pd.DataFrame:
    """Load sanctions timeline at scale from CSV, returning a standardized frame.

    Output columns:
      - sanction_date (datetime64)
      - country_iso3 (str)
      - label (str)
      - intensity (float)
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
                raise ValueError(f"Sanctions file missing required column: {c}")

        dt = pd.to_datetime(chunk[cfg.date_col], errors="coerce")
        country = chunk[cfg.country_col].astype(str).map(mapper.normalize)

        label = pd.Series("SANCTION", index=chunk.index)
        if cfg.label_col and cfg.label_col in chunk.columns:
            label = chunk[cfg.label_col].astype(str)

        intensity = pd.Series(1.0, index=chunk.index)
        if cfg.intensity_col and cfg.intensity_col in chunk.columns:
            intensity = pd.to_numeric(chunk[cfg.intensity_col], errors="coerce").fillna(1.0).astype(float)

        df = pd.DataFrame({
            "sanction_date": dt,
            "country_iso3": country,
            "label": label,
            "intensity": intensity,
        })
        df = df.dropna(subset=["sanction_date","country_iso3"])
        df = df[df["country_iso3"].str.len() >= 3]
        out_parts.append(df)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=["sanction_date","country_iso3","label","intensity"])
    return out
