from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from ..panel_builder import build_country_month_panel
from ...utils.hashing import sha256_file
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class Phase7Inputs:
    icews_path: Path
    eurepoc_path: Path
    sanctions_path: Optional[Path] = None
    iso3_mapping_csv: Optional[Path] = None

@dataclass
class Phase7Schema:
    # ICEWS
    icews_date_col: str
    icews_country_col: str
    icews_intensity_col: Optional[str] = None
    # EuRepoC
    eurepoc_date_col: str = "incident_date"
    eurepoc_country_col: str = "target_country_iso3"
    eurepoc_severity_col: Optional[str] = None
    # Sanctions
    sanctions_date_col: str = "sanction_date"
    sanctions_country_col: str = "country_iso3"
    sanctions_intensity_col: Optional[str] = None

def _missingness(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "missing_n": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [float(df[c].isna().mean()) for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
    })

def _dataset_fingerprint(inputs: Phase7Inputs) -> Dict[str, str]:
    fp = {
        "icews_sha256": sha256_file(Path(inputs.icews_path)),
        "eurepoc_sha256": sha256_file(Path(inputs.eurepoc_path)),
    }
    if inputs.sanctions_path and Path(inputs.sanctions_path).exists():
        fp["sanctions_sha256"] = sha256_file(Path(inputs.sanctions_path))
    if inputs.iso3_mapping_csv and Path(inputs.iso3_mapping_csv).exists():
        fp["iso3_mapping_sha256"] = sha256_file(Path(inputs.iso3_mapping_csv))
    return fp

def harmonize_build_audit_version(
    *,
    icews_df: pd.DataFrame,
    eurepoc_df: pd.DataFrame,
    sanctions_df: Optional[pd.DataFrame],
    export_dir: Path,
    inputs: Phase7Inputs,
    schema: Phase7Schema
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    panel = build_country_month_panel(
        icews_events=icews_df,
        eurepoc_incidents=eurepoc_df,
        sanctions=sanctions_df,
        icews_date_col="event_date",
        icews_country_col="country_iso3",
        icews_intensity_col="intensity",
        eurepoc_date_col="incident_date",
        eurepoc_country_col="target_country_iso3",
        eurepoc_severity_col="severity",
        sanctions_date_col="sanction_date",
        sanctions_country_col="country_iso3",
        sanctions_intensity_col="intensity",
    )

    audit_dir = export_dir / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Audit
    miss_panel = _missingness(panel)
    miss_panel.to_csv(audit_dir / "panel_missingness.csv", index=False)

    # Simple sanity checks
    checks = []
    checks.append({"check": "nonnegative_outcome", "passed": bool((panel["cyber_incidents"] >= 0).all())})
    checks.append({"check": "nonnegative_treatment", "passed": bool((panel["unrest_count"] >= 0).all())})
    checks.append({"check": "country_iso3_len", "passed": bool((panel["country_iso3"].astype(str).str.len() == 3).all())})

    # Versioning
    fp = _dataset_fingerprint(inputs)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ds_dir = export_dir / "datasets" / stamp
    ds_dir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(ds_dir / "panel_country_month.csv", index=False)
    (ds_dir / "fingerprint.json").write_text(pd.Series(fp).to_json(indent=2))
    (ds_dir / "schema.json").write_text(pd.Series(schema.__dict__).to_json(indent=2))

    summary = {
        "fingerprint": fp,
        "version_dir": str(ds_dir),
        "audit_dir": str(audit_dir),
        "panel_shape": [int(panel.shape[0]), int(panel.shape[1])],
        "months_min": str(panel["month"].min()) if len(panel) else "",
        "months_max": str(panel["month"].max()) if len(panel) else "",
        "countries_n": int(panel["country_iso3"].nunique()) if "country_iso3" in panel.columns else 0,
        "checks": checks,
    }
    (audit_dir / "phase7_audit_summary.json").write_text(pd.Series(summary).to_json(indent=2))

    return panel, summary
