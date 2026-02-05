from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

from ..data.adapters.icews_adapter_v2 import ICEWSV2Config, load_icews_v2
from ..data.adapters.eurepoc_adapter_v2 import EuRepoCV2Config, load_eurepoc_v2
from ..data.adapters.sanctions_adapter_v2 import SanctionsV2Config, load_sanctions_v2
from ..data.pipelines.harmonize_and_audit import Phase7Inputs, Phase7Schema, harmonize_build_audit_version
from .orchestrator_phase6 import run_phase6_russia_causal, Phase6Outputs

@dataclass
class Phase7Outputs:
    panel: pd.DataFrame
    phase6_outputs: Phase6Outputs
    ingest_audit: Dict[str, Any]

def run_phase7(
    *,
    icews_csv: Path,
    eurepoc_csv: Path,
    sanctions_csv: Optional[Path],
    iso3_mapping_csv: Optional[Path],
    schema: Phase7Schema,
    export_dir: Path,
    treated_country: str = "RUS",
    intervention_month: Optional[str] = None,
    x_cols: Optional[List[str]] = None,
) -> Phase7Outputs:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    icews = load_icews_v2(ICEWSV2Config(
        path=Path(icews_csv),
        date_col=schema.icews_date_col,
        country_col=schema.icews_country_col,
        intensity_col=schema.icews_intensity_col,
        chunksize=250_000,
        iso3_mapping_csv=iso3_mapping_csv,
    ))
    eurepoc = load_eurepoc_v2(EuRepoCV2Config(
        path=Path(eurepoc_csv),
        date_col=schema.eurepoc_date_col,
        country_col=schema.eurepoc_country_col,
        severity_col=schema.eurepoc_severity_col,
        chunksize=250_000,
        iso3_mapping_csv=iso3_mapping_csv,
    ))
    sanctions = None
    if sanctions_csv:
        sanctions = load_sanctions_v2(SanctionsV2Config(
            path=Path(sanctions_csv),
            date_col=schema.sanctions_date_col,
            country_col=schema.sanctions_country_col,
            intensity_col=schema.sanctions_intensity_col,
            label_col="label",
            chunksize=250_000,
            iso3_mapping_csv=iso3_mapping_csv,
        ))

    panel, audit = harmonize_build_audit_version(
        icews_df=icews,
        eurepoc_df=eurepoc,
        sanctions_df=sanctions,
        export_dir=export_dir,
        inputs=Phase7Inputs(icews_path=Path(icews_csv), eurepoc_path=Path(eurepoc_csv), sanctions_path=sanctions_csv, iso3_mapping_csv=iso3_mapping_csv),
        schema=schema,
    )

    # Run Russia causal analysis (phase 6 engine) but with sanctions_count as treatment
    p6 = run_phase6_russia_causal(
        panel,
        treated_country=treated_country,
        intervention_month=intervention_month,
        x_cols=x_cols or [],
        export_dir=export_dir,
        treatment_col="sanctions_count",
        outcome_col="cyber_incidents",
    )

    return Phase7Outputs(panel=panel, phase6_outputs=p6, ingest_audit=audit)
