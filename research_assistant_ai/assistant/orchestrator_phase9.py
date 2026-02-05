from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .dissertation_writer import export_results_docx

@dataclass
class Phase9Outputs:
    results_docx: str
    inputs_used: Dict[str, Any]

def run_phase9_from_phase8_exports(
    *,
    phase8_export_dir: Path,
    phase8_report: Dict[str, Any],
    out_dir: Path
) -> Phase9Outputs:
    phase8_export_dir = Path(phase8_export_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx = phase8_export_dir / "phase8_rigor_results.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(str(xlsx))

    event_study = pd.read_excel(xlsx, sheet_name="EventStudy")
    did_sensitivity = pd.read_excel(xlsx, sheet_name="DiD_Sensitivity")

    out_docx = export_results_docx(
        phase8_report=phase8_report,
        event_study_df=event_study,
        did_sensitivity_df=did_sensitivity,
        out_path=out_dir / "phase9_dissertation_results_draft.docx"
    )

    return Phase9Outputs(
        results_docx=str(out_docx),
        inputs_used={"phase8_results_xlsx": str(xlsx)}
    )
