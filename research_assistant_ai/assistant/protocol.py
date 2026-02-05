from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import datetime as dt

@dataclass
class ExperimentProtocol:
    hypothesis_id: str
    dataset_name: str
    unit_of_analysis: str
    model_family: str
    time_window: str
    covariates: List[str]
    max_lag: int
    correction: str = "BH"  # Benjamini-Hochberg by default
    created_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d["created_utc"]:
            d["created_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds")
        return d

class ProtocolLogger:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / "protocols.jsonl"

    def log(self, protocol: ExperimentProtocol) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(protocol.to_dict(), ensure_ascii=False) + "\n")
