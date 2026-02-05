from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time
import uuid

@dataclass
class DatasetFingerprint:
    name: str
    sha256: str
    schema: Dict[str, Any]

@dataclass
class RunRecord:
    run_id: str
    created_utc: str
    phase: str
    treated_country: str
    intervention_month: str
    method: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    tags: List[str]

class ExperimentRegistry:
    """Minimal reproducibility registry.

    Stores JSONL records for each run. This is intentionally simple and git-friendly.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.root / "runs.jsonl"

    def log_run(
        self,
        *,
        phase: str,
        treated_country: str,
        intervention_month: str,
        method: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> RunRecord:
        rid = str(uuid.uuid4())
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec = RunRecord(
            run_id=rid,
            created_utc=created,
            phase=str(phase),
            treated_country=str(treated_country),
            intervention_month=str(intervention_month),
            method=str(method),
            inputs=inputs or {},
            outputs=outputs or {},
            tags=tags or [],
        )
        with open(self.runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def list_runs(self, limit: int = 200) -> List[RunRecord]:
        if not self.runs_path.exists():
            return []
        rows = []
        with open(self.runs_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows.append(RunRecord(**d))
        return rows

    def find_runs(self, *, treated_country: Optional[str] = None, method: Optional[str] = None) -> List[RunRecord]:
        runs = self.list_runs(limit=10_000)
        out = []
        for r in runs:
            if treated_country and r.treated_country != treated_country:
                continue
            if method and r.method != method:
                continue
            out.append(r)
        return out
