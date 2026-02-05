from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import datetime as dt
import uuid

@dataclass
class HypothesisRecord:
    hypothesis_id: str
    statement: str
    unit_of_analysis: str
    outcome: str
    treatment: str
    created_utc: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class HypothesisRegistry:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / "hypotheses.jsonl"

    def create(self, statement: str, unit_of_analysis: str, outcome: str, treatment: str, tags: Optional[List[str]] = None) -> HypothesisRecord:
        hid = str(uuid.uuid4())
        rec = HypothesisRecord(
            hypothesis_id=hid,
            statement=statement,
            unit_of_analysis=unit_of_analysis,
            outcome=outcome,
            treatment=treatment,
            created_utc=dt.datetime.utcnow().isoformat(timespec="seconds"),
            tags=list(tags or []),
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return rec
