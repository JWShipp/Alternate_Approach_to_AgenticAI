from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import datetime as dt

@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    created_utc: str
    tags: List[str]

class ResearchMemory:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.hyp_path = self.root_dir / "hypotheses.jsonl"
        self.exp_path = self.root_dir / "experiments.jsonl"

    def add_hypothesis(self, hypothesis_id: str, statement: str, tags: Optional[List[str]] = None) -> Hypothesis:
        h = Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=statement,
            created_utc=dt.datetime.utcnow().isoformat(timespec="seconds"),
            tags=list(tags or []),
        )
        with self.hyp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(h), ensure_ascii=False) + "\n")
        return h

    def log_experiment(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("logged_utc", dt.datetime.utcnow().isoformat(timespec="seconds"))
        with self.exp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
