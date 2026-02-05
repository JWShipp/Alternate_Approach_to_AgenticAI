from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import uuid
import time

@dataclass
class Hypothesis:
    hypothesis_id: str
    created_utc: str
    statement: str
    treated_country: str
    outcome: str
    treatment: str
    expected_direction: str  # e.g., increase/decrease/ambiguous
    priors: Dict[str, Any]
    status: str  # proposed, tested, supported, refuted, inconclusive
    notes: str

class ResearchMemory:
    """Structured memory for hypotheses, decisions, and notes.

    This is not RAG. It's a small database of *claims* and *tests*.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"hypotheses": [], "notes": []}, indent=2))

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text())

    def _save(self, payload: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2))

    def add_hypothesis(
        self,
        *,
        statement: str,
        treated_country: str,
        outcome: str,
        treatment: str,
        expected_direction: str,
        priors: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ) -> Hypothesis:
        payload = self._load()
        h = Hypothesis(
            hypothesis_id=str(uuid.uuid4()),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            statement=str(statement),
            treated_country=str(treated_country),
            outcome=str(outcome),
            treatment=str(treatment),
            expected_direction=str(expected_direction),
            priors=priors or {},
            status="proposed",
            notes=str(notes),
        )
        payload["hypotheses"].append(asdict(h))
        self._save(payload)
        return h

    def list_hypotheses(self) -> List[Hypothesis]:
        payload = self._load()
        return [Hypothesis(**d) for d in payload.get("hypotheses", [])]

    def update_hypothesis_status(self, hypothesis_id: str, status: str, notes: str = "") -> None:
        payload = self._load()
        for h in payload.get("hypotheses", []):
            if h.get("hypothesis_id") == hypothesis_id:
                h["status"] = str(status)
                if notes:
                    h["notes"] = (h.get("notes","") + "\n" + notes).strip()
        self._save(payload)

    def add_note(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        payload = self._load()
        payload["notes"].append({
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "text": str(text),
            "meta": meta or {}
        })
        self._save(payload)
