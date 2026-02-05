from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Callable

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str

class Verifier:
    def __init__(self):
        self.checks: List[Callable[[Dict[str, Any]], CheckResult]] = []

    def add_check(self, fn: Callable[[Dict[str, Any]], CheckResult]) -> None:
        self.checks.append(fn)

    def run(self, artifact: Dict[str, Any]) -> List[CheckResult]:
        return [chk(artifact) for chk in self.checks]

def prob_bounds_check(field: str) -> Callable[[Dict[str, Any]], CheckResult]:
    def _chk(artifact: Dict[str, Any]) -> CheckResult:
        v = artifact.get(field)
        ok = (v is not None) and (0.0 <= float(v) <= 1.0)
        return CheckResult(
            name=f"prob_bounds({field})",
            passed=ok,
            message=f"{field}={v} is within [0,1]" if ok else f"{field}={v} is out of bounds or missing",
        )
    return _chk
