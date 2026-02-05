from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np
from .belief import entropy, normalize

@dataclass
class ActionResult:
    action: str
    expected_risk: float
    expected_uncertainty: float
    expected_free_energy: float

class ActiveInferencePlanner:
    def __init__(self, actions: List[str], transition_fn: Callable[[str, np.ndarray], np.ndarray],
                 preference: np.ndarray, uncertainty_weight: float = 0.5):
        self.actions = actions
        self.transition_fn = transition_fn
        self.preference = normalize(preference)
        self.uncertainty_weight = float(uncertainty_weight)

    def score_action(self, action: str, belief_state: np.ndarray) -> ActionResult:
        pred = normalize(self.transition_fn(action, belief_state))
        expected_good = float(np.dot(pred, self.preference))
        risk = 1.0 - expected_good
        unc = entropy(pred)
        fe = risk + self.uncertainty_weight * unc
        return ActionResult(action=action, expected_risk=risk, expected_uncertainty=unc, expected_free_energy=fe)

    def choose(self, belief_state: np.ndarray) -> Tuple[str, List[ActionResult]]:
        scores = [self.score_action(a, belief_state) for a in self.actions]
        best = min(scores, key=lambda x: x.expected_free_energy)
        return best.action, scores
