from __future__ import annotations
import numpy as np

def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum()

def entropy(p: np.ndarray) -> float:
    p = normalize(p)
    return float(-(p * np.log(p)).sum())

class DiscreteBelief:
    def __init__(self, p: np.ndarray):
        self.p = normalize(p)

    def update_bayes(self, likelihood: np.ndarray) -> None:
        self.p = normalize(self.p * np.asarray(likelihood, dtype=float))
