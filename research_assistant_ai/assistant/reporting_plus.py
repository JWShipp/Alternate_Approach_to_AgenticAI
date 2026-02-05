from __future__ import annotations
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

def plot_event_study(coef_by_k: Dict[int, float], p_by_k: Dict[int, float], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ks = sorted(coef_by_k.keys())
    vals = [coef_by_k[k] for k in ks]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axvline(x=0, linewidth=1)
    ax.plot(ks, vals, marker="o")
    ax.set_xlabel("Relative time k (months)")
    ax.set_ylabel("Estimated effect")
    ax.set_title("Event study coefficients (treated × relative time)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
