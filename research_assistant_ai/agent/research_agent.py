from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from ..registry.research_memory import ResearchMemory, Hypothesis
from ..registry.experiment_registry import ExperimentRegistry
from ..knowledge.knowledge_graph import KnowledgeGraph
from ..assistant.orchestrator_phase8 import run_phase8_rigor
from ..models.model_comparison import compare_did_models

@dataclass
class AgentOutputs:
    hypothesis: Dict[str, Any]
    model_comparison: List[Dict[str, Any]]
    phase8: Dict[str, Any]
    registry_run_id: str
    kg_path: str
    memory_path: str

class ResearchAgent:
    """A non-LLM research agent that manages hypotheses, runs tests, and records evidence.

    This is an MVP aimed at:
      - hypothesis tracking
      - test execution
      - updating 'status' based on evidence thresholds
      - building a knowledge graph of what's been tested
    """

    def __init__(self, *, workdir: Path) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.memory = ResearchMemory(self.workdir / "research_memory.json")
        self.registry = ExperimentRegistry(self.workdir / "experiment_registry")
        self.kg_path = self.workdir / "knowledge_graph.json"

        if self.kg_path.exists():
            self.kg = KnowledgeGraph.load(self.kg_path)
        else:
            self.kg = KnowledgeGraph()

    def propose_hypothesis(
        self,
        *,
        treated_country: str,
        outcome: str,
        treatment: str,
        expected_direction: str,
        statement: str,
        priors: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ) -> Hypothesis:
        h = self.memory.add_hypothesis(
            statement=statement,
            treated_country=treated_country,
            outcome=outcome,
            treatment=treatment,
            expected_direction=expected_direction,
            priors=priors or {},
            notes=notes,
        )
        self.kg.add_node(h.hypothesis_id, type="hypothesis", statement=statement, status=h.status)
        self.kg.add_edge(h.hypothesis_id, "ABOUT", treated_country, type="country")
        self.kg.add_edge(h.hypothesis_id, "OUTCOME", outcome, type="variable")
        self.kg.add_edge(h.hypothesis_id, "TREATMENT", treatment, type="variable")
        self._persist_kg()
        return h

    def test_hypothesis(
        self,
        *,
        panel: pd.DataFrame,
        hypothesis_id: str,
        intervention_month: str,
        covariate_candidates: Optional[List[List[str]]] = None,
        export_dir: Optional[Path] = None,
        alpha: float = 0.05,
    ) -> AgentOutputs:
        export_dir = Path(export_dir) if export_dir else (self.workdir / "agent_exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        hyps = {h.hypothesis_id: h for h in self.memory.list_hypotheses()}
        if hypothesis_id not in hyps:
            raise ValueError("Unknown hypothesis_id.")
        h = hyps[hypothesis_id]

        # Phase 8 rigor run
        p8 = run_phase8_rigor(
            panel,
            treated_country=h.treated_country,
            intervention_month=intervention_month,
            outcome_col=h.outcome,
            treatment_col=h.treatment,
            covariates=[],
            export_dir=export_dir,
        )

        # Model comparison across covariate sets (BIC weights)
        d = panel.copy()
        d["treated"] = (d["country_iso3"].astype(str) == str(h.treated_country)).astype(int)
        d["post"] = (d["month"].astype(str) >= str(intervention_month)).astype(int)

        if covariate_candidates is None:
            covariate_candidates = [[], [h.treatment], [h.treatment, "unrest_count"], [h.treatment, "unrest_intensity_sum"]]
            covariate_candidates = [c for c in covariate_candidates if all(col in d.columns for col in c)]

        scores = compare_did_models(
            d,
            unit_fe="country_iso3",
            time_fe="month",
            y_col=h.outcome,
            treated_col="treated",
            post_col="post",
            candidate_covariates=covariate_candidates,
            cluster_col="country_iso3",
        )
        scores_payload = [{"name": s.name, "bic": s.bic, "weight": s.weight, **s.meta} for s in scores]

        # Evidence rule (MVP): DiD p-value + direction consistency
        att = float(p8.did["att"])
        p = float(p8.did["p_value"])
        direction = "increase" if att > 0 else "decrease" if att < 0 else "none"

        status = "inconclusive"
        if p <= alpha:
            if h.expected_direction in ("increase", "decrease") and direction == h.expected_direction:
                status = "supported"
            elif h.expected_direction in ("increase", "decrease") and direction != h.expected_direction:
                status = "refuted"
            else:
                status = "tested"
        else:
            status = "inconclusive"

        self.memory.update_hypothesis_status(hypothesis_id, status, notes=f"Evidence: DiD ATT={att:.4f}, p={p:.4g}, direction={direction}.")
        self.kg.add_node(f"result:{hypothesis_id}", type="result", did_att=att, did_p=p, status=status, intervention_month=str(intervention_month))
        self.kg.add_edge(hypothesis_id, "TESTED_BY", "phase8_rigor", phase="8")
        self.kg.add_edge(hypothesis_id, "YIELDS", f"result:{hypothesis_id}", method="DiD")
        self._persist_kg()

        # Registry log
        rec = self.registry.log_run(
            phase="11",
            treated_country=h.treated_country,
            intervention_month=str(intervention_month),
            method="agent_hypothesis_test",
            inputs={"hypothesis_id": hypothesis_id, "covariate_candidates": covariate_candidates},
            outputs={"status": status, "did_att": att, "did_p": p, "model_comparison": scores_payload, "exports": p8.exports},
            tags=["phase11","agent","hypothesis_test"],
        )

        return AgentOutputs(
            hypothesis={"hypothesis_id": hypothesis_id, "statement": h.statement, "status": status, "expected_direction": h.expected_direction},
            model_comparison=scores_payload,
            phase8={"did": p8.did, "diagnostics": p8.diagnostics, "exports": p8.exports},
            registry_run_id=rec.run_id,
            kg_path=str(self.kg_path),
            memory_path=str(self.memory.path),
        )

    def _persist_kg(self) -> None:
        self.kg.save(self.kg_path)
