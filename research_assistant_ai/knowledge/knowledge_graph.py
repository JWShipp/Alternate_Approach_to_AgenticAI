from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import networkx as nx

@dataclass
class KGEdge:
    src: str
    rel: str
    dst: str
    meta: Dict[str, Any]

class KnowledgeGraph:
    """A tiny, pragmatic knowledge graph for dissertation research artifacts.

    Nodes can be: dataset, variable, country, method, hypothesis, result.
    Edges represent relations: USES, PREDICTS, AFFECTS, TESTED_BY, SUPPORTS, REFUTES, DERIVED_FROM.
    """

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.g.add_node(str(node_id), **attrs)

    def add_edge(self, src: str, rel: str, dst: str, **meta: Any) -> None:
        self.g.add_edge(str(src), str(dst), key=str(rel), rel=str(rel), **meta)

    def to_json(self) -> Dict[str, Any]:
        nodes = [{"id": n, **self.g.nodes[n]} for n in self.g.nodes]
        edges = []
        for u, v, k, d in self.g.edges(keys=True, data=True):
            edges.append({"src": u, "dst": v, "rel": d.get("rel", k), "meta": {kk: vv for kk, vv in d.items() if kk != "rel"}})
        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "KnowledgeGraph":
        kg = cls()
        for n in payload.get("nodes", []):
            nid = n.pop("id")
            kg.add_node(nid, **n)
        for e in payload.get("edges", []):
            kg.add_edge(e["src"], e["rel"], e["dst"], **(e.get("meta") or {}))
        return kg

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        payload = json.loads(Path(path).read_text())
        return cls.from_json(payload)

    def summarize(self) -> Dict[str, Any]:
        return {
            "nodes": int(self.g.number_of_nodes()),
            "edges": int(self.g.number_of_edges()),
            "node_types": self._count_attr("type"),
            "relations": self._count_edge_rel(),
        }

    def _count_attr(self, attr: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for n in self.g.nodes:
            v = self.g.nodes[n].get(attr, "unknown")
            out[str(v)] = out.get(str(v), 0) + 1
        return out

    def _count_edge_rel(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _, _, d in self.g.edges(data=True):
            r = d.get("rel", "unknown")
            out[str(r)] = out.get(str(r), 0) + 1
        return out
