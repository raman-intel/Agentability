"""Analysis engines for agent behavior.

This package provides analyzers for:
- Decision provenance (why decisions were made)
- Causal graphs (causality chains)
- Conflict analysis (multi-agent disagreements)
- Drift detection (quality degradation)
- Lineage tracing (information flow)
- Cost analysis (LLM optimization)
"""

from .provenance import ProvenanceAnalyzer, DecisionProvenance
from .causal_graph import CausalGraphBuilder, CausalNode, CausalEdge
from .conflict_analyzer import ConflictAnalyzer
from .drift_detector import DriftDetector, DriftReport
from .lineage_tracer import LineageTracer, InformationLineage
from .cost_analyzer import CostAnalyzer, CostOptimization

__all__ = [
    "ProvenanceAnalyzer",
    "DecisionProvenance",
    "CausalGraphBuilder",
    "CausalNode",
    "CausalEdge",
    "ConflictAnalyzer",
    "DriftDetector",
    "DriftReport",
    "LineageTracer",
    "InformationLineage",
    "CostAnalyzer",
    "CostOptimization",
]
