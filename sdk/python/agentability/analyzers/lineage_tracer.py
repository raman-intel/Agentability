"""Information lineage tracking.

Tracks how information flows from sources through agents to decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


@dataclass
class InformationLineage:
    """Tracks lineage of information."""
    lineage_id: str
    source: str
    destination: str
    path: List[str]
    transformations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class LineageTracer:
    """Traces information lineage through agent system."""
    
    def __init__(self):
        self.lineages: List[InformationLineage] = []
        self.graph: Dict[str, Set[str]] = {}
    
    def record_lineage(
        self,
        source: str,
        destination: str,
        path: List[str],
        transformations: Optional[List[str]] = None
    ) -> InformationLineage:
        """Record information lineage."""
        lineage = InformationLineage(
            lineage_id=f"lineage_{len(self.lineages)}",
            source=source,
            destination=destination,
            path=path,
            transformations=transformations or []
        )
        
        self.lineages.append(lineage)
        
        # Update graph
        for i in range(len(path) - 1):
            if path[i] not in self.graph:
                self.graph[path[i]] = set()
            self.graph[path[i]].add(path[i + 1])
        
        return lineage
    
    def trace_back(self, destination: str) -> List[InformationLineage]:
        """Trace back to find all sources for a destination."""
        return [l for l in self.lineages if l.destination == destination]
