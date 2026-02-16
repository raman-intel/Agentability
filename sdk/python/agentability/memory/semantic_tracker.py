"""Semantic memory tracking for knowledge graphs.

Tracks knowledge graph operations, relationship traversals,
and query complexity metrics.

Google Style Guide Compliant.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import statistics


@dataclass
class SemanticRetrievalMetric:
    """Metrics for semantic memory (knowledge graph) retrieval.
    
    Attributes:
        operation_id: Unique identifier.
        agent_id: Agent ID.
        timestamp: Operation timestamp.
        latency_ms: Query latency.
        knowledge_graph_nodes: Total nodes in graph.
        relationships_traversed: Number of relationships traversed.
        max_hop_distance: Maximum hops in query.
        graph_density: Graph density metric.
        query_complexity: Query complexity score.
        results_returned: Number of results.
        metadata: Additional metadata.
    """
    operation_id: str
    agent_id: str
    timestamp: datetime
    latency_ms: float
    knowledge_graph_nodes: int
    relationships_traversed: int
    max_hop_distance: int
    graph_density: float
    query_complexity: int
    results_returned: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticMemoryTracker:
    """Tracks semantic memory (knowledge graph) performance."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.operations: List[SemanticRetrievalMetric] = []
    
    def track_query(self) -> 'SemanticQueryContext':
        """Track a knowledge graph query."""
        operation_id = f"{self.agent_id}_sem_{len(self.operations)}"
        return SemanticQueryContext(self, operation_id)
    
    def record_operation(self, metric: SemanticRetrievalMetric) -> None:
        """Record an operation."""
        self.operations.append(metric)
    
    def get_avg_query_complexity(self, time_window_hours: Optional[int] = None) -> float:
        """Get average query complexity."""
        ops = self._filter_operations(time_window_hours)
        if not ops:
            return 0.0
        return statistics.mean(op.query_complexity for op in ops)
    
    def _filter_operations(self, time_window_hours: Optional[int]) -> List[SemanticRetrievalMetric]:
        if not time_window_hours:
            return self.operations
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        return [op for op in self.operations if op.timestamp > cutoff]


class SemanticQueryContext:
    """Context manager for semantic memory queries."""
    
    def __init__(self, tracker: SemanticMemoryTracker, operation_id: str):
        self.tracker = tracker
        self.operation_id = operation_id
        self.start_time: Optional[float] = None
        self.nodes: int = 0
        self.relationships: int = 0
        self.max_hops: int = 0
        self.density: float = 0.0
        self.complexity: int = 1
        self.results: List[Any] = []
    
    def __enter__(self) -> 'SemanticQueryContext':
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time
        latency_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0.0
        
        metric = SemanticRetrievalMetric(
            operation_id=self.operation_id,
            agent_id=self.tracker.agent_id,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            knowledge_graph_nodes=self.nodes,
            relationships_traversed=self.relationships,
            max_hop_distance=self.max_hops,
            graph_density=self.density,
            query_complexity=self.complexity,
            results_returned=len(self.results)
        )
        
        self.tracker.record_operation(metric)
    
    def record_query(
        self,
        nodes: int,
        relationships: int,
        max_hops: int,
        results: List[Any],
        complexity: int = 1
    ) -> None:
        """Record query details."""
        self.nodes = nodes
        self.relationships = relationships
        self.max_hops = max_hops
        self.complexity = complexity
        self.results = results
        if nodes > 0:
            self.density = relationships / (nodes * (nodes - 1)) if nodes > 1 else 0.0
