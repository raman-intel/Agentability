"""Episodic memory tracking for sequential experiences.

Tracks agent's sequential memory operations, temporal coherence,
and context window utilization.

Google Style Guide Compliant.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import statistics


@dataclass
class EpisodicRetrievalMetric:
    """Metrics for episodic memory retrieval.
    
    Attributes:
        operation_id: Unique identifier.
        agent_id: Agent ID.
        timestamp: Operation timestamp.
        latency_ms: Retrieval latency.
        time_range_start: Start of time range queried.
        time_range_end: End of time range queried.
        episodes_retrieved: Number of episodes retrieved.
        temporal_coherence: How well-ordered episodes are (0-1).
        context_tokens_used: Tokens used in context.
        context_tokens_limit: Maximum context tokens.
        context_utilization: Percentage of context used.
        metadata: Additional metadata.
    """
    operation_id: str
    agent_id: str
    timestamp: datetime
    latency_ms: float
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    episodes_retrieved: int = 0
    temporal_coherence: float = 1.0
    context_tokens_used: int = 0
    context_tokens_limit: int = 4096
    context_utilization: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemoryTracker:
    """Tracks episodic memory performance."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.operations: List[EpisodicRetrievalMetric] = []
    
    def track_retrieval(self) -> 'EpisodicRetrievalContext':
        """Track an episodic retrieval operation."""
        operation_id = f"{self.agent_id}_ep_{len(self.operations)}"
        return EpisodicRetrievalContext(self, operation_id)
    
    def record_operation(self, metric: EpisodicRetrievalMetric) -> None:
        """Record an operation."""
        self.operations.append(metric)
    
    def get_avg_context_utilization(self, time_window_hours: Optional[int] = None) -> float:
        """Get average context window utilization."""
        ops = self._filter_operations(time_window_hours)
        if not ops:
            return 0.0
        return statistics.mean(op.context_utilization for op in ops)
    
    def _filter_operations(self, time_window_hours: Optional[int]) -> List[EpisodicRetrievalMetric]:
        if not time_window_hours:
            return self.operations
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        return [op for op in self.operations if op.timestamp > cutoff]


class EpisodicRetrievalContext:
    """Context manager for episodic retrieval."""
    
    def __init__(self, tracker: EpisodicMemoryTracker, operation_id: str):
        self.tracker = tracker
        self.operation_id = operation_id
        self.start_time: Optional[float] = None
        self.episodes: List[Any] = []
        self.context_tokens_used: int = 0
        self.context_tokens_limit: int = 4096
    
    def __enter__(self) -> 'EpisodicRetrievalContext':
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time
        latency_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0.0
        
        metric = EpisodicRetrievalMetric(
            operation_id=self.operation_id,
            agent_id=self.tracker.agent_id,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            episodes_retrieved=len(self.episodes),
            context_tokens_used=self.context_tokens_used,
            context_tokens_limit=self.context_tokens_limit,
            context_utilization=self.context_tokens_used / self.context_tokens_limit if self.context_tokens_limit > 0 else 0.0
        )
        
        self.tracker.record_operation(metric)
    
    def record_episodes(self, episodes: List[Any], tokens_used: int) -> None:
        """Record retrieved episodes."""
        self.episodes = episodes
        self.context_tokens_used = tokens_used
