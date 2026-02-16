"""Memory subsystem metrics collection.

Tracks performance of different memory types:
- Vector memory (RAG/embeddings)
- Episodic memory (sequential experiences)
- Semantic memory (knowledge graphs)
- Working memory (active context)
"""

import time
from typing import Any, List, Optional
from uuid import UUID

from agentability.models import MemoryMetrics, MemoryOperation, MemoryType
from agentability.utils.logger import get_logger


logger = get_logger(__name__)


class MemoryMetricsCollector:
    """Collector for memory operation metrics.
    
    Usage:
        ```python
        collector = MemoryMetricsCollector(agent_id="my_agent")
        
        # Track a retrieval operation
        tracker = collector.start_operation(
            memory_type=MemoryType.VECTOR,
            operation=MemoryOperation.RETRIEVE
        )
        
        # ... perform memory operation ...
        results = vector_db.search(query, top_k=10)
        
        # Record metrics
        metrics = tracker.complete(
            items_processed=len(results),
            avg_similarity=0.82,
            retrieval_precision=0.85
        )
        ```
    """

    def __init__(self, agent_id: str):
        """Initialize collector.
        
        Args:
            agent_id: Agent performing memory operations
        """
        self.agent_id = agent_id

    def start_operation(
        self,
        memory_type: MemoryType,
        operation: MemoryOperation,
    ) -> "MemoryOperationTracker":
        """Start tracking a memory operation.
        
        Args:
            memory_type: Type of memory system
            operation: Type of operation
            
        Returns:
            MemoryOperationTracker instance
        """
        return MemoryOperationTracker(
            agent_id=self.agent_id,
            memory_type=memory_type,
            operation=operation,
        )


class MemoryOperationTracker:
    """Tracks a single memory operation."""

    def __init__(
        self,
        agent_id: str,
        memory_type: MemoryType,
        operation: MemoryOperation,
    ):
        """Initialize tracker."""
        self.agent_id = agent_id
        self.memory_type = memory_type
        self.operation = operation
        self.start_time = time.time()

    def complete(
        self,
        items_processed: int,
        bytes_processed: Optional[int] = None,
        **kwargs: Any,
    ) -> MemoryMetrics:
        """Complete tracking and return metrics.
        
        Args:
            items_processed: Number of items processed
            bytes_processed: Bytes processed (optional)
            **kwargs: Memory-type-specific metrics
            
        Returns:
            MemoryMetrics object
        """
        end_time = time.time()
        latency_ms = (end_time - self.start_time) * 1000
        
        return MemoryMetrics(
            agent_id=self.agent_id,
            memory_type=self.memory_type,
            operation=self.operation,
            latency_ms=latency_ms,
            items_processed=items_processed,
            bytes_processed=bytes_processed,
            **kwargs,
        )


# Helper functions for specific memory types

def calculate_retrieval_precision(
    retrieved_items: List[Any],
    relevant_items: List[Any],
) -> float:
    """Calculate precision for retrieval operation.
    
    Precision = relevant_retrieved / total_retrieved
    
    Args:
        retrieved_items: Items retrieved from memory
        relevant_items: Items that are actually relevant
        
    Returns:
        Precision score (0-1)
    """
    if not retrieved_items:
        return 0.0
    
    relevant_retrieved = len(set(retrieved_items) & set(relevant_items))
    return relevant_retrieved / len(retrieved_items)


def calculate_retrieval_recall(
    retrieved_items: List[Any],
    relevant_items: List[Any],
) -> float:
    """Calculate recall for retrieval operation.
    
    Recall = relevant_retrieved / total_relevant
    
    Args:
        retrieved_items: Items retrieved from memory
        relevant_items: Items that are actually relevant
        
    Returns:
        Recall score (0-1)
    """
    if not relevant_items:
        return 0.0
    
    relevant_retrieved = len(set(retrieved_items) & set(relevant_items))
    return relevant_retrieved / len(relevant_items)


def calculate_similarity_stats(similarities: List[float]) -> dict:
    """Calculate statistics for similarity scores.
    
    Args:
        similarities: List of similarity scores
        
    Returns:
        Dict with avg, min, max similarities
    """
    if not similarities:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    
    return {
        "avg_similarity": sum(similarities) / len(similarities),
        "min_similarity": min(similarities),
        "max_similarity": max(similarities),
    }
