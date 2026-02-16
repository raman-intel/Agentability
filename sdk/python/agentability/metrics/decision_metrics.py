"""Decision quality metrics tracking.

This module tracks decision-making quality, confidence, and performance metrics
for AI agents. It provides insights into decision accuracy, latency, and patterns.

Google Style Guide Compliant.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import time
import statistics


class DecisionType(Enum):
    """Types of decisions agents can make."""
    CLASSIFICATION = "classification"
    RANKING = "ranking"
    GENERATION = "generation"
    EXTRACTION = "extraction"
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    ROUTING = "routing"
    VALIDATION = "validation"


@dataclass
class DecisionMetric:
    """Metrics for a single decision.
    
    Attributes:
        decision_id: Unique identifier for the decision.
        agent_id: ID of the agent making the decision.
        decision_type: Type of decision made.
        timestamp: When the decision was made.
        latency_ms: Time taken to make the decision.
        confidence: Agent's confidence in the decision (0-1).
        success: Whether the decision was successful (if known).
        reasoning_steps: Number of reasoning steps taken.
        tool_calls: Number of tool calls made during decision.
        tokens_used: Total tokens consumed.
        cost_usd: Estimated cost in USD.
        metadata: Additional custom metadata.
    """
    decision_id: str
    agent_id: str
    decision_type: DecisionType
    timestamp: datetime
    latency_ms: float
    confidence: float
    success: Optional[bool] = None
    reasoning_steps: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionMetricsCollector:
    """Collects and aggregates decision metrics for agents.
    
    This collector tracks decision quality, patterns, and performance over time.
    It can identify drift, anomalies, and optimization opportunities.
    
    Example:
        >>> collector = DecisionMetricsCollector(agent_id="risk_agent")
        >>> with collector.track_decision("classification") as ctx:
        ...     result = agent.decide(input_data)
        ...     ctx.set_confidence(0.85)
        ...     ctx.set_success(True)
    """
    
    def __init__(self, agent_id: str):
        """Initialize the decision metrics collector.
        
        Args:
            agent_id: Unique identifier for the agent being tracked.
        """
        self.agent_id = agent_id
        self.decisions: List[DecisionMetric] = []
        self._active_decisions: Dict[str, Dict[str, Any]] = {}
        
    def track_decision(
        self,
        decision_type: str,
        decision_id: Optional[str] = None
    ) -> 'DecisionContext':
        """Context manager for tracking a decision.
        
        Args:
            decision_type: Type of decision being made.
            decision_id: Optional ID for the decision. Auto-generated if not provided.
            
        Returns:
            A DecisionContext that can be used to record decision details.
            
        Example:
            >>> with collector.track_decision("classification") as ctx:
            ...     result = make_decision()
            ...     ctx.set_confidence(0.9)
        """
        if decision_id is None:
            decision_id = f"{self.agent_id}_{int(time.time() * 1000)}"
            
        return DecisionContext(
            collector=self,
            decision_id=decision_id,
            decision_type=DecisionType(decision_type)
        )
    
    def record_decision(self, metric: DecisionMetric) -> None:
        """Record a completed decision metric.
        
        Args:
            metric: The decision metric to record.
        """
        self.decisions.append(metric)
    
    def get_success_rate(
        self,
        decision_type: Optional[DecisionType] = None,
        time_window_hours: Optional[int] = None
    ) -> float:
        """Calculate success rate for decisions.
        
        Args:
            decision_type: Filter by decision type.
            time_window_hours: Only consider decisions from last N hours.
            
        Returns:
            Success rate as a float between 0 and 1.
        """
        decisions = self._filter_decisions(decision_type, time_window_hours)
        decisions_with_outcome = [d for d in decisions if d.success is not None]
        
        if not decisions_with_outcome:
            return 0.0
            
        successful = sum(1 for d in decisions_with_outcome if d.success)
        return successful / len(decisions_with_outcome)
    
    def get_avg_confidence(
        self,
        decision_type: Optional[DecisionType] = None,
        time_window_hours: Optional[int] = None
    ) -> float:
        """Calculate average confidence for decisions.
        
        Args:
            decision_type: Filter by decision type.
            time_window_hours: Only consider decisions from last N hours.
            
        Returns:
            Average confidence score.
        """
        decisions = self._filter_decisions(decision_type, time_window_hours)
        
        if not decisions:
            return 0.0
            
        return statistics.mean(d.confidence for d in decisions)
    
    def get_latency_percentiles(
        self,
        decision_type: Optional[DecisionType] = None,
        time_window_hours: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate latency percentiles.
        
        Args:
            decision_type: Filter by decision type.
            time_window_hours: Only consider decisions from last N hours.
            
        Returns:
            Dictionary with p50, p95, p99 latencies in milliseconds.
        """
        decisions = self._filter_decisions(decision_type, time_window_hours)
        
        if not decisions:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            
        latencies = sorted(d.latency_ms for d in decisions)
        n = len(latencies)
        
        return {
            "p50": latencies[int(n * 0.50)],
            "p95": latencies[int(n * 0.95)] if n > 20 else latencies[-1],
            "p99": latencies[int(n * 0.99)] if n > 100 else latencies[-1],
        }
    
    def detect_drift(
        self,
        decision_type: Optional[DecisionType] = None,
        baseline_hours: int = 24,
        current_hours: int = 1,
        threshold: float = 0.15
    ) -> Dict[str, Any]:
        """Detect performance drift in decision quality.
        
        Compares recent performance to baseline to identify degradation.
        
        Args:
            decision_type: Filter by decision type.
            baseline_hours: Hours to use for baseline calculation.
            current_hours: Hours to use for current performance.
            threshold: Minimum change to trigger drift detection (0-1).
            
        Returns:
            Dictionary with drift analysis including:
                - drift_detected: Boolean
                - baseline_success_rate: Float
                - current_success_rate: Float
                - change: Float (negative means degradation)
        """
        baseline_rate = self.get_success_rate(
            decision_type=decision_type,
            time_window_hours=baseline_hours
        )
        
        current_rate = self.get_success_rate(
            decision_type=decision_type,
            time_window_hours=current_hours
        )
        
        change = current_rate - baseline_rate
        drift_detected = abs(change) > threshold
        
        return {
            "drift_detected": drift_detected,
            "baseline_success_rate": baseline_rate,
            "current_success_rate": current_rate,
            "change": change,
            "degradation": change < 0,
            "improvement": change > 0
        }
    
    def get_cost_analysis(
        self,
        decision_type: Optional[DecisionType] = None,
        time_window_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze costs for decisions.
        
        Args:
            decision_type: Filter by decision type.
            time_window_hours: Only consider decisions from last N hours.
            
        Returns:
            Dictionary with cost analysis metrics.
        """
        decisions = self._filter_decisions(decision_type, time_window_hours)
        
        if not decisions:
            return {
                "total_cost_usd": 0.0,
                "avg_cost_per_decision": 0.0,
                "total_tokens": 0,
                "avg_tokens_per_decision": 0.0
            }
        
        total_cost = sum(d.cost_usd for d in decisions)
        total_tokens = sum(d.tokens_used for d in decisions)
        
        return {
            "total_cost_usd": total_cost,
            "avg_cost_per_decision": total_cost / len(decisions),
            "total_tokens": total_tokens,
            "avg_tokens_per_decision": total_tokens / len(decisions),
            "decisions_count": len(decisions)
        }
    
    def _filter_decisions(
        self,
        decision_type: Optional[DecisionType],
        time_window_hours: Optional[int]
    ) -> List[DecisionMetric]:
        """Filter decisions by type and time window.
        
        Args:
            decision_type: Filter by decision type.
            time_window_hours: Only include decisions from last N hours.
            
        Returns:
            Filtered list of decision metrics.
        """
        decisions = self.decisions
        
        if decision_type:
            decisions = [d for d in decisions if d.decision_type == decision_type]
        
        if time_window_hours:
            cutoff = datetime.now().timestamp() - (time_window_hours * 3600)
            decisions = [
                d for d in decisions 
                if d.timestamp.timestamp() > cutoff
            ]
        
        return decisions


class DecisionContext:
    """Context manager for tracking a single decision.
    
    This class provides a convenient interface for recording decision metrics
    within a context manager.
    """
    
    def __init__(
        self,
        collector: DecisionMetricsCollector,
        decision_id: str,
        decision_type: DecisionType
    ):
        """Initialize the decision context.
        
        Args:
            collector: The parent collector.
            decision_id: Unique ID for this decision.
            decision_type: Type of decision being made.
        """
        self.collector = collector
        self.decision_id = decision_id
        self.decision_type = decision_type
        self.start_time: Optional[float] = None
        self.confidence: float = 0.0
        self.success: Optional[bool] = None
        self.reasoning_steps: int = 0
        self.tool_calls: int = 0
        self.tokens_used: int = 0
        self.cost_usd: float = 0.0
        self.metadata: Dict[str, Any] = {}
    
    def __enter__(self) -> 'DecisionContext':
        """Start tracking the decision."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Record the decision metric on exit."""
        latency_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0.0
        
        metric = DecisionMetric(
            decision_id=self.decision_id,
            agent_id=self.collector.agent_id,
            decision_type=self.decision_type,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            confidence=self.confidence,
            success=self.success,
            reasoning_steps=self.reasoning_steps,
            tool_calls=self.tool_calls,
            tokens_used=self.tokens_used,
            cost_usd=self.cost_usd,
            metadata=self.metadata
        )
        
        self.collector.record_decision(metric)
    
    def set_confidence(self, confidence: float) -> None:
        """Set the confidence score for the decision.
        
        Args:
            confidence: Confidence score between 0 and 1.
        """
        self.confidence = max(0.0, min(1.0, confidence))
    
    def set_success(self, success: bool) -> None:
        """Set whether the decision was successful.
        
        Args:
            success: True if decision was successful.
        """
        self.success = success
    
    def add_reasoning_step(self) -> None:
        """Increment the reasoning steps counter."""
        self.reasoning_steps += 1
    
    def add_tool_call(self) -> None:
        """Increment the tool calls counter."""
        self.tool_calls += 1
    
    def add_tokens(self, count: int) -> None:
        """Add to the token count.
        
        Args:
            count: Number of tokens to add.
        """
        self.tokens_used += count
    
    def add_cost(self, cost: float) -> None:
        """Add to the cost.
        
        Args:
            cost: Cost in USD to add.
        """
        self.cost_usd += cost
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata key-value pair.
        
        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value
