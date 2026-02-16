"""Multi-agent conflict tracking and analysis.

This module tracks conflicts, disagreements, and coordination issues between
multiple agents in a system. It uses game-theoretic concepts to analyze
agent interactions.

Google Style Guide Compliant.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import statistics
from collections import defaultdict


class ConflictType(Enum):
    """Types of conflicts that can occur between agents."""
    DECISION_DISAGREEMENT = "decision_disagreement"  # Agents reach different conclusions
    RESOURCE_CONTENTION = "resource_contention"  # Agents compete for same resource
    PRIORITY_CONFLICT = "priority_conflict"  # Agents have conflicting priorities
    CONSTRAINT_VIOLATION = "constraint_violation"  # Agent violates another's constraints
    TEMPORAL_CONFLICT = "temporal_conflict"  # Timing/sequencing conflicts
    DATA_INCONSISTENCY = "data_inconsistency"  # Agents have inconsistent data views


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    VOTING = "voting"  # Majority vote
    HIERARCHY = "hierarchy"  # Higher priority agent wins
    CONSENSUS = "consensus"  # All agents must agree
    ARBITRATION = "arbitration"  # External arbiter decides
    FIRST_COME = "first_come"  # First agent to decide wins
    CONFIDENCE_BASED = "confidence_based"  # Highest confidence wins


@dataclass
class AgentPosition:
    """Represents an agent's position in a conflict.
    
    Attributes:
        agent_id: ID of the agent.
        position: The agent's decision/stance.
        confidence: Confidence in the position (0-1).
        reasoning: List of reasoning steps.
        evidence: Supporting evidence.
        priority: Agent's priority level (higher = more important).
    """
    agent_id: str
    position: Any
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


@dataclass
class ConflictMetric:
    """Metrics for a single conflict between agents.
    
    Attributes:
        conflict_id: Unique identifier for the conflict.
        conflict_type: Type of conflict.
        timestamp: When the conflict occurred.
        agents_involved: IDs of agents in conflict.
        agent_positions: Each agent's position.
        resolution_strategy: How the conflict was resolved.
        resolution_time_ms: Time taken to resolve.
        consensus_reached: Whether consensus was achieved.
        final_decision: The final decision after resolution.
        conflict_severity: Severity score (0-1).
        metadata: Additional custom metadata.
    """
    conflict_id: str
    conflict_type: ConflictType
    timestamp: datetime
    agents_involved: Set[str]
    agent_positions: List[AgentPosition]
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolution_time_ms: Optional[float] = None
    consensus_reached: bool = False
    final_decision: Optional[Any] = None
    conflict_severity: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConflictMetricsCollector:
    """Collects and analyzes multi-agent conflict metrics.
    
    This collector tracks conflicts between agents, resolution patterns,
    and provides game-theoretic analysis of agent interactions.
    
    Example:
        >>> collector = ConflictMetricsCollector()
        >>> conflict_id = collector.record_conflict(
        ...     conflict_type="decision_disagreement",
        ...     agents=["agent_a", "agent_b"],
        ...     positions=[position_a, position_b]
        ... )
        >>> collector.resolve_conflict(
        ...     conflict_id=conflict_id,
        ...     strategy="voting",
        ...     final_decision="approve"
        ... )
    """
    
    def __init__(self):
        """Initialize the conflict metrics collector."""
        self.conflicts: List[ConflictMetric] = []
        self._agent_conflict_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        self._resolution_success: Dict[ResolutionStrategy, List[bool]] = defaultdict(list)
    
    def record_conflict(
        self,
        conflict_type: str,
        agents: List[str],
        positions: List[AgentPosition],
        conflict_id: Optional[str] = None
    ) -> str:
        """Record a new conflict between agents.
        
        Args:
            conflict_type: Type of conflict.
            agents: List of agent IDs involved.
            positions: Each agent's position.
            conflict_id: Optional conflict ID. Auto-generated if not provided.
            
        Returns:
            The conflict ID.
        """
        if conflict_id is None:
            conflict_id = f"conflict_{len(self.conflicts)}_{int(datetime.now().timestamp())}"
        
        # Calculate conflict severity based on disagreement
        severity = self._calculate_severity(positions)
        
        metric = ConflictMetric(
            conflict_id=conflict_id,
            conflict_type=ConflictType(conflict_type),
            timestamp=datetime.now(),
            agents_involved=set(agents),
            agent_positions=positions,
            conflict_severity=severity
        )
        
        self.conflicts.append(metric)
        
        # Update conflict matrix
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i+1:]:
                pair = tuple(sorted([agent_a, agent_b]))
                self._agent_conflict_matrix[pair] += 1
        
        return conflict_id
    
    def resolve_conflict(
        self,
        conflict_id: str,
        strategy: str,
        final_decision: Any,
        resolution_time_ms: float,
        consensus_reached: bool = False
    ) -> None:
        """Record the resolution of a conflict.
        
        Args:
            conflict_id: ID of the conflict being resolved.
            strategy: Resolution strategy used.
            final_decision: The final decision.
            resolution_time_ms: Time taken to resolve.
            consensus_reached: Whether consensus was achieved.
        """
        for conflict in self.conflicts:
            if conflict.conflict_id == conflict_id:
                conflict.resolution_strategy = ResolutionStrategy(strategy)
                conflict.final_decision = final_decision
                conflict.resolution_time_ms = resolution_time_ms
                conflict.consensus_reached = consensus_reached
                
                self._resolution_success[ResolutionStrategy(strategy)].append(
                    consensus_reached
                )
                break
    
    def get_conflict_rate(
        self,
        agent_id: Optional[str] = None,
        time_window_hours: Optional[int] = None
    ) -> float:
        """Calculate conflicts per hour.
        
        Args:
            agent_id: Filter by specific agent.
            time_window_hours: Time window for calculation.
            
        Returns:
            Conflicts per hour.
        """
        conflicts = self._filter_conflicts(agent_id, time_window_hours)
        
        if not conflicts:
            return 0.0
        
        if time_window_hours:
            return len(conflicts) / time_window_hours
        
        # Calculate based on actual time span
        if len(conflicts) < 2:
            return 0.0
        
        time_span_hours = (
            conflicts[-1].timestamp - conflicts[0].timestamp
        ).total_seconds() / 3600
        
        return len(conflicts) / max(time_span_hours, 1.0)
    
    def get_agent_conflict_matrix(self) -> Dict[Tuple[str, str], int]:
        """Get the conflict matrix showing which agents conflict most.
        
        Returns:
            Dictionary mapping agent pairs to conflict counts.
        """
        return dict(self._agent_conflict_matrix)
    
    def get_most_conflicting_pairs(self, top_n: int = 5) -> List[Tuple[Tuple[str, str], int]]:
        """Get the agent pairs with most conflicts.
        
        Args:
            top_n: Number of top pairs to return.
            
        Returns:
            List of ((agent_a, agent_b), count) tuples, sorted by count.
        """
        return sorted(
            self._agent_conflict_matrix.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
    
    def get_resolution_effectiveness(
        self,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze resolution strategy effectiveness.
        
        Args:
            strategy: Specific strategy to analyze, or all if None.
            
        Returns:
            Dictionary with effectiveness metrics.
        """
        if strategy:
            strategies = [ResolutionStrategy(strategy)]
        else:
            strategies = list(ResolutionStrategy)
        
        results = {}
        
        for strat in strategies:
            successes = self._resolution_success.get(strat, [])
            if successes:
                results[strat.value] = {
                    "total_uses": len(successes),
                    "consensus_rate": sum(successes) / len(successes),
                    "avg_resolution_time_ms": self._get_avg_resolution_time(strat)
                }
            else:
                results[strat.value] = {
                    "total_uses": 0,
                    "consensus_rate": 0.0,
                    "avg_resolution_time_ms": 0.0
                }
        
        return results
    
    def get_conflict_by_type(
        self,
        time_window_hours: Optional[int] = None
    ) -> Dict[str, int]:
        """Get conflict counts by type.
        
        Args:
            time_window_hours: Time window for counting.
            
        Returns:
            Dictionary mapping conflict types to counts.
        """
        conflicts = self._filter_conflicts(None, time_window_hours)
        
        counts = defaultdict(int)
        for conflict in conflicts:
            counts[conflict.conflict_type.value] += 1
        
        return dict(counts)
    
    def get_avg_severity(
        self,
        agent_id: Optional[str] = None,
        conflict_type: Optional[str] = None,
        time_window_hours: Optional[int] = None
    ) -> float:
        """Calculate average conflict severity.
        
        Args:
            agent_id: Filter by specific agent.
            conflict_type: Filter by conflict type.
            time_window_hours: Time window for calculation.
            
        Returns:
            Average severity score (0-1).
        """
        conflicts = self._filter_conflicts(agent_id, time_window_hours)
        
        if conflict_type:
            conflicts = [
                c for c in conflicts 
                if c.conflict_type == ConflictType(conflict_type)
            ]
        
        if not conflicts:
            return 0.0
        
        return statistics.mean(c.conflict_severity for c in conflicts)
    
    def analyze_agent_behavior(
        self,
        agent_id: str,
        time_window_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze a specific agent's conflict behavior.
        
        Args:
            agent_id: Agent to analyze.
            time_window_hours: Time window for analysis.
            
        Returns:
            Dictionary with behavioral analysis.
        """
        conflicts = self._filter_conflicts(agent_id, time_window_hours)
        
        if not conflicts:
            return {
                "total_conflicts": 0,
                "conflict_rate": 0.0,
                "avg_severity": 0.0,
                "most_common_conflict_type": None,
                "most_conflicting_agents": []
            }
        
        # Find conflicts by type
        type_counts = defaultdict(int)
        for conflict in conflicts:
            type_counts[conflict.conflict_type.value] += 1
        
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        
        # Find most conflicting partners
        partner_counts = defaultdict(int)
        for conflict in conflicts:
            for other_agent in conflict.agents_involved:
                if other_agent != agent_id:
                    partner_counts[other_agent] += 1
        
        most_conflicting = sorted(
            partner_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "total_conflicts": len(conflicts),
            "conflict_rate": self.get_conflict_rate(agent_id, time_window_hours),
            "avg_severity": statistics.mean(c.conflict_severity for c in conflicts),
            "most_common_conflict_type": most_common_type,
            "most_conflicting_agents": most_conflicting,
            "conflict_types_distribution": dict(type_counts)
        }
    
    def get_game_theoretic_analysis(
        self,
        conflict_id: str
    ) -> Dict[str, Any]:
        """Perform game-theoretic analysis of a conflict.
        
        Analyzes the conflict as a strategic game between agents.
        
        Args:
            conflict_id: ID of the conflict to analyze.
            
        Returns:
            Dictionary with game-theoretic insights.
        """
        conflict = None
        for c in self.conflicts:
            if c.conflict_id == conflict_id:
                conflict = c
                break
        
        if not conflict:
            return {}
        
        # Calculate Nash equilibrium likelihood
        positions = conflict.agent_positions
        
        # Check for dominant strategies
        dominant_strategy = None
        if positions:
            max_confidence_pos = max(positions, key=lambda p: p.confidence)
            if max_confidence_pos.confidence > 0.8:
                dominant_strategy = max_confidence_pos.agent_id
        
        # Calculate position diversity (disagreement level)
        if len(positions) > 1:
            confidences = [p.confidence for p in positions]
            diversity = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        else:
            diversity = 0.0
        
        # Identify potential Pareto improvements
        # (situations where one agent could be better off without making others worse)
        pareto_improvements = []
        for i, pos_a in enumerate(positions):
            for pos_b in positions[i+1:]:
                if pos_a.confidence < 0.5 and pos_b.confidence > 0.7:
                    pareto_improvements.append({
                        "agent": pos_a.agent_id,
                        "could_defer_to": pos_b.agent_id,
                        "potential_gain": pos_b.confidence - pos_a.confidence
                    })
        
        return {
            "agents_count": len(positions),
            "dominant_strategy_agent": dominant_strategy,
            "position_diversity": diversity,
            "consensus_likelihood": 1.0 - diversity,
            "pareto_improvements_possible": len(pareto_improvements),
            "pareto_improvements": pareto_improvements,
            "is_zero_sum": conflict.conflict_type in [
                ConflictType.RESOURCE_CONTENTION
            ],
            "recommended_resolution": self._recommend_resolution(conflict)
        }
    
    def _calculate_severity(self, positions: List[AgentPosition]) -> float:
        """Calculate conflict severity based on position disagreement.
        
        Args:
            positions: Agent positions in the conflict.
            
        Returns:
            Severity score between 0 and 1.
        """
        if len(positions) < 2:
            return 0.0
        
        # Higher confidence disagreement = higher severity
        confidences = [p.confidence for p in positions]
        
        # If all agents are confident in different positions, severity is high
        avg_confidence = statistics.mean(confidences)
        
        # High average confidence with disagreement = high severity
        return min(1.0, avg_confidence)
    
    def _get_avg_resolution_time(self, strategy: ResolutionStrategy) -> float:
        """Get average resolution time for a strategy.
        
        Args:
            strategy: The resolution strategy.
            
        Returns:
            Average resolution time in milliseconds.
        """
        times = [
            c.resolution_time_ms
            for c in self.conflicts
            if c.resolution_strategy == strategy and c.resolution_time_ms is not None
        ]
        
        return statistics.mean(times) if times else 0.0
    
    def _recommend_resolution(self, conflict: ConflictMetric) -> str:
        """Recommend a resolution strategy for a conflict.
        
        Args:
            conflict: The conflict to analyze.
            
        Returns:
            Recommended resolution strategy name.
        """
        # Simple heuristics for recommendation
        if conflict.conflict_type == ConflictType.RESOURCE_CONTENTION:
            return "hierarchy"  # Use priority levels
        
        positions = conflict.agent_positions
        if not positions:
            return "arbitration"
        
        # Check if there's a clear high-confidence position
        max_confidence = max(p.confidence for p in positions)
        if max_confidence > 0.9:
            return "confidence_based"
        
        # Check for high priority agent
        max_priority = max(p.priority for p in positions)
        if max_priority > sum(p.priority for p in positions) / len(positions) * 1.5:
            return "hierarchy"
        
        # Default to voting
        return "voting"
    
    def _filter_conflicts(
        self,
        agent_id: Optional[str],
        time_window_hours: Optional[int]
    ) -> List[ConflictMetric]:
        """Filter conflicts by agent and time window.
        
        Args:
            agent_id: Filter by specific agent.
            time_window_hours: Time window in hours.
            
        Returns:
            Filtered list of conflicts.
        """
        conflicts = self.conflicts
        
        if agent_id:
            conflicts = [
                c for c in conflicts
                if agent_id in c.agents_involved
            ]
        
        if time_window_hours:
            cutoff = datetime.now().timestamp() - (time_window_hours * 3600)
            conflicts = [
                c for c in conflicts
                if c.timestamp.timestamp() > cutoff
            ]
        
        return conflicts
