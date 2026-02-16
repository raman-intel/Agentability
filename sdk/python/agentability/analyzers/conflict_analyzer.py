"""Multi-agent conflict analyzer - understand WHY agents disagree.

Critical for multi-agent systems where agents have competing objectives.
Tracks conflicts, resolution patterns, and provides recommendations for
better agent coordination.

Copyright (c) 2026 Agentability
Licensed under MIT License
Google Python Style Guide Compliant
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import Counter


class ConflictResolutionMethod(Enum):
    """Methods for resolving agent conflicts."""
    PRIORITY_HIERARCHY = "priority_hierarchy"  # Higher priority agent wins
    CONSENSUS = "consensus"  # Vote/average across agents
    HUMAN_ESCALATION = "human_escalation"  # Defer to human
    CONFIDENCE_BASED = "confidence_based"  # Highest confidence wins
    CUSTOM_LOGIC = "custom_logic"  # Custom resolution function


@dataclass
class AgentConflict:
    """Record of a conflict between agents.
    
    Attributes:
        conflict_id: Unique identifier.
        timestamp: When conflict occurred.
        agents: Agent IDs involved in conflict.
        outputs: What each agent decided.
        confidences: Each agent's confidence.
        resolution_method: How it was resolved.
        winning_agent: Which agent's decision was used.
        final_decision: The resolved decision.
        context: Input data that caused disagreement.
    """
    conflict_id: str
    timestamp: datetime
    agents: List[str]
    outputs: Dict[str, Any]
    confidences: Dict[str, float]
    resolution_method: ConflictResolutionMethod
    winning_agent: str
    final_decision: Any
    context: Dict[str, Any] = field(default_factory=dict)


class ConflictAnalyzer:
    """Analyzes conflicts in multi-agent systems.
    
    When multiple agents disagree on a decision, this analyzer helps you understand:
    - Which agents conflict most often
    - What types of decisions cause conflicts
    - Whether certain agents are systematically ignored
    - How to optimize conflict resolution
    
    Key Capabilities:
        - Track all agent conflicts
        - Detect systematic biases in resolution
        - Recommend threshold/priority adjustments
        - Identify problematic agent pairs
        - Game-theoretic conflict analysis
    
    Example Usage:
        >>> analyzer = ConflictAnalyzer()
        >>> 
        >>> # Record a conflict
        >>> analyzer.record_conflict(
        ...     agents=["risk_agent", "sales_agent"],
        ...     outputs={
        ...         "risk_agent": {"decision": "deny", "score": 0.3},
        ...         "sales_agent": {"decision": "approve", "score": 0.8}
        ...     },
        ...     confidences={"risk_agent": 0.85, "sales_agent": 0.92},
        ...     resolution_method="priority_hierarchy",
        ...     winning_agent="risk_agent",  # Risk wins despite lower score
        ...     final_decision={"decision": "deny"}
        ... )
        >>> 
        >>> # Analyze patterns
        >>> patterns = analyzer.get_conflict_patterns(days=30)
        >>> print(f"Most conflicts: {patterns['most_common_pairs'][0]}")
        >>> # → "risk_agent vs sales_agent: 18 conflicts (risk wins 72%)"
    """
    
    def __init__(self):
        """Initialize the conflict analyzer."""
        self.conflicts: List[AgentConflict] = []
    
    def record_conflict(
        self,
        agents: List[str],
        outputs: Dict[str, Any],
        confidences: Dict[str, float],
        resolution_method: str,
        winning_agent: str,
        final_decision: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentConflict:
        """Record a conflict between agents.
        
        Args:
            agents: List of agent IDs involved.
            outputs: Dictionary mapping agent_id -> their decision.
            confidences: Dictionary mapping agent_id -> confidence.
            resolution_method: How the conflict was resolved.
            winning_agent: Which agent's decision was used.
            final_decision: The resolved decision.
            context: Input data/context that caused disagreement.
            
        Returns:
            The created AgentConflict record.
        """
        conflict = AgentConflict(
            conflict_id=f"conflict_{len(self.conflicts)}_{datetime.now().isoformat()}",
            timestamp=datetime.now(),
            agents=agents,
            outputs=outputs,
            confidences=confidences,
            resolution_method=ConflictResolutionMethod(resolution_method),
            winning_agent=winning_agent,
            final_decision=final_decision,
            context=context or {}
        )
        
        self.conflicts.append(conflict)
        return conflict
    
    def get_conflict_patterns(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze conflict patterns over time.
        
        This is THE KEY METHOD for understanding multi-agent dynamics.
        
        Args:
            days: Number of days to analyze.
            
        Returns:
            Dictionary containing:
            - total_conflicts: Number of conflicts
            - most_common_pairs: Agent pairs that conflict most
            - win_rates: How often each agent wins
            - resolution_methods: Distribution of resolution methods
            - conflict_rate: Conflicts per day
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_conflicts = [
            c for c in self.conflicts if c.timestamp >= cutoff
        ]
        
        if not recent_conflicts:
            return {"total_conflicts": 0}
        
        # Count agent pairs
        pair_counts: Counter = Counter()
        win_counts: Counter = Counter()
        resolution_counts: Counter = Counter()
        
        for conflict in recent_conflicts:
            # Create sorted tuple for consistent pairing
            if len(conflict.agents) == 2:
                pair = tuple(sorted(conflict.agents))
                pair_counts[pair] += 1
            
            win_counts[conflict.winning_agent] += 1
            resolution_counts[conflict.resolution_method.value] += 1
        
        # Most common pairs
        most_common_pairs = []
        for pair, count in pair_counts.most_common(10):
            # Calculate win rates for this pair
            pair_conflicts = [
                c for c in recent_conflicts
                if set(c.agents) == set(pair)
            ]
            
            agent1_wins = sum(1 for c in pair_conflicts if c.winning_agent == pair[0])
            agent2_wins = sum(1 for c in pair_conflicts if c.winning_agent == pair[1])
            
            most_common_pairs.append({
                "agents": list(pair),
                "count": count,
                "agent_wins": {
                    pair[0]: agent1_wins,
                    pair[1]: agent2_wins
                },
                "win_rates": {
                    pair[0]: agent1_wins / count if count > 0 else 0,
                    pair[1]: agent2_wins / count if count > 0 else 0
                }
            })
        
        # Overall win rates
        total_wins = sum(win_counts.values())
        win_rates = {
            agent: count / total_wins if total_wins > 0 else 0
            for agent, count in win_counts.items()
        }
        
        return {
            "total_conflicts": len(recent_conflicts),
            "conflict_rate_per_day": len(recent_conflicts) / days,
            "most_common_pairs": most_common_pairs,
            "win_rates": win_rates,
            "resolution_methods": dict(resolution_counts),
            "unique_agents": len(set(c.winning_agent for c in recent_conflicts))
        }
    
    def detect_systematic_bias(
        self,
        agent_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Detect if an agent is systematically ignored or favored.
        
        Args:
            agent_id: Agent to analyze.
            days: Days of history to check.
            
        Returns:
            Dictionary with bias analysis.
        """
        cutoff = datetime.now() - timedelta(days=days)
        relevant_conflicts = [
            c for c in self.conflicts
            if c.timestamp >= cutoff and agent_id in c.agents
        ]
        
        if not relevant_conflicts:
            return {"error": "No conflicts involving this agent"}
        
        # Calculate metrics
        total_conflicts = len(relevant_conflicts)
        wins = sum(1 for c in relevant_conflicts if c.winning_agent == agent_id)
        win_rate = wins / total_conflicts
        
        # Expected win rate (if fair) would be ~50% in 2-agent conflicts
        # or 1/n in n-agent conflicts
        avg_agents_per_conflict = sum(
            len(c.agents) for c in relevant_conflicts
        ) / total_conflicts
        
        expected_win_rate = 1.0 / avg_agents_per_conflict
        
        # Calculate bias score (-1 to +1, where 0 is fair)
        bias_score = (win_rate - expected_win_rate) / expected_win_rate
        
        # Determine bias type
        if bias_score < -0.3:
            bias_type = "systematically_ignored"
        elif bias_score > 0.3:
            bias_type = "systematically_favored"
        else:
            bias_type = "fair"
        
        return {
            "agent_id": agent_id,
            "total_conflicts": total_conflicts,
            "wins": wins,
            "win_rate": win_rate,
            "expected_win_rate": expected_win_rate,
            "bias_score": bias_score,
            "bias_type": bias_type,
            "recommendation": self._generate_bias_recommendation(
                agent_id, bias_type, win_rate
            )
        }
    
    def recommend_resolution_changes(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Recommend changes to conflict resolution strategy.
        
        Args:
            days: Days of history to analyze.
            
        Returns:
            List of recommendations with rationale.
        """
        patterns = self.get_conflict_patterns(days)
        recommendations = []
        
        if not patterns.get("most_common_pairs"):
            return recommendations
        
        # Analyze each common pair
        for pair_data in patterns["most_common_pairs"][:5]:
            agents = pair_data["agents"]
            win_rates = pair_data["win_rates"]
            
            # Check for severe imbalance
            max_win_rate = max(win_rates.values())
            min_win_rate = min(win_rates.values())
            
            if max_win_rate > 0.8:  # One agent wins 80%+ of the time
                dominant_agent = max(win_rates, key=win_rates.get)
                
                recommendations.append({
                    "type": "rebalance_priorities",
                    "agents": agents,
                    "issue": f"{dominant_agent} wins {max_win_rate:.0%} of conflicts",
                    "recommendation": (
                        f"Consider: (1) Increasing {agents[0] if agents[1] == dominant_agent else agents[1]} "
                        f"priority, (2) Using confidence-based resolution, "
                        f"(3) Adding a tie-breaker agent"
                    ),
                    "confidence_count": pair_data["count"]
                })
        
        # Check resolution method distribution
        methods = patterns.get("resolution_methods", {})
        if "human_escalation" in methods:
            escalation_rate = methods["human_escalation"] / sum(methods.values())
            if escalation_rate > 0.3:  # 30%+ escalations
                recommendations.append({
                    "type": "reduce_escalations",
                    "issue": f"{escalation_rate:.0%} of conflicts escalate to humans",
                    "recommendation": (
                        "High escalation rate suggests: (1) Agents lack clear priorities, "
                        "(2) Need better tie-breaking logic, (3) Insufficient training data"
                    )
                })
        
        return recommendations
    
    def get_conflict_by_decision_type(
        self,
        days: int = 30
    ) -> Dict[str, int]:
        """Group conflicts by decision type/category.
        
        Args:
            days: Days of history.
            
        Returns:
            Dictionary mapping decision types to conflict counts.
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_conflicts = [
            c for c in self.conflicts if c.timestamp >= cutoff
        ]
        
        type_counts: Counter = Counter()
        
        for conflict in recent_conflicts:
            # Try to extract decision type from context or outputs
            decision_type = conflict.context.get("decision_type", "unknown")
            type_counts[decision_type] += 1
        
        return dict(type_counts)
    
    def analyze_confidence_correlation(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze if higher confidence actually wins more often.
        
        Tests the hypothesis: "Does the most confident agent win?"
        
        Args:
            days: Days of history.
            
        Returns:
            Dictionary with correlation analysis.
        """
        cutoff = datetime.now() - timedelta(days=days)
        relevant_conflicts = [
            c for c in self.conflicts if c.timestamp >= cutoff
        ]
        
        if not relevant_conflicts:
            return {"error": "No conflicts to analyze"}
        
        # Count: how often does highest-confidence agent win?
        highest_conf_wins = 0
        total_analyzed = 0
        
        for conflict in relevant_conflicts:
            if not conflict.confidences:
                continue
            
            highest_conf_agent = max(
                conflict.confidences,
                key=conflict.confidences.get
            )
            
            if highest_conf_agent == conflict.winning_agent:
                highest_conf_wins += 1
            
            total_analyzed += 1
        
        if total_analyzed == 0:
            return {"error": "No confidence data available"}
        
        correlation_rate = highest_conf_wins / total_analyzed
        
        return {
            "total_conflicts": total_analyzed,
            "highest_confidence_wins": highest_conf_wins,
            "correlation_rate": correlation_rate,
            "interpretation": (
                "Strong correlation - confidence drives resolution"
                if correlation_rate > 0.7 else
                "Weak correlation - other factors dominate"
                if correlation_rate < 0.4 else
                "Moderate correlation - mixed resolution strategy"
            )
        }
    
    def _generate_bias_recommendation(
        self,
        agent_id: str,
        bias_type: str,
        win_rate: float
    ) -> str:
        """Generate recommendation for addressing bias.
        
        Args:
            agent_id: Agent being analyzed.
            bias_type: Type of bias detected.
            win_rate: Agent's win rate.
            
        Returns:
            Actionable recommendation.
        """
        if bias_type == "systematically_ignored":
            return (
                f"{agent_id} wins only {win_rate:.0%} of conflicts. "
                f"Consider: (1) Increasing priority, (2) Reviewing confidence "
                f"calibration, (3) Investigating if agent provides unique value"
            )
        elif bias_type == "systematically_favored":
            return (
                f"{agent_id} wins {win_rate:.0%} of conflicts. "
                f"Verify this is intentional. If too dominant, consider: "
                f"(1) Decreasing priority, (2) Adding checks/balances, "
                f"(3) Ensuring other agents' input is valued"
            )
        else:
            return f"{agent_id} has balanced conflict resolution ({win_rate:.0%} win rate)"


# Example usage for documentation
if __name__ == "__main__":
    analyzer = ConflictAnalyzer()
    
    # Simulate conflicts between risk and sales agents
    for i in range(20):
        analyzer.record_conflict(
            agents=["risk_agent", "sales_agent"],
            outputs={
                "risk_agent": {"decision": "deny"},
                "sales_agent": {"decision": "approve"}
            },
            confidences={"risk_agent": 0.85, "sales_agent": 0.90},
            resolution_method="priority_hierarchy",
            winning_agent="risk_agent",  # Risk always wins
            final_decision={"decision": "deny"}
        )
    
    # Analyze
    patterns = analyzer.get_conflict_patterns(days=30)
    print(f"Total conflicts: {patterns['total_conflicts']}")
    print(f"Most common pair: {patterns['most_common_pairs'][0]}")
    
    # Check for bias
    bias = analyzer.detect_systematic_bias("sales_agent")
    print(f"\nSales agent bias: {bias['bias_type']}")
    print(f"Win rate: {bias['win_rate']:.0%}")
    print(f"\nRecommendation: {bias['recommendation']}")
    
    # Get recommendations
    recommendations = analyzer.recommend_resolution_changes()
    for rec in recommendations:
        print(f"\n🎯 {rec['type']}: {rec['recommendation']}")
