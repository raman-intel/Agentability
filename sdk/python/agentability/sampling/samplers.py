"""
Sampling system - cost controls at scale.

Copyright (c) 2026 Agentability
Licensed under MIT License
"""

from enum import Enum
from typing import Optional, Dict
import random

from agentability.models import Decision


class SamplingStrategy(Enum):
    """Sampling strategies."""
    ALWAYS = "always"
    NEVER = "never"
    HEAD_BASED = "head_based"
    TAIL_BASED = "tail_based"
    PROBABILISTIC = "probabilistic"
    IMPORTANCE_BASED = "importance_based"
    COST_AWARE = "cost_aware"
    ADAPTIVE = "adaptive"


class TraceSampler:
    """
    Intelligent trace sampling for cost control.
    
    Example:
        sampler = TraceSampler(
            strategy=SamplingStrategy.PROBABILISTIC,
            sample_rate=0.1  # 10% of traces
        )
        
        if sampler.should_sample_head(context):
            # Trace this decision
            pass
    """
    
    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.ALWAYS,
        sample_rate: float = 1.0,
        cost_budget_per_day: Optional[float] = None
    ):
        self.strategy = strategy
        self.sample_rate = sample_rate
        self.cost_budget_per_day = cost_budget_per_day
        self.daily_cost_spent = 0.0
    
    def should_sample_head(self, trace_context: Dict) -> bool:
        """
        Decide if trace should be sampled (head-based).
        
        Called at trace start, before decision is made.
        """
        if self.strategy == SamplingStrategy.ALWAYS:
            return True
        
        elif self.strategy == SamplingStrategy.NEVER:
            return False
        
        elif self.strategy in [SamplingStrategy.HEAD_BASED, 
                              SamplingStrategy.PROBABILISTIC]:
            return random.random() < self.sample_rate
        
        elif self.strategy == SamplingStrategy.COST_AWARE:
            if self.cost_budget_per_day:
                return self.daily_cost_spent < self.cost_budget_per_day
            return True
        
        elif self.strategy == SamplingStrategy.IMPORTANCE_BASED:
            importance = trace_context.get('importance', 0.5)
            return random.random() < importance
        
        else:
            # For TAIL_BASED and ADAPTIVE, sample everything initially
            return True
    
    def should_sample_tail(
        self,
        trace,
        decision: Decision
    ) -> bool:
        """
        Decide if trace should be sampled (tail-based).
        
        Called after decision is made, with full information.
        """
        if self.strategy != SamplingStrategy.TAIL_BASED:
            # For non-tail strategies, we already decided
            return True
        
        # Tail-based: ALWAYS sample important traces
        should_sample = (
            # Always sample errors
            decision.error is not None or
            
            # Always sample low confidence
            decision.confidence < 0.5 or
            
            # Always sample high cost
            (decision.llm_metrics and decision.llm_metrics.cost > 0.10) or
            
            # Always sample policy violations
            (decision.policy_violations and len(decision.policy_violations) > 0) or
            
            # Sample others probabilistically
            random.random() < self.sample_rate
        )
        
        return should_sample
    
    def record_cost(self, cost: float):
        """Record cost for budget tracking."""
        self.daily_cost_spent += cost
    
    def reset_daily_budget(self):
        """Reset daily cost counter (call at midnight)."""
        self.daily_cost_spent = 0.0


class ImportanceScorer:
    """
    Score trace importance for importance-based sampling.
    
    Higher importance = more likely to be sampled.
    """
    
    def score(self, trace_context: Dict) -> float:
        """
        Score importance (0-1).
        
        Factors:
        - User tier (premium users = higher importance)
        - Task criticality
        - Historical error rate
        """
        score = 0.5  # Base importance
        
        # User tier
        if trace_context.get('user_tier') == 'premium':
            score += 0.2
        
        # Critical tasks
        if trace_context.get('critical', False):
            score += 0.3
        
        # High error rate agents
        if trace_context.get('error_rate', 0) > 0.1:
            score += 0.2
        
        return min(1.0, score)
