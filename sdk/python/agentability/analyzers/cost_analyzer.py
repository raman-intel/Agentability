"""LLM cost analysis and optimization.

Analyzes LLM usage costs and provides optimization recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import statistics


@dataclass
class CostOptimization:
    """Cost optimization recommendation."""
    optimization_type: str
    description: str
    estimated_savings_usd: float
    confidence: float


class CostAnalyzer:
    """Analyzes and optimizes LLM costs."""
    
    # Pricing per 1M tokens (approximate as of Feb 2026)
    MODEL_PRICING = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }
    
    def __init__(self):
        self.costs: List[Dict[str, Any]] = []
    
    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        timestamp: Optional[datetime] = None
    ) -> float:
        """Record an LLM call and return its cost."""
        pricing = self.MODEL_PRICING.get(model, {"input": 10.0, "output": 30.0})
        
        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        
        self.costs.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "timestamp": timestamp or datetime.now()
        })
        
        return cost
    
    def get_total_cost(self, time_window_hours: Optional[int] = None) -> float:
        """Calculate total cost."""
        costs = self._filter_costs(time_window_hours)
        return sum(c["cost_usd"] for c in costs)
    
    def get_cost_by_model(self, time_window_hours: Optional[int] = None) -> Dict[str, float]:
        """Get costs broken down by model."""
        costs = self._filter_costs(time_window_hours)
        by_model: Dict[str, float] = {}
        
        for cost in costs:
            model = cost["model"]
            if model not in by_model:
                by_model[model] = 0.0
            by_model[model] += cost["cost_usd"]
        
        return by_model
    
    def suggest_optimizations(self) -> List[CostOptimization]:
        """Suggest cost optimizations."""
        optimizations = []
        by_model = self.get_cost_by_model(time_window_hours=24)
        
        # Check if using expensive models unnecessarily
        if "gpt-4" in by_model and by_model["gpt-4"] > 10.0:
            savings = by_model["gpt-4"] * 0.8  # 80% savings with cheaper model
            optimizations.append(CostOptimization(
                optimization_type="model_downgrade",
                description="Consider using GPT-3.5-turbo for simple tasks instead of GPT-4",
                estimated_savings_usd=savings,
                confidence=0.7
            ))
        
        return optimizations
    
    def _filter_costs(self, time_window_hours: Optional[int]) -> List[Dict[str, Any]]:
        if not time_window_hours:
            return self.costs
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        return [c for c in self.costs if c["timestamp"] >= cutoff]
