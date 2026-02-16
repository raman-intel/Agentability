"""
Capability scoring system - measures agent capabilities across 8 dimensions.

Copyright (c) 2026 Agentability
Licensed under MIT License
"""

from typing import Dict, List
import numpy as np

from agentability.models import (
    CapabilityDimension,
    CapabilityScore,
    Decision,
    MemoryOperation,
    PolicyViolation,
)


class AgentabilityScorer:
    """
    Computes Agentability scores across all dimensions.
    
    Dimensions:
    - Reasoning: Decision quality, confidence calibration
    - Memory: Retrieval precision/recall, freshness
    - Tool Use: Selection accuracy, error recovery
    - Autonomy: Self-correction, planning horizon
    - Robustness: Error handling, edge cases
    - Safety: Policy compliance, harmful prevention
    - Efficiency: Cost per decision, latency
    - Collaboration: Communication, conflict resolution
    """
    
    def __init__(self):
        self.dimension_weights = {
            CapabilityDimension.REASONING: 0.20,
            CapabilityDimension.MEMORY: 0.15,
            CapabilityDimension.TOOL_USE: 0.15,
            CapabilityDimension.AUTONOMY: 0.15,
            CapabilityDimension.ROBUSTNESS: 0.10,
            CapabilityDimension.SAFETY: 0.10,
            CapabilityDimension.EFFICIENCY: 0.10,
            CapabilityDimension.COLLABORATION: 0.05,
        }
    
    def score_reasoning(self, decisions: List[Decision]) -> CapabilityScore:
        """Score reasoning capability."""
        if not decisions:
            return CapabilityScore(
                dimension=CapabilityDimension.REASONING,
                score=0.0,
                confidence=0.0,
                evidence_count=0
            )
        
        conf_scores = [d.confidence for d in decisions if d.confidence]
        avg_confidence = np.mean(conf_scores) if conf_scores else 0.5
        
        avg_steps = np.mean([len(d.reasoning_steps) for d in decisions])
        depth_score = min(avg_steps / 5.0, 1.0)
        
        with_uncertainties = sum(1 for d in decisions if d.uncertainties)
        uncertainty_score = with_uncertainties / len(decisions)
        
        raw_score = (
            0.30 * avg_confidence +
            0.30 * depth_score +
            0.40 * uncertainty_score
        )
        
        return CapabilityScore(
            dimension=CapabilityDimension.REASONING,
            score=raw_score * 100,
            confidence=self._calculate_confidence(len(decisions)),
            evidence_count=len(decisions)
        )
    
    def score_memory(self, memory_ops: List[MemoryOperation]) -> CapabilityScore:
        """Score memory subsystem capability."""
        if not memory_ops:
            return CapabilityScore(
                dimension=CapabilityDimension.MEMORY,
                score=0.0,
                confidence=0.0,
                evidence_count=0
            )
        
        precisions = [op.retrieval_precision for op in memory_ops if op.retrieval_precision]
        avg_precision = np.mean(precisions) if precisions else 0.7
        
        latencies = [op.latency_ms for op in memory_ops]
        avg_latency = np.mean(latencies)
        latency_score = max(0, 1.0 - (avg_latency / 1000))
        
        raw_score = 0.60 * avg_precision + 0.40 * latency_score
        
        return CapabilityScore(
            dimension=CapabilityDimension.MEMORY,
            score=raw_score * 100,
            confidence=self._calculate_confidence(len(memory_ops)),
            evidence_count=len(memory_ops)
        )
    
    def score_safety(self, decisions: List[Decision], 
                    policy_violations: List[PolicyViolation]) -> CapabilityScore:
        """Score safety capability."""
        if not decisions:
            return CapabilityScore(
                dimension=CapabilityDimension.SAFETY,
                score=100.0,
                confidence=0.0,
                evidence_count=0
            )
        
        decisions_with_violations = sum(1 for d in decisions if d.policy_violations)
        compliance_rate = 1.0 - (decisions_with_violations / len(decisions))
        
        critical_violations = sum(1 for v in policy_violations 
                                if v.severity.value == "critical")
        severity_penalty = min(critical_violations * 0.2, 0.5)
        
        raw_score = max(0, compliance_rate - severity_penalty)
        
        return CapabilityScore(
            dimension=CapabilityDimension.SAFETY,
            score=raw_score * 100,
            confidence=self._calculate_confidence(len(decisions)),
            evidence_count=len(decisions)
        )
    
    def score_efficiency(self, decisions: List[Decision]) -> CapabilityScore:
        """Score efficiency capability."""
        if not decisions:
            return CapabilityScore(
                dimension=CapabilityDimension.EFFICIENCY,
                score=0.0,
                confidence=0.0,
                evidence_count=0
            )
        
        costs = [d.llm_metrics.cost for d in decisions if d.llm_metrics]
        avg_cost = np.mean(costs) if costs else 0.01
        
        # Cost score: lower is better
        cost_score = max(0, 1.0 - (avg_cost / 0.10))  # $0.10 as threshold
        
        latencies = [d.latency_ms for d in decisions if d.latency_ms]
        avg_latency = np.mean(latencies) if latencies else 1000
        latency_score = max(0, 1.0 - (avg_latency / 5000))  # 5s threshold
        
        raw_score = 0.50 * cost_score + 0.50 * latency_score
        
        return CapabilityScore(
            dimension=CapabilityDimension.EFFICIENCY,
            score=raw_score * 100,
            confidence=self._calculate_confidence(len(decisions)),
            evidence_count=len(decisions)
        )
    
    def compute_composite_score(
        self,
        dimension_scores: Dict[CapabilityDimension, CapabilityScore]
    ) -> float:
        """Compute weighted composite Agentability score (0-100)."""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in self.dimension_weights.items():
            if dimension in dimension_scores:
                score_obj = dimension_scores[dimension]
                effective_weight = weight * score_obj.confidence
                total_score += score_obj.score * effective_weight
                total_weight += effective_weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def _calculate_confidence(self, evidence_count: int) -> float:
        """Calculate confidence based on evidence count."""
        return min(1.0, np.log1p(evidence_count) / np.log1p(100))
