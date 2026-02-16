"""Decision provenance analyzer - answers "WHY did this happen?".

This module provides complete decision lineage and provenance tracking,
enabling root cause analysis and debugging of agent decisions.

Copyright (c) 2026 Agentability
Licensed under MIT License
Google Python Style Guide Compliant
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum


class ProvenanceType(Enum):
    """Types of provenance relationships."""
    INPUT_DATA = "input_data"              # Input data used
    REASONING_STEP = "reasoning_step"      # Reasoning process
    MEMORY_RETRIEVAL = "memory_retrieval"  # Memory accessed
    TOOL_CALL = "tool_call"                # External tool used
    CONSTRAINT = "constraint"              # Constraint checked
    DEPENDENCY = "dependency"              # Dependent decision
    ASSUMPTION = "assumption"              # Assumption made
    UNCERTAINTY = "uncertainty"            # Uncertainty identified


@dataclass
class ProvenanceRecord:
    """A single provenance record tracking influence on a decision.
    
    Attributes:
        record_id: Unique identifier.
        provenance_type: Type of provenance.
        source: Where this came from (data source, agent, tool, etc).
        content: The actual data/reasoning/information.
        confidence: Confidence in this information (0-1).
        timestamp: When this was created/accessed.
        impact: How much this influenced the decision (0-1).
        metadata: Additional context.
    """
    record_id: str
    provenance_type: ProvenanceType
    source: str
    content: Any
    confidence: float
    timestamp: datetime
    impact: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionProvenance:
    """Complete provenance for a single decision.
    
    Attributes:
        decision_id: Decision being analyzed.
        agent_id: Agent that made the decision.
        output: Final decision output.
        output_confidence: Final decision confidence.
        records: All provenance records.
        critical_points: Key decision points that had major impact.
        alternatives_considered: Other options that were rejected.
        bottleneck_records: Records that limited confidence.
    """
    decision_id: str
    agent_id: str
    output: Any
    output_confidence: float
    records: List[ProvenanceRecord] = field(default_factory=list)
    critical_points: List[Dict[str, Any]] = field(default_factory=list)
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    bottleneck_records: List[str] = field(default_factory=list)


class ProvenanceAnalyzer:
    """Analyzes decision provenance to answer "WHY did this happen?".
    
    This analyzer tracks the complete lineage of a decision including:
    - Input data sources
    - Reasoning steps taken
    - Memory accessed
    - Tools called
    - Constraints checked
    - Dependencies on other decisions
    - Assumptions made
    - Uncertainties encountered
    
    Key Capabilities:
        - Trace complete decision lineage
        - Identify critical decision points
        - Find confidence bottlenecks
        - Explain why alternatives were rejected
        - Generate human-readable explanations
    
    Example Usage:
        >>> analyzer = ProvenanceAnalyzer()
        >>> 
        >>> # Record provenance for a decision
        >>> prov = analyzer.create_provenance(
        ...     decision_id="dec_001",
        ...     agent_id="risk_agent",
        ...     output={"decision": "DENY"},
        ...     output_confidence=0.42
        ... )
        >>> 
        >>> # Add input data provenance
        >>> analyzer.add_record(
        ...     "dec_001",
        ...     provenance_type="input_data",
        ...     source="customer_db",
        ...     content={"income": 75000},
        ...     confidence=0.90,
        ...     impact=0.8
        ... )
        >>> 
        >>> # Add reasoning step
        >>> analyzer.add_record(
        ...     "dec_001",
        ...     provenance_type="reasoning_step",
        ...     source="risk_agent",
        ...     content="Income verification failed - data is 6 months old",
        ...     confidence=0.42,
        ...     impact=0.95
        ... )
        >>> 
        >>> # Analyze
        >>> explanation = analyzer.explain_decision("dec_001")
        >>> print(explanation["summary"])
        >>> # "Decision: DENY. Confidence reduced to 0.42 due to stale income data"
    """
    
    def __init__(self):
        """Initialize the provenance analyzer."""
        self.provenances: Dict[str, DecisionProvenance] = {}
    
    def create_provenance(
        self,
        decision_id: str,
        agent_id: str,
        output: Any,
        output_confidence: float
    ) -> DecisionProvenance:
        """Create a new provenance record for a decision.
        
        Args:
            decision_id: Unique decision identifier.
            agent_id: Agent that made the decision.
            output: Final decision output.
            output_confidence: Final decision confidence.
            
        Returns:
            The created DecisionProvenance object.
        """
        prov = DecisionProvenance(
            decision_id=decision_id,
            agent_id=agent_id,
            output=output,
            output_confidence=output_confidence
        )
        
        self.provenances[decision_id] = prov
        return prov
    
    def add_record(
        self,
        decision_id: str,
        provenance_type: str,
        source: str,
        content: Any,
        confidence: float,
        impact: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ProvenanceRecord]:
        """Add a provenance record to a decision.
        
        Args:
            decision_id: Decision to add record to.
            provenance_type: Type of provenance (input_data, reasoning_step, etc).
            source: Where this came from.
            content: The actual data/information.
            confidence: Confidence in this information (0-1).
            impact: How much this influenced the decision (0-1).
            metadata: Additional context.
            
        Returns:
            The created ProvenanceRecord, or None if decision doesn't exist.
        """
        if decision_id not in self.provenances:
            return None
        
        record = ProvenanceRecord(
            record_id=f"{decision_id}_rec_{len(self.provenances[decision_id].records)}",
            provenance_type=ProvenanceType(provenance_type),
            source=source,
            content=content,
            confidence=confidence,
            timestamp=datetime.now(),
            impact=impact,
            metadata=metadata or {}
        )
        
        self.provenances[decision_id].records.append(record)
        return record
    
    def get_provenance(self, decision_id: str) -> Optional[DecisionProvenance]:
        """Get complete provenance for a decision.
        
        Args:
            decision_id: Decision to retrieve.
            
        Returns:
            DecisionProvenance if exists, None otherwise.
        """
        return self.provenances.get(decision_id)
    
    def explain_decision(self, decision_id: str) -> Dict[str, Any]:
        """Generate human-readable explanation of a decision.
        
        This is the KEY function that answers "WHY did this happen?".
        
        Args:
            decision_id: Decision to explain.
            
        Returns:
            Dictionary containing:
            - summary: One-sentence explanation
            - critical_points: Key decision points
            - confidence_analysis: Why confidence is what it is
            - bottlenecks: What limited performance
            - alternatives: What was considered and rejected
            - timeline: Chronological provenance
        """
        prov = self.provenances.get(decision_id)
        if not prov:
            return {"error": "Decision not found"}
        
        # Analyze confidence bottlenecks
        bottlenecks = self._identify_bottlenecks(prov)
        
        # Identify critical decision points
        critical_points = self._identify_critical_points(prov)
        
        # Generate summary
        summary = self._generate_summary(prov, bottlenecks)
        
        # Timeline
        timeline = sorted(prov.records, key=lambda r: r.timestamp)
        
        return {
            "summary": summary,
            "decision": prov.output,
            "confidence": prov.output_confidence,
            "agent_id": prov.agent_id,
            "critical_points": critical_points,
            "bottlenecks": bottlenecks,
            "alternatives": prov.alternatives_considered,
            "timeline": [
                {
                    "type": rec.provenance_type.value,
                    "source": rec.source,
                    "content": str(rec.content)[:200],  # Truncate for display
                    "confidence": rec.confidence,
                    "impact": rec.impact,
                    "timestamp": rec.timestamp.isoformat()
                }
                for rec in timeline
            ]
        }
    
    def find_confidence_bottleneck(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Find what caused low confidence in a decision.
        
        Args:
            decision_id: Decision to analyze.
            
        Returns:
            Dictionary describing the main bottleneck, or None if confidence is high.
        """
        prov = self.provenances.get(decision_id)
        if not prov or prov.output_confidence >= 0.7:
            return None
        
        # Find record with lowest confidence that had high impact
        bottleneck = None
        min_score = 1.0
        
        for record in prov.records:
            if record.impact and record.impact >= 0.5:
                score = record.confidence
                if score < min_score:
                    min_score = score
                    bottleneck = record
        
        if not bottleneck:
            return None
        
        return {
            "type": bottleneck.provenance_type.value,
            "source": bottleneck.source,
            "content": bottleneck.content,
            "confidence": bottleneck.confidence,
            "impact": bottleneck.impact,
            "explanation": self._explain_bottleneck(bottleneck)
        }
    
    def trace_information_flow(
        self,
        decision_id: str,
        information_key: str
    ) -> List[Dict[str, Any]]:
        """Trace how a specific piece of information flowed through the decision.
        
        Args:
            decision_id: Decision to analyze.
            information_key: Key to trace (e.g., "income", "credit_score").
            
        Returns:
            List of provenance records that touched this information.
        """
        prov = self.provenances.get(decision_id)
        if not prov:
            return []
        
        flow = []
        for record in prov.records:
            # Check if this record contains the information
            if isinstance(record.content, dict) and information_key in record.content:
                flow.append({
                    "type": record.provenance_type.value,
                    "source": record.source,
                    "value": record.content[information_key],
                    "confidence": record.confidence,
                    "timestamp": record.timestamp.isoformat()
                })
            elif information_key.lower() in str(record.content).lower():
                flow.append({
                    "type": record.provenance_type.value,
                    "source": record.source,
                    "mention": str(record.content)[:200],
                    "confidence": record.confidence,
                    "timestamp": record.timestamp.isoformat()
                })
        
        return flow
    
    def compare_decisions(
        self,
        decision_id_1: str,
        decision_id_2: str
    ) -> Dict[str, Any]:
        """Compare provenance of two decisions.
        
        Useful for understanding why similar inputs led to different outputs.
        
        Args:
            decision_id_1: First decision.
            decision_id_2: Second decision.
            
        Returns:
            Dictionary with comparison analysis.
        """
        prov1 = self.provenances.get(decision_id_1)
        prov2 = self.provenances.get(decision_id_2)
        
        if not prov1 or not prov2:
            return {"error": "One or both decisions not found"}
        
        # Compare provenance types used
        types1 = set(r.provenance_type for r in prov1.records)
        types2 = set(r.provenance_type for r in prov2.records)
        
        # Compare sources
        sources1 = set(r.source for r in prov1.records)
        sources2 = set(r.source for r in prov2.records)
        
        return {
            "decision_1": {
                "id": decision_id_1,
                "output": prov1.output,
                "confidence": prov1.output_confidence
            },
            "decision_2": {
                "id": decision_id_2,
                "output": prov2.output,
                "confidence": prov2.output_confidence
            },
            "differences": {
                "provenance_types_only_in_1": list(types1 - types2),
                "provenance_types_only_in_2": list(types2 - types1),
                "sources_only_in_1": list(sources1 - sources2),
                "sources_only_in_2": list(sources2 - sources1),
                "confidence_delta": prov2.output_confidence - prov1.output_confidence
            }
        }
    
    def get_dependency_chain(self, decision_id: str) -> List[str]:
        """Get chain of decisions this decision depends on.
        
        Args:
            decision_id: Decision to analyze.
            
        Returns:
            List of decision IDs in dependency order.
        """
        prov = self.provenances.get(decision_id)
        if not prov:
            return []
        
        dependencies = []
        for record in prov.records:
            if record.provenance_type == ProvenanceType.DEPENDENCY:
                if isinstance(record.content, str):
                    dependencies.append(record.content)
                elif isinstance(record.content, dict) and "decision_id" in record.content:
                    dependencies.append(record.content["decision_id"])
        
        return dependencies
    
    def _identify_bottlenecks(self, prov: DecisionProvenance) -> List[Dict[str, Any]]:
        """Identify what limited decision confidence.
        
        Args:
            prov: Decision provenance.
            
        Returns:
            List of bottleneck records.
        """
        bottlenecks = []
        
        for record in prov.records:
            # Low confidence + high impact = bottleneck
            if record.confidence < 0.6 and record.impact and record.impact >= 0.5:
                bottlenecks.append({
                    "type": record.provenance_type.value,
                    "source": record.source,
                    "confidence": record.confidence,
                    "impact": record.impact,
                    "content": str(record.content)[:200]
                })
        
        # Sort by impact * (1 - confidence) to prioritize worst bottlenecks
        bottlenecks.sort(
            key=lambda b: b["impact"] * (1 - b["confidence"]),
            reverse=True
        )
        
        return bottlenecks
    
    def _identify_critical_points(self, prov: DecisionProvenance) -> List[Dict[str, Any]]:
        """Identify critical decision points that had major impact.
        
        Args:
            prov: Decision provenance.
            
        Returns:
            List of critical point records.
        """
        critical = []
        
        for record in prov.records:
            # High impact = critical
            if record.impact and record.impact >= 0.7:
                critical.append({
                    "type": record.provenance_type.value,
                    "source": record.source,
                    "confidence": record.confidence,
                    "impact": record.impact,
                    "content": str(record.content)[:200]
                })
        
        critical.sort(key=lambda c: c["impact"], reverse=True)
        return critical
    
    def _generate_summary(
        self,
        prov: DecisionProvenance,
        bottlenecks: List[Dict[str, Any]]
    ) -> str:
        """Generate one-sentence summary of decision.
        
        Args:
            prov: Decision provenance.
            bottlenecks: Identified bottlenecks.
            
        Returns:
            Human-readable summary string.
        """
        output_str = str(prov.output)[:50]
        
        if prov.output_confidence >= 0.8:
            return f"Decision: {output_str}. High confidence ({prov.output_confidence:.0%})"
        
        if bottlenecks:
            bottleneck = bottlenecks[0]
            return (
                f"Decision: {output_str}. "
                f"Confidence reduced to {prov.output_confidence:.0%} due to "
                f"{bottleneck['source']} ({bottleneck['confidence']:.0%} confidence)"
            )
        
        return f"Decision: {output_str}. Confidence: {prov.output_confidence:.0%}"
    
    def _explain_bottleneck(self, record: ProvenanceRecord) -> str:
        """Explain why a record is a bottleneck.
        
        Args:
            record: Provenance record.
            
        Returns:
            Human-readable explanation.
        """
        return (
            f"This {record.provenance_type.value} from {record.source} had "
            f"low confidence ({record.confidence:.0%}) but high impact "
            f"({record.impact:.0%}), limiting overall decision confidence."
        )


# Example usage for documentation
if __name__ == "__main__":
    # Create analyzer
    analyzer = ProvenanceAnalyzer()
    
    # Track a decision
    prov = analyzer.create_provenance(
        decision_id="dec_001",
        agent_id="risk_agent",
        output={"decision": "DENY"},
        output_confidence=0.42
    )
    
    # Add input data
    analyzer.add_record(
        "dec_001",
        provenance_type="input_data",
        source="customer_db",
        content={"income": 75000, "age_months": 6},
        confidence=0.50,
        impact=0.9
    )
    
    # Add reasoning
    analyzer.add_record(
        "dec_001",
        provenance_type="reasoning_step",
        source="risk_agent",
        content="Income verification failed - data is stale",
        confidence=0.42,
        impact=0.95
    )
    
    # Explain
    explanation = analyzer.explain_decision("dec_001")
    print(explanation["summary"])
    print(f"Bottlenecks: {len(explanation['bottlenecks'])}")
