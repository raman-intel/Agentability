"""Causal graph builder for temporal causality analysis.

This module builds temporal causal graphs showing how decisions and events
causally influence each other over time. Unlike simple traces, this captures
TRUE CAUSALITY RELATIONSHIPS.

Copyright (c) 2026 Agentability
Licensed under MIT License
Google Python Style Guide Compliant
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import json


class CausalRelationType(Enum):
    """Types of causal relationships between decisions."""
    DIRECT = "direct"              # A directly causes B
    INDIRECT = "indirect"          # A causes B through intermediaries
    CONTRIBUTORY = "contributory"  # A contributes to B
    PREVENTIVE = "preventive"      # A prevents B
    CORRELATION = "correlation"    # A and B correlate but unclear causation


@dataclass
class CausalNode:
    """Node in the causal graph representing a decision or event.
    
    Attributes:
        node_id: Unique identifier for this node.
        node_type: Type of node (decision, event, action, constraint).
        label: Human-readable label for display.
        timestamp: When this decision/event occurred.
        agent_id: Agent responsible for this decision.
        confidence: Decision confidence score (0-1).
        metadata: Additional context and data.
    """
    node_id: str
    node_type: str
    label: str
    timestamp: datetime
    agent_id: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    """Edge representing a causal relationship between two nodes.
    
    Attributes:
        edge_id: Unique identifier for this edge.
        source_id: Source node ID (cause).
        target_id: Target node ID (effect).
        relation_type: Type of causal relationship.
        strength: Strength of causation (0-1, where 1 = deterministic).
        time_delta_seconds: Time elapsed between cause and effect.
        confidence: Confidence in this causal link (0-1).
        evidence: Supporting evidence for this causal relationship.
        mechanism: Explanation of HOW source caused target.
    """
    edge_id: str
    source_id: str
    target_id: str
    relation_type: CausalRelationType
    strength: float
    time_delta_seconds: float
    confidence: float
    evidence: List[str] = field(default_factory=list)
    mechanism: Optional[str] = None


class CausalGraphBuilder:
    """Builds and analyzes temporal causal graphs for agent decisions.
    
    This is THE killer feature that differentiates Agentability from basic loggers.
    It answers "WHY did this decision happen?" not just "what happened?".
    
    Key Capabilities:
        - Build decision causality graphs
        - Find root causes of failures
        - Identify bottleneck decisions
        - Trace causal chains
        - Detect causal loops
        - Calculate causal impact
    
    Example Usage:
        >>> builder = CausalGraphBuilder()
        >>> 
        >>> # Add decisions as nodes
        >>> risk_node = builder.add_node(
        ...     "dec_001", "decision", "Risk Assessment",
        ...     agent_id="risk_agent", confidence=0.42
        ... )
        >>> 
        >>> approval_node = builder.add_node(
        ...     "dec_002", "decision", "Loan Approval",
        ...     agent_id="approval_agent", confidence=0.85
        ... )
        >>> 
        >>> # Link them causally
        >>> builder.add_causal_edge(
        ...     "dec_001", "dec_002", "direct", strength=0.9,
        ...     mechanism="Low risk confidence forced conservative approval"
        ... )
        >>> 
        >>> # Analyze
        >>> bottlenecks = builder.find_bottlenecks()
        >>> root_causes = builder.get_root_causes("dec_002")
    """
    
    def __init__(self):
        """Initialize the causal graph builder."""
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self._adjacency: Dict[str, List[str]] = {}
        self._reverse_adjacency: Dict[str, List[str]] = {}
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        timestamp: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CausalNode:
        """Add a node to the causal graph.
        
        Args:
            node_id: Unique identifier.
            node_type: Type (decision, event, action, constraint).
            label: Human-readable description.
            timestamp: When this occurred (defaults to now).
            agent_id: Agent responsible.
            confidence: Decision confidence (0-1).
            metadata: Additional context.
            
        Returns:
            The created CausalNode.
        """
        node = CausalNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            timestamp=timestamp or datetime.now(),
            agent_id=agent_id,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self._adjacency[node_id] = []
        self._reverse_adjacency[node_id] = []
        
        return node
    
    def add_causal_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        strength: float,
        confidence: float = 1.0,
        evidence: Optional[List[str]] = None,
        mechanism: Optional[str] = None
    ) -> Optional[CausalEdge]:
        """Add a causal edge between two nodes.
        
        Args:
            source_id: Source node ID (the cause).
            target_id: Target node ID (the effect).
            relation_type: Type of causal relationship.
            strength: Causation strength (0-1).
            confidence: Confidence in this link (0-1).
            evidence: Supporting evidence.
            mechanism: Explanation of causation mechanism.
            
        Returns:
            The created CausalEdge, or None if nodes don't exist.
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        source_node = self.nodes[source_id]
        target_node = self.nodes[target_id]
        
        # Calculate time delta
        time_delta = (target_node.timestamp - source_node.timestamp).total_seconds()
        
        edge = CausalEdge(
            edge_id=f"{source_id}_to_{target_id}",
            source_id=source_id,
            target_id=target_id,
            relation_type=CausalRelationType(relation_type),
            strength=strength,
            time_delta_seconds=time_delta,
            confidence=confidence,
            evidence=evidence or [],
            mechanism=mechanism
        )
        
        self.edges.append(edge)
        self._adjacency[source_id].append(target_id)
        self._reverse_adjacency[target_id].append(source_id)
        
        return edge
    
    def get_causal_chain(
        self,
        from_node_id: str,
        to_node_id: str,
        max_depth: int = 10
    ) -> List[List[str]]:
        """Find all causal paths from one node to another.
        
        Args:
            from_node_id: Starting node.
            to_node_id: Ending node.
            max_depth: Maximum path length to explore.
            
        Returns:
            List of paths, where each path is a list of node IDs.
        """
        paths: List[List[str]] = []
        
        def dfs(current: str, target: str, path: List[str], 
                visited: Set[str], depth: int) -> None:
            if depth > max_depth:
                return
            
            if current == target:
                paths.append(path.copy())
                return
            
            visited.add(current)
            
            for neighbor in self._adjacency.get(current, []):
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited, depth + 1)
                    path.pop()
            
            visited.remove(current)
        
        dfs(from_node_id, to_node_id, [from_node_id], set(), 0)
        return paths
    
    def get_root_causes(self, node_id: str) -> List[str]:
        """Get root causes for a node.
        
        Root causes are nodes with no incoming causal edges in the
        causal chain leading to this node.
        
        Args:
            node_id: Node to analyze.
            
        Returns:
            List of root cause node IDs.
        """
        root_causes = []
        visited = set()
        
        def find_roots(current: str) -> None:
            if current in visited:
                return
            
            visited.add(current)
            sources = self._reverse_adjacency.get(current, [])
            
            if not sources:
                root_causes.append(current)
            else:
                for source in sources:
                    find_roots(source)
        
        find_roots(node_id)
        return root_causes
    
    def get_downstream_effects(
        self,
        node_id: str,
        max_depth: int = 10
    ) -> List[str]:
        """Get all nodes causally influenced by this node.
        
        Args:
            node_id: Source node.
            max_depth: Maximum path depth to explore.
            
        Returns:
            List of affected node IDs.
        """
        affected = set()
        
        def dfs(current: str, depth: int) -> None:
            if depth >= max_depth:
                return
            
            for target in self._adjacency.get(current, []):
                if target not in affected:
                    affected.add(target)
                    dfs(target, depth + 1)
        
        dfs(node_id, 0)
        return list(affected)
    
    def find_bottlenecks(self) -> List[Dict[str, Any]]:
        """Find bottleneck decisions that limit system confidence.
        
        Bottlenecks are decisions with:
        - Low confidence (<0.5)
        - High downstream impact (many dependent decisions)
        - Critical position in causal graph
        
        Returns:
            List of bottleneck nodes with impact analysis.
        """
        bottlenecks = []
        
        for node_id, node in self.nodes.items():
            # Only consider decisions with confidence scores
            if node.confidence is None or node.confidence >= 0.5:
                continue
            
            # Calculate downstream impact
            affected = self.get_downstream_effects(node_id)
            
            if len(affected) >= 2:  # Affects multiple downstream decisions
                bottlenecks.append({
                    "node_id": node_id,
                    "label": node.label,
                    "confidence": node.confidence,
                    "affected_count": len(affected),
                    "affected_nodes": affected,
                    "agent_id": node.agent_id,
                    "impact": "high" if len(affected) >= 5 else "medium"
                })
        
        # Sort by impact (affected count * severity)
        bottlenecks.sort(
            key=lambda b: b["affected_count"] * (1 - b["confidence"]),
            reverse=True
        )
        
        return bottlenecks
    
    def analyze_causal_strength(
        self,
        from_node_id: str,
        to_node_id: str
    ) -> Dict[str, Any]:
        """Analyze the strength of causal relationship between two nodes.
        
        Args:
            from_node_id: Source node.
            to_node_id: Target node.
            
        Returns:
            Dictionary with causality analysis including:
            - has_causal_relationship: bool
            - paths_count: Number of causal paths
            - strongest_path_strength: Maximum strength across all paths
            - average_path_strength: Mean strength
            - paths: List of paths with strengths
        """
        paths = self.get_causal_chain(from_node_id, to_node_id)
        
        if not paths:
            return {
                "has_causal_relationship": False,
                "paths_count": 0,
                "strongest_path_strength": 0.0,
                "average_path_strength": 0.0
            }
        
        path_strengths = []
        path_details = []
        
        for path in paths:
            # Multiply strengths along path (chain rule for causation)
            path_strength = 1.0
            path_edges = []
            
            for i in range(len(path) - 1):
                edge = self._get_edge(path[i], path[i + 1])
                if edge:
                    path_strength *= edge.strength
                    path_edges.append({
                        "from": path[i],
                        "to": path[i + 1],
                        "strength": edge.strength,
                        "mechanism": edge.mechanism
                    })
            
            path_strengths.append(path_strength)
            path_details.append({
                "path": path,
                "strength": path_strength,
                "edges": path_edges
            })
        
        return {
            "has_causal_relationship": True,
            "paths_count": len(paths),
            "strongest_path_strength": max(path_strengths),
            "average_path_strength": sum(path_strengths) / len(path_strengths),
            "paths": path_details
        }
    
    def detect_causal_loops(self) -> List[List[str]]:
        """Detect circular causal relationships (feedback loops).
        
        Returns:
            List of loops, where each loop is a list of node IDs.
        """
        loops = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._adjacency.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a loop
                    loop_start = path.index(neighbor)
                    loops.append(path[loop_start:])
            
            path.pop()
            rec_stack.remove(node)
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id, [])
        
        return loops
    
    def get_temporal_analysis(self) -> Dict[str, Any]:
        """Analyze temporal patterns in causal relationships.
        
        Returns:
            Dictionary with temporal statistics:
            - total_edges: Number of causal edges
            - avg_time_delta_seconds: Mean time between cause and effect
            - min/max_time_delta_seconds: Time range
            - instant_causations: Count of immediate effects (<1s)
            - delayed_causations: Count of delayed effects (>60s)
        """
        time_deltas = [edge.time_delta_seconds for edge in self.edges]
        
        if not time_deltas:
            return {}
        
        return {
            "total_edges": len(time_deltas),
            "avg_time_delta_seconds": sum(time_deltas) / len(time_deltas),
            "min_time_delta_seconds": min(time_deltas),
            "max_time_delta_seconds": max(time_deltas),
            "instant_causations": sum(1 for td in time_deltas if td < 1.0),
            "delayed_causations": sum(1 for td in time_deltas if td >= 60.0),
            "median_time_delta_seconds": sorted(time_deltas)[len(time_deltas) // 2]
        }
    
    def build_graph(self) -> Dict[str, Any]:
        """Build the complete graph structure for visualization.
        
        Returns:
            Dictionary representation suitable for D3.js visualization:
            - nodes: List of node dictionaries
            - edges: List of edge dictionaries
            - metadata: Graph statistics
        """
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.node_type,
                    "label": node.label,
                    "timestamp": node.timestamp.isoformat(),
                    "agent_id": node.agent_id,
                    "confidence": node.confidence,
                    "metadata": node.metadata
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "id": edge.edge_id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.relation_type.value,
                    "strength": edge.strength,
                    "confidence": edge.confidence,
                    "time_delta": edge.time_delta_seconds,
                    "mechanism": edge.mechanism,
                    "evidence": edge.evidence
                }
                for edge in self.edges
            ],
            "metadata": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "temporal_stats": self.get_temporal_analysis()
            }
        }
    
    def export_to_json(self, filepath: str) -> None:
        """Export graph to JSON file for persistence.
        
        Args:
            filepath: Path to save JSON file.
        """
        graph = self.build_graph()
        with open(filepath, 'w') as f:
            json.dump(graph, f, indent=2, default=str)
    
    def _get_edge(self, source_id: str, target_id: str) -> Optional[CausalEdge]:
        """Get edge between two nodes.
        
        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            
        Returns:
            CausalEdge if exists, None otherwise.
        """
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None


# Example usage for documentation
if __name__ == "__main__":
    # Build a sample causal graph
    builder = CausalGraphBuilder()
    
    # Add nodes
    builder.add_node(
        "dec_001", "decision", "Risk Assessment",
        agent_id="risk_agent", confidence=0.42
    )
    builder.add_node(
        "dec_002", "decision", "Compliance Check",
        agent_id="compliance_agent", confidence=0.85
    )
    builder.add_node(
        "dec_003", "decision", "Final Approval",
        agent_id="approval_agent", confidence=0.42
    )
    
    # Add causal edges
    builder.add_causal_edge(
        "dec_001", "dec_003", "direct", strength=0.9,
        mechanism="Low risk confidence forced conservative approval"
    )
    builder.add_causal_edge(
        "dec_002", "dec_003", "contributory", strength=0.6,
        mechanism="Compliance approval influenced but didn't determine outcome"
    )
    
    # Analyze
    bottlenecks = builder.find_bottlenecks()
    print(f"Found {len(bottlenecks)} bottlenecks")
    
    # Export
    builder.export_to_json("causal_graph.json")
