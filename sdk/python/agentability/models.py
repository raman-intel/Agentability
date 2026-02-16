"""Core data models for Agentability.

This module defines all core data structures used throughout the platform.
Follows Google Python Style Guide and uses Pydantic for validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class DecisionType(str, Enum):
    """Types of agent decisions that can be tracked."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    PLANNING = "planning"
    EXECUTION = "execution"
    DELEGATION = "delegation"
    COORDINATION = "coordination"


class MemoryType(str, Enum):
    """Types of memory subsystems that can be tracked."""

    VECTOR = "vector"  # RAG/embeddings
    EPISODIC = "episodic"  # Sequential experiences
    SEMANTIC = "semantic"  # Knowledge graphs
    WORKING = "working"  # Active context
    PROCEDURAL = "procedural"  # Learned skills


class MemoryOperation(str, Enum):
    """Memory operations that can be tracked."""

    RETRIEVE = "retrieve"
    STORE = "store"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"


class ConflictType(str, Enum):
    """Types of multi-agent conflicts."""

    GOAL_CONFLICT = "goal_conflict"  # Incompatible objectives
    RESOURCE_CONFLICT = "resource_conflict"  # Shared resource contention
    BELIEF_CONFLICT = "belief_conflict"  # Different world models
    PRIORITY_CONFLICT = "priority_conflict"  # Different urgency
    STRATEGY_CONFLICT = "strategy_conflict"  # Different approaches


class Decision(BaseModel):
    """Represents a single agent decision with complete provenance."""

    # Identity
    decision_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(..., description="Unique identifier for the agent")
    session_id: Optional[str] = Field(None, description="Session/conversation ID")
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = Field(None, description="Decision latency")
    
    # Decision metadata
    decision_type: DecisionType
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Provenance - The "Why"
    reasoning: List[str] = Field(
        default_factory=list,
        description="Complete reasoning chain"
    )
    uncertainties: List[str] = Field(
        default_factory=list,
        description="What the agent was uncertain about"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="What the agent assumed"
    )
    constraints_checked: List[str] = Field(
        default_factory=list,
        description="Constraints validated"
    )
    constraints_violated: List[str] = Field(
        default_factory=list,
        description="Constraints violated"
    )
    
    # Quality metrics
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Agent's confidence score"
    )
    quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="External quality assessment"
    )
    
    # Relationships
    parent_decision_id: Optional[UUID] = Field(
        None,
        description="Parent decision that triggered this one"
    )
    child_decision_ids: List[UUID] = Field(
        default_factory=list,
        description="Decisions triggered by this one"
    )
    
    # Information lineage
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources consulted"
    )
    memory_operations: List[UUID] = Field(
        default_factory=list,
        description="Memory operations performed"
    )
    
    # LLM usage
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    
    # Tags and metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class MemoryMetrics(BaseModel):
    """Metrics for a memory operation."""

    # Identity
    operation_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    memory_type: MemoryType
    operation: MemoryOperation
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float
    
    # Volume
    items_processed: int = Field(ge=0)
    bytes_processed: Optional[int] = Field(None, ge=0)
    
    # Vector memory specific (RAG)
    vector_dimension: Optional[int] = Field(None, ge=0)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    avg_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    retrieval_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    retrieval_recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Episodic memory specific
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    episodes_retrieved: Optional[int] = Field(None, ge=0)
    temporal_coherence: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_tokens_used: Optional[int] = Field(None, ge=0)
    context_tokens_limit: Optional[int] = Field(None, ge=0)
    
    # Semantic memory specific (Knowledge graphs)
    knowledge_graph_nodes: Optional[int] = Field(None, ge=0)
    relationships_traversed: Optional[int] = Field(None, ge=0)
    max_hop_distance: Optional[int] = Field(None, ge=0)
    graph_density: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Freshness
    oldest_item_age_hours: Optional[float] = Field(None, ge=0.0)
    average_item_age_hours: Optional[float] = Field(None, ge=0.0)
    
    # Quality
    cache_hit_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class LLMMetrics(BaseModel):
    """Metrics for an LLM API call."""

    # Identity
    call_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    decision_id: Optional[UUID] = None
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float
    time_to_first_token_ms: Optional[float] = None
    
    # Model
    provider: str  # "openai", "anthropic", "google", etc.
    model: str
    
    # Token usage
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    
    # Cost
    cost_usd: float = Field(ge=0.0)
    
    # Quality
    finish_reason: Optional[str] = None
    
    # Streaming
    is_streaming: bool = Field(default=False)
    chunks_received: Optional[int] = Field(None, ge=0)
    
    # Rate limiting
    rate_limited: bool = Field(default=False)
    retry_count: int = Field(default=0, ge=0)
    
    # Metadata
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class AgentConflict(BaseModel):
    """Represents a conflict between multiple agents."""

    # Identity
    conflict_id: UUID = Field(default_factory=uuid4)
    session_id: str
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Conflict details
    conflict_type: ConflictType
    involved_agents: List[str] = Field(min_items=2)
    
    # Positions
    agent_positions: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Each agent's position/goal"
    )
    
    # Analysis
    severity: float = Field(ge=0.0, le=1.0, description="Conflict severity")
    resolution_strategy: Optional[str] = None
    resolution_outcome: Optional[str] = None
    
    # Game theory metrics
    nash_equilibrium: Optional[Dict[str, Any]] = None
    pareto_optimal: Optional[bool] = None
    
    # Resolution
    resolved: bool = Field(default=False)
    resolution_time_ms: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class AgentMetrics(BaseModel):
    """Aggregate metrics for an agent."""

    agent_id: str
    time_window_start: datetime
    time_window_end: datetime
    
    # Decision metrics
    total_decisions: int = Field(default=0, ge=0)
    avg_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    avg_latency_ms: Optional[float] = Field(None, ge=0.0)
    
    # Quality metrics
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    avg_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # LLM usage
    total_llm_calls: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    
    # Memory usage
    total_memory_operations: int = Field(default=0, ge=0)
    avg_memory_latency_ms: Optional[float] = Field(None, ge=0.0)
    
    # Conflicts
    conflicts_initiated: int = Field(default=0, ge=0)
    conflicts_involved: int = Field(default=0, ge=0)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class CausalRelationship(BaseModel):
    """Represents a causal relationship between decisions."""

    source_decision_id: UUID
    target_decision_id: UUID
    relationship_type: str  # "causes", "enables", "prevents", "influences"
    strength: float = Field(ge=0.0, le=1.0, description="Causal strength")
    time_delta_ms: float = Field(description="Time between decisions")
    
    # Evidence
    evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            UUID: lambda v: str(v),
        }
