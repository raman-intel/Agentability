"""Core tracer for instrumenting AI agents.

This is the main entry point for developers using Agentability.
It provides a simple API for tracking decisions, memory operations, and LLM calls.
"""

import contextlib
import time
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional
from uuid import UUID, uuid4

from agentability.models import (
    Decision,
    DecisionType,
    LLMMetrics,
    MemoryMetrics,
    MemoryOperation,
    MemoryType,
    AgentConflict,
    ConflictType,
)
from agentability.storage.sqlite_store import SQLiteStore
from agentability.utils.logger import get_logger


logger = get_logger(__name__)


class Tracer:
    """Main instrumentation class for tracking agent behavior.
    
    Usage:
        ```python
        from agentability import Tracer
        
        tracer = Tracer(offline_mode=True)
        
        with tracer.trace_decision(
            agent_id="my_agent",
            decision_type="classification"
        ):
            result = my_agent.predict(input_data)
            tracer.record_decision(
                output=result,
                confidence=0.92,
                reasoning=["Feature X > threshold", "Pattern Y detected"]
            )
        ```
    """

    def __init__(
        self,
        offline_mode: bool = True,
        storage_backend: str = "sqlite",
        database_path: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        auto_flush: bool = True,
        flush_interval_seconds: int = 10,
    ):
        """Initialize the tracer.
        
        Args:
            offline_mode: If True, uses SQLite. If False, streams to API.
            storage_backend: "sqlite", "duckdb", or "timescaledb"
            database_path: Path to local database file (for offline mode)
            api_endpoint: API endpoint URL (for cloud mode)
            api_key: API authentication key (for cloud mode)
            auto_flush: Whether to automatically flush data periodically
            flush_interval_seconds: How often to flush data (in seconds)
        """
        self.offline_mode = offline_mode
        self.storage_backend = storage_backend
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        
        # Initialize storage
        if offline_mode or storage_backend == "sqlite":
            self.store = SQLiteStore(database_path or "agentability.db")
        else:
            # TODO: Implement other storage backends
            raise NotImplementedError(
                f"Storage backend '{storage_backend}' not yet implemented"
            )
        
        # Context tracking
        self._current_decision: Optional[Dict[str, Any]] = None
        self._decision_start_time: Optional[float] = None
        
        logger.info(
            f"Tracer initialized: offline_mode={offline_mode}, "
            f"storage={storage_backend}"
        )

    @contextlib.contextmanager
    def trace_decision(
        self,
        agent_id: str,
        decision_type: DecisionType,
        session_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        parent_decision_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[UUID, None, None]:
        """Context manager for tracing a decision.
        
        Args:
            agent_id: Unique identifier for the agent making the decision
            decision_type: Type of decision being made
            session_id: Optional session/conversation identifier
            input_data: Input data for the decision
            parent_decision_id: Parent decision that triggered this one
            tags: Optional tags for categorization
            metadata: Additional metadata
            
        Yields:
            decision_id: UUID for this decision
            
        Example:
            ```python
            with tracer.trace_decision(
                agent_id="risk_agent",
                decision_type=DecisionType.CLASSIFICATION,
                input_data={"loan_amount": 50000}
            ) as decision_id:
                # Make decision
                result = agent.assess(input_data)
                
                # Record result
                tracer.record_decision(
                    output={"approved": True},
                    confidence=0.85
                )
            ```
        """
        decision_id = uuid4()
        self._decision_start_time = time.time()
        
        self._current_decision = {
            "decision_id": decision_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "decision_type": decision_type,
            "input_data": input_data or {},
            "parent_decision_id": parent_decision_id,
            "tags": tags or [],
            "metadata": metadata or {},
            "memory_operations": [],
            "llm_calls": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }
        
        try:
            yield decision_id
        finally:
            # Calculate latency
            if self._decision_start_time:
                latency_ms = (time.time() - self._decision_start_time) * 1000
                self._current_decision["latency_ms"] = latency_ms
            
            # If decision wasn't explicitly recorded, record with minimal info
            if "output_data" not in self._current_decision:
                logger.warning(
                    f"Decision {decision_id} completed without recording output"
                )
                self._current_decision["output_data"] = {}
            
            # Save decision
            decision = Decision(**self._current_decision)
            self.store.save_decision(decision)
            
            # Reset context
            self._current_decision = None
            self._decision_start_time = None

    def record_decision(
        self,
        output: Any,
        confidence: Optional[float] = None,
        reasoning: Optional[List[str]] = None,
        uncertainties: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        constraints_checked: Optional[List[str]] = None,
        constraints_violated: Optional[List[str]] = None,
        quality_score: Optional[float] = None,
        data_sources: Optional[List[str]] = None,
    ) -> None:
        """Record the output and provenance of the current decision.
        
        Must be called within a trace_decision() context.
        
        Args:
            output: The decision output
            confidence: Agent's confidence score (0-1)
            reasoning: List of reasoning steps
            uncertainties: What the agent was uncertain about
            assumptions: What the agent assumed
            constraints_checked: Constraints that were validated
            constraints_violated: Constraints that were violated
            quality_score: External quality assessment (0-1)
            data_sources: Data sources consulted
        """
        if not self._current_decision:
            raise RuntimeError(
                "record_decision() must be called within trace_decision() context"
            )
        
        self._current_decision.update({
            "output_data": output if isinstance(output, dict) else {"result": output},
            "confidence": confidence,
            "reasoning": reasoning or [],
            "uncertainties": uncertainties or [],
            "assumptions": assumptions or [],
            "constraints_checked": constraints_checked or [],
            "constraints_violated": constraints_violated or [],
            "quality_score": quality_score,
            "data_sources": data_sources or [],
        })

    def record_memory_operation(
        self,
        agent_id: str,
        memory_type: MemoryType,
        operation: MemoryOperation,
        latency_ms: float,
        items_processed: int,
        **kwargs: Any,
    ) -> UUID:
        """Record a memory operation.
        
        Args:
            agent_id: Agent performing the operation
            memory_type: Type of memory system
            operation: Type of operation
            latency_ms: Operation latency
            items_processed: Number of items processed
            **kwargs: Additional memory-specific metrics
            
        Returns:
            operation_id: UUID for this operation
        """
        metrics = MemoryMetrics(
            agent_id=agent_id,
            memory_type=memory_type,
            operation=operation,
            latency_ms=latency_ms,
            items_processed=items_processed,
            **kwargs,
        )
        
        operation_id = self.store.save_memory_metrics(metrics)
        
        # Link to current decision if in context
        if self._current_decision:
            self._current_decision["memory_operations"].append(operation_id)
        
        return operation_id

    def record_llm_call(
        self,
        agent_id: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost_usd: float,
        time_to_first_token_ms: Optional[float] = None,
        finish_reason: Optional[str] = None,
        is_streaming: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> UUID:
        """Record an LLM API call.
        
        Args:
            agent_id: Agent making the call
            provider: LLM provider (e.g., "openai", "anthropic")
            model: Model name
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            latency_ms: Total latency
            cost_usd: Cost in USD
            time_to_first_token_ms: Streaming latency
            finish_reason: Why generation stopped
            is_streaming: Whether response was streamed
            temperature: Sampling temperature
            max_tokens: Max tokens setting
            **kwargs: Additional metadata
            
        Returns:
            call_id: UUID for this LLM call
        """
        metrics = LLMMetrics(
            agent_id=agent_id,
            decision_id=self._current_decision["decision_id"] if self._current_decision else None,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            time_to_first_token_ms=time_to_first_token_ms,
            finish_reason=finish_reason,
            is_streaming=is_streaming,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=kwargs,
        )
        
        call_id = self.store.save_llm_metrics(metrics)
        
        # Update current decision
        if self._current_decision:
            self._current_decision["llm_calls"] += 1
            self._current_decision["total_tokens"] += metrics.total_tokens
            self._current_decision["total_cost_usd"] += cost_usd
        
        return call_id

    def record_conflict(
        self,
        session_id: str,
        conflict_type: ConflictType,
        involved_agents: List[str],
        agent_positions: Dict[str, Dict[str, Any]],
        severity: float,
        **kwargs: Any,
    ) -> UUID:
        """Record a multi-agent conflict.
        
        Args:
            session_id: Session identifier
            conflict_type: Type of conflict
            involved_agents: Agents involved in conflict
            agent_positions: Each agent's position
            severity: Conflict severity (0-1)
            **kwargs: Additional conflict data
            
        Returns:
            conflict_id: UUID for this conflict
        """
        conflict = AgentConflict(
            session_id=session_id,
            conflict_type=conflict_type,
            involved_agents=involved_agents,
            agent_positions=agent_positions,
            severity=severity,
            **kwargs,
        )
        
        return self.store.save_conflict(conflict)

    def get_decision(self, decision_id: UUID) -> Optional[Decision]:
        """Retrieve a decision by ID."""
        return self.store.get_decision(decision_id)

    def query_decisions(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        decision_type: Optional[DecisionType] = None,
        limit: int = 100,
    ) -> List[Decision]:
        """Query decisions with filters."""
        return self.store.query_decisions(
            agent_id=agent_id,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            decision_type=decision_type,
            limit=limit,
        )

    def close(self) -> None:
        """Close the tracer and flush any pending data."""
        if hasattr(self.store, "close"):
            self.store.close()
        logger.info("Tracer closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
