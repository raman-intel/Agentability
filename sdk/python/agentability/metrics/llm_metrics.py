"""LLM metrics collection and cost tracking.

This module provides advanced LLM performance tracking including:
- Token usage and cost calculation
- Latency tracking (total and time-to-first-token)
- Rate limiting detection
- Model-specific pricing
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from agentability.models import LLMMetrics
from agentability.utils.logger import get_logger


logger = get_logger(__name__)


# Pricing per 1M tokens (as of Feb 2026)
LLM_PRICING = {
    # OpenAI
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    
    # Anthropic
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-haiku-4": {"input": 0.25, "output": 1.25},
    
    # Google
    "gemini-pro": {"input": 0.5, "output": 1.5},
    "gemini-ultra": {"input": 10.0, "output": 30.0},
    
    # Default fallback
    "default": {"input": 1.0, "output": 2.0},
}


class LLMMetricsCollector:
    """Collector for LLM API metrics.
    
    Usage:
        ```python
        collector = LLMMetricsCollector(agent_id="my_agent")
        
        # Start tracking a call
        tracker = collector.start_call(
            provider="anthropic",
            model="claude-sonnet-4"
        )
        
        # ... make LLM API call ...
        
        # Record completion
        metrics = tracker.complete(
            prompt_tokens=1500,
            completion_tokens=800,
            finish_reason="stop"
        )
        ```
    """

    def __init__(self, agent_id: str, decision_id: Optional[UUID] = None):
        """Initialize collector.
        
        Args:
            agent_id: Agent making LLM calls
            decision_id: Optional decision ID to link calls to
        """
        self.agent_id = agent_id
        self.decision_id = decision_id

    def start_call(
        self,
        provider: str,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        is_streaming: bool = False,
    ) -> "LLMCallTracker":
        """Start tracking an LLM call.
        
        Args:
            provider: LLM provider (e.g., "openai", "anthropic")
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens setting
            is_streaming: Whether response is streamed
            
        Returns:
            LLMCallTracker instance
        """
        return LLMCallTracker(
            agent_id=self.agent_id,
            decision_id=self.decision_id,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            is_streaming=is_streaming,
        )

    @staticmethod
    def calculate_cost(
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost for an LLM call.
        
        Args:
            model: Model name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            
        Returns:
            Cost in USD
        """
        # Normalize model name
        model_key = model.lower()
        for key in LLM_PRICING:
            if key in model_key:
                pricing = LLM_PRICING[key]
                break
        else:
            pricing = LLM_PRICING["default"]
            logger.warning(f"Unknown model '{model}', using default pricing")
        
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost


class LLMCallTracker:
    """Tracks a single LLM API call.
    
    Automatically records timing and calculates costs.
    """

    def __init__(
        self,
        agent_id: str,
        provider: str,
        model: str,
        decision_id: Optional[UUID] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        is_streaming: bool = False,
    ):
        """Initialize tracker."""
        self.agent_id = agent_id
        self.decision_id = decision_id
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_streaming = is_streaming
        
        self.start_time = time.time()
        self.first_token_time: Optional[float] = None
        self.chunks_received = 0
        self.retry_count = 0
        self.rate_limited = False

    def record_first_token(self) -> None:
        """Record when first token is received (for streaming)."""
        if not self.first_token_time:
            self.first_token_time = time.time()

    def record_chunk(self) -> None:
        """Record a streaming chunk received."""
        self.chunks_received += 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retry_count += 1

    def record_rate_limit(self) -> None:
        """Record that request was rate limited."""
        self.rate_limited = True

    def complete(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: Optional[str] = None,
        **metadata: Any,
    ) -> LLMMetrics:
        """Complete tracking and return metrics.
        
        Args:
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            finish_reason: Why generation stopped
            **metadata: Additional metadata
            
        Returns:
            LLMMetrics object
        """
        end_time = time.time()
        latency_ms = (end_time - self.start_time) * 1000
        
        time_to_first_token_ms = None
        if self.first_token_time:
            time_to_first_token_ms = (self.first_token_time - self.start_time) * 1000
        
        cost_usd = LLMMetricsCollector.calculate_cost(
            self.model,
            prompt_tokens,
            completion_tokens,
        )
        
        return LLMMetrics(
            agent_id=self.agent_id,
            decision_id=self.decision_id,
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            cost_usd=cost_usd,
            finish_reason=finish_reason,
            is_streaming=self.is_streaming,
            chunks_received=self.chunks_received if self.is_streaming else None,
            rate_limited=self.rate_limited,
            retry_count=self.retry_count,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata=metadata,
        )
