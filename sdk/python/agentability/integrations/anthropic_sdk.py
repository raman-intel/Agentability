"""Anthropic SDK wrapper with Agentability tracking."""

from typing import Any, Optional
from ..tracer import Tracer


class AnthropicInstrumentation:
    """Wraps Anthropic SDK calls with tracking."""
    
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
    
    def wrap_client(self, client: Any) -> Any:
        """Wrap an Anthropic client."""
        # Intercept messages.create calls
        original_create = client.messages.create
        
        def tracked_create(*args, **kwargs):
            with self.tracer.trace_decision(
                agent_id="anthropic_claude",
                decision_type="generation"
            ) as ctx:
                response = original_create(*args, **kwargs)
                
                # Extract tokens and cost
                if hasattr(response, 'usage'):
                    ctx.add_tokens(response.usage.input_tokens + response.usage.output_tokens)
                
                return response
        
        client.messages.create = tracked_create
        return client
