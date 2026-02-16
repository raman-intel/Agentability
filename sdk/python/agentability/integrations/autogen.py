"""AutoGen auto-instrumentation."""

from typing import Any
from ..tracer import Tracer


class AutoGenInstrumentation:
    """Auto-instruments AutoGen agents."""
    
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
    
    def instrument_agent(self, agent: Any) -> Any:
        """Instrument an AutoGen agent."""
        return agent
