"""LlamaIndex auto-instrumentation."""

from typing import Any
from ..tracer import Tracer


class LlamaIndexInstrumentation:
    """Auto-instruments LlamaIndex components."""
    
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
    
    def instrument_index(self, index: Any) -> Any:
        """Instrument a LlamaIndex index."""
        return index
