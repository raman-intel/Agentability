"""LangChain auto-instrumentation.

Automatically instruments LangChain chains and agents with Agentability tracking.

Google Style Guide Compliant.
"""

from typing import Any, Optional
from ..tracer import Tracer


class LangChainInstrumentation:
    """Auto-instruments LangChain components.
    
    Example:
        >>> from agentability import Tracer
        >>> from agentability.integrations import LangChainInstrumentation
        >>> 
        >>> tracer = Tracer(offline_mode=True)
        >>> instrumenter = LangChainInstrumentation(tracer)
        >>> 
        >>> # Instrument your chain
        >>> chain = instrumenter.instrument_chain(my_langchain)
    """
    
    def __init__(self, tracer: Tracer):
        """Initialize the instrumentation.
        
        Args:
            tracer: Agentability tracer instance.
        """
        self.tracer = tracer
    
    def instrument_chain(self, chain: Any) -> Any:
        """Instrument a LangChain chain.
        
        Args:
            chain: LangChain chain to instrument.
            
        Returns:
            Instrumented chain.
        """
        # Wrap chain's __call__ method
        original_call = getattr(chain, '__call__', None)
        
        if not original_call:
            return chain
        
        def instrumented_call(*args, **kwargs):
            with self.tracer.trace_decision(
                agent_id=f"langchain_{chain.__class__.__name__}",
                decision_type="generation"
            ) as ctx:
                result = original_call(*args, **kwargs)
                ctx.set_confidence(0.8)  # Default confidence
                return result
        
        chain.__call__ = instrumented_call
        return chain
    
    def instrument_agent(self, agent: Any) -> Any:
        """Instrument a LangChain agent.
        
        Args:
            agent: LangChain agent to instrument.
            
        Returns:
            Instrumented agent.
        """
        # Similar to chain instrumentation
        return self.instrument_chain(agent)
