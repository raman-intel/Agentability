"""Framework integrations for Agentability.

This package provides auto-instrumentation for popular agent frameworks:
- LangChain
- CrewAI
- AutoGen
- LlamaIndex
- Anthropic SDK
"""

from .langchain import LangChainInstrumentation
from .crewai import CrewAIInstrumentation
from .autogen import AutoGenInstrumentation
from .llamaindex import LlamaIndexInstrumentation
from .anthropic_sdk import AnthropicInstrumentation

__all__ = [
    "LangChainInstrumentation",
    "CrewAIInstrumentation",
    "AutoGenInstrumentation",
    "LlamaIndexInstrumentation",
    "AnthropicInstrumentation",
]
