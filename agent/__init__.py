"""
AI Agent package for tradehandler-api
Provides LangGraph-based agentic trading capabilities
"""

from .config import AgentConfig, get_agent_config

# Lazy imports for graph to avoid requiring langgraph at module level
def create_agent_graph(*args, **kwargs):
    from .graph import create_agent_graph as _create_agent_graph
    return _create_agent_graph(*args, **kwargs)

def get_agent_instance(*args, **kwargs):
    from .graph import get_agent_instance as _get_agent_instance
    return _get_agent_instance(*args, **kwargs)

__all__ = [
    "create_agent_graph",
    "get_agent_instance",
    "AgentConfig",
    "get_agent_config",
]

