"""
Agent state definition for LangGraph
"""
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from operator import add


class AgentState(TypedDict):
    """State for the trading agent"""
    # Messages
    messages: Annotated[List[BaseMessage], add]
    
    # Current context
    user_query: str
    agent_response: Optional[str]
    intent: Optional[str]
    entities: Optional[Dict[str, Any]]
    
    # Tool execution
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    
    # Trading context
    positions: List[Dict[str, Any]]
    orders: List[Dict[str, Any]]
    balance: Optional[Dict[str, Any]]
    
    # Risk and approval
    risk_assessment: Optional[Dict[str, Any]]
    requires_approval: bool
    approval_id: Optional[str]
    
    # Agent metadata
    reasoning: List[str]
    errors: List[str]
    
    # Configuration
    config: Optional[Dict[str, Any]]

