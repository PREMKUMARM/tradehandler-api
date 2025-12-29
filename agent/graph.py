"""
LangGraph agent workflow
"""
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from agent.state import AgentState
from agent.nodes import (
    analyze_request_node,
    select_tools_node,
    execute_tools_node,
    risk_assessment_node,
    approval_check_node,
    execute_trade_node,
    generate_response_node,
)
from agent.llm_factory import create_llm
from agent.tools import ALL_TOOLS
from agent.memory import AgentMemory


# Get all tools from agent.tools


def create_agent_graph(llm: Optional[BaseChatModel] = None) -> StateGraph:
    """
    Create the LangGraph agent workflow
    
    Args:
        llm: Optional LLM instance (creates one if not provided)
        
    Returns:
        StateGraph instance
    """
    if llm is None:
        llm = create_llm()
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze_request", analyze_request_node)
    workflow.add_node("select_tools", select_tools_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("risk_assessment", risk_assessment_node)
    workflow.add_node("approval_check", approval_check_node)
    workflow.add_node("execute_trade", execute_trade_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # Define edges
    workflow.set_entry_point("analyze_request")
    workflow.add_edge("analyze_request", "select_tools")
    workflow.add_edge("select_tools", "execute_tools")
    workflow.add_edge("execute_tools", "risk_assessment")
    workflow.add_edge("risk_assessment", "approval_check")
    workflow.add_edge("approval_check", "execute_trade")
    workflow.add_edge("execute_trade", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()


# Global agent instance
_agent_instance: Optional[StateGraph] = None
_agent_memory: Optional[AgentMemory] = None


def get_agent_instance() -> StateGraph:
    """Get or create agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = create_agent_graph()
    return _agent_instance


def get_agent_memory() -> AgentMemory:
    """Get or create agent memory instance"""
    global _agent_memory
    if _agent_memory is None:
        _agent_memory = AgentMemory()
    return _agent_memory


async def run_agent(user_query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the agent with a user query
    
    Args:
        user_query: User's natural language query
        context: Optional context (positions, balance, etc.)
        
    Returns:
        Agent response and state
    """
    from datetime import datetime
    from agent.ws_manager import add_agent_log
    from agent.task_tracker import create_task, update_task
    
    agent = get_agent_instance()
    memory = get_agent_memory()
    
    # Create task tracking
    task_id = create_task(
        agent_id="main_trading_agent",
        input_data={
            "query": user_query,
            "context": context or {}
        }
    )
    task_start_time = datetime.now()
    
    # Initialize state
    initial_state: AgentState = {
        "messages": [],
        "user_query": user_query,
        "agent_response": None,
        "intent": None,
        "entities": {},
        "tool_calls": [],
        "tool_results": [],
        "positions": context.get("positions", []) if context else [],
        "orders": context.get("orders", []) if context else [],
        "balance": context.get("balance") if context else None,
        "risk_assessment": None,
        "requires_approval": False,
        "approval_id": None,
        "reasoning": [],
        "errors": [],
        "config": None,
    }
    
    # Add user message to memory
    memory.add_message("user", user_query)
    
    # Run agent
    try:
        final_state = agent.invoke(initial_state)
        
        # Add agent response to memory
        if final_state.get("agent_response"):
            memory.add_message("assistant", final_state["agent_response"])
        
        # Update task tracking
        task_end_time = datetime.now()
        update_task(
            task_id=task_id,
            status="completed",
            output={
                "response": final_state.get("agent_response", ""),
                "requires_approval": final_state.get("requires_approval", False),
                "approval_id": final_state.get("approval_id")
            },
            tool_calls=final_state.get("tool_calls", []),
            completed_at=task_end_time.isoformat()
        )
        
        return {
            "status": "success",
            "response": final_state.get("agent_response", "No response generated"),
            "state": final_state,
            "requires_approval": final_state.get("requires_approval", False),
            "approval_id": final_state.get("approval_id"),
            "task_id": task_id
        }
    except Exception as e:
        # Update task tracking with error
        task_end_time = datetime.now()
        update_task(
            task_id=task_id,
            status="failed",
            error=str(e),
            completed_at=task_end_time.isoformat()
        )
        
        add_agent_log(f"Agent task {task_id} failed: {str(e)}", "error")
        
        return {
            "status": "error",
            "error": str(e),
            "response": f"Error processing request: {str(e)}",
            "task_id": task_id
        }

