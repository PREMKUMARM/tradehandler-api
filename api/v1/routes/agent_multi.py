"""
Enhanced Agent Chat API with Multi-Agent Support
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import asyncio

from agents.agent_orchestrator import AgentOrchestrator
from utils.logger import log_info, log_error

router = APIRouter(prefix="/agent", tags=["agent-chat"])

# Global orchestrator instance
orchestrator = None

class ChatRequest(BaseModel):
    """Chat request with multi-agent support"""
    message: str = Field(..., description="User message")
    request_type: str = Field(default="chat", description="Request type")
    session_id: Optional[str] = Field(None, description="Session ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    priority: int = Field(default=5, description="Request priority (1-10)")
    agent_preferences: Optional[List[str]] = Field(None, description="Preferred agents to use")

class ChatResponse(BaseModel):
    """Chat response with multi-agent information"""
    session_id: Optional[str] = None
    message_id: str
    response: str
    agents_used: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any]

def get_orchestrator() -> AgentOrchestrator:
    """Get orchestrator instance"""
    global orchestrator, orchestrator_initialized
    if orchestrator is None:
        orchestrator = AgentOrchestrator()
        orchestrator_initialized = False
    return orchestrator

@router.post("/chat-multi")
async def chat_with_multi_agents(
    request: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Process chat request through multi-agent system"""
    
    # Initialize orchestrator if not already initialized
    global orchestrator_initialized
    if not orchestrator_initialized:
        await orchestrator.initialize()
        orchestrator_initialized = True
    
    try:
        start_time = datetime.now()
        
        # Create user request for multi-agent system
        multi_agent_request = {
            "request_id": str(uuid.uuid4()),
            "message": request.message,
            "request_type": request.request_type or "chat",
            "context": {
                **request.context,
                "session_id": request.session_id,
                "agent_preferences": request.agent_preferences
            },
            "priority": request.priority,
            "timestamp": start_time.isoformat()
        }
        
        # Process through multi-agent orchestrator
        result = await orchestrator.process_user_request(multi_agent_request)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        response = ChatResponse(
            session_id=request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            message_id=result.get("request_id", str(uuid.uuid4())),
            response=result.get("summary", "No response generated"),
            agents_used=result.get("agents_used", []),
            processing_time=processing_time,
            timestamp=datetime.now(),
            metadata={
                "status": result.get("status", "completed"),
                "recommendations": result.get("recommendations", []),
                "responses": result.get("responses", []),
                "agent_count": len(result.get("agents_used", []))
            }
        )
        
        # Save chat message to database
        await save_chat_message(request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}", request.message, response.dict())
        
        return {
            "success": True,
            "data": response.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to process multi-agent chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat-hybrid")
async def chat_with_hybrid_system(
    request: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Process chat request with hybrid routing (multi-agent + existing system)"""
    
    try:
        start_time = datetime.now()
        
        # First try to determine if this should go to multi-agent system
        should_use_multi_agent = await should_route_to_multi_agent(request.message)
        
        if should_use_multi_agent:
            # Use multi-agent system
            return await chat_with_multi_agents(request, orchestrator)
        else:
            # Fall back to existing hybrid agent system
            from utils.hybrid_agent import hybrid_agent
            
            result = await hybrid_agent.process_prompt(request.message, request.context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = ChatResponse(
                session_id=request.session_id,
                message_id=str(uuid.uuid4()),
                response=result.get("result", {}).get("response", "No response generated"),
                agents_used=[{
                    "agent_id": "hybrid_agent",
                    "agent_name": "Hybrid Agent System",
                    "task_type": "chat_processing",
                    "status": "completed"
                }],
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={
                    "source": result.get("source", "unknown"),
                    "classification": result.get("classification", {}),
                    "tool": result.get("tool", "none")
                }
            )
            
            # Save chat message to database
            await save_chat_message(request.session_id, request.message, response.dict())
            
            return {
                "success": True,
                "data": response.dict(),
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        log_error(f"Failed to process hybrid chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def should_route_to_multi_agent(message: str) -> bool:
    """Determine if message should be routed to multi-agent system"""
    
    message_lower = message.lower()
    
    # Keywords that suggest multi-agent processing
    multi_agent_keywords = [
        "analyze", "strategy", "build", "create", "optimize", "backtest",
        "premarket", "portfolio", "risk", "monitor", "coordinate",
        "multiple agents", "team", "collaborate", "comprehensive"
    ]
    
    # Check for multi-agent keywords
    if any(keyword in message_lower for keyword in multi_agent_keywords):
        return True
    
    # Check for complex requests that might benefit from multiple agents
    if len(message_lower.split()) > 10:  # Longer, more complex requests
        return True
    
    # Check for requests that involve multiple aspects
    aspects = ["market", "portfolio", "risk", "strategy"]
    mentioned_aspects = sum(1 for aspect in aspects if aspect in message_lower)
    if mentioned_aspects >= 2:
        return True
    
    return False

async def save_chat_message(session_id: str, user_message: str, response_data: Dict[str, Any]):
    """Save chat message to database"""
    
    try:
        from database.repositories import get_chat_repository
        from database.models import ChatMessage
        
        chat_repo = get_chat_repository()
        
        # Save user message
        user_message_obj = ChatMessage(
            message_id=str(uuid.uuid4()),
            session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            role="user",
            content=user_message,
            timestamp=datetime.now()
        )
        await chat_repo.save(user_message_obj)
        
        # Save assistant response
        assistant_message_obj = ChatMessage(
            message_id=response_data.get("message_id", str(uuid.uuid4())),
            session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            role="assistant",
            content=response_data.get("response", ""),
            timestamp=response_data.get("timestamp", datetime.now()),
            metadata={
                "agents_used": response_data.get("agents_used", []),
                "processing_time": response_data.get("processing_time", 0),
                "metadata": response_data.get("metadata", {})
            }
        )
        await chat_repo.save(assistant_message_obj)
        
    except Exception as e:
        log_error(f"Failed to save chat message: {e}")

@router.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    
    try:
        from database.repositories import get_chat_repository
        
        chat_repo = get_chat_repository()
        messages = await chat_repo.get_session_messages(session_id)
        
        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "messages": messages,
                "total_count": len(messages)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent-capabilities")
async def get_agent_capabilities(orchestrator: AgentOrchestrator = Depends(get_orchestrator)):
    """Get capabilities of all agents"""
    
    try:
        agents = await orchestrator.get_agent_list()
        
        capabilities = {}
        for agent in agents:
            capabilities[agent["agent_id"]] = {
                "name": agent["name"],
                "type": agent["type"],
                "capabilities": agent["capabilities"],
                "status": agent["status"]
            }
        
        return {
            "success": True,
            "data": capabilities,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to get agent capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent-preferences")
async def set_agent_preferences(
    session_id: str,
    preferences: Dict[str, Any],
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Set agent preferences for a session"""
    
    try:
        # Save preferences to database or session storage
        # This would be implemented based on your preference storage system
        
        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "preferences": preferences,
                "saved_at": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to set agent preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent-performance")
async def get_agent_performance(
    agent_id: Optional[str] = None,
    days: int = 7,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get agent performance metrics"""
    
    try:
        # Get performance metrics from orchestrator
        metrics = await orchestrator.data_store.get_agent_metrics(
            agent_id, None, days
        )
        
        # Group metrics by agent and metric name
        performance_data = {}
        for metric in metrics:
            agent_id = metric["agent_id"]
            metric_name = metric["metric_name"]
            
            if agent_id not in performance_data:
                performance_data[agent_id] = {}
            
            if metric_name not in performance_data[agent_id]:
                performance_data[agent_id][metric_name] = []
            
            performance_data[agent_id][metric_name].append(metric["metric_value"])
        
        # Calculate averages
        for agent_id in performance_data:
            for metric_name in performance_data[agent_id]:
                values = performance_data[agent_id][metric_name]
                performance_data[agent_id][metric_name] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return {
            "success": True,
            "data": {
                "performance": performance_data,
                "period_days": days,
                "agent_count": len(performance_data)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to get agent performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
