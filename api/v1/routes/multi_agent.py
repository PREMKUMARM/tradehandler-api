"""
Multi-Agent API Routes
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import uuid

from agents.agent_orchestrator import AgentOrchestrator
from agents.agent_types import AgentType
from utils.logger import log_info, log_error

router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])

# Global orchestrator instance
orchestrator = None

class UserRequest(BaseModel):
    """User request for multi-agent processing"""
    message: str = Field(..., description="User message or request")
    request_type: str = Field(default="general", description="Type of request")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    priority: int = Field(default=5, description="Request priority (1-10)")
    
class AgentTaskRequest(BaseModel):
    """Request to create agent task"""
    agent_id: str = Field(..., description="Target agent ID")
    task_type: str = Field(..., description="Type of task")
    task_data: Dict[str, Any] = Field(..., description="Task data")
    priority: int = Field(default=5, description="Task priority")
    scheduled_time: Optional[datetime] = Field(None, description="Schedule time")

class AgentScheduleRequest(BaseModel):
    """Request to schedule agent"""
    agent_id: str = Field(..., description="Agent ID")
    schedule_type: str = Field(..., description="Schedule type (cron/interval)")
    schedule_config: Dict[str, Any] = Field(..., description="Schedule configuration")
    enabled: bool = Field(default=True, description="Enable/disable schedule")

class StrategyBuilderRequest(BaseModel):
    """Request for premium strategy building"""
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    requirements: Dict[str, Any] = Field(..., description="Strategy requirements")
    market_conditions: Dict[str, Any] = Field(default_factory=dict, description="Market conditions")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    instruments: List[str] = Field(default_factory=list, description="Trading instruments")
    timeframes: List[str] = Field(default_factory=list, description="Timeframes")

async def get_orchestrator() -> AgentOrchestrator:
    """Get orchestrator instance"""
    global orchestrator, orchestrator_initialized
    if orchestrator is None:
        orchestrator = AgentOrchestrator()
        # Initialize the orchestrator
        await orchestrator.initialize()
        orchestrator_initialized = True
    return orchestrator

def get_orchestrator_sync() -> AgentOrchestrator:
    """Get orchestrator instance (sync version for dependency injection)"""
    global orchestrator, orchestrator_initialized
    if orchestrator is None or not orchestrator_initialized:
        # Return a dummy orchestrator that will be initialized later
        orchestrator = AgentOrchestrator()
        orchestrator_initialized = False
    return orchestrator

@router.post("/process")
async def process_user_request(
    request: UserRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator_sync)
):
    """Process user request through multi-agent system"""
    
    # Initialize orchestrator if not already initialized
    global orchestrator_initialized
    if not orchestrator_initialized:
        await orchestrator.initialize()
        orchestrator_initialized = True
    
    try:
        # Add request ID
        request_data = request.dict()
        request_data["request_id"] = str(uuid.uuid4())
        
        # Process request
        result = await orchestrator.process_user_request(request_data)
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to process user request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def get_agents(orchestrator: AgentOrchestrator = Depends(get_orchestrator)):
    """Get list of all agents"""
    
    try:
        agents = await orchestrator.get_agent_list()
        
        return {
            "success": True,
            "data": {
                "agents": agents,
                "total_count": len(agents)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to get agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}/status")
async def get_agent_status(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get status of a specific agent"""
    
    try:
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        agent = orchestrator.agents[agent_id]
        status = await agent.get_status()
        
        return {
            "success": True,
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/tasks")
async def create_agent_task(
    agent_id: str,
    task_request: AgentTaskRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Create a task for a specific agent"""
    
    try:
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Schedule task
        task_id = await orchestrator.schedule_agent_task(
            agent_id, 
            task_request.task_data,
            task_request.scheduled_time
        )
        
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "agent_id": agent_id,
                "task_type": task_request.task_type,
                "scheduled_time": task_request.scheduled_time.isoformat() if task_request.scheduled_time else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Failed to create agent task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/start")
async def start_agent(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Start a specific agent"""
    
    try:
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        agent = orchestrator.agents[agent_id]
        await agent.start()
        
        return {
            "success": True,
            "data": {
                "agent_id": agent_id,
                "status": "started"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Failed to start agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/stop")
async def stop_agent(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Stop a specific agent"""
    
    try:
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        agent = orchestrator.agents[agent_id]
        await agent.stop()
        
        return {
            "success": True,
            "data": {
                "agent_id": agent_id,
                "status": "stopped"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Failed to stop agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/schedule")
async def schedule_agent(
    agent_id: str,
    schedule_request: AgentScheduleRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Schedule an agent"""
    
    try:
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        agent = orchestrator.agents[agent_id]
        
        # Update agent configuration
        agent.config.schedule = {
            "type": schedule_request.schedule_type,
            **schedule_request.schedule_config
        }
        
        # Restart agent to apply new schedule
        if agent.status.value == "running":
            await agent.stop()
            await agent.start()
        
        return {
            "success": True,
            "data": {
                "agent_id": agent_id,
                "schedule": agent.config.schedule,
                "enabled": schedule_request.enabled
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Failed to schedule agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategies/build")
async def build_strategy(
    strategy_request: StrategyBuilderRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Build a premium strategy"""
    
    try:
        # Create request for premium strategy agent
        request_data = {
            "request_id": str(uuid.uuid4()),
            "message": f"Build strategy: {strategy_request.name}",
            "request_type": "strategy_building",
            "context": {
                "strategy_request": strategy_request.dict()
            }
        }
        
        # Process through orchestrator
        result = await orchestrator.process_user_request(request_data)
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to build strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategies/{strategy_id}/backtest")
async def backtest_strategy(
    strategy_id: str,
    backtest_config: Dict[str, Any],
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Backtest a strategy"""
    
    try:
        # Create backtest request
        request_data = {
            "request_id": str(uuid.uuid4()),
            "message": f"Backtest strategy: {strategy_id}",
            "request_type": "backtesting",
            "context": {
                "strategy_id": strategy_id,
                "backtest_config": backtest_config
            }
        }
        
        # Process through orchestrator
        result = await orchestrator.process_user_request(request_data)
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to backtest strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies")
async def get_strategies(orchestrator: AgentOrchestrator = Depends(get_orchestrator)):
    """Get all strategies"""
    
    try:
        strategies = await orchestrator.data_store.get_all_strategies()
        
        return {
            "success": True,
            "data": {
                "strategies": strategies,
                "total_count": len(strategies)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to get strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies/{strategy_id}")
async def get_strategy(
    strategy_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get a specific strategy"""
    
    try:
        strategy = await orchestrator.data_store.get_strategy(strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
        
        return {
            "success": True,
            "data": strategy,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Failed to get strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/premarket-data")
async def get_premarket_data(
    date: Optional[str] = None,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get pre-market data"""
    
    try:
        premarket_data = await orchestrator.data_store.get_premarket_data(date)
        
        return {
            "success": True,
            "data": premarket_data or {},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to get premarket data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/status")
async def get_system_status(orchestrator: AgentOrchestrator = Depends(get_orchestrator_sync)):
    """Get overall system status"""
    
    # Initialize orchestrator if not already initialized
    global orchestrator_initialized
    if not orchestrator_initialized:
        await orchestrator.initialize()
        orchestrator_initialized = True
    
    try:
        status = await orchestrator.get_system_status()
        
        return {
            "success": True,
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/start")
async def start_system(orchestrator: AgentOrchestrator = Depends(get_orchestrator)):
    """Start the multi-agent system"""
    
    try:
        await orchestrator.start_all_agents()
        
        return {
            "success": True,
            "data": {
                "status": "started",
                "agents_started": len(orchestrator.agents)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to start system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/stop")
async def stop_system(orchestrator: AgentOrchestrator = Depends(get_orchestrator)):
    """Stop the multi-agent system"""
    
    try:
        await orchestrator.stop_all_agents()
        
        return {
            "success": True,
            "data": {
                "status": "stopped",
                "agents_stopped": len(orchestrator.agents)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Failed to stop system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks")
async def get_tasks(
    agent_id: Optional[str] = None,
    limit: int = 100,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get tasks"""
    
    try:
        if agent_id:
            if agent_id not in orchestrator.agents:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            
            tasks = await orchestrator.data_store.get_agent_tasks(agent_id, limit)
        else:
            # Get tasks from all agents
            all_tasks = []
            for agent_id in orchestrator.agents:
                agent_tasks = await orchestrator.data_store.get_agent_tasks(agent_id, limit)
                all_tasks.extend(agent_tasks)
            
            # Sort by created_at and limit
            all_tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            tasks = all_tasks[:limit]
        
        return {
            "success": True,
            "data": {
                "tasks": tasks,
                "total_count": len(tasks)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Failed to get tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/communications")
async def get_communications(
    agent_id: Optional[str] = None,
    limit: int = 100,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get agent communications"""
    
    try:
        if agent_id:
            if agent_id not in orchestrator.agents:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            
            communications = await orchestrator.data_store.get_agent_messages(agent_id, limit)
        else:
            # Get communications from all agents
            all_comms = []
            for agent_id in orchestrator.agents:
                agent_comms = await orchestrator.data_store.get_agent_messages(agent_id, limit)
                all_comms.extend(agent_comms)
            
            # Sort by timestamp and limit
            all_comms.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            communications = all_comms[:limit]
        
        return {
            "success": True,
            "data": {
                "communications": communications,
                "total_count": len(communications)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Failed to get communications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics(
    agent_id: Optional[str] = None,
    metric_name: Optional[str] = None,
    days: int = 30,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get agent metrics"""
    
    try:
        if agent_id:
            if agent_id not in orchestrator.agents:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            
            metrics = await orchestrator.data_store.get_agent_metrics(agent_id, metric_name, days * 100)
        else:
            # Get metrics from all agents
            all_metrics = []
            for agent_id in orchestrator.agents:
                agent_metrics = await orchestrator.data_store.get_agent_metrics(agent_id, metric_name, days * 100)
                all_metrics.extend(agent_metrics)
            
            # Sort by timestamp and limit
            all_metrics.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            metrics = all_metrics[:days * 100]
        
        return {
            "success": True,
            "data": {
                "metrics": metrics,
                "total_count": len(metrics),
                "period_days": days
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
