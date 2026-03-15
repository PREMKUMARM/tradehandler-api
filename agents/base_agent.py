"""
Base Agent Class for Multi-Agent Architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, time
from enum import Enum
import asyncio
import json
import uuid
from dataclasses import dataclass, field
import logging

from utils.logger import log_info, log_error, log_warning, log_debug

class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"
    SCHEDULED = "scheduled"

class AgentCapability(Enum):
    """Agent capabilities"""
    PREMARKET_ANALYSIS = "premarket_analysis"
    MARKET_ANALYSIS = "market_analysis"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    STRATEGY_BUILDING = "strategy_building"
    STRATEGY_EXECUTION = "strategy_execution"
    RISK_MANAGEMENT = "risk_management"
    ORDER_EXECUTION = "order_execution"
    MONITORING = "monitoring"
    BACKTESTING = "backtesting"
    DATA_COLLECTION = "data_collection"

@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_id: str
    agent_type: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    schedule: Optional[Dict[str, Any]] = None
    enabled: bool = True
    priority: int = 5  # 1-10, 1 being highest
    max_concurrent_tasks: int = 1
    dependencies: List[str] = field(default_factory=list)
    mcp_tools: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    output_format: str = "json"  # json, text, structured
    
@dataclass
class AgentTask:
    """Agent task definition"""
    task_id: str
    agent_id: str
    task_type: str
    priority: int
    data: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.status = AgentStatus.IDLE
        self.current_tasks: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        self.communication_layer = None
        self.data_store = None
        self.mcp_clients = {}
        self.logger = logging.getLogger(f"agent.{config.agent_type}")
        
        # Performance metrics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        self.last_execution = None
        
    async def initialize(self, communication_layer, data_store):
        """Initialize agent with communication and data storage"""
        self.communication_layer = communication_layer
        self.data_store = data_store
        
        # Register with communication layer
        await self.communication_layer.register_agent(self)
        
        # Initialize MCP clients
        await self._initialize_mcp_clients()
        
        # Load agent-specific data
        await self._load_agent_data()
        
        log_info(f"Agent {self.config.name} initialized successfully")
        
    async def _initialize_mcp_clients(self):
        """Initialize MCP clients for this agent"""
        for mcp_tool in self.config.mcp_tools:
            try:
                if mcp_tool == "zerodha":
                    from utils.mcp_zerodha_client import ZerodhaMCPClient
                    self.mcp_clients["zerodha"] = ZerodhaMCPClient()
                    await self.mcp_clients["zerodha"].initialize()
                # Add more MCP clients as needed
            except Exception as e:
                log_error(f"Failed to initialize MCP client {mcp_tool}: {e}")
                
    async def _load_agent_data(self):
        """Load agent-specific data from storage"""
        try:
            if self.data_store:
                agent_data = await self.data_store.get_agent_data(self.config.agent_id)
                if agent_data:
                    self.tasks_completed = agent_data.get("tasks_completed", 0)
                    self.tasks_failed = agent_data.get("tasks_failed", 0)
                    self.total_execution_time = agent_data.get("total_execution_time", 0.0)
        except Exception as e:
            log_error(f"Failed to load agent data: {e}")
            
    async def save_agent_data(self):
        """Save agent data to storage"""
        try:
            if self.data_store:
                agent_data = {
                    "tasks_completed": self.tasks_completed,
                    "tasks_failed": self.tasks_failed,
                    "total_execution_time": self.total_execution_time,
                    "last_execution": self.last_execution.isoformat() if self.last_execution else None
                }
                await self.data_store.save_agent_data(self.config.agent_id, agent_data)
        except Exception as e:
            log_error(f"Failed to save agent data: {e}")
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a task - must be implemented by each agent"""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        pass
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a task with error handling and metrics"""
        start_time = datetime.now()
        
        try:
            self.status = AgentStatus.RUNNING
            task.status = AgentStatus.RUNNING
            
            log_info(f"Agent {self.config.name} executing task {task.task_id}")
            
            # Check dependencies
            if task.dependencies:
                await self._check_task_dependencies(task)
            
            # Process the task
            result = await self.process_task(task)
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.tasks_completed += 1
            self.total_execution_time += execution_time
            self.last_execution = datetime.now()
            
            # Update task
            task.status = AgentStatus.IDLE
            task.result = result
            task.execution_time = execution_time
            
            # Save agent data
            await self.save_agent_data()
            
            log_info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            log_error(f"Task {task.task_id} failed: {error_msg}")
            
            # Update metrics
            self.tasks_failed += 1
            
            # Update task
            task.status = AgentStatus.ERROR
            task.error = error_msg
            
            # Save agent data
            await self.save_agent_data()
            
            # Notify other agents of failure
            await self.communication_layer.broadcast_task_failure(task)
            
            raise e
            
        finally:
            self.status = AgentStatus.IDLE
    
    async def _check_task_dependencies(self, task: AgentTask):
        """Check if task dependencies are satisfied"""
        for dep_task_id in task.dependencies:
            dep_task = await self.communication_layer.get_task(dep_task_id)
            if not dep_task or dep_task.status != AgentStatus.IDLE:
                raise Exception(f"Task dependency {dep_task_id} not satisfied")
    
    async def schedule_task(self, task: AgentTask):
        """Schedule a task for execution"""
        task.status = AgentStatus.SCHEDULED
        await self.communication_layer.schedule_task(task)
    
    async def communicate_with_agent(self, agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Communicate with another agent"""
        return await self.communication_layer.send_message(self.config.agent_id, agent_id, message)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status and metrics"""
        return {
            "agent_id": self.config.agent_id,
            "name": self.config.name,
            "type": self.config.agent_type,
            "status": self.status.value,
            "capabilities": [cap.value for cap in await self.get_capabilities()],
            "current_tasks": len(self.current_tasks),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.total_execution_time / max(self.tasks_completed, 1),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "enabled": self.config.enabled
        }
    
    async def start(self):
        """Start the agent"""
        if not self.config.enabled:
            log_warning(f"Agent {self.config.name} is disabled")
            return
            
        self.status = AgentStatus.RUNNING
        log_info(f"Agent {self.config.name} started")
        
        # Start scheduled tasks
        if self.config.schedule:
            await self._start_scheduled_tasks()
    
    async def stop(self):
        """Stop the agent"""
        self.status = AgentStatus.STOPPED
        log_info(f"Agent {self.config.name} stopped")
        
        # Stop all current tasks
        for task in self.current_tasks:
            task.status = AgentStatus.STOPPED
    
    async def _start_scheduled_tasks(self):
        """Start scheduled tasks"""
        schedule_config = self.config.schedule
        
        if schedule_config.get("type") == "cron":
            # Handle cron-like scheduling
            await self._handle_cron_schedule(schedule_config)
        elif schedule_config.get("type") == "interval":
            # Handle interval-based scheduling
            await self._handle_interval_schedule(schedule_config)
    
    async def _handle_cron_schedule(self, schedule_config: Dict[str, Any]):
        """Handle cron-based scheduling"""
        # Implementation for cron scheduling
        pass
    
    async def _handle_interval_schedule(self, schedule_config: Dict[str, Any]):
        """Handle interval-based scheduling"""
        interval_seconds = schedule_config.get("interval_seconds", 3600)
        
        while self.status == AgentStatus.RUNNING:
            try:
                # Create scheduled task
                task = AgentTask(
                    task_id=str(uuid.uuid4()),
                    agent_id=self.config.agent_id,
                    task_type="scheduled",
                    priority=5,
                    data=schedule_config.get("task_data", {}),
                    created_at=datetime.now()
                )
                
                # Execute task
                await self.execute_task(task)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                log_error(f"Scheduled task failed: {e}")
                await asyncio.sleep(min(interval_seconds, 300))  # Wait up to 5 minutes on error
    
    def __str__(self):
        return f"Agent({self.config.name}, {self.config.agent_type}, {self.status.value})"
    
    def __repr__(self):
        return self.__str__()
