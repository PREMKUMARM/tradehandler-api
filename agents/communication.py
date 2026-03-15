"""
Agent Communication Layer for Multi-Agent System
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

from .base_agent import AgentTask, AgentStatus
from .agent_types import can_communicate, AgentType
from utils.logger import log_info, log_error, log_warning, log_debug

class MessageType(Enum):
    """Message types for agent communication"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    STATUS_UPDATE = "status_update"
    ALERT = "alert"
    COLLABORATION = "collaboration"
    BROADCAST = "broadcast"

@dataclass
class AgentMessage:
    """Message between agents"""
    message_id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 5
    correlation_id: Optional[str] = None
    requires_response: bool = False
    response_timeout: int = 30  # seconds

@dataclass
class AgentSubscription:
    """Agent subscription to events"""
    agent_id: str
    event_type: str
    callback: Callable
    active: bool = True

class AgentCommunicationLayer:
    """Communication layer for multi-agent system"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}  # agent_id -> agent_instance
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.subscriptions: Dict[str, List[AgentSubscription]] = {}
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, AgentTask] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        
        # Performance metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.tasks_coordinated = 0
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
    async def initialize(self):
        """Initialize the communication layer"""
        self.running = True
        
        # Start message processing
        asyncio.create_task(self._process_messages())
        
        # Start task coordination
        asyncio.create_task(self._coordinate_tasks())
        
        log_info("Agent communication layer initialized")
    
    async def register_agent(self, agent):
        """Register an agent with the communication layer"""
        self.agents[agent.config.agent_id] = agent
        
        # Register message handlers
        await self._register_agent_handlers(agent)
        
        log_info(f"Agent {agent.config.name} registered with communication layer")
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            log_info(f"Agent {agent_id} unregistered")
    
    async def _register_agent_handlers(self, agent):
        """Register message handlers for an agent"""
        agent_id = agent.config.agent_id
        
        # Task request handler
        self.message_handlers[f"{agent_id}_task_request"] = self._handle_task_request
        
        # Data request handler
        self.message_handlers[f"{agent_id}_data_request"] = self._handle_data_request
        
        # Status update handler
        self.message_handlers[f"{agent_id}_status_update"] = self._handle_status_update
        
        # Data response handler
        self.message_handlers[f"{agent_id}_data_response"] = self._handle_data_response
    
    async def send_message(self, from_agent: str, to_agent: str, message_type: MessageType, 
                          data: Dict[str, Any], priority: int = 5, 
                          requires_response: bool = False, correlation_id: str = None) -> Optional[Dict[str, Any]]:
        """Send a message to another agent"""
        
        # Check if communication is allowed
        from_agent_type = self.agents.get(from_agent)
        to_agent_type = self.agents.get(to_agent)
        
        if not from_agent_type or not to_agent_type:
            log_error(f"Agent not found: {from_agent} or {to_agent}")
            return None
        
        # Create message
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            data=data,
            timestamp=datetime.now(),
            priority=priority,
            correlation_id=correlation_id,
            requires_response=requires_response
        )
        
        # Add to queue
        await self.message_queue.put(message)
        self.messages_sent += 1
        
        log_debug(f"Message queued: {from_agent} -> {to_agent} ({message_type.value})")
        
        # Wait for response if required
        if requires_response:
            return await self._wait_for_response(message.message_id, timeout=30)
        
        return None
    
    async def broadcast_message(self, from_agent: str, message_type: MessageType, 
                               data: Dict[str, Any], priority: int = 5):
        """Broadcast message to all agents"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent="broadcast",
            message_type=message_type,
            data=data,
            timestamp=datetime.now(),
            priority=priority
        )
        
        await self.message_queue.put(message)
        self.messages_sent += 1
        
        log_info(f"Broadcast message from {from_agent}: {message_type.value}")
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while self.running:
            try:
                # Get message from queue
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                self.messages_received += 1
                
                # Process message
                if message.to_agent == "broadcast":
                    await self._handle_broadcast(message)
                else:
                    await self._handle_direct_message(message)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log_error(f"Error processing message: {e}")
    
    async def _handle_broadcast(self, message: AgentMessage):
        """Handle broadcast message"""
        for agent_id, agent in self.agents.items():
            if agent_id != message.from_agent:  # Don't send to self
                try:
                    await self._deliver_message(agent, message)
                except Exception as e:
                    log_error(f"Failed to deliver broadcast to {agent_id}: {e}")
    
    async def _handle_direct_message(self, message: AgentMessage):
        """Handle direct message"""
        to_agent = self.agents.get(message.to_agent)
        if to_agent:
            await self._deliver_message(to_agent, message)
        else:
            log_warning(f"Target agent {message.to_agent} not found")
    
    async def _deliver_message(self, agent, message: AgentMessage):
        """Deliver message to agent"""
        handler_key = f"{agent.config.agent_id}_{message.message_type.value}"
        handler = self.message_handlers.get(handler_key)
        
        if handler:
            try:
                await handler(agent, message)
            except Exception as e:
                log_error(f"Message handler error: {e}")
        else:
            log_warning(f"No handler found for {handler_key}")
    
    async def _handle_task_request(self, agent, message: AgentMessage):
        """Handle task request"""
        task_data = message.data
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_id=agent.config.agent_id,
            task_type=task_data.get("task_type", "unknown"),
            priority=task_data.get("priority", 5),
            data=task_data,
            created_at=datetime.now()
        )
        
        # Add to active tasks
        self.active_tasks[task.task_id] = task
        self.tasks_coordinated += 1
        
        # Execute task
        try:
            result = await agent.execute_task(task)
            
            # Send response
            if message.requires_response:
                await self.send_message(
                    from_agent=agent.config.agent_id,
                    to_agent=message.from_agent,
                    message_type=MessageType.TASK_RESPONSE,
                    data={
                        "task_id": task.task_id,
                        "result": result,
                        "status": "completed"
                    },
                    correlation_id=message.correlation_id
                )
                
        except Exception as e:
            log_error(f"Task execution failed: {e}")
            
            # Send error response
            if message.requires_response:
                await self.send_message(
                    from_agent=agent.config.agent_id,
                    to_agent=message.from_agent,
                    message_type=MessageType.TASK_RESPONSE,
                    data={
                        "task_id": task.task_id,
                        "error": str(e),
                        "status": "failed"
                    },
                    correlation_id=message.correlation_id
                )
    
    async def _handle_data_request(self, agent, message: AgentMessage):
        """Handle data request"""
        # This would be implemented based on what data agents can share
        pass
    
    async def _handle_status_update(self, agent, message: AgentMessage):
        """Handle status update"""
        # Log status update
        status_data = message.data
        log_debug(f"Status update from {agent.config.name}: {status_data}")
    
    async def _handle_data_response(self, agent, message: AgentMessage):
        """Handle data response"""
        # Log data response
        data_response = message.data
        log_debug(f"Data response from {agent.config.name}: {data_response}")
    
    async def _wait_for_response(self, message_id: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for response to a message"""
        # Implementation would depend on how you want to handle responses
        # For now, return None
        return None
    
    async def schedule_task(self, task: AgentTask):
        """Schedule a task for execution"""
        self.active_tasks[task.task_id] = task
        log_info(f"Task {task.task_id} scheduled")
    
    async def get_task(self, task_id: str) -> Optional[AgentTask]:
        """Get a task by ID"""
        return self.active_tasks.get(task_id) or self.completed_tasks.get(task_id)
    
    async def broadcast_task_failure(self, task: AgentTask):
        """Broadcast task failure to other agents"""
        await self.broadcast_message(
            from_agent=task.agent_id,
            message_type=MessageType.ALERT,
            data={
                "type": "task_failure",
                "task_id": task.task_id,
                "error": task.error,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    async def _coordinate_tasks(self):
        """Coordinate task execution across agents"""
        while self.running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Check for tasks that need coordination
                await self._check_task_dependencies()
                await self._cleanup_completed_tasks()
                
            except Exception as e:
                log_error(f"Task coordination error: {e}")
    
    async def _check_task_dependencies(self):
        """Check if tasks have satisfied dependencies"""
        for task_id, task in list(self.active_tasks.items()):
            if task.status == AgentStatus.SCHEDULED:
                dependencies_met = True
                
                for dep_task_id in task.dependencies:
                    dep_task = self.active_tasks.get(dep_task_id)
                    if not dep_task or dep_task.status != AgentStatus.IDLE:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    # Move task to ready state
                    task.status = AgentStatus.IDLE
                    log_info(f"Task {task_id} dependencies satisfied")
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed tasks"""
        current_time = datetime.now()
        
        # Move completed tasks to completed dict
        for task_id, task in list(self.active_tasks.items()):
            if task.status in [AgentStatus.IDLE, AgentStatus.ERROR, AgentStatus.STOPPED]:
                # Check if task is old enough to cleanup
                if (current_time - task.created_at).total_seconds() > 3600:  # 1 hour
                    self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system-wide status"""
        agent_statuses = {}
        
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = await agent.get_status()
        
        return {
            "total_agents": len(self.agents),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "tasks_coordinated": self.tasks_coordinated,
            "agent_statuses": agent_statuses
        }
    
    async def shutdown(self):
        """Shutdown the communication layer"""
        self.running = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        log_info("Agent communication layer shutdown")
