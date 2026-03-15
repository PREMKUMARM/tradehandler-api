"""
Agent Registry for managing multi-agent system
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio
import logging
from datetime import datetime

from .base_agent import BaseAgent, AgentStatus
from .agent_types import AgentType

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Registry for managing all agents in the system"""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_configs: Dict[str, Dict[str, Any]] = {}
        self._orchestrator = None
    
    def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent in the registry"""
        try:
            self._agents[agent.agent_id] = agent
            logger.info(f"Agent {agent.agent_id} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry"""
        try:
            if agent_id in self._agents:
                del self._agents[agent_id]
                logger.info(f"Agent {agent_id} unregistered successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID"""
        return self._agents.get(agent_id)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents"""
        return list(self._agents.values())
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get agents by type"""
        return [agent for agent in self._agents.values() if agent.agent_type == agent_type]
    
    def get_agents_by_status(self, status: AgentStatus) -> List[BaseAgent]:
        """Get agents by status"""
        return [agent for agent in self._agents.values() if agent.status == status]
    
    def start_agent(self, agent_id: str) -> bool:
        """Start an agent"""
        agent = self.get_agent(agent_id)
        if agent:
            try:
                asyncio.create_task(agent.start())
                return True
            except Exception as e:
                logger.error(f"Failed to start agent {agent_id}: {e}")
                return False
        return False
    
    def stop_agent(self, agent_id: str) -> bool:
        """Stop an agent"""
        agent = self.get_agent(agent_id)
        if agent:
            try:
                asyncio.create_task(agent.stop())
                return True
            except Exception as e:
                logger.error(f"Failed to stop agent {agent_id}: {e}")
                return False
        return False
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get agent status"""
        agent = self.get_agent(agent_id)
        return agent.status if agent else None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        agents = self.get_all_agents()
        return {
            "total_agents": len(agents),
            "running_agents": len(self.get_agents_by_status(AgentStatus.RUNNING)),
            "idle_agents": len(self.get_agents_by_status(AgentStatus.IDLE)),
            "error_agents": len(self.get_agents_by_status(AgentStatus.ERROR)),
            "agent_statuses": {
                agent.agent_id: {
                    "status": agent.status.value,
                    "type": agent.agent_type.value,
                    "last_execution": agent.last_execution,
                    "tasks_completed": getattr(agent, 'tasks_completed', 0),
                    "tasks_failed": getattr(agent, 'tasks_failed', 0)
                }
                for agent in agents
            },
            "orchestrator_status": "running" if self._orchestrator else "stopped",
            "data_store_status": {"status": "healthy"},
            "communication_layer_status": {"status": "healthy"},
            "uptime_seconds": 3600,  # Mock uptime
            "tasks_processed": sum(getattr(agent, 'tasks_completed', 0) for agent in agents),
            "tasks_failed": sum(getattr(agent, 'tasks_failed', 0) for agent in agents)
        }
    
    def set_orchestrator(self, orchestrator):
        """Set the orchestrator instance"""
        self._orchestrator = orchestrator
    
    async def initialize_default_agents(self):
        """Initialize default agents"""
        try:
            # Import agents here to avoid circular imports
            from .premarket_agent import PreMarketAgent
            from .premium_strategy_agent import PremiumStrategyAgent
            
            # Create premarket agent
            premarket_agent = PreMarketAgent(
                agent_id="premarket_analyzer_001",
                name="Pre-Market Analyzer",
                agent_type=AgentType.PREMARKET_ANALYZER
            )
            self.register_agent(premarket_agent)
            
            # Create premium strategy agent
            premium_agent = PremiumStrategyAgent(
                agent_id="premium_strategy_agent_001",
                name="Premium Strategy Agent",
                agent_type=AgentType.STRATEGY_AGENT
            )
            self.register_agent(premium_agent)
            
            logger.info("Default agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize default agents: {e}")

# Global registry instance
agent_registry = AgentRegistry()
