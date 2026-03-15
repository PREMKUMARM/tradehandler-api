"""
Multi-Agent Architecture for AlgoFeast Trading Platform
"""

from .base_agent import BaseAgent, AgentCapability, AgentStatus
from .agent_registry import AgentRegistry
from .agent_orchestrator import AgentOrchestrator
from .communication import AgentCommunicationLayer
from .agent_types import AgentType

# Import all agent types
from .premarket_agent import PreMarketAgent
from .market_agent import MarketAgent
from .portfolio_agent import PortfolioAgent
from .strategy_agent import StrategyAgent
from .risk_agent import RiskAgent
from .execution_agent import ExecutionAgent
from .monitoring_agent import MonitoringAgent
from .premium_strategy_agent import PremiumStrategyAgent

__version__ = "1.0.0"
__all__ = [
    "BaseAgent",
    "AgentCapability", 
    "AgentStatus",
    "AgentRegistry",
    "AgentOrchestrator",
    "AgentCommunicationLayer",
    "AgentType",
    "PreMarketAgent",
    "MarketAgent", 
    "PortfolioAgent",
    "StrategyAgent",
    "RiskAgent",
    "ExecutionAgent",
    "MonitoringAgent",
    "PremiumStrategyAgent"
]
