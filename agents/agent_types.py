"""
Agent Types and Definitions for Multi-Agent Architecture
"""

from enum import Enum
from typing import List, Dict, Any

from .base_agent import AgentCapability, AgentConfig

class AgentType(Enum):
    """Agent types in the system"""
    PREMARKET_ANALYZER = "premarket_analyzer"
    MARKET_ANALYZER = "market_analyzer"
    PORTFOLIO_MANAGER = "portfolio_manager"
    STRATEGY_BUILDER = "strategy_builder"
    STRATEGY_EXECUTOR = "strategy_executor"
    RISK_MANAGER = "risk_manager"
    ORDER_EXECUTOR = "order_executor"
    MONITORING_AGENT = "monitoring_agent"
    PREMIUM_STRATEGY_AGENT = "premium_strategy_agent"

# Agent configurations for each type
AGENT_CONFIGURATIONS = {
    AgentType.PREMARKET_ANALYZER: AgentConfig(
        agent_id="premarket_analyzer_001",
        agent_type="premarket_analyzer",
        name="Pre-Market Analysis Agent",
        description="Analyzes pre-market data, news, and sentiment before market opens",
        capabilities=[
            AgentCapability.PREMARKET_ANALYSIS,
            AgentCapability.DATA_COLLECTION,
            AgentCapability.MARKET_ANALYSIS
        ],
        schedule={
            "type": "cron",
            "expression": "0 8 * * 1-5",  # 8 AM on weekdays
            "timezone": "Asia/Kolkata"
        },
        enabled=True,
        priority=8,
        max_concurrent_tasks=3,
        mcp_tools=["zerodha", "news_api"],
        data_sources=["nse", "bse", "news_feeds", "global_markets"]
    ),
    
    AgentType.MARKET_ANALYZER: AgentConfig(
        agent_id="market_analyzer_001",
        agent_type="market_analyzer",
        name="Market Analysis Agent",
        description="Real-time market analysis, technical indicators, and opportunity detection",
        capabilities=[
            AgentCapability.MARKET_ANALYSIS,
            AgentCapability.DATA_COLLECTION,
            AgentCapability.MONITORING
        ],
        schedule={
            "type": "interval",
            "interval_seconds": 300  # Every 5 minutes during market hours
        },
        enabled=True,
        priority=7,
        max_concurrent_tasks=5,
        mcp_tools=["zerodha", "binance"],
        data_sources=["nse", "bse", "crypto", "commodities"]
    ),
    
    AgentType.PORTFOLIO_MANAGER: AgentConfig(
        agent_id="portfolio_manager_001",
        agent_type="portfolio_manager",
        name="Portfolio Management Agent",
        description="Manages portfolio, tracks performance, and provides portfolio insights",
        capabilities=[
            AgentCapability.PORTFOLIO_MANAGEMENT,
            AgentCapability.RISK_MANAGEMENT,
            AgentCapability.MONITORING
        ],
        schedule={
            "type": "interval",
            "interval_seconds": 600  # Every 10 minutes
        },
        enabled=True,
        priority=6,
        max_concurrent_tasks=2,
        mcp_tools=["zerodha"],
        data_sources=["portfolio_db", "market_data"]
    ),
    
    AgentType.STRATEGY_BUILDER: AgentConfig(
        agent_id="strategy_builder_001",
        agent_type="strategy_builder",
        name="Strategy Building Agent",
        description="Builds and optimizes trading strategies based on market conditions",
        capabilities=[
            AgentCapability.STRATEGY_BUILDING,
            AgentCapability.BACKTESTING,
            AgentCapability.MARKET_ANALYSIS
        ],
        enabled=True,
        priority=5,
        max_concurrent_tasks=3,
        mcp_tools=["zerodha"],
        data_sources=["historical_data", "market_indicators"]
    ),
    
    AgentType.STRATEGY_EXECUTOR: AgentConfig(
        agent_id="strategy_executor_001",
        agent_type="strategy_executor",
        name="Strategy Execution Agent",
        description="Executes trading strategies and manages automated trading",
        capabilities=[
            AgentCapability.STRATEGY_EXECUTION,
            AgentCapability.RISK_MANAGEMENT,
            AgentCapability.MONITORING
        ],
        enabled=True,
        priority=4,
        max_concurrent_tasks=10,
        mcp_tools=["zerodha"],
        data_sources=["strategies", "market_data", "portfolio"]
    ),
    
    AgentType.RISK_MANAGER: AgentConfig(
        agent_id="risk_manager_001",
        agent_type="risk_manager",
        name="Risk Management Agent",
        description="Monitors and manages trading risks across all positions and strategies",
        capabilities=[
            AgentCapability.RISK_MANAGEMENT,
            AgentCapability.MONITORING,
            AgentCapability.PORTFOLIO_MANAGEMENT
        ],
        schedule={
            "type": "interval",
            "interval_seconds": 60  # Every minute
        },
        enabled=True,
        priority=2,  # High priority for risk management
        max_concurrent_tasks=5,
        mcp_tools=["zerodha"],
        data_sources=["portfolio", "market_data", "risk_limits"]
    ),
    
    AgentType.ORDER_EXECUTOR: AgentConfig(
        agent_id="order_executor_001",
        agent_type="order_executor",
        name="Order Execution Agent",
        description="Handles order placement, modification, and cancellation",
        capabilities=[
            AgentCapability.ORDER_EXECUTION,
            AgentCapability.MONITORING
        ],
        enabled=True,
        priority=1,  # Highest priority for order execution
        max_concurrent_tasks=20,
        mcp_tools=["zerodha"],
        data_sources=["order_queue", "market_data"]
    ),
    
    AgentType.MONITORING_AGENT: AgentConfig(
        agent_id="monitoring_agent_001",
        agent_type="monitoring_agent",
        name="System Monitoring Agent",
        description="Monitors all agents, system health, and performance metrics",
        capabilities=[
            AgentCapability.MONITORING,
            AgentCapability.DATA_COLLECTION
        ],
        schedule={
            "type": "interval",
            "interval_seconds": 30  # Every 30 seconds
        },
        enabled=True,
        priority=3,
        max_concurrent_tasks=5,
        mcp_tools=[],
        data_sources=["agent_registry", "system_metrics"]
    ),
    
    AgentType.PREMIUM_STRATEGY_AGENT: AgentConfig(
        agent_id="premium_strategy_agent_001",
        agent_type="premium_strategy_agent",
        name="Premium Strategy Agent",
        description="Advanced strategy building with AI, backtesting, and optimization",
        capabilities=[
            AgentCapability.STRATEGY_BUILDING,
            AgentCapability.BACKTESTING,
            AgentCapability.MARKET_ANALYSIS,
            AgentCapability.RISK_MANAGEMENT
        ],
        enabled=True,
        priority=5,
        max_concurrent_tasks=2,
        mcp_tools=["zerodha", "advanced_analytics"],
        data_sources=["historical_data", "market_indicators", "alternative_data"]
    )
}

# Agent dependencies
AGENT_DEPENDENCIES = {
    AgentType.STRATEGY_EXECUTOR: [AgentType.RISK_MANAGER, AgentType.MARKET_ANALYZER],
    AgentType.PORTFOLIO_MANAGER: [AgentType.MARKET_ANALYZER, AgentType.RISK_MANAGER],
    AgentType.PREMIUM_STRATEGY_AGENT: [AgentType.MARKET_ANALYZER, AgentType.STRATEGY_BUILDER]
}

# Agent communication patterns
AGENT_COMMUNICATION_PATTERNS = {
    # Which agents can communicate with which
    "broadcast_from": [
        AgentType.MARKET_ANALYZER,  # Broadcasts market updates
        AgentType.RISK_MANAGER,     # Broadcasts risk alerts
        AgentType.MONITORING_AGENT  # Broadcasts system alerts
    ],
    
    "request_to": {
        AgentType.STRATEGY_EXECUTOR: [AgentType.ORDER_EXECUTOR],  # Requests order execution
        AgentType.PORTFOLIO_MANAGER: [AgentType.MARKET_ANALYZER],  # Requests market data
        AgentType.RISK_MANAGER: [AgentType.STRATEGY_EXECUTOR, AgentType.PORTFOLIO_MANAGER]  # Risk checks
    },
    
    "collaborative_pairs": [
        (AgentType.STRATEGY_BUILDER, AgentType.PREMIUM_STRATEGY_AGENT),  # Strategy collaboration
        (AgentType.MARKET_ANALYZER, AgentType.PORTFOLIO_MANAGER),  # Market portfolio sync
    ]
}

def get_agent_config(agent_type: AgentType) -> AgentConfig:
    """Get configuration for an agent type"""
    return AGENT_CONFIGURATIONS.get(agent_type)

def get_agent_dependencies(agent_type: AgentType) -> List[AgentType]:
    """Get dependencies for an agent type"""
    return AGENT_DEPENDENCIES.get(agent_type, [])

def can_communicate(from_agent: AgentType, to_agent: AgentType) -> bool:
    """Check if two agents can communicate"""
    # All agents can communicate with monitoring agent
    if to_agent == AgentType.MONITORING_AGENT:
        return True
    
    # Check broadcast patterns
    if from_agent in AGENT_COMMUNICATION_PATTERNS["broadcast_from"]:
        return True
    
    # Check request patterns
    if to_agent in AGENT_COMMUNICATION_PATTERNS["request_to"].get(from_agent, []):
        return True
    
    # Check collaborative pairs
    pair = (from_agent, to_agent)
    reverse_pair = (to_agent, from_agent)
    if pair in AGENT_COMMUNICATION_PATTERNS["collaborative_pairs"] or \
       reverse_pair in AGENT_COMMUNICATION_PATTERNS["collaborative_pairs"]:
        return True
    
    return False
