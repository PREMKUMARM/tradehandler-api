"""
Agent Registry - Manages multiple specialized AI agents
"""
from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph
from agent.memory import AgentMemory

# Tool categorization mapping
TOOL_CATEGORIES = {
    "Trading": [
        "place_order_tool", "modify_order_tool", "cancel_order_tool",
        "get_positions_tool", "get_orders_tool", "get_balance_tool",
        "exit_position_tool", "place_gtt_tool", "modify_gtt_tool",
        "delete_gtt_tool", "get_gtt_tool", "get_gtts_tool"
    ],
    "Market Data": [
        "get_quote_tool", "get_historical_data_tool", "analyze_trend_tool",
        "get_nifty_options_tool", "calculate_indicators_tool"
    ],
    "Strategy": [
        "backtest_strategy_tool", "get_strategy_signal_tool",
        "optimize_strategy_tool", "create_strategy_tool"
    ],
    "Risk Management": [
        "calculate_risk_tool", "check_risk_limits_tool",
        "get_portfolio_risk_tool", "suggest_position_size_tool",
        "get_portfolio_summary_tool", "analyze_performance_tool",
        "rebalance_portfolio_tool"
    ],
    "Analysis": [
        "find_indicator_threshold_crossings", "get_indicator_history",
        "find_indicator_based_trading_opportunities", "analyze_gap_probability",
        "find_candlestick_patterns"
    ],
    "Simulation": [
        "download_historical_data_to_local_tool", "run_simulation_on_local_data_tool"
    ]
}


def get_tools_by_category(category: str) -> List[Any]:
    """Get tools for a specific category"""
    from agent.tools.kite_tools import (
        place_order_tool, modify_order_tool, cancel_order_tool,
        get_positions_tool, get_orders_tool, get_balance_tool,
        exit_position_tool, place_gtt_tool, modify_gtt_tool,
        delete_gtt_tool, get_gtt_tool, get_gtts_tool,
    )
    from agent.tools.market_tools import (
        get_quote_tool, get_historical_data_tool, analyze_trend_tool,
        get_nifty_options_tool, calculate_indicators_tool,
    )
    from agent.tools.strategy_tools import (
        backtest_strategy_tool, get_strategy_signal_tool,
        optimize_strategy_tool, create_strategy_tool,
    )
    from agent.tools.risk_tools import (
        calculate_risk_tool, check_risk_limits_tool,
        get_portfolio_risk_tool, suggest_position_size_tool,
    )
    from agent.tools.portfolio_tools import (
        get_portfolio_summary_tool, analyze_performance_tool,
        rebalance_portfolio_tool,
    )
    from agent.tools.indicator_query_tool import (
        find_indicator_threshold_crossings, get_indicator_history,
    )
    from agent.tools.trading_opportunities_tool import (
        find_indicator_based_trading_opportunities,
    )
    from agent.tools.gap_analysis_tool import (
        analyze_gap_probability,
    )
    from agent.tools.candlestick_patterns_tool import (
        find_candlestick_patterns,
    )
    from agent.tools.simulation_tools import (
        download_historical_data_to_local_tool, run_simulation_on_local_data_tool
    )
    
    tool_map = {
        "place_order_tool": place_order_tool,
        "modify_order_tool": modify_order_tool,
        "cancel_order_tool": cancel_order_tool,
        "get_positions_tool": get_positions_tool,
        "get_orders_tool": get_orders_tool,
        "get_balance_tool": get_balance_tool,
        "exit_position_tool": exit_position_tool,
        "place_gtt_tool": place_gtt_tool,
        "modify_gtt_tool": modify_gtt_tool,
        "delete_gtt_tool": delete_gtt_tool,
        "get_gtt_tool": get_gtt_tool,
        "get_gtts_tool": get_gtts_tool,
        "get_quote_tool": get_quote_tool,
        "get_historical_data_tool": get_historical_data_tool,
        "analyze_trend_tool": analyze_trend_tool,
        "get_nifty_options_tool": get_nifty_options_tool,
        "calculate_indicators_tool": calculate_indicators_tool,
        "backtest_strategy_tool": backtest_strategy_tool,
        "get_strategy_signal_tool": get_strategy_signal_tool,
        "optimize_strategy_tool": optimize_strategy_tool,
        "create_strategy_tool": create_strategy_tool,
        "calculate_risk_tool": calculate_risk_tool,
        "check_risk_limits_tool": check_risk_limits_tool,
        "get_portfolio_risk_tool": get_portfolio_risk_tool,
        "suggest_position_size_tool": suggest_position_size_tool,
        "get_portfolio_summary_tool": get_portfolio_summary_tool,
        "analyze_performance_tool": analyze_performance_tool,
        "rebalance_portfolio_tool": rebalance_portfolio_tool,
        "find_indicator_threshold_crossings": find_indicator_threshold_crossings,
        "get_indicator_history": get_indicator_history,
        "find_indicator_based_trading_opportunities": find_indicator_based_trading_opportunities,
        "analyze_gap_probability": analyze_gap_probability,
        "find_candlestick_patterns": find_candlestick_patterns,
        "download_historical_data_to_local_tool": download_historical_data_to_local_tool,
        "run_simulation_on_local_data_tool": run_simulation_on_local_data_tool,
    }
    
    tool_names = TOOL_CATEGORIES.get(category, [])
    return [tool_map[name] for name in tool_names if name in tool_map]


class SpecializedAgent:
    """Specialized agent with specific tools"""
    
    def __init__(self, agent_id: str, name: str, category: str, tools: List[Any], description: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.category = category
        self.tools = tools
        self.description = description
        self._graph_instance: Optional[StateGraph] = None
        self._memory: Optional[AgentMemory] = None
    
    def get_graph(self) -> StateGraph:
        """Get or create agent graph instance"""
        if self._graph_instance is None:
            from agent.graph import create_agent_graph
            # Create a custom graph with only this agent's tools
            self._graph_instance = create_agent_graph_with_tools(self.tools)
        return self._graph_instance
    
    def get_memory(self) -> AgentMemory:
        """Get or create agent memory instance"""
        if self._memory is None:
            self._memory = AgentMemory()
        return self._memory
    
    def get_tool_names(self) -> List[str]:
        """Get list of tool names"""
        return [tool.name for tool in self.tools]


def create_agent_graph_with_tools(tools: List[Any]) -> StateGraph:
    """Create agent graph with specific tools - uses main graph but filters tools in select_tools_node"""
    # For now, use the main graph but we'll filter tools in the select_tools_node
    # This maintains backward compatibility
    # Lazy import to avoid circular dependency
    from agent.graph import create_agent_graph
    return create_agent_graph()


# Agent Registry
_agent_registry: Dict[str, SpecializedAgent] = {}
_initialized = False


def initialize_agents():
    """Initialize all specialized agents"""
    global _agent_registry, _initialized
    
    if _initialized:
        return _agent_registry
    
    # Create specialized agents
    agents = [
        SpecializedAgent(
            agent_id="trading_agent",
            name="Trading Agent",
            category="Trading",
            tools=get_tools_by_category("Trading"),
            description="Handles order placement, modification, cancellation, and position management"
        ),
        SpecializedAgent(
            agent_id="market_data_agent",
            name="Market Data Agent",
            category="Market Data",
            tools=get_tools_by_category("Market Data"),
            description="Fetches market quotes, historical data, and performs market analysis"
        ),
        SpecializedAgent(
            agent_id="strategy_agent",
            name="Strategy Agent",
            category="Strategy",
            tools=get_tools_by_category("Strategy"),
            description="Creates, optimizes, and backtests trading strategies"
        ),
        SpecializedAgent(
            agent_id="risk_management_agent",
            name="Risk Management Agent",
            category="Risk Management",
            tools=get_tools_by_category("Risk Management"),
            description="Manages portfolio risk, calculates position sizes, and monitors limits"
        ),
        SpecializedAgent(
            agent_id="analysis_agent",
            name="Analysis Agent",
            category="Analysis",
            tools=get_tools_by_category("Analysis"),
            description="Performs technical analysis, finds trading opportunities, and analyzes patterns"
        ),
        SpecializedAgent(
            agent_id="simulation_agent",
            name="Simulation Agent",
            category="Simulation",
            tools=get_tools_by_category("Simulation"),
            description="Runs simulations and backtests on historical data"
        ),
    ]
    
    # Register all agents
    for agent in agents:
        _agent_registry[agent.agent_id] = agent
    
    _initialized = True
    return _agent_registry


def get_agent_registry() -> Dict[str, SpecializedAgent]:
    """Get agent registry, initializing if needed"""
    global _agent_registry
    if not _agent_registry:
        initialize_agents()
    return _agent_registry


def get_agent(agent_id: str) -> Optional[SpecializedAgent]:
    """Get a specific agent by ID"""
    registry = get_agent_registry()
    return registry.get(agent_id)


def get_all_agents() -> List[SpecializedAgent]:
    """Get all registered agents"""
    registry = get_agent_registry()
    return list(registry.values())


def select_agent_for_query(user_query: str, intent: Optional[str] = None) -> Optional[SpecializedAgent]:
    """
    Select the most appropriate agent for a user query
    
    Args:
        user_query: User's query
        intent: Detected intent (optional)
        
    Returns:
        Best matching agent or None (use main agent)
    """
    query_lower = user_query.lower()
    
    # Trading keywords
    if any(keyword in query_lower for keyword in ["buy", "sell", "order", "place", "cancel", "modify", "position", "gtt"]):
        return get_agent("trading_agent")
    
    # Market data keywords
    if any(keyword in query_lower for keyword in ["quote", "price", "historical", "candle", "market data", "ohlc"]):
        return get_agent("market_data_agent")
    
    # Strategy keywords
    if any(keyword in query_lower for keyword in ["strategy", "backtest", "optimize", "signal"]):
        return get_agent("strategy_agent")
    
    # Risk keywords
    if any(keyword in query_lower for keyword in ["risk", "portfolio", "position size", "limit", "exposure"]):
        return get_agent("risk_management_agent")
    
    # Analysis keywords
    if any(keyword in query_lower for keyword in ["analyze", "indicator", "rsi", "macd", "trend", "pattern", "gap", "opportunity"]):
        return get_agent("analysis_agent")
    
    # Simulation keywords
    if any(keyword in query_lower for keyword in ["simulate", "simulation", "download data", "local data"]):
        return get_agent("simulation_agent")
    
    # Default: use main agent (has all tools)
    return None

