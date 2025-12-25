"""
Agent tools for trading operations
"""
from .kite_tools import (
    place_order_tool,
    modify_order_tool,
    cancel_order_tool,
    get_positions_tool,
    get_orders_tool,
    get_balance_tool,
    exit_position_tool,
)
from .market_tools import (
    get_quote_tool,
    get_historical_data_tool,
    analyze_trend_tool,
    get_nifty_options_tool,
    calculate_indicators_tool,
)
from .strategy_tools import (
    backtest_strategy_tool,
    get_strategy_signal_tool,
    optimize_strategy_tool,
    create_strategy_tool,
)
from .risk_tools import (
    calculate_risk_tool,
    check_risk_limits_tool,
    get_portfolio_risk_tool,
    suggest_position_size_tool,
)
from .portfolio_tools import (
    get_portfolio_summary_tool,
    analyze_performance_tool,
    rebalance_portfolio_tool,
)
from .indicator_query_tool import (
    find_indicator_threshold_crossings,
    get_indicator_history,
)
from .trading_opportunities_tool import (
    find_indicator_based_trading_opportunities,
)
from .gap_analysis_tool import (
    analyze_gap_probability,
)
from .candlestick_patterns_tool import (
    find_candlestick_patterns,
)

__all__ = [
    # Kite tools
    "place_order_tool",
    "modify_order_tool",
    "cancel_order_tool",
    "get_positions_tool",
    "get_orders_tool",
    "get_balance_tool",
    "exit_position_tool",
    # Market tools
    "get_quote_tool",
    "get_historical_data_tool",
    "analyze_trend_tool",
    "get_nifty_options_tool",
    "calculate_indicators_tool",
    # Strategy tools
    "backtest_strategy_tool",
    "get_strategy_signal_tool",
    "optimize_strategy_tool",
    "create_strategy_tool",
    # Risk tools
    "calculate_risk_tool",
    "check_risk_limits_tool",
    "get_portfolio_risk_tool",
    "suggest_position_size_tool",
    # Portfolio tools
    "get_portfolio_summary_tool",
    "analyze_performance_tool",
    "rebalance_portfolio_tool",
    # Indicator query tools
    "find_indicator_threshold_crossings",
    "get_indicator_history",
    # Trading opportunities tools
    "find_indicator_based_trading_opportunities",
    # Gap analysis tools
    "analyze_gap_probability",
    # Candlestick pattern tools
    "find_candlestick_patterns",
    "ALL_TOOLS",
]

# Aggregate all tools for easy access
ALL_TOOLS = [
    place_order_tool,
    modify_order_tool,
    cancel_order_tool,
    get_positions_tool,
    get_orders_tool,
    get_balance_tool,
    exit_position_tool,
    get_quote_tool,
    get_historical_data_tool,
    analyze_trend_tool,
    get_nifty_options_tool,
    calculate_indicators_tool,
    calculate_risk_tool,
    check_risk_limits_tool,
    get_portfolio_risk_tool,
    suggest_position_size_tool,
    get_portfolio_summary_tool,
    analyze_performance_tool,
    rebalance_portfolio_tool,
    find_indicator_based_trading_opportunities,
    analyze_gap_probability,
]

