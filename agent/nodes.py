"""
LangGraph nodes for the trading agent
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.state import AgentState
from agent.tools import (
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
    backtest_strategy_tool,
    get_strategy_signal_tool,
    optimize_strategy_tool,
    create_strategy_tool,
    calculate_risk_tool,
    check_risk_limits_tool,
    get_portfolio_risk_tool,
    suggest_position_size_tool,
    get_portfolio_summary_tool,
    analyze_performance_tool,
    rebalance_portfolio_tool,
    find_indicator_threshold_crossings,
    get_indicator_history,
    find_indicator_based_trading_opportunities,
    analyze_gap_probability,
    find_candlestick_patterns,
    download_historical_data_to_local_tool,
    run_simulation_on_local_data_tool,
)
from agent.safety import get_safety_manager
from agent.approval import get_approval_queue
from agent.memory import AgentMemory
from agent.llm_factory import create_llm
from agent.tools import ALL_TOOLS
from utils.logger import log_tool_interaction, log_agent_activity
from database.repositories import get_tool_repository
from database.models import ToolExecution
import time


def resolve_date_to_range(date_str: str) -> tuple:
    """Resolve relative date strings like 'last 1 week' to (from_date, to_date)"""
    import re
    today = datetime.now().date()
    
    if not date_str:
        return str(today), str(today)
    
    date_str = date_str.lower().strip()
    
    if date_str == "today":
        return str(today), str(today)
    elif date_str == "yesterday":
        yesterday = today - timedelta(days=1)
        return str(yesterday), str(yesterday)
    elif date_str == "tomorrow":
        tomorrow = today + timedelta(days=1)
        return str(tomorrow), str(tomorrow)
    
    # Handle "last N days" or "last N weeks"
    match_days = re.search(r'last (\d+) day', date_str)
    if match_days:
        n = int(match_days.group(1))
        from_dt = today - timedelta(days=n)
        return str(from_dt), str(today)
    
    match_weeks = re.search(r'last (\d+) week', date_str)
    if match_weeks:
        n = int(match_weeks.group(1))
        from_dt = today - timedelta(weeks=n)
        return str(from_dt), str(today)
    
    # Handle "last N month(s)"
    match_months = re.search(r'last (\d+) month', date_str)
    if match_months:
        n = int(match_months.group(1))
        # Simple approximation for months (30 days per month)
        from_dt = today - timedelta(days=n * 30)
        return str(from_dt), str(today)
    
    # Plural support
    match_months_pl = re.search(r'last (\d+) months', date_str)
    if match_months_pl:
        n = int(match_months_pl.group(1))
        from_dt = today - timedelta(days=n * 30)
        return str(from_dt), str(today)
    
    # Handle "last month" (meaning previous month or last 30 days)
    if "last month" in date_str:
        from_dt = today - timedelta(days=30)
        return str(from_dt), str(today)

    # Handle "last year" or "last 1 year"
    match_years = re.search(r'last (\d+) year', date_str)
    if match_years:
        n = int(match_years.group(1))
        from_dt = today - timedelta(days=n * 365)
        return str(from_dt), str(today)
    
    if "last year" in date_str:
        from_dt = today - timedelta(days=365)
        return str(from_dt), str(today)
    
    # Handle "this week"
    if "this week" in date_str:
        from_dt = today - timedelta(days=today.weekday())
        return str(from_dt), str(today)
        
    # Handle "this month"
    if "this month" in date_str:
        from_dt = today.replace(day=1)
        return str(from_dt), str(today)
        
    # Default: assume it's already a YYYY-MM-DD date
    return date_str, date_str


def analyze_request_node(state: AgentState) -> AgentState:
    """Analyze user request and understand intent using LLM - Pure LLM approach"""
    user_query = state.get("user_query", "")
    print(f"\n[DEBUG] NODE: analyze_request_node | Query: '{user_query}'")
    log_agent_activity(f"New User Request: {user_query}", "info")
    state["reasoning"] = state.get("reasoning", [])
    state["reasoning"].append(f"Analyzing user request: {user_query}")
    
    # Use LLM with structured output to determine intent and extract entities
    try:
        llm = create_llm()
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_day = datetime.now().strftime("%A")
        
        system_prompt = f"""You are an intelligent trading assistant. 
IMPORTANT: The current real-world date is {current_date} ({current_day}). 
Analyze the user's query semantically and determine:
1. Intent: One of TRADE, QUERY, ANALYSIS, INDICATOR_QUERY, TRADING_OPPORTUNITIES, GAP_ANALYSIS, PREDICTION, SIMULATION, or CONVERSATION
2. Extract all relevant entities from the query

Intent categories:
- TRADE: User wants to buy/sell/place orders (e.g., "buy 100 shares of reliance", "sell nifty ce")
- QUERY: User wants to check positions, portfolio, balance (e.g., "show my positions", "what is my balance")
- ANALYSIS: User wants market analysis, trends, quotes, indicators (e.g., "price of tcs", "trend in nifty")
- INDICATOR_QUERY: User asks about specific indicator values or crossings (e.g., "when did rsi cross 30")
- TRADING_OPPORTUNITIES: User asks about historical trade possibilities (e.g., "what trades today in reliance")
- GAP_ANALYSIS: User asks about opening gaps (e.g., "gapup or gapdown tomorrow")
- PREDICTION: User asks for future price movements (e.g., "where will reliance be next week")
- SIMULATION: User wants to download data or run simulation (e.g., "download data for yesterday", "run simulation")
- CONVERSATION: General greetings, small talk, or help (e.g., "hi", "hello", "who are you", "help")

Respond ONLY with valid JSON, no other text:
{{
    "intent": "TRADE|QUERY|ANALYSIS|TRADING_OPPORTUNITIES|GAP_ANALYSIS|PREDICTION|SIMULATION|CONVERSATION",
    "entities": {{
        "instrument": "instrument name or group if mentioned (e.g., reliance, nifty, tcs, top 10 nifty50 stocks).",
        "date": "date or period mentioned (YYYY-MM-DD, 'today', 'tomorrow', 'yesterday', 'last 1 week', 'last 5 days', 'this month', 'last 1 month'). LEAVE EMPTY if not mentioned.",
        "action": "buy/sell if mentioned"
    }}
}}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        print(f"[DEBUG] analyze_request_node | LLM Content: {content}")
        
        # Parse JSON response
        import json
        import re
        
        # Look for the JSON block
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                # Clean up any potential markdown code block artifacts
                json_str = json_str.replace('```json', '').replace('```', '').strip()
                
                parsed = json.loads(json_str)
                # Normalize keys to lowercase for robust lookup
                parsed_normalized = {k.lower(): v for k, v in parsed.items()}
                
                intent = parsed_normalized.get("intent", "QUERY").upper()
                entities = parsed_normalized.get("entities", {})
                
                print(f"[DEBUG] analyze_request_node | Final Intent: {intent}")
                state["reasoning"].append(f"LLM detected intent: {intent}")
            except Exception as parse_error:
                print(f"[DEBUG] analyze_request_node | Parse Error: {parse_error}")
                # Fallback to simple keyword check if JSON fails
                if "TRADING_OPPORTUNITIES" in content.upper() or "trades" in user_query.lower():
                    intent = "TRADING_OPPORTUNITIES"
                elif "GAP_ANALYSIS" in content.upper() or "gap" in user_query.lower():
                    intent = "GAP_ANALYSIS"
                elif "download" in user_query.lower() or "simulation" in user_query.lower() or "simulate" in user_query.lower():
                    intent = "SIMULATION"
                elif "hi" in user_query.lower() or "hello" in user_query.lower():
                    intent = "CONVERSATION"
                else:
                    intent = "QUERY"
                entities = {}
        else:
            # Fallback for simple greetings if no JSON braces found
            if user_query.lower().strip() in ["hi", "hello", "hey", "help"]:
                intent = "CONVERSATION"
            elif "trades" in user_query.lower() or "opportunities" in user_query.lower():
                intent = "TRADING_OPPORTUNITIES"
            else:
                intent = "QUERY"
            entities = {}
        
        state["intent"] = intent
        state["entities"] = entities
        
    except Exception as e:
        print(f"[DEBUG] analyze_request_node | Exception: {str(e)}")
        state["intent"] = "QUERY"
        state["entities"] = {}
    
    return state


def select_tools_node(state: AgentState) -> AgentState:
    """Select appropriate tools using LLM with tool calling - Pure LLM approach"""
    intent = state.get("intent", "QUERY")
    user_query = state.get("user_query", "")
    entities = state.get("entities", {})
    print(f"\n[DEBUG] NODE: select_tools_node | Intent: {intent}")
    state["tool_calls"] = []
    
    # Skip tool selection for general conversation
    if intent == "CONVERSATION":
        print("[DEBUG] select_tools_node | CONVERSATION detected, skipping tools")
        state["reasoning"].append("General conversation detected, skipping tool selection")
        return state
    
    # Use LLM with tool binding to select and parameterize tools
    try:
        llm = create_llm()
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_day = datetime.now().strftime("%A")
        
        # Create a comprehensive prompt for tool selection
        system_prompt = f"""You are a trading assistant. 
IMPORTANT: The current real-world date is {current_date} ({current_day}).
Based on the user's query and the DETECTED INTENT, you MUST select the correct tool.

User Query: {user_query}
Detected Intent: {intent}
Extracted Entities: {entities}

Available tools include:
- get_quote_tool: Get current price/quote for an instrument
- get_historical_data_tool: Get historical candle data
- analyze_trend_tool: Analyze trend across timeframes
- calculate_indicators_tool: Calculate technical indicators (RSI, MACD, BB)
- find_indicator_based_trading_opportunities: Find trading opportunities using the Institutional VWAP Strategy. 
  This is the ONLY strategy used for trade decisions. 
  Default interval is "5minute" (5-minute candles). 
  Supports date ranges (e.g., "last 1 week", "last 5 days").
  CRITICAL: This tool can take a 'local_data_file' parameter to run offline simulations. LEAVE 'local_data_file' EMPTY if you don't know the path; the system will fill it automatically.
- analyze_gap_probability: Analyze gap up/down probability for opening
- get_positions_tool: Get current positions
- get_balance_tool: Get account balance
- get_portfolio_summary_tool: Get portfolio summary
- download_historical_data_to_local_tool: Download market data for offline simulation
- run_simulation_on_local_data_tool: Run a simulated trading session on local data

STRICT TOOL SELECTION RULES:
1. If Intent is TRADING_OPPORTUNITIES, you MUST call 'find_indicator_based_trading_opportunities'.
2. If Intent is GAP_ANALYSIS, you MUST call 'analyze_gap_probability'.
3. If Intent is ANALYSIS, use 'get_quote_tool', 'analyze_trend_tool', or 'calculate_indicators_tool'.
4. If Intent is QUERY and the user asks for balance/positions, use 'get_positions_tool' or 'get_balance_tool'.
5. If the user asks for "trades", "what trades", or "opportunities", you MUST use 'find_indicator_based_trading_opportunities'.
6. If Intent is SIMULATION and user asks to "download", use 'download_historical_data_to_local_tool'.
7. If the user asks to "run simulation", "simulate", or "what trades on local data", you MUST call 'run_simulation_on_local_data_tool' and LEAVE 'file_path' EMPTY.

Extract instrument names or groups from the query. Available groups include:
- "top 10 nifty50 stocks" or "nifty top 10" - Predefined top 10 Nifty stocks
- "selected stocks", "my stocks", "watchlist", or "selected" - User's selected stocks from the stock selection page
If no specific instrument is mentioned and the query is about trading opportunities or analysis, use "selected stocks" as the default.

Call the tools with proper parameters. You can call multiple tools if needed.

CRITICAL: If the intent is CONVERSATION or the query is just a greeting (hi, hello, etc.), DO NOT CALL ANY TOOLS. Just return without any tool calls.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        
        response = llm_with_tools.invoke(messages)
        
        # Extract tool calls from response
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                
                state["tool_calls"].append({
                    "tool": tool_name,
                    "args": tool_args,
                    "id": tool_call.get("id", "")
                })
            state["reasoning"].append(f"LLM selected {len(response.tool_calls)} tool(s)")
        else:
            # Pure LLM approach - if no tool calls, retry with more explicit prompt
            state["reasoning"].append("LLM did not return tool calls, retrying with explicit request")
            
            # Retry with more explicit instruction
            retry_prompt = f"""The user asked: {user_query}

You MUST call at least one tool to answer this query. Select the most appropriate tool(s) and call them with proper parameters extracted from the query."""
            
            retry_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=retry_prompt)
            ]
            
            try:
                retry_response = llm_with_tools.invoke(retry_messages)
                if hasattr(retry_response, 'tool_calls') and retry_response.tool_calls:
                    for tool_call in retry_response.tool_calls:
                        state["tool_calls"].append({
                            "tool": tool_call.get("name", ""),
                            "args": tool_call.get("args", {}),
                            "id": tool_call.get("id", "")
                        })
                    state["reasoning"].append(f"LLM selected {len(retry_response.tool_calls)} tool(s) on retry")
                else:
                    state["reasoning"].append("LLM still did not return tool calls after retry")
                    state["tool_calls"] = []  # Let execute_tools_node handle empty tool calls
            except Exception as retry_e:
                state["reasoning"].append(f"Retry failed: {str(retry_e)}")
                state["tool_calls"] = []
    except Exception as e:
        # Pure LLM approach - log error but don't use keyword fallback
        state["reasoning"].append(f"LLM tool selection failed: {str(e)}")
        state["tool_calls"] = []  # Empty tool calls - will be handled gracefully
    
    return state


def execute_tools_node(state: AgentState) -> AgentState:
    """Execute selected tools with proper parameters"""
    tool_calls = state.get("tool_calls", [])
    entities = state.get("entities", {})
    user_query = state.get("user_query", "")
    print(f"\n[DEBUG] NODE: execute_tools_node | Executing {len(tool_calls)} tools")
    log_agent_activity(f"Executing tools for query: '{user_query}'", "info")
    
    tool_results = []

    # Map tool names to actual tool functions
    tool_map = {
        "place_order_tool": place_order_tool,
        "modify_order_tool": modify_order_tool,
        "cancel_order_tool": cancel_order_tool,
        "get_positions_tool": get_positions_tool,
        "get_orders_tool": get_orders_tool,
        "get_balance_tool": get_balance_tool,
        "exit_position_tool": exit_position_tool,
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
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("tool", "")
        tool_args = tool_call.get("args", {})
        print(f"[DEBUG] execute_tools_node | Executing: {tool_name} with args: {tool_args}")
        
        try:
            # Get the actual tool function
            tool_func = tool_map.get(tool_name)
            
            if not tool_func:
                print(f"[DEBUG] execute_tools_node | Error: Unknown tool {tool_name}")
                result = {"status": "error", "error": f"Unknown tool: {tool_name}"}
            else:
                # 1. UNIVERSAL DATE RESOLUTION
                # If tool has 'date' or 'from_date'/'to_date', resolve it from entities
                date_val = tool_args.get("date") or entities.get("date")
                if date_val:
                    from_dt, to_dt = resolve_date_to_range(str(date_val))
                    
                    # If tool expects from_date/to_date (like trading_opportunities)
                    if tool_name == "find_indicator_based_trading_opportunities":
                        tool_args["from_date"] = from_dt
                        tool_args["to_date"] = to_dt
                        if "date" in tool_args: del tool_args["date"]
                    else:
                        # Otherwise, if it's a range but tool only takes 'date', use 'from_dt' (the start of the range)
                        # or just keep it as 'today' if that's what was resolved.
                        tool_args["date"] = from_dt
                
                # 2. INSTRUMENT RESOLUTION (Multiple support & Groups)
                # Check if the instrument is a predefined group first
                from agent.tools.instrument_resolver import INSTRUMENT_GROUPS
                inst_val = tool_args.get("instrument_name") or entities.get("instrument")
                
                if inst_val:
                    inst_lower = str(inst_val).lower().strip()
                    if inst_lower in INSTRUMENT_GROUPS:
                        group_value = INSTRUMENT_GROUPS[inst_lower]
                        # Handle lambda functions for dynamic groups (selected stocks)
                        if callable(group_value):
                            tool_args["instrument_name"] = group_value()
                        else:
                            tool_args["instrument_name"] = group_value
                    elif not tool_args.get("instrument_name") and entities.get("instrument"):
                        inst = entities.get("instrument")
                        if isinstance(inst, list):
                            tool_args["instrument_name"] = inst
                        elif isinstance(inst, str) and (" and " in inst.lower() or "," in inst):
                            import re
                            tool_args["instrument_name"] = [i.strip() for i in re.split(r'\s+and\s+|,', inst, flags=re.IGNORECASE)]
                        else:
                            tool_args["instrument_name"] = inst
                
                # 2.5. DEFAULT TO SELECTED STOCKS if no instrument specified
                if not tool_args.get("instrument_name") and not entities.get("instrument"):
                    # For trading opportunities, analysis, and simulation tools, default to selected stocks
                    if tool_name in ["find_indicator_based_trading_opportunities", "analyze_gap_probability", 
                                   "calculate_indicators_tool", "analyze_trend_tool", "find_indicator_threshold_crossings",
                                   "find_candlestick_patterns", "download_historical_data_to_local_tool"]:
                        from agent.tools.instrument_resolver import get_selected_stocks_cached
                        selected_stocks = get_selected_stocks_cached()
                        if selected_stocks:
                            tool_args["instrument_name"] = selected_stocks
                            print(f"[DEBUG] execute_tools_node | No instrument specified, using selected stocks: {selected_stocks}")
                
                # 3. TOOL-SPECIFIC OVERRIDES
                if tool_name == "find_indicator_based_trading_opportunities":
                    # SMART LOCAL FILE PICKING: If path is generic placeholder OR empty, and requested simulation
                    current_path = tool_args.get("local_data_file", "")
                    if (not current_path or "path_from" in str(current_path)) and "simulation" in user_query.lower():
                        import os
                        sim_dir = "data/simulation"
                        if os.path.exists(sim_dir):
                            files = sorted([f for f in os.listdir(sim_dir) if f.endswith(".json")], reverse=True)
                            if files:
                                tool_args["local_data_file"] = os.path.join(sim_dir, files[0])
                                print(f"[DEBUG] execute_tools_node | Auto-selected local data file: {tool_args['local_data_file']}")
                                
                                # SMART INTERVAL PICKING: Detect '5minute' or 'minute' correctly
                                if "5minute" in files[0]:
                                    tool_args["interval"] = "5minute"
                                    print(f"[DEBUG] execute_tools_node | Auto-selected interval '5minute'")
                                elif "minute" in files[0]:
                                    tool_args["interval"] = "minute"
                                    print(f"[DEBUG] execute_tools_node | Auto-selected interval 'minute'")
                                
                                # SMART INSTRUMENT PICKING: If no instruments provided, extract them from the simulation file
                                if not tool_args.get("instrument_name"):
                                    try:
                                        import json
                                        with open(tool_args["local_data_file"], "r") as f:
                                            sim_json = json.load(f)
                                            sim_instruments = list(sim_json.get("data", {}).keys())
                                            if sim_instruments:
                                                tool_args["instrument_name"] = sim_instruments
                                                print(f"[DEBUG] execute_tools_node | Auto-selected instruments from file: {sim_instruments}")
                                    except Exception as e:
                                        print(f"[DEBUG] execute_tools_node | Error reading instruments from sim file: {e}")

                elif tool_name == "run_simulation_on_local_data_tool":
                    # SMART LOCAL FILE PICKING FOR WRAPPER
                    if not tool_args.get("file_path"):
                        import os
                        sim_dir = "data/simulation"
                        if os.path.exists(sim_dir):
                            files = sorted([f for f in os.listdir(sim_dir) if f.endswith(".json")], reverse=True)
                            if files:
                                tool_args["file_path"] = os.path.join(sim_dir, files[0])
                                print(f"[DEBUG] execute_tools_node | Auto-selected wrapper file: {tool_args['file_path']}")

                    if not tool_args.get("indicators") and entities.get("indicators"):
                        tool_args["indicators"] = entities.get("indicators")
                    if not tool_args.get("conditions") and entities.get("conditions"):
                        tool_args["conditions"] = entities.get("conditions")
                    if "use_risk_reward" not in tool_args:
                        tool_args["use_risk_reward"] = True
                
                elif tool_name == "find_indicator_threshold_crossings":
                    if not tool_args.get("indicator") and entities.get("indicator"):
                        tool_args["indicator"] = entities.get("indicator")
                    if not tool_args.get("threshold") and entities.get("threshold"):
                        try:
                            tool_args["threshold"] = float(entities.get("threshold"))
                        except (ValueError, TypeError): pass
                    if not tool_args.get("direction") and entities.get("direction"):
                        tool_args["direction"] = entities.get("direction")
                
                elif tool_name == "find_candlestick_patterns":
                    if not tool_args.get("instrument_names") and tool_args.get("instrument_name"):
                         # Standardize parameter name
                         tool_args["instrument_names"] = tool_args["instrument_name"] if isinstance(tool_args["instrument_name"], list) else [tool_args["instrument_name"]]
                    
                    if not tool_args.get("pattern") and entities.get("pattern"):
                        tool_args["pattern"] = entities.get("pattern")
                    elif not tool_args.get("pattern"):
                        tool_args["pattern"] = "doji"
                    
                    if "reversal" in user_query.lower() or "reversed" in user_query.lower():
                        tool_args["check_reversal"] = True
                
                # Invoke the tool with arguments
                log_agent_activity(f"Executing tool: {tool_name} with args: {tool_args}", "debug")
                start_time = time.time()
                result = tool_func.invoke(tool_args)
                execution_time = time.time() - start_time

                # Log detailed interaction for file-based debugging
                log_tool_interaction(tool_name, tool_args, result)
                log_agent_activity(f"Tool {tool_name} finished successfully in {execution_time:.2f}s", "debug")
            
            # Save tool execution to database
            try:
                tool_repo = get_tool_repository()
                execution_time = time.time() - start_time if 'start_time' in locals() else 0
                execution = ToolExecution(
                    execution_id=f"{tool_name}_{int(time.time())}",
                    tool_name=tool_name,
                    inputs=tool_args,
                    outputs=result,
                    execution_time=execution_time,
                    success=True,
                    timestamp=datetime.now()
                )
                tool_repo.save(execution)
            except Exception as e:
                print(f"Error saving tool execution: {e}")

            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": result
            })
        except Exception as e:
            print(f"[DEBUG] execute_tools_node | Error executing {tool_name}: {str(e)}")

            # Save failed tool execution to database
            try:
                tool_repo = get_tool_repository()
                execution_time = time.time() - start_time if 'start_time' in locals() else 0
                execution = ToolExecution(
                    execution_id=f"{tool_name}_error_{int(time.time())}",
                    tool_name=tool_name,
                    inputs=tool_args,
                    outputs={"status": "error", "error": str(e)},
                    execution_time=execution_time,
                    success=False,
                    error_message=str(e),
                    timestamp=datetime.now()
                )
                tool_repo.save(execution)
            except Exception as save_error:
                print(f"Error saving failed tool execution: {save_error}")

            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": {"status": "error", "error": str(e)}
            })
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Error executing {tool_name}: {str(e)}")
    
    state["tool_results"] = tool_results
    return state


def risk_assessment_node(state: AgentState) -> AgentState:
    """Assess risk for trading actions"""
    intent = state.get("intent")
    tool_results = state.get("tool_results", [])
    
    if intent != "TRADE":
        state["risk_assessment"] = {"status": "not_required"}
        return state
    
    # Extract risk information from tool results
    risk_info = {}
    for result in tool_results:
        if "risk" in result.get("tool", "").lower():
            risk_info = result.get("result", {})
            break
    
    # Get safety validation
    safety = get_safety_manager()
    
    # Extract trade value and risk amount (would come from tool results)
    trade_value = 0.0  # Would be extracted from tool results
    risk_amount = 0.0  # Would be extracted from tool results
    
    validation = safety.validate_trade(trade_value, risk_amount)
    
    state["risk_assessment"] = {
        "validation": validation,
        "risk_info": risk_info,
        "can_proceed": validation.get("is_valid", False)
    }
    
    return state


def approval_check_node(state: AgentState) -> AgentState:
    """Check if approval is needed"""
    intent = state.get("intent")
    risk_assessment = state.get("risk_assessment", {})
    
    if intent != "TRADE":
        state["requires_approval"] = False
        return state
    
    approval_queue = get_approval_queue()
    
    # Extract trade details (would come from tool results)
    trade_value = 0.0
    risk_amount = 0.0
    
    needs_approval = approval_queue.needs_approval(trade_value, risk_amount)
    state["requires_approval"] = needs_approval
    
    if needs_approval:
        # Create approval request
        approval_id = approval_queue.create_approval(
            action="PLACE_ORDER",
            details={},  # Would contain order details
            trade_value=trade_value,
            risk_amount=risk_amount,
            reasoning="Trade requires approval due to size/risk"
        )
        state["approval_id"] = approval_id
    
    return state


def execute_trade_node(state: AgentState) -> AgentState:
    """Execute approved trades"""
    requires_approval = state.get("requires_approval", False)
    approval_id = state.get("approval_id")
    
    if requires_approval and approval_id:
        # Check if approved
        approval_queue = get_approval_queue()
        approval = approval_queue.get_approval(approval_id)
        
        if not approval or approval["status"] != "APPROVED":
            state["agent_response"] = f"Trade pending approval (ID: {approval_id})"
            return state
    
    # Execute trade (would use place_order_tool)
    # For now, just set response
    state["agent_response"] = "Trade execution would happen here"
    
    return state


def generate_response_node(state: AgentState) -> AgentState:
    """Generate natural language response using LLM"""
    intent = state.get("intent", "QUERY")
    tool_results = state.get("tool_results", [])
    risk_assessment = state.get("risk_assessment", {})
    requires_approval = state.get("requires_approval", False)
    approval_id = state.get("approval_id")
    user_query = state.get("user_query", "")
    
    print(f"\n--- NODE: generate_response_node ---")
    print(f"Intent: {intent}, Tool Results: {len(tool_results)}")
    
    # Handle simple conversation directly to avoid hallucinations
    if intent == "CONVERSATION" and not tool_results:
        print("[DEBUG] generate_response_node | Handling as direct conversation")
        try:
            llm = create_llm()
            current_date = datetime.now().strftime("%Y-%m-%d")
            system_prompt = f"You are a helpful trading assistant. The current real-world date is {current_date}. The user is just greeting you or making small talk. Respond naturally and briefly using Markdown. Do not mention any specific stocks unless the user did."
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
            response = llm.invoke(messages)
            state["agent_response"] = response.content
            return state
        except Exception as e:
            print(f"Conversation response failed: {e}")
            state["agent_response"] = "Hello! How can I help you with your trading today?"
            return state

    # Build response from tool results using LLM
    try:
        llm = create_llm()
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_day = datetime.now().strftime("%A")
        
        # Format tool results for LLM
        results_summary = []
        for result in tool_results:
            tool_name = result.get("tool", "")
            tool_result = result.get("result", {})
            if tool_result.get("status") == "success":
                # Format successful results
                if tool_name in ["find_indicator_based_trading_opportunities", "run_simulation_on_local_data_tool"]:
                    # Handle Sequential Global Analysis (Multi-instrument)
                    if tool_result.get("is_sequential"):
                        total_opps = tool_result.get("total_opportunities", 0)
                        total_pnl = tool_result.get("total_pnl", 0)
                        opportunities = tool_result.get("opportunities", [])
                        
                        results_summary.append(f"**Global Sequential Analysis** across {tool_result.get('instruments_analyzed')} stocks:")
                        results_summary.append(f"- This analysis ensures only one trade is active at a time to respect capital limits.")
                        results_summary.append(f"- Total Trades: {total_opps}, Combined P&L: **₹{total_pnl:.2f}**")
                        
                        if opportunities:
                            results_summary.append("\n### Chronological Trade List (Sequential Capital Use)")
                            results_summary.append("| # | Stock | Type | Qty | Entry Price | Entry Time | Exit Price | Exit Time | P&L | Available Funds |")
                            results_summary.append("|---|-------|------|-----|-------------|------------|------------|-----------|-----|-----------------|")
                            for i, opp in enumerate(opportunities, 1):
                                results_summary.append(f"| {i} | {opp.get('instrument')} | {opp.get('signal_type')} | {opp.get('suggested_quantity')} | {opp.get('entry_price'):.2f} | {opp.get('entry_time')} | {opp.get('exit_price'):.2f} | {opp.get('exit_time')} | ₹{opp.get('pnl'):.2f} | **₹{opp.get('available_funds'):.2f}** |")
                        else:
                            results_summary.append(f"No trades were possible across the group of {tool_result.get('instruments_analyzed')} stocks during this period based on our strict VWAP criteria.")
                    
                    # Handle siloed results (fallback or single instrument)
                    elif "results" in tool_result:
                        # Multiple instruments
                        results_data = tool_result.get("results", {})
                        total_opps = tool_result.get("total_opportunities", 0)
                        total_pnl = tool_result.get("total_pnl", 0)
                        indicators = tool_result.get("indicators", [])
                        results_summary.append(f"Found {total_opps} total opportunities across {tool_result.get('instruments_analyzed', 0)} instrument(s) using {', '.join(indicators)}:")
                        results_summary.append(f"  Total P&L: ₹{total_pnl:.2f}")
                        for inst_name, inst_result in results_data.items():
                            if isinstance(inst_result, dict) and inst_result.get("status") == "success":
                                summary = inst_result.get("summary", {})
                                inst_opps = inst_result.get("opportunities", [])
                                results_summary.append(f"\n- {inst_result.get('instrument', inst_name)}: {summary.get('total_opportunities', 0)} opportunities, P&L: ₹{summary.get('total_pnl', 0):.2f}")
                                if inst_opps:
                                    results_summary.append(f"  Trades: {inst_opps}")
                        # Single instrument
                        opportunities = tool_result.get("opportunities", [])
                        summary = tool_result.get("summary", {})
                        indicators = tool_result.get("indicators", [])
                        conditions = tool_result.get("conditions", [])
                        
                        if opportunities:
                            results_summary.append(f"Found {summary.get('total_opportunities', 0)} trading opportunities using {', '.join(indicators)}:")
                            results_summary.append(f"  Conditions: {', '.join(conditions)}")
                            
                            # Add Detailed Trades Table for LLM to format
                            results_summary.append("\nDetailed Trade List:")
                            results_summary.append("| # | Type | Qty | Entry Price | Entry Time | Exit Price | Exit Time | P&L | Available Funds |")
                            results_summary.append("|---|------|-----|-------------|------------|------------|-----------|-----|-----------------|")
                            for i, opp in enumerate(opportunities, 1):
                                pnl = opp.get('pnl', 0)
                                results_summary.append(f"| {i} | {opp.get('signal_type')} | {opp.get('suggested_quantity')} | {opp.get('entry_price'):.2f} | {opp.get('entry_time')} | {opp.get('exit_price'):.2f} | {opp.get('exit_time')} | ₹{pnl:.2f} | **₹{opp.get('available_funds'):.2f}** |")
                            
                            results_summary.append("\nStrategy Performance Summary:")
                            results_summary.append(f"  Win Rate: {summary.get('win_rate', 0):.1f}% ({summary.get('winning_trades', 0)} wins, {summary.get('losing_trades', 0)} losses)")
                            results_summary.append(f"  Total P&L: ₹{summary.get('total_pnl', 0):.2f} ({summary.get('total_pnl_percent', 0):.2f}%)")
                            results_summary.append(f"  Average P&L per trade: ₹{summary.get('avg_pnl_per_trade', 0):.2f}")
                            results_summary.append(f"  Average Risk/Reward Ratio: {summary.get('avg_risk_reward_ratio', 0):.2f}:1")
                        else:
                            results_summary.append(summary.get("message", "No opportunities found"))
                elif tool_name == "find_indicator_threshold_crossings":
                    # Handle single or multiple instruments
                    if "results" in tool_result:
                        # Multiple instruments
                        results_data = tool_result.get("results", {})
                        total_crossings = tool_result.get("total_crossings", 0)
                        results_summary.append(f"Found {total_crossings} total crossings across {tool_result.get('instruments_analyzed', 0)} instrument(s):")
                        for inst_name, inst_result in results_data.items():
                            if isinstance(inst_result, dict) and inst_result.get("status") == "success":
                                crossings = inst_result.get("crossings", [])
                                results_summary.append(f"  {inst_result.get('instrument', inst_name)}: {len(crossings)} crossings")
                    else:
                        # Single instrument
                        crossings = tool_result.get("crossings", [])
                        if crossings:
                            results_summary.append(f"Found {len(crossings)} times when {tool_result.get('indicator')} crossed {tool_result.get('threshold')} {tool_result.get('direction')}:")
                            for crossing in crossings:  # No limit
                                results_summary.append(f"  - {crossing.get('timestamp')}: RSI={crossing.get('rsi'):.2f}, Price={crossing.get('close'):.2f}")
                        else:
                            results_summary.append(f"No crossings found for {tool_result.get('indicator')} {tool_result.get('direction')} {tool_result.get('threshold')} on {tool_result.get('date')}")
                elif tool_name == "find_candlestick_patterns":
                    results_data = tool_result.get("results", {})
                    pattern = tool_result.get("pattern", "")
                    total_patterns = tool_result.get("total_patterns_found", 0)
                    total_reversals = tool_result.get("total_reversals", 0)
                    
                    results_summary.append(f"Found {total_patterns} {pattern} pattern(s) with {total_reversals} reversal(s):")
                    for inst_name, inst_result in results_data.items():
                        if isinstance(inst_result, dict) and inst_result.get("status") == "success":
                            patterns = inst_result.get("patterns_found", [])
                            reversals = inst_result.get("reversals_count", 0)
                            results_summary.append(f"  {inst_result.get('instrument', inst_name)}: {inst_result.get('count', 0)} patterns, {reversals} reversals")
                            for pattern_info in patterns:
                                if pattern_info.get("reversal"):
                                    rev = pattern_info["reversal"]
                                    results_summary.append(f"    - {pattern_info.get('pattern')} at {pattern_info.get('timestamp')} → {rev.get('reversal_type')} reversal")
                elif tool_name == "analyze_gap_probability":
                    # Handle single or multiple instruments
                    if "results" in tool_result:
                        # Multiple instruments
                        results_data = tool_result.get("results", {})
                        results_summary.append(f"Gap Analysis for {tool_result.get('instruments_analyzed', 0)} instruments:")
                        for inst_name, inst_result in results_data.items():
                            if isinstance(inst_result, dict) and inst_result.get("status") == "success":
                                actual = inst_result.get("actual_gap_today")
                                prob = inst_result.get("gap_analysis", {})
                                if actual:
                                    results_summary.append(f"  {inst_result.get('instrument', inst_name)}: Actual Today: {actual.get('direction')} ({actual.get('gap_percentage', 0):.2f}%)")
                                results_summary.append(f"  {inst_result.get('instrument', inst_name)}: Probabilistic Prediction for Next Open: {prob.get('likely_direction')} ({prob.get('probability', 0):.1f}%)")
                    else:
                        # Single instrument
                        actual = tool_result.get("actual_gap_today")
                        prob = tool_result.get("gap_analysis", {})
                        if actual:
                            results_summary.append(f"Actual Gap Today for {tool_result.get('instrument')}: {actual.get('direction')} of {actual.get('gap_percentage', 0):.2f}% (Open: {actual.get('open')}, Prev Close: {actual.get('prev_close')})")
                        
                        results_summary.append(f"Probabilistic Prediction for Next Open for {tool_result.get('instrument')}:")
                        results_summary.append(f"  Likely Direction: {prob.get('likely_direction')}")
                        results_summary.append(f"  Probability: {prob.get('probability', 0):.1f}%")
                        results_summary.append(f"  Expected Gap: {prob.get('estimated_gap_percentage', 0):.2f}%")
                        results_summary.append(f"  Based on {prob.get('total_days_analyzed')} days of history ({prob.get('gap_up_days')} gap up, {prob.get('gap_down_days')} gap down)")
                else:
                    results_summary.append(f"{tool_name}: {tool_result}")
            else:
                results_summary.append(f"{tool_name} failed: {tool_result.get('error', 'Unknown error')}")
        
        system_prompt = f"""You are a professional trading assistant. 
The current real-world date is {current_date} ({current_day}).
Generate a professional response based on the tool results.
WE ONLY USE THE INSTITUTIONAL VWAP STRATEGY FOR ALL TRADE DECISIONS.

STRICT INSTRUCTIONS:
1. If the results contain multiple instruments OR a sequential simulation, provide a combined strategy summary followed by the full chronological trade table.
2. For sequential simulations, you MUST include the "Available Funds" column to show the compounding effect.
3. If no trades were found, explain that the VWAP setup was not triggered during the specific window (9:45 AM - 3:00 PM).
4. Use **bold** for important values (prices, symbols, P&L).
5. CRITICAL: Always include the full trade details (Entry, Exit, P&L) in a table if trades are present. 
6. DO NOT TRIM OR TRUNCATE THE TABLE. YOU MUST LIST EVERY SINGLE TRADE FOUND.
7. Use professional trading terminology."""
        
        results_summary_str = chr(10).join(results_summary) if results_summary else 'No tool results'
        user_prompt = f"""User asked: {user_query}

Tool Results:
{results_summary_str}

Generate a helpful response. 
CRITICAL: If the tool results contain a detailed trade table or data list, you MUST include that table/list in your final response. 
Do not summarize it into text; keep the tabular structure. 
You can add a brief summary before and after the table."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        agent_response = response.content
        
        # MANDATORY TABLE OVERRIDE: If the LLM somehow forgot the table, append it manually
        if "### Chronological Trade List" in results_summary_str and "### Chronological Trade List" not in agent_response:
            print("[DEBUG] generate_response_node | LLM omitted table, appending manually")
            # Extract the table part from results_summary
            table_part = results_summary_str.split("### Chronological Trade List")[1]
            agent_response += "\n\n### Chronological Trade List" + table_part
        elif "Detailed Trade List:" in results_summary_str and "Detailed Trade List:" not in agent_response:
            print("[DEBUG] generate_response_node | LLM omitted table, appending manually")
            table_part = results_summary_str.split("Detailed Trade List:")[1]
            agent_response += "\n\nDetailed Trade List:" + table_part

        # Add approval notice if needed
        if requires_approval:
            agent_response += f"\n\n⚠️ This action requires your approval. Approval ID: {approval_id}"
        
        state["agent_response"] = agent_response
        
    except Exception as e:
        # Fallback to simple response
        response_parts = []
        for result in tool_results:
            tool_result = result.get("result", {})
            if tool_result.get("status") == "success":
                tool_name = result.get("tool", "")
                if tool_name in ["find_indicator_based_trading_opportunities", "run_simulation_on_local_data_tool"]:
                    # Handle Sequential Global Analysis (Multi-instrument)
                    if tool_result.get("is_sequential"):
                        total_opps = tool_result.get("total_opportunities", 0)
                        total_pnl = tool_result.get("total_pnl", 0)
                        opportunities = tool_result.get("opportunities", [])
                        
                        response_parts.append(f"**Global Sequential Analysis** across {tool_result.get('instruments_analyzed')} stocks:")
                        response_parts.append(f"  • Total Trades: {total_opps}")
                        response_parts.append(f"  • Combined P&L: **₹{total_pnl:.2f}**")
                        
                        if opportunities:
                            response_parts.append("\n### Chronological Trade List (Sequential Capital Use)")
                            response_parts.append("| # | Stock | Type | Qty | Entry Price | Entry Time | Exit Price | Exit Time | P&L | Available Funds |")
                            response_parts.append("|---|-------|------|-----|-------------|------------|------------|-----------|-----|-----------------|")
                            for i, opp in enumerate(opportunities, 1):
                                response_parts.append(f"| {i} | {opp.get('instrument')} | **{opp.get('signal_type')}** | {opp.get('suggested_quantity')} | {opp.get('entry_price'):.2f} | {opp.get('entry_time')} | {opp.get('exit_price'):.2f} | {opp.get('exit_time')} | ₹{opp.get('pnl'):.2f} | **₹{opp.get('available_funds'):.2f}** |")
                        else:
                            response_parts.append(f"No sequential trades found.")
                    
                    # Handle siloed results (fallback or single instrument)
                    elif "results" in tool_result:
                        # Multiple instruments
                        results_data = tool_result.get("results", {})
                        total_opps = tool_result.get("total_opportunities", 0)
                        total_pnl = tool_result.get("total_pnl", 0)
                        indicators = tool_result.get("indicators", [])
                        conditions = tool_result.get("conditions", [])
                        response_parts.append(f"Found {total_opps} total opportunities across {tool_result.get('instruments_analyzed', 0)} instrument(s) using {', '.join(indicators)}:")
                        response_parts.append(f"  Total P&L: ₹{total_pnl:.2f}")
                        for inst_name, inst_result in results_data.items():
                            if isinstance(inst_result, dict) and inst_result.get("status") == "success":
                                summary = inst_result.get("summary", {})
                                opps = inst_result.get("opportunities", [])
                                response_parts.append(f"\n{inst_result.get('instrument', inst_name)}: {summary.get('total_opportunities', 0)} opportunities, P&L: ₹{summary.get('total_pnl', 0):.2f}")
                    else:
                        # Single instrument
                        opportunities = tool_result.get("opportunities", [])
                        summary = tool_result.get("summary", {})
                        indicators = tool_result.get("indicators", [])
                        conditions = tool_result.get("conditions", [])
                        
                        if opportunities:
                            response_parts.append(f"Found {summary.get('total_opportunities', 0)} trading opportunities using {', '.join(indicators)}:")
                            response_parts.append(f"  Conditions: {', '.join(conditions)}")
                            
                            # Add Detailed Trades Table
                            response_parts.append("\n### Detailed Trade Log")
                            response_parts.append("| # | Type | Qty | Entry (Price/Time) | Exit (Price/Time) | P&L (₹) | Available Funds |")
                            response_parts.append("|---|------|-----|-------------------|-------------------|---------|-----------------|")
                            for i, opp in enumerate(opportunities, 1):
                                entry = f"₹{opp.get('entry_price'):.2f}<br>{opp.get('entry_time')}"
                                exit = f"₹{opp.get('exit_price'):.2f}<br>{opp.get('exit_time')}"
                                pnl = f"₹{opp.get('pnl'):.2f}"
                                funds = f"**₹{opp.get('available_funds'):.2f}**"
                                response_parts.append(f"| {i} | **{opp.get('signal_type')}** | {opp.get('suggested_quantity')} | {entry} | {exit} | {pnl} | {funds} |")

                            # Mention created approvals for group analysis
                            response_parts.append(f"\n> 💡 **Simulation Note**: I have generated **{len(opportunities)} simulated approval requests** in the Approvals queue for this group. You can review the chronological sequence in the Approvals tab.")

                            response_parts.append("\n### Strategy Performance Summary")
                            response_parts.append(f"  • **Win Rate**: {summary.get('win_rate', 0):.1f}% ({summary.get('winning_trades', 0)} wins, {summary.get('losing_trades', 0)} losses)")
                            response_parts.append(f"  • **Total P&L**: ₹{summary.get('total_pnl', 0):.2f} ({summary.get('total_pnl_percent', 0):.2f}%)")
                            response_parts.append(f"  • **Average**: ₹{summary.get('avg_pnl_per_trade', 0):.2f} per trade")
                            response_parts.append(f"  • **Avg Risk/Reward**: {summary.get('avg_risk_reward_ratio', 0):.2f}:1")
                            
                            # Mention created approvals
                            response_parts.append(f"\n> 💡 **Simulation Note**: I have generated **{len(opportunities)} simulated approval requests** in the Approvals queue. You can review them to see exactly how these trades would have been presented in the live market.")
                        else:
                            response_parts.append(summary.get("message", "No opportunities found"))
                elif tool_name == "find_indicator_threshold_crossings":
                    # Handle single or multiple instruments
                    if "results" in tool_result:
                        # Multiple instruments
                        results_data = tool_result.get("results", {})
                        total_crossings = tool_result.get("total_crossings", 0)
                        response_parts.append(f"Found {total_crossings} total crossings across {tool_result.get('instruments_analyzed', 0)} instrument(s):")
                        for inst_name, inst_result in results_data.items():
                            if isinstance(inst_result, dict) and inst_result.get("status") == "success":
                                crossings = inst_result.get("crossings", [])
                                response_parts.append(f"\n{inst_result.get('instrument', inst_name)}: {len(crossings)} crossings")
                                for crossing in crossings[:3]:
                                    response_parts.append(f"  • {crossing.get('timestamp')}: RSI={crossing.get('rsi'):.2f}, Price=₹{crossing.get('close'):.2f}")
                    else:
                        # Single instrument
                        crossings = tool_result.get("crossings", [])
                        if crossings:
                            response_parts.append(f"Found {len(crossings)} time(s) when {tool_result.get('indicator')} reached {tool_result.get('direction')} {tool_result.get('threshold')}:")
                            for crossing in crossings[:5]:
                                response_parts.append(f"  • {crossing.get('timestamp')}: RSI={crossing.get('rsi'):.2f}, Price=₹{crossing.get('close'):.2f}")
                        else:
                            response_parts.append(f"No times found when {tool_result.get('indicator')} reached {tool_result.get('direction')} {tool_result.get('threshold')} on {tool_result.get('date')}")
                elif tool_name == "find_candlestick_patterns":
                    results_data = tool_result.get("results", {})
                    pattern = tool_result.get("pattern", "")
                    total_patterns = tool_result.get("total_patterns_found", 0)
                    total_reversals = tool_result.get("total_reversals", 0)
                    
                    if total_patterns > 0:
                        response_parts.append(f"Found {total_patterns} {pattern} pattern(s) with {total_reversals} reversal(s):")
                        for inst_name, inst_result in results_data.items():
                            if isinstance(inst_result, dict) and inst_result.get("status") == "success":
                                patterns = inst_result.get("patterns_found", [])
                                reversals = inst_result.get("reversals_count", 0)
                                response_parts.append(f"\n{inst_result.get('instrument', inst_name)}:")
                                response_parts.append(f"  • {inst_result.get('count', 0)} {pattern} pattern(s), {reversals} reversal(s)")
                                for i, pattern_info in enumerate(patterns, 1):
                                    response_parts.append(f"\n  {i}. {pattern_info.get('pattern')} at {pattern_info.get('timestamp')}")
                                    response_parts.append(f"     Price: ₹{pattern_info.get('close', 0):.2f}")
                                    if pattern_info.get("reversal"):
                                        rev = pattern_info["reversal"]
                                        response_parts.append(f"     ✓ {rev.get('reversal_type')} Reversal after {rev.get('candles_after')} candle(s)")
                                        response_parts.append(f"     Reversal Price: ₹{rev.get('reversal_price', 0):.2f} ({rev.get('price_change_pct', 0):.2f}%)")
                    else:
                        response_parts.append(f"No {pattern} patterns found in the specified instruments")
                elif tool_name == "analyze_gap_probability":
                    # Handle single or multiple instruments
                    if "results" in tool_result:
                        # Multiple instruments
                        results_data = tool_result.get("results", {})
                        response_parts.append(f"Gap Analysis for {tool_result.get('instruments_analyzed', 0)} instruments:")
                        for inst_name, inst_result in results_data.items():
                            if isinstance(inst_result, dict) and inst_result.get("status") == "success":
                                actual = inst_result.get("actual_gap_today")
                                prob = inst_result.get("gap_analysis", {})
                                if actual:
                                    response_parts.append(f"  • {inst_result.get('instrument', inst_name)}: Actual Today {actual.get('direction')} ({actual.get('gap_percentage', 0):.2f}%)")
                                response_parts.append(f"  • {inst_result.get('instrument', inst_name)}: Probabilistic Prediction {prob.get('likely_direction')} ({prob.get('probability', 0):.1f}%)")
                    else:
                        # Single instrument
                        actual = tool_result.get("actual_gap_today")
                        prob = tool_result.get("gap_analysis", {})
                        if actual:
                            response_parts.append(f"Actual Gap Today for {tool_result.get('instrument')}: {actual.get('direction')} of {actual.get('gap_percentage', 0):.2f}%")
                        
                        response_parts.append(f"\nProbabilistic Prediction for Next Open for {tool_result.get('instrument')}:")
                        response_parts.append(f"  • Likely Direction: {prob.get('likely_direction')}")
                        response_parts.append(f"  • Probability: {prob.get('probability', 0):.1f}%")
                        response_parts.append(f"  • Expected Gap: {prob.get('estimated_gap_percentage', 0):.2f}%")
                elif tool_name == "download_historical_data_to_local_tool":
                    response_parts.append(f"✓ **Data Download Successful**")
                    response_parts.append(f"  • File: `{tool_result.get('file_path')}`")
                    response_parts.append(f"  • Instruments: {', '.join(tool_result.get('instruments', []))}")
                    response_parts.append(f"  • Total Candles: {tool_result.get('total_candles')}")
                elif tool_name == "run_simulation_on_local_data_tool":
                    response_parts.append(f"✓ **Simulation Loaded**")
                    response_parts.append(f"  • File: `{tool_result.get('metadata', {}).get('file_path') or 'Local Data'}`")
                    response_parts.append(f"  • Instruments: {', '.join(tool_result.get('instruments', []))}")
                    response_parts.append(f"  • Instruction: {tool_result.get('instruction')}")
                else:
                    response_parts.append(f"✓ {result.get('tool')} completed")
            else:
                response_parts.append(f"✗ {result.get('tool')} failed: {tool_result.get('error', 'Unknown error')}")
        
        if requires_approval:
            response_parts.append(f"\n⚠️ Trade requires approval (ID: {approval_id})")
        
        state["agent_response"] = "\n".join(response_parts) if response_parts else "No action taken"
        state["errors"] = state.get("errors", [])
        state["errors"].append(f"LLM response generation failed: {str(e)}")
    
    return state

