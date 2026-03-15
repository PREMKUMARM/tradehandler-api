"""
Zerodha MCP Server Integration for AlgoFeast
Works alongside existing AlgoFeast backend to enhance chat capabilities
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys

# MCP imports
try:
    from mcp.server import Server
    from mcp.types import Tool, Resource, TextContent
    from mcp.server.stdio import stdio_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not installed. Install with: pip install mcp")

# Existing AlgoFeast imports
from utils.kite_utils import get_kite_instance
from utils.trade_limits import trade_limits
from utils.order_monitor import order_monitor
from utils.strategy_executor import strategy_executor
from core.exceptions import AlgoFeastException
from utils.logger import log_info, log_error


class ZerodhaMCPServer:
    """Zerodha MCP Server that integrates with AlgoFeast backend"""
    
    def __init__(self):
        if not MCP_AVAILABLE:
            raise ImportError("MCP library not installed")
        
        self.server = Server("zerodha-algofeast")
        self.kite = None
        self.setup_tools()
        self.setup_resources()
    
    def setup_tools(self):
        """Setup MCP tools for Zerodha trading operations"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                # Market Data Tools
                Tool(
                    name="get_market_price",
                    description="Get current market price for a symbol",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Trading symbol (e.g., RELIANCE, NIFTY 50)"},
                            "exchange": {"type": "string", "enum": ["NSE", "BSE"], "default": "NSE"}
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="get_portfolio",
                    description="Get current portfolio positions and balance",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_positions",
                    description="Get detailed current positions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "product": {"type": "string", "enum": ["MIS", "CNC", "NRML"], "description": "Filter by product type"}
                        }
                    }
                ),
                
                # Trading Tools
                Tool(
                    name="place_order",
                    description="Place a trading order with optional stoploss and target",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Trading symbol"},
                            "transaction_type": {"type": "string", "enum": ["BUY", "SELL"]},
                            "quantity": {"type": "integer", "minimum": 1},
                            "order_type": {"type": "string", "enum": ["MARKET", "LIMIT", "SL", "SL-M"], "default": "MARKET"},
                            "product": {"type": "string", "enum": ["MIS", "CNC", "NRML"], "default": "MIS"},
                            "price": {"type": "number", "description": "Price for LIMIT orders"},
                            "trigger_price": {"type": "number", "description": "Trigger price for SL orders"},
                            "stoploss": {"type": "number", "description": "Stoploss price for automatic exit"},
                            "target": {"type": "number", "description": "Target price for automatic exit"},
                            "trailing_stoploss": {"type": "number", "description": "Trailing stoploss amount"}
                        },
                        "required": ["symbol", "transaction_type", "quantity"]
                    }
                ),
                Tool(
                    name="modify_order",
                    description="Modify an existing order",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "price": {"type": "number"},
                            "trigger_price": {"type": "number"}
                        },
                        "required": ["order_id"]
                    }
                ),
                Tool(
                    name="cancel_order",
                    description="Cancel an existing order",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"}
                        },
                        "required": ["order_id"]
                    }
                ),
                
                # Account Tools
                Tool(
                    name="get_balance",
                    description="Get account balance and margins",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_order_history",
                    description="Get order history",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "default": 50},
                            "status": {"type": "string", "enum": ["OPEN", "COMPLETE", "CANCELLED"]}
                        }
                    }
                ),
                
                # Trade Limits Tools
                Tool(
                    name="get_trade_limits",
                    description="Get current trade limits status",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="set_trade_limits",
                    description="Configure daily trade limits",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "max_trades_per_day": {"type": "integer", "minimum": 1},
                            "max_profit_per_day_pct": {"type": "number", "minimum": 0, "maximum": 1},
                            "max_loss_per_day_pct": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                ),
                
                # Strategy Tools
                Tool(
                    name="start_strategy",
                    description="Start automated trading strategy",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "strategy_type": {"type": "string", "enum": [
                                "915_candle_break", "mean_reversion", "momentum_breakout",
                                "support_resistance", "rsi_reversal", "macd_crossover", "ema_cross"
                            ]},
                            "symbol": {"type": "string"},
                            "quantity": {"type": "integer", "minimum": 1},
                            "product": {"type": "string", "enum": ["MIS", "CNC", "NRML"]},
                            "stoploss_pct": {"type": "number", "minimum": 0, "maximum": 1},
                            "target_pct": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["strategy_type", "symbol", "quantity"]
                    }
                ),
                Tool(
                    name="get_strategy_status",
                    description="Get status of all trading strategies",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                # Initialize Kite connection if needed
                if not self.kite:
                    self.kite = get_kite_instance()
                
                result = await self.handle_tool_call(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                
            except Exception as e:
                log_error(f"MCP tool error ({name}): {e}")
                error_response = {
                    "error": str(e),
                    "tool": name,
                    "timestamp": datetime.now().isoformat()
                }
                return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    def setup_resources(self):
        """Setup MCP resources"""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="zerodha://portfolio",
                    name="Current Portfolio",
                    description="Current trading portfolio with positions and P&L",
                    mimeType="application/json"
                ),
                Resource(
                    uri="zerodha://orders",
                    name="Order History",
                    description="Recent trading orders and their status",
                    mimeType="application/json"
                ),
                Resource(
                    uri="zerodha://limits",
                    name="Trade Limits",
                    description="Current trade limits and usage",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource data"""
            try:
                if not self.kite:
                    self.kite = get_kite_instance()
                
                if uri == "zerodha://portfolio":
                    positions = self.kite.positions()
                    margins = self.kite.margins()
                    return json.dumps({
                        "positions": positions,
                        "margins": margins,
                        "timestamp": datetime.now().isoformat()
                    }, default=str)
                
                elif uri == "zerodha://orders":
                    orders = self.kite.orders()
                    return json.dumps({
                        "orders": orders,
                        "timestamp": datetime.now().isoformat()
                    }, default=str)
                
                elif uri == "zerodha://limits":
                    limits_status = trade_limits.get_limits_status()
                    return json.dumps(limits_status, default=str)
                
                else:
                    raise ValueError(f"Unknown resource: {uri}")
                    
            except Exception as e:
                log_error(f"Resource read error ({uri}): {e}")
                return json.dumps({"error": str(e)})
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle individual tool calls"""
        
        # Market Data Tools
        if name == "get_market_price":
            symbol = arguments["symbol"]
            exchange = arguments.get("exchange", "NSE")
            
            # Get instrument token
            instruments = self.kite.instruments(exchange)
            instrument = next((inst for inst in instruments if inst["tradingsymbol"] == symbol), None)
            
            if not instrument:
                raise ValueError(f"Instrument {symbol} not found")
            
            # Get quote
            quote = self.kite.quote(instrument["instrument_token"])
            return {
                "symbol": symbol,
                "price": quote[instrument["instrument_token"]]["last_price"],
                "timestamp": datetime.now().isoformat()
            }
        
        elif name == "get_portfolio":
            positions = self.kite.positions()
            margins = self.kite.margins()
            
            # Transform positions
            all_positions = positions.get("net", []) + positions.get("day", [])
            transformed_positions = []
            
            for pos in all_positions:
                transformed_positions.append({
                    "symbol": pos.get("tradingsymbol"),
                    "quantity": pos.get("quantity"),
                    "average_price": pos.get("average_price"),
                    "last_price": pos.get("last_price"),
                    "pnl": pos.get("pnl"),
                    "product": pos.get("product")
                })
            
            return {
                "positions": transformed_positions,
                "margins": margins,
                "timestamp": datetime.now().isoformat()
            }
        
        elif name == "get_positions":
            positions = self.kite.positions()
            product_filter = arguments.get("product")
            
            all_positions = positions.get("net", []) + positions.get("day", [])
            if product_filter:
                all_positions = [pos for pos in all_positions if pos.get("product") == product_filter]
            
            return {
                "positions": all_positions,
                "timestamp": datetime.now().isoformat()
            }
        
        # Trading Tools
        elif name == "place_order":
            # Check trade limits first
            limits_status = trade_limits.get_limits_status()
            if not limits_status.get("can_trade", True):
                raise ValueError("Daily trade limits reached")
            
            # Place order using existing AlgoFeast logic
            order_params = {
                "tradingsymbol": arguments["symbol"],
                "exchange": "NSE",
                "transaction_type": arguments["transaction_type"],
                "quantity": arguments["quantity"],
                "order_type": arguments["order_type"],
                "product": arguments["product"]
            }
            
            # Add optional parameters
            if "price" in arguments:
                order_params["price"] = arguments["price"]
            if "trigger_price" in arguments:
                order_params["trigger_price"] = arguments["trigger_price"]
            
            # Place order
            order_id = self.kite.place_order(**order_params)
            
            # Add to order monitor if stoploss/target specified
            if "stoploss" in arguments or "target" in arguments:
                await order_monitor.add_order({
                    "order_id": order_id,
                    "symbol": arguments["symbol"],
                    "transaction_type": arguments["transaction_type"],
                    "quantity": arguments["quantity"],
                    "stoploss": arguments.get("stoploss"),
                    "target": arguments.get("target"),
                    "trailing_stoploss": arguments.get("trailing_stoploss"),
                    "created_at": datetime.now()
                })
            
            # Record trade
            trade_limits.record_trade(order_id, arguments["quantity"])
            
            return {
                "order_id": order_id,
                "status": "placed",
                "timestamp": datetime.now().isoformat()
            }
        
        elif name == "modify_order":
            order_id = arguments["order_id"]
            modify_params = {}
            
            for param in ["quantity", "price", "trigger_price"]:
                if param in arguments:
                    modify_params[param] = arguments[param]
            
            self.kite.modify_order(order_id=order_id, **modify_params)
            
            return {
                "order_id": order_id,
                "status": "modified",
                "timestamp": datetime.now().isoformat()
            }
        
        elif name == "cancel_order":
            order_id = arguments["order_id"]
            self.kite.cancel_order(order_id=order_id)
            
            # Remove from order monitor
            await order_monitor.remove_order(order_id)
            
            return {
                "order_id": order_id,
                "status": "cancelled",
                "timestamp": datetime.now().isoformat()
            }
        
        # Account Tools
        elif name == "get_balance":
            margins = self.kite.margins()
            return {
                "balance": margins,
                "timestamp": datetime.now().isoformat()
            }
        
        elif name == "get_order_history":
            orders = self.kite.orders()
            
            # Apply filters
            status_filter = arguments.get("status")
            limit = arguments.get("limit", 50)
            
            if status_filter:
                orders = [order for order in orders if order.get("status") == status_filter]
            
            orders = orders[:limit]
            
            return {
                "orders": orders,
                "timestamp": datetime.now().isoformat()
            }
        
        # Trade Limits Tools
        elif name == "get_trade_limits":
            return trade_limits.get_limits_status()
        
        elif name == "set_trade_limits":
            config = {}
            
            for param in ["max_trades_per_day", "max_profit_per_day_pct", "max_loss_per_day_pct"]:
                if param in arguments:
                    config[param] = arguments[param]
            
            trade_limits.configure_limits(config)
            
            return {
                "status": "configured",
                "config": config,
                "timestamp": datetime.now().isoformat()
            }
        
        # Strategy Tools
        elif name == "start_strategy":
            strategy_config = {
                "strategy_type": arguments["strategy_type"],
                "symbol": arguments["symbol"],
                "quantity": arguments["quantity"],
                "product": arguments["product"],
                "stoploss_pct": arguments.get("stoploss_pct", 0.02),
                "target_pct": arguments.get("target_pct", 0.04),
                "enabled": True
            }
            
            strategy_id = await strategy_executor.add_strategy(strategy_config)
            
            return {
                "strategy_id": strategy_id,
                "status": "started",
                "timestamp": datetime.now().isoformat()
            }
        
        elif name == "get_strategy_status":
            return strategy_executor.get_strategy_status()
        
        else:
            raise ValueError(f"Unknown tool: {name}")


async def main():
    """Main MCP server function"""
    if not MCP_AVAILABLE:
        print("Error: MCP library not installed")
        print("Install with: pip install mcp")
        sys.exit(1)
    
    try:
        server = ZerodhaMCPServer()
        
        # Run the MCP server
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except Exception as e:
        log_error(f"MCP server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
