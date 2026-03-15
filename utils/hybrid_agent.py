"""
Hybrid Agent Handler: Combines AlgoFeast agent with Zerodha MCP
Routes chat prompts between existing AlgoFeast logic and MCP tools using smart intent classification
"""
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

# Existing AlgoFeast imports
from utils.logger import log_info, log_error, log_warning
from utils.intent_classifier import intent_classifier


class HybridAgentHandler:
    """Hybrid agent that combines AlgoFeast and MCP capabilities"""
    
    def __init__(self):
        self.mcp_available = self._check_mcp_availability()
        self.routing_rules = self._setup_routing_rules()
        
    def _check_mcp_availability(self) -> bool:
        """Check if MCP server is available"""
        try:
            # Try importing MCP components
            import mcp
            return True
        except Exception as e:
            log_warning(f"MCP availability check failed: {e}")
            return False
    
    def _setup_routing_rules(self) -> Dict[str, str]:
        """Setup routing rules for different types of prompts"""
        return {
            # Direct trading commands -> MCP
            "place_order": "mcp",
            "buy": "mcp", 
            "sell": "mcp",
            "cancel": "mcp",
            "modify": "mcp",
            
            # Portfolio queries -> MCP
            "portfolio": "mcp",
            "positions": "mcp",
            "balance": "mcp",
            "holdings": "mcp",
            
            # Market data -> MCP
            "price": "mcp",
            "quote": "mcp",
            "market": "mcp",
            
            # Strategy automation -> MCP
            "start strategy": "mcp",
            "automated": "mcp",
            "execute": "mcp",
            
            # Analysis and advice -> AlgoFeast
            "analyze": "algofeast",
            "recommend": "algofeast",
            "suggest": "algofeast",
            "what should": "algofeast",
            "advice": "algofeast",
            
            # Complex queries -> AlgoFeast
            "why": "algofeast",
            "how": "algofeast",
            "explain": "algofeast",
            "compare": "algofeast",
            
            # Conversational -> AlgoFeast
            "hello": "algofeast",
            "hi": "algofeast",
            "thanks": "algofeast"
        }
    
    async def process_prompt(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user prompt using smart intent classification"""
        
        try:
            # Use smart intent classification
            classification = await intent_classifier.classify_intent(user_query, context)
            
            intent = classification.get("intent", "GENERAL_INFO")
            confidence = classification.get("confidence", 0)
            tool = classification.get("tool")
            entities = classification.get("entities", {})
            parameters = classification.get("parameters", {})
            
            log_info(f"Smart Classification: Intent='{intent}' (confidence: {confidence:.2f}), Tool='{tool}'")
            
            # Determine routing based on classification
            if intent_classifier.should_route_to_mcp(classification):
                log_info(f"Routing to MCP: {user_query[:50]}...")
                return await self._process_with_mcp_smart(user_query, classification, context)
            else:
                log_info(f"Routing to AlgoFeast: {user_query[:50]}...")
                return await self._process_with_algofeast(user_query, context)
                
        except Exception as e:
            log_error(f"Hybrid Agent error: {e}")
            # Fallback to AlgoFeast
            return await self._process_with_algofeast(user_query, context)
    
    def _determine_route(self, query: str) -> str:
        """Determine which handler should process the query"""
        query_lower = query.lower()
        
        # FIRST: Check for trading-specific patterns (highest priority)
        trading_patterns = [
            "buy", "sell", "order", "trade", "position", "portfolio",
            "balance", "margin", "stoploss", "target", "nifty", "reliance",
            "tcs", "infosys", "hdfc", "bank", "stock", "share"
        ]
        
        if any(pattern in query_lower for pattern in trading_patterns):
            return "mcp" if self.mcp_available else "algofeast"
        
        # THEN: Check routing rules
        for keyword, route in self.routing_rules.items():
            if keyword in query_lower:
                return route
        
        # Default to AlgoFeast for complex queries
        return "algofeast"
    
    async def _process_with_mcp_smart(self, query: str, classification: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query using MCP with smart classification"""
        
        if not self.mcp_available:
            log_warning("MCP not available, falling back to AlgoFeast")
            return await self._process_with_algofeast(query, context)
        
        try:
            tool = classification.get("tool")
            parameters = classification.get("parameters", {})
            
            if not tool:
                # If no tool determined, try to parse
                tool_call = self._parse_to_mcp_tool(query)
                if tool_call:
                    tool = tool_call["name"]
                    parameters = tool_call["arguments"]
                else:
                    # Fallback to AlgoFeast
                    return await self._process_with_algofeast(query, context)
            
            # Execute MCP tool call
            result = await self._execute_mcp_tool({"name": tool, "arguments": parameters})
            
            return {
                "source": "mcp",
                "query": query,
                "classification": classification,
                "tool": tool,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"Smart MCP processing error: {e}")
            return await self._process_with_algofeast(query, context)
    
    async def _process_with_mcp(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query using MCP server (legacy method for fallback)"""
        
        if not self.mcp_available:
            log_warning("MCP not available, falling back to AlgoFeast")
            return await self._process_with_algofeast(query, context)
        
        try:
            # Convert natural language to MCP tool call
            tool_call = self._parse_to_mcp_tool(query)
            
            if not tool_call:
                # If can't parse, try AlgoFeast
                return await self._process_with_algofeast(query, context)
            
            # Execute MCP tool call
            result = await self._execute_mcp_tool(tool_call)
            
            return {
                "source": "mcp",
                "query": query,
                "tool": tool_call["name"],
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"MCP processing error: {e}")
            return await self._process_with_algofeast(query, context)
    
    async def _process_with_algofeast(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query using existing AlgoFeast agent"""
        
        try:
            # Import here to avoid circular imports
            from agent.graph import run_agent
            result = await run_agent(query, context or {})
            
            return {
                "source": "algofeast",
                "query": query,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"AlgoFeast processing error: {e}")
            return {
                "source": "error",
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _process_with_both(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query using both handlers and combine results"""
        
        # Try AlgoFeast first (usually better for complex queries)
        algofeast_result = await self._process_with_algofeast(query, context)
        
        # If AlgoFeast gives a good result, use it
        if algofeast_result.get("source") == "algofeast" and not algofeast_result.get("error"):
            return algofeast_result
        
        # Otherwise, try MCP
        if self.mcp_available:
            mcp_result = await self._process_with_mcp(query, context)
            if not mcp_result.get("error"):
                return mcp_result
        
        # Return the better result
        return algofeast_result if not algofeast_result.get("error") else mcp_result
    
    def _parse_to_mcp_tool(self, query: str) -> Optional[Dict[str, Any]]:
        """Parse natural language query to MCP tool call"""
        
        query_lower = query.lower()
        
        # Order placement
        if any(word in query_lower for word in ["buy", "sell"]):
            return self._parse_order_query(query)
        
        # Portfolio queries
        elif any(word in query_lower for word in ["portfolio", "positions", "holdings"]):
            return {"name": "get_portfolio", "arguments": {}}
        
        elif "balance" in query_lower:
            return {"name": "get_balance", "arguments": {}}
        
        # Market data
        elif "price" in query_lower or "quote" in query_lower:
            return self._parse_price_query(query)
        
        # Order management
        elif "cancel" in query_lower:
            return self._parse_cancel_query(query)
        
        elif "modify" in query_lower:
            return self._parse_modify_query(query)
        
        # Strategy
        elif "start strategy" in query_lower:
            return self._parse_strategy_query(query)
        
        return None
    
    def _parse_order_query(self, query: str) -> Dict[str, Any]:
        """Parse order placement query"""
        import re
        
        # Extract transaction type
        transaction_type = "BUY" if "buy" in query.lower() else "SELL"
        
        # Extract symbol (basic pattern matching)
        symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI", "KOTAK", "SBIN", "NIFTY", "BANKNIFTY"]
        symbol = None
        for sym in symbols:
            if sym.lower() in query.lower():
                symbol = sym
                break
        
        if not symbol:
            return None
        
        # Extract quantity
        quantity_match = re.search(r'(\d+)\s*(?:shares?|stocks?)', query, re.IGNORECASE)
        quantity = int(quantity_match.group(1)) if quantity_match else 1
        
        # Extract price (if specified)
        price_match = re.search(r'(?:at|@)\s*(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)', query, re.IGNORECASE)
        price = float(price_match.group(1)) if price_match else None
        
        # Extract stoploss/target
        stoploss_match = re.search(r'stoploss\s*(?:at|@)\s*(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)', query, re.IGNORECASE)
        stoploss = float(stoploss_match.group(1)) if stoploss_match else None
        
        target_match = re.search(r'target\s*(?:at|@)\s*(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)', query, re.IGNORECASE)
        target = float(target_match.group(1)) if target_match else None
        
        # Build tool call
        arguments = {
            "symbol": symbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "product": "MIS"  # Default to MIS
        }
        
        if price:
            arguments["order_type"] = "LIMIT"
            arguments["price"] = price
        else:
            arguments["order_type"] = "MARKET"
        
        if stoploss:
            arguments["stoploss"] = stoploss
        
        if target:
            arguments["target"] = target
        
        return {
            "name": "place_order",
            "arguments": arguments
        }
    
    def _parse_price_query(self, query: str) -> Dict[str, Any]:
        """Parse price query"""
        symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI", "KOTAK", "SBIN", "NIFTY", "BANKNIFTY"]
        symbol = None
        for sym in symbols:
            if sym.lower() in query.lower():
                symbol = sym
                break
        
        if not symbol:
            return None
        
        return {
            "name": "get_market_price",
            "arguments": {"symbol": symbol}
        }
    
    def _parse_cancel_query(self, query: str) -> Dict[str, Any]:
        """Parse cancel order query"""
        import re
        order_id_match = re.search(r'order\s*([A-Z0-9]+)', query, re.IGNORECASE)
        
        if order_id_match:
            return {
                "name": "cancel_order",
                "arguments": {"order_id": order_id_match.group(1)}
            }
        
        return None
    
    def _parse_modify_query(self, query: str) -> Dict[str, Any]:
        """Parse modify order query"""
        import re
        order_id_match = re.search(r'order\s*([A-Z0-9]+)', query, re.IGNORECASE)
        
        if not order_id_match:
            return None
        
        arguments = {"order_id": order_id_match.group(1)}
        
        # Extract quantity
        quantity_match = re.search(r'quantity\s*(?:to\s*)?(\d+)', query, re.IGNORECASE)
        if quantity_match:
            arguments["quantity"] = int(quantity_match.group(1))
        
        # Extract price
        price_match = re.search(r'price\s*(?:to\s*)?(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)', query, re.IGNORECASE)
        if price_match:
            arguments["price"] = float(price_match.group(1))
        
        return {
            "name": "modify_order",
            "arguments": arguments
        }
    
    def _parse_strategy_query(self, query: str) -> Dict[str, Any]:
        """Parse strategy query"""
        # Extract strategy type
        strategies = {
            "915": "915_candle_break",
            "candle": "915_candle_break",
            "breakout": "momentum_breakout",
            "mean": "mean_reversion",
            "reversal": "rsi_reversal",
            "macd": "macd_crossover",
            "ema": "ema_cross"
        }
        
        strategy_type = None
        for key, strategy in strategies.items():
            if key in query.lower():
                strategy_type = strategy
                break
        
        if not strategy_type:
            strategy_type = "915_candle_break"  # Default
        
        # Extract symbol
        symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]
        symbol = None
        for sym in symbols:
            if sym.lower() in query.lower():
                symbol = sym
                break
        
        if not symbol:
            symbol = "NIFTY 50"  # Default
        
        return {
            "name": "start_strategy",
            "arguments": {
                "strategy_type": strategy_type,
                "symbol": symbol,
                "quantity": 75,  # Default for indices
                "product": "MIS"
            }
        }
    
    async def _execute_mcp_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool call"""
        
        try:
            # For now, simulate the result since MCP server integration needs work
            # In production, this would connect to actual MCP server
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            
            log_info(f"Simulating MCP tool execution: {tool_name} with args: {arguments}")
            
            # Simulate different tool responses
            if tool_name == "place_order":
                return {
                    "order_id": f"ORD{datetime.now().strftime('%H%M%S')}",
                    "status": "placed",
                    "message": f"Order placed successfully for {arguments.get('symbol', 'UNKNOWN')}"
                }
            elif tool_name == "get_portfolio":
                return {
                    "positions": [
                        {"symbol": "RELIANCE", "quantity": 10, "pnl": 500.0},
                        {"symbol": "TCS", "quantity": 5, "pnl": -200.0}
                    ],
                    "total_pnl": 300.0
                }
            elif tool_name == "get_balance":
                return {
                    "available_margin": 50000.0,
                    "used_margin": 10000.0,
                    "total_margin": 60000.0
                }
            elif tool_name == "get_market_price":
                symbol = arguments.get("symbol", "UNKNOWN")
                return {
                    "symbol": symbol,
                    "price": 2500.0,
                    "change": 50.0,
                    "change_percent": 2.04
                }
            else:
                return {
                    "status": "simulated",
                    "tool": tool_name,
                    "message": "MCP server simulation - tool executed successfully"
                }
                
        except Exception as e:
            log_error(f"MCP tool execution error: {e}")
            # Return error response
            return {
                "error": str(e),
                "tool": tool_call.get("name"),
                "status": "error"
            }
    
    def _simulate_mcp_result(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate MCP result for demonstration"""
        
        tool_name = tool_call["name"]
        
        if tool_name == "get_market_price":
            return {
                "symbol": tool_call["arguments"]["symbol"],
                "price": 2500.0,
                "change": 50.0,
                "change_percent": 2.04
            }
        
        elif tool_name == "get_portfolio":
            return {
                "positions": [
                    {"symbol": "RELIANCE", "quantity": 10, "pnl": 500.0},
                    {"symbol": "TCS", "quantity": 5, "pnl": -200.0}
                ],
                "total_pnl": 300.0
            }
        
        elif tool_name == "place_order":
            return {
                "order_id": f"ORD{datetime.now().strftime('%H%M%S')}",
                "status": "placed",
                "message": f"Order placed successfully"
            }
        
        else:
            return {
                "status": "simulated",
                "tool": tool_name,
                "message": "MCP server not fully configured, using simulation"
            }


# Global instance
hybrid_agent = HybridAgentHandler()
