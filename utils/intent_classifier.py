"""
Smart Intent Classifier for MCP Tool Recognition
Uses LLM to classify user intent and determine appropriate routing
"""
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from utils.logger import log_info, log_error, log_warning


class IntentClassifier:
    """Smart intent classifier using LLM for dynamic prompt understanding"""
    
    def __init__(self):
        self.intent_patterns = {
            # Trading operations that should route to MCP
            "TRADING_OPERATION": [
                "place_order", "modify_order", "cancel_order", "get_market_price",
                "get_portfolio", "get_positions", "get_balance", "get_order_history"
            ],
            # Analysis and advice that should route to AlgoFeast
            "MARKET_ANALYSIS": [
                "analyze", "predict", "recommend", "suggest", "compare", "explain",
                "should", "why", "how", "what if", "opinion", "view"
            ],
            # Strategy automation
            "STRATEGY_EXECUTION": [
                "start_strategy", "stop_strategy", "get_strategy_status"
            ],
            # Risk management
            "RISK_MANAGEMENT": [
                "get_trade_limits", "set_trade_limits"
            ],
            # General information
            "GENERAL_INFO": [
                "help", "status", "health", "info", "about"
            ]
        }
    
    async def classify_intent(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Classify user intent using LLM analysis"""
        
        try:
            # First, try LLM-based classification
            llm_result = await self._llm_classify(user_query, context)
            
            if llm_result and llm_result.get("confidence", 0) > 0.7:
                log_info(f"LLM classified intent: {llm_result['intent']} (confidence: {llm_result['confidence']})")
                return llm_result
            
            # Fallback to pattern-based classification
            pattern_result = self._pattern_classify(user_query)
            log_info(f"Pattern classified intent: {pattern_result['intent']} (confidence: {pattern_result['confidence']})")
            return pattern_result
            
        except Exception as e:
            log_error(f"Intent classification error: {e}")
            # Safe fallback
            return {
                "intent": "GENERAL_INFO",
                "confidence": 0.1,
                "entities": {},
                "tool": None,
                "parameters": {}
            }
    
    async def _llm_classify(self, user_query: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Use LLM to classify intent"""
        
        try:
            # Import here to avoid circular imports
            from agent.graph import run_agent
            
            # Create classification prompt
            classification_prompt = f"""
            Analyze this user query and classify the intent:

            Query: "{user_query}"
            
            Possible intents:
            1. TRADING_OPERATION - User wants to execute a trading action (buy, sell, check portfolio, get prices, etc.)
            2. MARKET_ANALYSIS - User wants analysis, prediction, or advice
            3. STRATEGY_EXECUTION - User wants to start/stop trading strategies
            4. RISK_MANAGEMENT - User wants to manage risk limits or settings
            5. GENERAL_INFO - General questions or help requests
            
            Also extract:
            - trading symbols (RELIANCE, TCS, NIFTY, etc.)
            - transaction type (BUY/SELL)
            - quantity (numbers)
            - price information
            - specific action requested
            
            Respond in JSON format:
            {{
                "intent": "TRADING_OPERATION",
                "confidence": 0.95,
                "entities": {{
                    "symbol": "RELIANCE",
                    "action": "buy",
                    "quantity": 10,
                    "price": 2500
                }},
                "tool": "place_order",
                "parameters": {{
                    "symbol": "RELIANCE",
                    "transaction_type": "BUY",
                    "quantity": 10,
                    "order_type": "LIMIT",
                    "price": 2500
                }}
            }}
            """
            
            # Use AlgoFeast agent for classification
            result = await run_agent(classification_prompt, context or {})
            
            if result and "response" in result:
                response = result["response"]
                
                # Try to extract JSON from response
                import json
                import re
                
                # Look for JSON pattern in response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        classified = json.loads(json_match.group())
                        
                        # Validate and enhance the result
                        return self._validate_classification(classified, user_query)
                        
                    except json.JSONDecodeError:
                        log_warning(f"Failed to parse LLM JSON response: {response}")
            
            return None
            
        except Exception as e:
            log_error(f"LLM classification error: {e}")
            return None
    
    def _pattern_classify(self, user_query: str) -> Dict[str, Any]:
        """Fallback pattern-based classification"""
        
        query_lower = user_query.lower()
        
        # Trading operation patterns
        trading_keywords = [
            "buy", "sell", "order", "trade", "portfolio", "balance", "positions",
            "price", "quote", "market", "cancel", "modify", "holdings", "shares", "stock"
        ]
        
        analysis_keywords = [
            "analyze", "predict", "recommend", "suggest", "should", "why", "how",
            "compare", "explain", "opinion", "view", "advice", "good", "bad"
        ]
        
        strategy_keywords = [
            "strategy", "automated", "start", "stop", "execute", "algorithm"
        ]
        
        risk_keywords = [
            "limit", "risk", "margin", "exposure", "stoploss", "target"
        ]
        
        # Count keyword matches
        trading_score = sum(1 for kw in trading_keywords if kw in query_lower)
        analysis_score = sum(1 for kw in analysis_keywords if kw in query_lower)
        strategy_score = sum(1 for kw in strategy_keywords if kw in query_lower)
        risk_score = sum(1 for kw in risk_keywords if kw in query_lower)
        
        # Determine intent with confidence
        scores = {
            "TRADING_OPERATION": trading_score,
            "MARKET_ANALYSIS": analysis_score,
            "STRATEGY_EXECUTION": strategy_score,
            "RISK_MANAGEMENT": risk_score
        }
        
        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]
        
        # Calculate confidence
        total_keywords = sum(scores.values())
        confidence = max_score / max(total_keywords, 1)
        
        # Extract entities
        entities = self._extract_entities_pattern(user_query)
        
        # Determine tool and parameters
        tool, parameters = self._determine_tool_parameters(max_intent, entities)
        
        return {
            "intent": max_intent,
            "confidence": min(confidence * 1.2, 0.8),  # Boost confidence slightly
            "entities": entities,
            "tool": tool,
            "parameters": parameters
        }
    
    def _extract_entities_pattern(self, user_query: str) -> Dict[str, Any]:
        """Extract entities using pattern matching"""
        import re
        
        entities = {}
        query_lower = user_query.lower()
        
        # Extract symbols
        symbols = ["reliance", "tcs", "infosys", "hdfc", "icici", "kotak", "sbin", 
                  "nifty", "banknifty", "sensex", "lt", "axisbank", "bhartiartl"]
        
        for symbol in symbols:
            if symbol in query_lower:
                entities["symbol"] = symbol.upper()
                break
        
        # Extract transaction type
        if "buy" in query_lower:
            entities["action"] = "BUY"
        elif "sell" in query_lower:
            entities["action"] = "SELL"
        
        # Extract quantity
        quantity_match = re.search(r'(\d+)\s*(?:shares?|stocks?)', user_query, re.IGNORECASE)
        if quantity_match:
            entities["quantity"] = int(quantity_match.group(1))
        
        # Extract price
        price_match = re.search(r'(?:at|@)\s*(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)', user_query, re.IGNORECASE)
        if price_match:
            entities["price"] = float(price_match.group(1))
        
        return entities
    
    def _determine_tool_parameters(self, intent: str, entities: Dict[str, Any]) -> tuple:
        """Determine MCP tool and parameters based on intent and entities"""
        
        if intent == "TRADING_OPERATION":
            action = entities.get("action", "")
            
            if action == "BUY" or action == "SELL":
                return "place_order", {
                    "symbol": entities.get("symbol", "RELIANCE"),
                    "transaction_type": action,
                    "quantity": entities.get("quantity", 1),
                    "order_type": "LIMIT" if entities.get("price") else "MARKET",
                    "price": entities.get("price"),
                    "product": "MIS"
                }
            
            elif "portfolio" in entities.get("symbol", "").lower() or "portfolio" in entities.get("action", "").lower():
                return "get_portfolio", {}
            
            elif "balance" in entities.get("action", "").lower():
                return "get_balance", {}
            
            elif "price" in entities.get("action", "").lower() and entities.get("symbol"):
                return "get_market_price", {"symbol": entities["symbol"]}
        
        elif intent == "MARKET_ANALYSIS":
            return None, {}  # Route to AlgoFeast
        
        elif intent == "STRATEGY_EXECUTION":
            return "start_strategy", {
                "strategy_type": "915_candle_break",
                "symbol": entities.get("symbol", "NIFTY 50"),
                "quantity": entities.get("quantity", 75)
            }
        
        elif intent == "RISK_MANAGEMENT":
            return "get_trade_limits", {}
        
        return None, {}
    
    def _validate_classification(self, classified: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Validate and enhance LLM classification"""
        
        # Ensure required fields exist
        if "intent" not in classified:
            classified["intent"] = "GENERAL_INFO"
        
        if "confidence" not in classified:
            classified["confidence"] = 0.5
        
        if "entities" not in classified:
            classified["entities"] = {}
        
        if "tool" not in classified:
            classified["tool"] = None
        
        if "parameters" not in classified:
            classified["parameters"] = {}
        
        # Validate intent
        valid_intents = list(self.intent_patterns.keys())
        if classified["intent"] not in valid_intents:
            classified["intent"] = "GENERAL_INFO"
            classified["confidence"] = 0.3
        
        # Ensure tool matches intent
        if classified["intent"] == "TRADING_OPERATION" and not classified["tool"]:
            # Try to infer tool from entities
            tool, params = self._determine_tool_parameters(classified["intent"], classified["entities"])
            classified["tool"] = tool
            classified["parameters"] = params
        
        return classified
    
    def should_route_to_mcp(self, classification: Dict[str, Any]) -> bool:
        """Determine if query should route to MCP based on classification"""
        
        intent = classification.get("intent", "")
        confidence = classification.get("confidence", 0)
        
        # High confidence trading operations go to MCP
        if intent == "TRADING_OPERATION" and confidence >= 0.6:
            return True
        
        # Strategy execution goes to MCP
        if intent == "STRATEGY_EXECUTION" and confidence >= 0.6:
            return True
        
        # Risk management goes to MCP
        if intent == "RISK_MANAGEMENT" and confidence >= 0.6:
            return True
        
        # Market analysis goes to AlgoFeast
        if intent == "MARKET_ANALYSIS":
            return False
        
        # Low confidence queries go to AlgoFeast for better understanding
        if confidence < 0.5:
            return False
        
        return False


# Global instance
intent_classifier = IntentClassifier()
