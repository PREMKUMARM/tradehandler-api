"""
Pre-Market Analysis Agent
Analyzes pre-market data, news, and sentiment before market opens
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd

from .base_agent import BaseAgent, AgentTask, AgentCapability, AgentConfig
from .agent_types import AgentType
from .communication import MessageType
from utils.logger import log_info, log_error, log_warning

class PreMarketAgent(BaseAgent):
    """Pre-Market Analysis Agent"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.pre_market_data = {}
        self.news_sentiment = {}
        self.global_market_data = {}
        
    async def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return [
            AgentCapability.PREMARKET_ANALYSIS,
            AgentCapability.DATA_COLLECTION,
            AgentCapability.MARKET_ANALYSIS
        ]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process pre-market analysis task"""
        
        task_type = task.data.get("task_type", "full_analysis")
        
        if task_type == "full_analysis":
            return await self._perform_full_premarket_analysis(task.data)
        elif task_type == "news_analysis":
            return await self._analyze_news_sentiment(task.data)
        elif task_type == "global_markets":
            return await self._analyze_global_markets(task.data)
        elif task_type == "premarket_indicators":
            return await self._analyze_premarket_indicators(task.data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _perform_full_premarket_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive pre-market analysis"""
        
        log_info("Starting full pre-market analysis")
        
        # Collect all data
        news_data = await self._analyze_news_sentiment(task_data)
        global_data = await self._analyze_global_markets(task_data)
        indicators_data = await self._analyze_premarket_indicators(task_data)
        
        # Generate insights
        insights = await self._generate_premarket_insights(news_data, global_data, indicators_data)
        
        # Prepare recommendations
        recommendations = await self._generate_recommendations(insights)
        
        # Store data for other agents
        await self._store_premarket_data({
            "news": news_data,
            "global_markets": global_data,
            "indicators": indicators_data,
            "insights": insights,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        })
        
        # Notify other agents
        await self._notify_agents_of_premarket_data()
        
        return {
            "status": "completed",
            "analysis_type": "full_premarket_analysis",
            "timestamp": datetime.now().isoformat(),
            "news_sentiment": news_data,
            "global_markets": global_data,
            "premarket_indicators": indicators_data,
            "insights": insights,
            "recommendations": recommendations,
            "market_outlook": insights.get("market_outlook", "neutral")
        }
    
    async def _analyze_news_sentiment(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze news sentiment"""
        
        try:
            # Get news from MCP or other sources
            news_sources = task_data.get("news_sources", ["economic_times", "moneycontrol", "reuters"])
            
            news_items = []
            sentiment_scores = []
            
            for source in news_sources:
                if source in self.mcp_clients:
                    # Use MCP to get news
                    news_data = await self.mcp_clients[source].get_latest_news(limit=50)
                    for item in news_data:
                        sentiment = await self._analyze_sentiment(item.get("text", ""))
                        news_items.append({
                            "source": source,
                            "title": item.get("title", ""),
                            "text": item.get("text", ""),
                            "sentiment": sentiment,
                            "timestamp": item.get("timestamp", datetime.now().isoformat())
                        })
                        sentiment_scores.append(sentiment)
            
            # Calculate overall sentiment
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Categorize sentiment
            if overall_sentiment > 0.2:
                sentiment_category = "positive"
            elif overall_sentiment < -0.2:
                sentiment_category = "negative"
            else:
                sentiment_category = "neutral"
            
            # Analyze sector-wise sentiment
            sector_sentiment = await self._analyze_sector_sentiment(news_items)
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_category": sentiment_category,
                "news_count": len(news_items),
                "sector_sentiment": sector_sentiment,
                "key_news": news_items[:10],  # Top 10 news items
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"News sentiment analysis failed: {e}")
            return {"error": str(e), "sentiment_category": "neutral"}
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        # Simple sentiment analysis - in production, use proper NLP model
        positive_words = ["good", "great", "positive", "bullish", "growth", "profit", "gain"]
        negative_words = ["bad", "poor", "negative", "bearish", "loss", "decline", "fall"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment))
    
    async def _analyze_sector_sentiment(self, news_items: List[Dict]) -> Dict[str, float]:
        """Analyze sentiment by sector"""
        sector_keywords = {
            "technology": ["tech", "software", "it", "computer", "digital"],
            "banking": ["bank", "finance", "loan", "interest", "credit"],
            "pharma": ["pharma", "drug", "medicine", "health", "pharmaceutical"],
            "energy": ["oil", "gas", "energy", "petroleum", "power"],
            "automobile": ["car", "auto", "vehicle", "motor", "automobile"]
        }
        
        sector_sentiments = {}
        
        for sector, keywords in sector_keywords.items():
            sector_news = [item for item in news_items 
                          if any(keyword in item["text"].lower() for keyword in keywords)]
            
            if sector_news:
                sentiments = [item["sentiment"] for item in sector_news]
                sector_sentiments[sector] = sum(sentiments) / len(sentiments)
            else:
                sector_sentiments[sector] = 0.0
        
        return sector_sentiments
    
    async def _analyze_global_markets(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze global market indicators"""
        
        try:
            global_indices = {
                "US": ["S&P 500", "NASDAQ", "DOW"],
                "Europe": ["FTSE", "DAX", "CAC"],
                "Asia": ["Nikkei", "Hang Seng", "Shanghai"]
            }
            
            global_data = {}
            
            for region, indices in global_indices.items():
                region_data = {}
                for index in indices:
                    if "zerodha" in self.mcp_clients:
                        # Get global index data
                        index_data = await self.mcp_clients["zerodha"].get_global_index(index)
                        region_data[index] = {
                            "last_price": index_data.get("last_price", 0),
                            "change": index_data.get("change", 0),
                            "change_percent": index_data.get("change_percent", 0)
                        }
                
                global_data[region] = region_data
            
            # Calculate global sentiment
            all_changes = []
            for region_data in global_data.values():
                for index_data in region_data.values():
                    all_changes.append(index_data.get("change_percent", 0))
            
            global_sentiment = sum(all_changes) / len(all_changes) if all_changes else 0
            
            return {
                "global_data": global_data,
                "global_sentiment": global_sentiment,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"Global market analysis failed: {e}")
            return {"error": str(e), "global_sentiment": 0}
    
    async def _analyze_premarket_indicators(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pre-market indicators"""
        
        try:
            indicators = {}
            
            if "zerodha" in self.mcp_clients:
                # Get pre-market indicators
                premarket_data = await self.mcp_clients["zerodha"].get_premarket_indicators()
                
                indicators = {
                    "nifty_premarket": premarket_data.get("nifty_premarket", {}),
                    "banknifty_premarket": premarket_data.get("banknifty_premarket", {}),
                    "sgx_nifty": premarket_data.get("sgx_nifty", {}),
                    "vix": premarket_data.get("vix", {}),
                    "fii_dii_data": premarket_data.get("fii_dii_data", {})
                }
            
            # Analyze pre-market trends
            premarket_trend = await self._analyze_premarket_trend(indicators)
            
            return {
                "indicators": indicators,
                "premarket_trend": premarket_trend,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"Pre-market indicators analysis failed: {e}")
            return {"error": str(e), "premarket_trend": "neutral"}
    
    async def _analyze_premarket_trend(self, indicators: Dict) -> str:
        """Analyze pre-market trend"""
        
        try:
            # Get key indicators
            nifty_premarket = indicators.get("nifty_premarket", {})
            sgx_nifty = indicators.get("sgx_nifty", {})
            
            # Calculate trend based on multiple indicators
            trend_signals = []
            
            # SGX Nifty signal
            if sgx_nifty.get("change_percent", 0) > 0.5:
                trend_signals.append("positive")
            elif sgx_nifty.get("change_percent", 0) < -0.5:
                trend_signals.append("negative")
            
            # Pre-market Nifty signal
            if nifty_premarket.get("change_percent", 0) > 0.3:
                trend_signals.append("positive")
            elif nifty_premarket.get("change_percent", 0) < -0.3:
                trend_signals.append("negative")
            
            # Determine overall trend
            positive_signals = trend_signals.count("positive")
            negative_signals = trend_signals.count("negative")
            
            if positive_signals > negative_signals:
                return "positive"
            elif negative_signals > positive_signals:
                return "negative"
            else:
                return "neutral"
                
        except Exception as e:
            log_error(f"Pre-market trend analysis failed: {e}")
            return "neutral"
    
    async def _generate_premarket_insights(self, news_data: Dict, global_data: Dict, 
                                         indicators_data: Dict) -> Dict[str, Any]:
        """Generate insights from pre-market analysis"""
        
        insights = {
            "market_outlook": "neutral",
            "key_factors": [],
            "risk_factors": [],
            "opportunity_factors": [],
            "sector_outlook": {}
        }
        
        # Analyze news sentiment
        news_sentiment = news_data.get("sentiment_category", "neutral")
        if news_sentiment == "positive":
            insights["opportunity_factors"].append("Positive news sentiment")
        elif news_sentiment == "negative":
            insights["risk_factors"].append("Negative news sentiment")
        
        # Analyze global markets
        global_sentiment = global_data.get("global_sentiment", 0)
        if global_sentiment > 1.0:
            insights["opportunity_factors"].append("Positive global market sentiment")
        elif global_sentiment < -1.0:
            insights["risk_factors"].append("Negative global market sentiment")
        
        # Analyze pre-market indicators
        premarket_trend = indicators_data.get("premarket_trend", "neutral")
        if premarket_trend == "positive":
            insights["opportunity_factors"].append("Positive pre-market indicators")
        elif premarket_trend == "negative":
            insights["risk_factors"].append("Negative pre-market indicators")
        
        # Determine overall market outlook
        opportunity_count = len(insights["opportunity_factors"])
        risk_count = len(insights["risk_factors"])
        
        if opportunity_count > risk_count:
            insights["market_outlook"] = "positive"
        elif risk_count > opportunity_count:
            insights["market_outlook"] = "negative"
        else:
            insights["market_outlook"] = "neutral"
        
        # Add sector outlook
        sector_sentiment = news_data.get("sector_sentiment", {})
        for sector, sentiment in sector_sentiment.items():
            if sentiment > 0.2:
                insights["sector_outlook"][sector] = "positive"
            elif sentiment < -0.2:
                insights["sector_outlook"][sector] = "negative"
            else:
                insights["sector_outlook"][sector] = "neutral"
        
        return insights
    
    async def _generate_recommendations(self, insights: Dict) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on insights"""
        
        recommendations = []
        
        market_outlook = insights.get("market_outlook", "neutral")
        
        if market_outlook == "positive":
            recommendations.append({
                "type": "market_strategy",
                "action": "consider_buying",
                "reasoning": "Positive pre-market indicators and news sentiment",
                "confidence": 0.7
            })
        elif market_outlook == "negative":
            recommendations.append({
                "type": "market_strategy",
                "action": "consider_selling",
                "reasoning": "Negative pre-market indicators and news sentiment",
                "confidence": 0.7
            })
        
        # Sector-specific recommendations
        sector_outlook = insights.get("sector_outlook", {})
        for sector, outlook in sector_outlook.items():
            if outlook == "positive":
                recommendations.append({
                    "type": "sector_strategy",
                    "sector": sector,
                    "action": "consider_buying",
                    "reasoning": f"Positive sentiment in {sector} sector",
                    "confidence": 0.6
                })
            elif outlook == "negative":
                recommendations.append({
                    "type": "sector_strategy",
                    "sector": sector,
                    "action": "consider_selling",
                    "reasoning": f"Negative sentiment in {sector} sector",
                    "confidence": 0.6
                })
        
        return recommendations
    
    async def _store_premarket_data(self, data: Dict[str, Any]):
        """Store pre-market data for other agents"""
        self.pre_market_data = data
        
        # Store in data store if available
        if self.data_store:
            await self.data_store.save_premarket_data(data)
        
        log_info("Pre-market data stored successfully")
    
    async def _notify_agents_of_premarket_data(self):
        """Notify other agents about pre-market data availability"""
        
        if self.communication_layer:
            await self.communication_layer.broadcast_message(
                from_agent=self.config.agent_id,
                message_type=MessageType.DATA_RESPONSE,
                data={
                    "type": "premarket_data_available",
                    "data": self.pre_market_data,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def get_premarket_summary(self) -> Dict[str, Any]:
        """Get summary of pre-market analysis"""
        
        if not self.pre_market_data:
            return {"status": "no_data_available"}
        
        return {
            "status": "data_available",
            "market_outlook": self.pre_market_data.get("insights", {}).get("market_outlook", "neutral"),
            "news_sentiment": self.pre_market_data.get("news", {}).get("sentiment_category", "neutral"),
            "global_sentiment": self.pre_market_data.get("global_markets", {}).get("global_sentiment", 0),
            "premarket_trend": self.pre_market_data.get("indicators", {}).get("premarket_trend", "neutral"),
            "recommendations_count": len(self.pre_market_data.get("recommendations", [])),
            "last_updated": self.pre_market_data.get("timestamp")
        }
