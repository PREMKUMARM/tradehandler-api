"""
Market Agent for real-time market data analysis
"""

from .base_agent import BaseAgent, AgentStatus, AgentTask, AgentCapability
from .agent_types import AgentType
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class MarketAgent(BaseAgent):
    """Agent for real-time market data analysis"""
    
    def __init__(self, agent_id: str, name: str, **kwargs):
        super().__init__(agent_id, name, AgentType.MARKET_ANALYZER, **kwargs)
        self.capabilities = ["market_analysis", "price_monitoring", "trend_detection"]
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return [
            AgentCapability.MARKET_ANALYSIS,
            AgentCapability.DATA_COLLECTION,
            AgentCapability.TECHNICAL_ANALYSIS
        ]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a task"""
        try:
            task_type = task.data.get("task_type", "market_analysis")
            
            if task_type == "market_analysis":
                return await self.analyze_market(task.data)
            elif task_type == "price_monitoring":
                return await self.monitor_prices(task.data)
            elif task_type == "trend_detection":
                return await self.detect_trends(task.data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            logger.error(f"Error executing market task: {e}")
            raise
    
    async def execute_task(self, task_type: str, task_data: dict) -> dict:
        """Execute market analysis task (legacy method)"""
        task = AgentTask(
            task_id="legacy_task",
            agent_id=self.agent_id,
            task_type=task_type,
            priority=5,
            data=task_data,
            created_at=datetime.now()
        )
        return await self.process_task(task)
    
    async def analyze_market(self, data: dict) -> dict:
        """Analyze market conditions"""
        return {
            "status": "success",
            "analysis": "Market analysis complete",
            "trend": "bullish",
            "volatility": "medium"
        }
    
    async def monitor_prices(self, data: dict) -> dict:
        """Monitor price movements"""
        return {
            "status": "success",
            "prices": {"NIFTY": 19500.50, "BANKNIFTY": 45000.25},
            "changes": {"NIFTY": "+0.5%", "BANKNIFTY": "+0.8%"}
        }
    
    async def detect_trends(self, data: dict) -> dict:
        """Detect market trends"""
        return {
            "status": "success",
            "trends": {
                "short_term": "bullish",
                "medium_term": "neutral",
                "long_term": "bullish"
            },
            "confidence": 0.75
        }
