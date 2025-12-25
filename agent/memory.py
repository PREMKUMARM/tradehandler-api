"""
Memory management for the agent
"""
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class AgentMemory:
    """Manages conversation history and context for the agent"""
    
    def __init__(self, window_size: int = 10):
        self.messages: List[BaseMessage] = []
        self.window_size = window_size
        self.positions_history: List[Dict[str, Any]] = []
        self.trades_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        if role == "user":
            self.messages.append(HumanMessage(content=content))
        else:
            self.messages.append(AIMessage(content=content))
        
        # Maintain window size
        if len(self.messages) > self.window_size * 2:  # *2 because user+ai pairs
            self.messages = self.messages[-(self.window_size * 2):]
    
    def get_messages(self) -> List[BaseMessage]:
        """Get conversation history"""
        return self.messages
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
    
    def add_position(self, position: Dict[str, Any]):
        """Track position"""
        self.positions_history.append(position)
    
    def add_trade(self, trade: Dict[str, Any]):
        """Track trade"""
        self.trades_history.append(trade)
    
    def get_context_summary(self) -> str:
        """Get summary of recent context"""
        summary_parts = []
        
        if self.positions_history:
            recent_positions = self.positions_history[-5:]
            summary_parts.append(f"Recent positions: {len(recent_positions)}")
        
        if self.trades_history:
            recent_trades = self.trades_history[-5:]
            summary_parts.append(f"Recent trades: {len(recent_trades)}")
        
        return " | ".join(summary_parts) if summary_parts else "No recent context"

