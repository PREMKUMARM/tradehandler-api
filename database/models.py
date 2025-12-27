"""
Database models for the trading agent application
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


class User(BaseModel):
    """Model for user information"""
    user_id: str
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    google_id: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class AgentApproval(BaseModel):
    """Model for agent approvals and trade decisions"""
    approval_id: str
    action: str
    details: Dict[str, Any]
    trade_value: float
    risk_amount: float
    reward_amount: float = 0.0
    risk_percentage: float
    rr_ratio: float = 0.0
    reasoning: str
    status: str  # PENDING, APPROVED, REJECTED
    created_at: datetime
    approved_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    rejected_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    symbol: Optional[str] = None
    entry_price: Optional[float] = None
    quantity: Optional[int] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None


class AgentLog(BaseModel):
    """Model for agent activity logs"""
    id: Optional[int] = None
    timestamp: datetime
    level: str  # DEBUG, INFO, WARNING, ERROR
    message: str
    component: str  # agent, tools, autonomous, etc.
    metadata: Optional[Dict[str, Any]] = None


class AgentConfig(BaseModel):
    """Model for agent configuration settings"""
    key: str
    user_id: str = "default"  # User identifier for multi-user support
    value: str
    value_type: str  # str, int, float, bool
    category: str  # ai, autonomous, strategy, capital, market
    description: Optional[str] = None
    updated_at: datetime


class SimulationResult(BaseModel):
    """Model for storing simulation results"""
    simulation_id: str
    instrument_name: str
    date_range: str
    strategy: str
    trades: Dict[str, Any]  # JSON data of trades
    summary: Dict[str, Any]  # Performance summary
    created_at: datetime
    file_path: Optional[str] = None


class ToolExecution(BaseModel):
    """Model for tracking tool executions"""
    execution_id: str
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    execution_time: float  # in seconds
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime


class ChatMessage(BaseModel):
    """Model for storing chat conversation history"""
    message_id: str
    session_id: str
    role: str  # user, assistant
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
