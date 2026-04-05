"""
Database models for trading agent application
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, Float
from sqlalchemy.ext.declarative import declarative_base

# SQLAlchemy Base for database tables
SQLAlchemyBase = declarative_base()

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
    """Model for agent activity logs (matches agent_logs table and AgentLogRepository)"""
    id: Optional[int] = None  # SQLite AUTOINCREMENT when read back; omit on insert
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR
    message: str
    component: str
    metadata: Optional[Dict[str, Any]] = None

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


# SQLAlchemy Models for Scheduler
class ScheduledTask(SQLAlchemyBase):
    """Model for storing scheduled notification tasks"""
    __tablename__ = 'scheduled_tasks'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    schedule_type = Column(String, nullable=False)  # once, daily, weekly, monthly, interval, cron
    operation_type = Column(String, nullable=False)  # custom_message, fetch_price, fetch_news, etc.
    schedule_config = Column(JSON, nullable=False)  # Time, day, minutes, cron expression, etc.
    operation_config = Column(JSON, nullable=False)  # Operation-specific parameters
    enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True)
    run_count = Column(Integer, default=0, nullable=False)
    timezone = Column(String, default='UTC', nullable=False)


class TaskExecutionLog(SQLAlchemyBase):
    """Model for logging task executions"""
    __tablename__ = 'task_execution_logs'

    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False)
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)  # Execution time in seconds
    result_data = Column(JSON, nullable=True)  # Store operation results

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
