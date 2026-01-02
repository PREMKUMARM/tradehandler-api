"""
Agent-related request/response schemas
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from datetime import datetime


class ChatRequest(BaseModel):
    """Chat request schema"""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    session_id: str = Field(default="default", description="Session ID for conversation context")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ChatResponse(BaseModel):
    """Chat response schema"""
    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ApprovalRequest(BaseModel):
    """Approval request schema"""
    approved_by: Optional[str] = Field(default="user", description="Who approved the trade")
    notes: Optional[str] = Field(None, max_length=500, description="Approval notes")


class RejectionRequest(BaseModel):
    """Rejection request schema"""
    reason: str = Field(..., min_length=1, max_length=500, description="Rejection reason")
    rejected_by: Optional[str] = Field(default="user", description="Who rejected the trade")


class ConfigUpdateRequest(BaseModel):
    """Configuration update request schema"""
    # LLM Settings
    llm_provider: Optional[str] = Field(None, description="LLM provider")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    ollama_base_url: Optional[str] = Field(None, description="Ollama base URL")
    agent_model: Optional[str] = Field(None, description="Agent model name")
    agent_temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Agent temperature")
    max_tokens: Optional[int] = Field(None, ge=1000, le=8000, description="Maximum tokens for responses")
    
    # Trading Settings
    trading_capital: Optional[float] = Field(None, gt=0, description="Trading capital")
    risk_per_trade_pct: Optional[float] = Field(None, gt=0, le=10, description="Risk per trade percentage")
    reward_per_trade_pct: Optional[float] = Field(None, gt=0, le=50, description="Reward per trade percentage")
    auto_trade_threshold: Optional[float] = Field(None, ge=0, description="Auto-trade threshold")
    max_position_size: Optional[float] = Field(None, gt=0, description="Maximum position size")
    daily_loss_limit: Optional[float] = Field(None, ge=0, description="Daily loss limit")
    max_trades_per_day: Optional[int] = Field(None, ge=1, le=100, description="Maximum trades per day")
    
    # GTT Settings
    use_gtt_orders: Optional[bool] = Field(None, description="Use GTT orders")
    gtt_for_intraday: Optional[bool] = Field(None, description="Use GTT for intraday trades")
    gtt_for_positional: Optional[bool] = Field(None, description="Use GTT for positional trades")
    
    # Zerodha Kite Connect Settings
    kite_api_key: Optional[str] = Field(None, description="Zerodha Kite Connect API Key")
    kite_api_secret: Optional[str] = Field(None, description="Zerodha Kite Connect API Secret")
    kite_redirect_uri: Optional[str] = Field(None, description="OAuth redirect URI")
    
    # Strategy Settings
    vwap_proximity_pct: Optional[float] = Field(None, gt=0, le=5, description="VWAP proximity percentage")
    vwap_group_proximity_pct: Optional[float] = Field(None, gt=0, le=5, description="VWAP group proximity percentage")
    rejection_shadow_pct: Optional[float] = Field(None, gt=0, le=100, description="Rejection shadow percentage")
    prime_session_start: Optional[str] = Field(None, description="Prime session start time (HH:MM)")
    prime_session_end: Optional[str] = Field(None, description="Prime session end time (HH:MM)")
    intraday_square_off_time: Optional[str] = Field(None, description="Intraday square-off time (HH:MM)")
    trading_start_time: Optional[str] = Field(None, description="Trading start time (HH:MM)")
    trading_end_time: Optional[str] = Field(None, description="Trading end time (HH:MM)")
    
    @validator("prime_session_start", "prime_session_end")
    def validate_time_format(cls, v):
        if v:
            try:
                from datetime import datetime
                datetime.strptime(v, "%H:%M")
            except ValueError:
                raise ValueError("Time must be in HH:MM format")
        return v

