"""
Agent configuration and settings
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class AgentConfig(BaseSettings):
    """Agent configuration settings"""
    
    # LLM Configuration
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    
    # Model Configuration
    agent_model: str = "gpt-4-turbo-preview"
    agent_temperature: float = 0.3
    max_tokens: int = 4000
    
    # Safety Limits
    auto_trade_threshold: float = 5000.0  # Auto-approve trades below this amount (INR)
    max_position_size: float = 200000.0  # Maximum position size (INR)
    trading_capital: float = 200000.0  # Default trading capital to use for sizing (INR)
    daily_loss_limit: float = 5000.0  # Daily loss limit (INR)
    max_trades_per_day: int = 10  # Maximum number of trades per day
    
    # Trading Strategy & Execution
    active_strategies: str = ""  # Comma-separated list of active strategies
    is_auto_trade_enabled: bool = False
    
    # Risk Management
    risk_per_trade_pct: float = 1.0  # Risk 1% per trade
    reward_per_trade_pct: float = 3.0  # Target 3% reward per trade
    
    # Autonomous Mode
    autonomous_mode: bool = False
    autonomous_scan_interval_mins: int = 5
    autonomous_target_group: str = "top 10 nifty50 stocks"
    
    # Trading Hours (IST)
    trading_start_time: str = "09:15"
    trading_end_time: str = "15:30"
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_loss_threshold: float = 10000.0  # Stop trading if loss exceeds this
    
    # Strategy Parameters (Static values moved to config)
    vwap_proximity_pct: float = 0.5
    vwap_group_proximity_pct: float = 0.75
    rejection_shadow_pct: float = 35.0
    prime_session_start: str = "10:15"
    prime_session_end: str = "14:45"
    intraday_square_off_time: str = "15:15"
    
    # GTT (Good Till Triggered) Configuration
    use_gtt_orders: bool = False  # Use GTT for stop-loss/target instead of regular orders
    gtt_for_intraday: bool = False  # Use GTT for intraday trades (MIS)
    gtt_for_positional: bool = True  # Use GTT for positional trades (CNC)
    
    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False


# Global config instance
_agent_config: Optional[AgentConfig] = None


def get_agent_config() -> AgentConfig:
    """Get or create agent configuration instance"""
    global _agent_config
    if _agent_config is None:
        _agent_config = AgentConfig()
    return _agent_config

