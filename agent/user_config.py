"""
User-based configuration management
Loads and saves user-specific configurations from database
"""
from typing import Optional, Dict, Any
from datetime import datetime
from agent.config import AgentConfig, LLMProvider
from database.repositories import get_config_repository
from database.models import AgentConfig as AgentConfigModel


def get_user_config(user_id: str = "default") -> AgentConfig:
    """
    Get agent configuration for a specific user.
    Loads from database first, then falls back to .env defaults.
    """
    config = AgentConfig()  # Start with defaults from .env
    
    # Load user-specific overrides from database
    config_repo = get_config_repository()
    db_configs = config_repo.get_all(user_id=user_id)
    
    # Apply database configs over defaults
    for db_config in db_configs:
        key = db_config.key
        value = db_config.value
        value_type = db_config.value_type
        
        # Convert value to appropriate type
        if value_type == "int":
            value = int(value)
        elif value_type == "float":
            value = float(value)
        elif value_type == "bool":
            value = value.lower() in ("true", "1", "yes")
        # str is already string
        
        # Set the attribute if it exists in AgentConfig
        if hasattr(config, key):
            # Special handling for LLMProvider enum
            if key == "llm_provider" and isinstance(value, str):
                try:
                    from agent.config import LLMProvider
                    value = LLMProvider(value)
                except (ValueError, AttributeError):
                    # Invalid provider, skip
                    continue
            setattr(config, key, value)
    
    return config


def save_user_config(user_id: str, config: AgentConfig) -> bool:
    """
    Save agent configuration for a specific user to database.
    Only saves fields that differ from defaults or are explicitly set.
    """
    config_repo = get_config_repository()
    default_config = AgentConfig()  # Get defaults
    
    # List of config fields to save
    config_fields = [
        "llm_provider", "openai_api_key", "anthropic_api_key", "ollama_base_url",
        "agent_model", "agent_temperature", "max_tokens",
        "auto_trade_threshold", "max_position_size", "trading_capital",
        "daily_loss_limit", "max_trades_per_day",
        "risk_per_trade_pct", "reward_per_trade_pct",
        "autonomous_mode", "autonomous_scan_interval_mins", "autonomous_target_group",
        "active_strategies", "is_auto_trade_enabled",
        "vwap_proximity_pct", "vwap_group_proximity_pct", "rejection_shadow_pct",
        "prime_session_start", "prime_session_end", "intraday_square_off_time",
        "trading_start_time", "trading_end_time",
        "circuit_breaker_enabled", "circuit_breaker_loss_threshold",
        "use_gtt_orders", "gtt_for_intraday", "gtt_for_positional",
        "kite_api_key", "kite_api_secret", "kite_redirect_uri"
    ]
    
    saved_count = 0
    for field in config_fields:
        user_value = getattr(config, field, None)
        default_value = getattr(default_config, field, None)
        
        # Save if different from default or explicitly set
        if user_value != default_value or user_value is not None:
            # Determine value type
            if isinstance(user_value, bool):
                value_type = "bool"
                value = str(user_value)
            elif isinstance(user_value, int):
                value_type = "int"
                value = str(user_value)
            elif isinstance(user_value, float):
                value_type = "float"
                value = str(user_value)
            else:
                value_type = "str"
                value = str(user_value) if user_value is not None else ""
            
            # Determine category
            if field in ["llm_provider", "openai_api_key", "anthropic_api_key", "ollama_base_url", "agent_model", "agent_temperature", "max_tokens"]:
                category = "ai"
            elif field in ["autonomous_mode", "autonomous_scan_interval_mins", "autonomous_target_group"]:
                category = "autonomous"
            elif field in ["vwap_proximity_pct", "vwap_group_proximity_pct", "rejection_shadow_pct", "prime_session_start", "prime_session_end", "intraday_square_off_time", "active_strategies"]:
                category = "strategy"
            elif field in ["trading_capital", "max_position_size", "daily_loss_limit", "auto_trade_threshold", "max_trades_per_day", "risk_per_trade_pct", "reward_per_trade_pct", "circuit_breaker_loss_threshold"]:
                category = "capital"
            elif field in ["trading_start_time", "trading_end_time", "circuit_breaker_enabled"]:
                category = "market"
            elif field in ["use_gtt_orders", "gtt_for_intraday", "gtt_for_positional"]:
                category = "gtt"
            elif field in ["kite_api_key", "kite_api_secret", "kite_redirect_uri"]:
                category = "kite"
            else:
                category = "general"
            
            # Special handling for LLMProvider enum
            if field == "llm_provider" and isinstance(user_value, LLMProvider):
                value = user_value.value
            
            # Save to database
            config_model = AgentConfigModel(
                key=field,
                user_id=user_id,
                value=value,
                value_type=value_type,
                category=category,
                description=f"User-specific {field}",
                updated_at=datetime.now()
            )
            
            if config_repo.save(config_model):
                saved_count += 1
    
    return saved_count > 0

