"""
Enhanced configuration management with validation
"""
import os
import json
from pathlib import Path
from typing import Optional, Any, List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from enum import Enum

# Import existing agent config
from agent.config import AgentConfig, LLMProvider

class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Settings(BaseSettings):
    """Enterprise-level application settings"""
    
    # Pydantic V2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True
    )
    
    # Application
    app_name: str = "AlgoFeast API"
    app_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    api_prefix: str = "/api/v1"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Security
    secret_key: str = "change-me-in-production"
    # Using Any to prevent pydantic-settings from trying to parse the list itself
    allowed_hosts: Any = ["*"]
    cors_origins: Any = ["http://localhost:4200", "https://algofeast.com", "https://www.algofeast.com"]
    
    # Database
    database_path: str = "data/algofeast.db"
    database_pool_size: int = 10
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_max_bytes: int = 10485760
    log_backup_count: int = 5
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Kite Connect
    kite_api_key: Optional[str] = None
    kite_api_secret: Optional[str] = None
    kite_redirect_uri: str = "https://algofeast.com/auth-token"
    
    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            v = v.strip()
            # Handle JSON list format
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except:
                    pass
            # Handle comma-separated string
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        try:
            _settings = Settings()
        except Exception as e:
            print(f"Error loading settings: {e}")
            # Fallback to default settings without .env if parsing fails
            _settings = Settings(_env_file=None)
    return _settings

def get_agent_config() -> AgentConfig:
    """Get agent configuration (wrapper for compatibility)"""
    from agent.config import get_agent_config as _get_agent_config
    return _get_agent_config()
