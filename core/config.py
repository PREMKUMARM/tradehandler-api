"""
Enhanced configuration management with validation
"""
import os
import json
from pathlib import Path
from typing import Optional, Any, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
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
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="AlgoFeast API")
    app_version: str = Field(default="1.0.0")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    api_prefix: str = Field(default="/api/v1")
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)
    
    # Security
    secret_key: str = Field(default="change-me-in-production")
    allowed_hosts: Any = Field(default_factory=lambda: ["*"])
    cors_origins: Any = Field(
        default_factory=lambda: ["http://localhost:4200", "https://algofeast.com", "https://www.algofeast.com"]
    )
    
    # Database
    database_path: str = Field(default="data/algofeast.db")
    database_pool_size: int = Field(default=10)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/app.log")
    log_max_bytes: int = Field(default=10485760)  # 10MB
    log_backup_count: int = Field(default=5)
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_per_minute: int = Field(default=60)
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    
    # Kite Connect (from existing)
    kite_api_key: Optional[str] = Field(default=None)
    kite_api_secret: Optional[str] = Field(default=None)
    kite_redirect_uri: str = Field(
        default="https://algofeast.com/auth-token"
    )
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except:
                    pass
            return [host.strip() for host in v.split(",") if host.strip()]
        return v
    
    # Remove the old Config class at the bottom


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_agent_config() -> AgentConfig:
    """Get agent configuration (wrapper for compatibility)"""
    from agent.config import get_agent_config as _get_agent_config
    return _get_agent_config()

