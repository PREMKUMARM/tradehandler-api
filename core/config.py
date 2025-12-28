"""
Enhanced configuration management with validation
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
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
    
    # Application
    app_name: str = "AlgoFeast API"
    app_version: str = "1.0.0"
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    api_prefix: str = "/api/v1"
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Security
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    allowed_hosts: list[str] = Field(default_factory=lambda: ["*"], env="ALLOWED_HOSTS")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:4200", "https://algofeast.com", "https://www.algofeast.com"],
        env="CORS_ORIGINS"
    )
    
    # Database
    database_path: str = Field(default="data/algofeast.db", env="DATABASE_PATH")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    log_max_bytes: int = Field(default=10485760, env="LOG_MAX_BYTES")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Kite Connect (from existing)
    kite_api_key: Optional[str] = Field(default=None, env="KITE_API_KEY")
    kite_api_secret: Optional[str] = Field(default=None, env="KITE_API_SECRET")
    kite_redirect_uri: str = Field(
        default="https://algofeast.com/auth-token",
        env="KITE_REDIRECT_URI"
    )
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env (agent config fields are handled separately)


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

