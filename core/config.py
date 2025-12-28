"""
Enhanced configuration management with validation
"""
import os
import json
from pathlib import Path
from typing import Optional, Any, Union, List
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
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Crucial: ignore extra environment variables
    )
    
    # Application
    app_name: str = Field(default="AlgoFeast API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: Environment = Field(default=Environment.DEVELOPMENT, alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")
    
    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    reload: bool = Field(default=False, alias="RELOAD")
    
    # Security
    secret_key: str = Field(default="change-me-in-production", alias="SECRET_KEY")
    allowed_hosts: List[str] = Field(default_factory=lambda: ["*"], alias="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:4200", "https://algofeast.com", "https://www.algofeast.com"],
        alias="CORS_ORIGINS"
    )
    
    # Database
    database_path: str = Field(default="data/algofeast.db", alias="DATABASE_PATH")
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", alias="LOG_FILE")
    log_max_bytes: int = Field(default=10485760, alias="LOG_MAX_BYTES")
    log_backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")
    
    # Kite Connect
    kite_api_key: Optional[str] = Field(default=None, alias="KITE_API_KEY")
    kite_api_secret: Optional[str] = Field(default=None, alias="KITE_API_SECRET")
    kite_redirect_uri: str = Field(
        default="https://algofeast.com/auth-token",
        alias="KITE_REDIRECT_URI"
    )
    
    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except:
                    pass
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
            # Fallback to defaults if .env parsing fails
            _settings = Settings(_env_file=None)
    return _settings


def get_agent_config() -> AgentConfig:
    """Get agent configuration (wrapper for compatibility)"""
    from agent.config import get_agent_config as _get_agent_config
    return _get_agent_config()
