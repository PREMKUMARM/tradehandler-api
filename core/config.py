"""
Enhanced configuration management with validation
"""
import os
import json
from pathlib import Path
from typing import Optional, Any, List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, AliasChoices
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
        extra="ignore",        # Explicitly ignore extra environment variables
        populate_by_name=True
    )
    
    # Application
    app_name: str = Field(default="AlgoFeast API", validation_alias=AliasChoices("app_name", "APP_NAME"))
    app_version: str = Field(default="1.0.0", validation_alias=AliasChoices("app_version", "APP_VERSION"))
    environment: Environment = Field(default=Environment.DEVELOPMENT, validation_alias=AliasChoices("environment", "ENVIRONMENT"))
    debug: bool = Field(default=False, validation_alias=AliasChoices("debug", "DEBUG"))
    api_prefix: str = Field(default="/api/v1", validation_alias=AliasChoices("api_prefix", "API_PREFIX"))
    
    # Server
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("host", "HOST"))
    port: int = Field(default=8000, validation_alias=AliasChoices("port", "PORT"))
    reload: bool = Field(default=False, validation_alias=AliasChoices("reload", "RELOAD"))
    
    # Security
    secret_key: str = Field(default="change-me-in-production", validation_alias=AliasChoices("secret_key", "SECRET_KEY"))
    allowed_hosts: Any = Field(default_factory=lambda: ["*"], validation_alias=AliasChoices("allowed_hosts", "ALLOWED_HOSTS"))
    cors_origins: Any = Field(
        default_factory=lambda: ["http://localhost:4200", "https://algofeast.com", "https://www.algofeast.com"],
        validation_alias=AliasChoices("cors_origins", "CORS_ORIGINS")
    )
    
    # Database
    database_path: str = Field(default="data/algofeast.db", validation_alias=AliasChoices("database_path", "DATABASE_PATH"))
    database_pool_size: int = Field(default=10, validation_alias=AliasChoices("database_pool_size", "DATABASE_POOL_SIZE"))
    
    # Logging
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("log_level", "LOG_LEVEL"))
    log_file: str = Field(default="logs/app.log", validation_alias=AliasChoices("log_file", "LOG_FILE"))
    log_max_bytes: int = Field(default=10485760, validation_alias=AliasChoices("log_max_bytes", "LOG_MAX_BYTES"))
    log_backup_count: int = Field(default=5, validation_alias=AliasChoices("log_backup_count", "LOG_BACKUP_COUNT"))
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, validation_alias=AliasChoices("rate_limit_enabled", "RATE_LIMIT_ENABLED"))
    rate_limit_per_minute: int = Field(default=60, validation_alias=AliasChoices("rate_limit_per_minute", "RATE_LIMIT_PER_MINUTE"))
    
    # Monitoring
    enable_metrics: bool = Field(default=True, validation_alias=AliasChoices("enable_metrics", "ENABLE_METRICS"))
    metrics_port: int = Field(default=9090, validation_alias=AliasChoices("metrics_port", "METRICS_PORT"))
    
    # Kite Connect
    kite_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("kite_api_key", "KITE_API_KEY"))
    kite_api_secret: Optional[str] = Field(default=None, validation_alias=AliasChoices("kite_api_secret", "KITE_API_SECRET"))
    kite_redirect_uri: str = Field(
        default="https://algofeast.com/auth-token",
        validation_alias=AliasChoices("kite_redirect_uri", "KITE_REDIRECT_URI")
    )
    
    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            v = v.strip()
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
        # Load settings without passing manual arguments to avoid Extra Forbidden errors
        _settings = Settings()
    return _settings

def get_agent_config() -> AgentConfig:
    """Get agent configuration (wrapper for compatibility)"""
    from agent.config import get_agent_config as _get_agent_config
    return _get_agent_config()
