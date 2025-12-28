"""
Enhanced configuration management with validation for AlgoFeast
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
    """
    Enterprise-level application settings.
    Highly resilient to Pydantic V2 parsing strictness.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",         # Strictly ignore any extra variables
        populate_by_name=True
    )
    
    # Application Settings
    app_name: str = Field(default="AlgoFeast API", validation_alias=AliasChoices("APP_NAME", "app_name"))
    app_version: str = Field(default="1.0.0", validation_alias=AliasChoices("APP_VERSION", "app_version"))
    environment: Environment = Field(default=Environment.DEVELOPMENT, validation_alias=AliasChoices("ENVIRONMENT", "environment"))
    debug: bool = Field(default=False, validation_alias=AliasChoices("DEBUG", "debug"))
    api_prefix: str = Field(default="/api/v1", validation_alias=AliasChoices("API_PREFIX", "api_prefix"))
    
    # Server Settings
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("HOST", "host"))
    port: int = Field(default=8000, validation_alias=AliasChoices("PORT", "port"))
    reload: bool = Field(default=False, validation_alias=AliasChoices("RELOAD", "reload"))
    
    # Security & CORS
    secret_key: str = Field(default="change-me-in-production", validation_alias=AliasChoices("SECRET_KEY", "secret_key"))
    # Use Any to bypass Pydantic's internal list-parsing which causes the crash
    allowed_hosts: Any = Field(default=["*"], validation_alias=AliasChoices("ALLOWED_HOSTS", "allowed_hosts"))
    cors_origins: Any = Field(
        default=["http://localhost:4200", "https://algofeast.com", "https://www.algofeast.com"],
        validation_alias=AliasChoices("CORS_ORIGINS", "cors_origins")
    )
    
    # Database Settings
    database_path: str = Field(default="data/algofeast.db", validation_alias=AliasChoices("DATABASE_PATH", "database_path"))
    database_pool_size: int = Field(default=10, validation_alias=AliasChoices("DATABASE_POOL_SIZE", "database_pool_size"))
    
    # Logging Settings
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL", "log_level"))
    log_file: str = Field(default="logs/app.log", validation_alias=AliasChoices("LOG_FILE", "log_file"))
    log_max_bytes: int = Field(default=10485760, validation_alias=AliasChoices("LOG_MAX_BYTES", "log_max_bytes"))
    log_backup_count: int = Field(default=5, validation_alias=AliasChoices("LOG_BACKUP_COUNT", "log_backup_count"))
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, validation_alias=AliasChoices("RATE_LIMIT_ENABLED", "rate_limit_enabled"))
    rate_limit_per_minute: int = Field(default=60, validation_alias=AliasChoices("RATE_LIMIT_PER_MINUTE", "rate_limit_per_minute"))
    
    # Monitoring
    enable_metrics: bool = Field(default=True, validation_alias=AliasChoices("ENABLE_METRICS", "enable_metrics"))
    metrics_port: int = Field(default=9090, validation_alias=AliasChoices("METRICS_PORT", "metrics_port"))
    
    # Kite Connect Settings
    kite_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("KITE_API_KEY", "kite_api_key"))
    kite_api_secret: Optional[str] = Field(default=None, validation_alias=AliasChoices("KITE_API_SECRET", "kite_api_secret"))
    kite_redirect_uri: str = Field(
        default="https://algofeast.com/auth-token",
        validation_alias=AliasChoices("KITE_REDIRECT_URI", "kite_redirect_uri")
    )
    
    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def handle_list_strings(cls, v: Any) -> List[str]:
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
            # Fallback to hardcoded defaults if .env is fundamentally broken
            # This prevents the whole app from being unable to start
            print(f"CRITICAL: Failed to load settings from environment: {e}")
            _settings = Settings(_env_file=None)
    return _settings

def get_agent_config() -> AgentConfig:
    """Get agent configuration (wrapper for compatibility)"""
    from agent.config import get_agent_config as _get_agent_config
    return _get_agent_config()
