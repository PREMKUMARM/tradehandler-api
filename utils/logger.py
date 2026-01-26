"""
Enterprise logging utility
Centralized logging with structured format
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure root logger
def setup_logging(log_level: str = "INFO"):
    """Setup enterprise-level logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        ]
    )


class RequestContextFilter(logging.Filter):
    """Add request ID to log records"""
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'N/A')
        return True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with request context support"""
    logger = logging.getLogger(name)
    logger.addFilter(RequestContextFilter())
    return logger


def log_agent_activity(message: str, level: str = "info", request_id: Optional[str] = None):
    """
    Log agent activity with structured format
    
    Args:
        message: Log message
        level: Log level (info, error, warning, debug)
        request_id: Optional request ID for tracing
    """
    logger = get_logger("agent")
    
    # Add request ID to log record
    extra = {"request_id": request_id or "N/A"}
    
    level_map = {
        "info": logger.info,
        "error": logger.error,
        "warning": logger.warning,
        "debug": logger.debug,
        "critical": logger.critical
    }
    
    log_func = level_map.get(level.lower(), logger.info)
    log_func(message, extra=extra)


def log_tool_execution(tool_name: str, input_data: dict, output_data: dict, 
                      request_id: Optional[str] = None, duration_ms: Optional[float] = None):
    """
    Log tool execution with input/output
    
    Args:
        tool_name: Name of the tool
        input_data: Tool input parameters
        output_data: Tool output
        request_id: Optional request ID
        duration_ms: Execution duration in milliseconds
    """
    logger = get_logger("tools")
    extra = {"request_id": request_id or "N/A"}
    
    log_message = f"Tool: {tool_name}"
    if duration_ms:
        log_message += f" | Duration: {duration_ms:.2f}ms"
    
    logger.debug(f"{log_message} | Input: {input_data} | Output: {output_data}", extra=extra)


def log_tool_interaction(tool_name: str, input_data: dict, output_data: dict, 
                        request_id: Optional[str] = None):
    """
    Log tool interaction (alias for log_tool_execution for backward compatibility)
    
    Args:
        tool_name: Name of the tool
        input_data: Tool input parameters
        output_data: Tool output
        request_id: Optional request ID
    """
    log_tool_execution(tool_name, input_data, output_data, request_id)


# Convenience functions for easy print() replacement
def log_info(message: str, request_id: Optional[str] = None, **kwargs):
    """Log info message - replacement for print() statements"""
    logger = get_logger("app")
    extra = {"request_id": request_id or "N/A"}
    if kwargs:
        message = f"{message} | {kwargs}"
    logger.info(message, extra=extra)


def log_error(message: str, request_id: Optional[str] = None, **kwargs):
    """Log error message - replacement for print() statements"""
    logger = get_logger("app")
    extra = {"request_id": request_id or "N/A"}
    if kwargs:
        message = f"{message} | {kwargs}"
    logger.error(message, extra=extra)


def log_warning(message: str, request_id: Optional[str] = None, **kwargs):
    """Log warning message - replacement for print() statements"""
    logger = get_logger("app")
    extra = {"request_id": request_id or "N/A"}
    if kwargs:
        message = f"{message} | {kwargs}"
    logger.warning(message, extra=extra)


def log_debug(message: str, request_id: Optional[str] = None, **kwargs):
    """Log debug message - replacement for print() statements"""
    logger = get_logger("app")
    extra = {"request_id": request_id or "N/A"}
    if kwargs:
        message = f"{message} | {kwargs}"
    logger.debug(message, extra=extra)
