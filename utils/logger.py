"""
Enterprise logging utility
Centralized logging with structured format
"""
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# --- Filters ---
class RequestContextFilter(logging.Filter):
    """Ensure request_id exists on every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return True


class RateLimitFilter(logging.Filter):
    """
    Drop repeated identical log lines for chatty loops.

    This is intentionally simple: it rate-limits by (logger name + level + message).
    """

    def __init__(self, window_seconds: float) -> None:
        super().__init__()
        self.window_seconds = max(0.0, float(window_seconds or 0.0))
        self._last: Dict[Tuple[str, int, str], float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if self.window_seconds <= 0:
            return True
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(getattr(record, "msg", ""))  # best-effort
        key = (record.name, int(record.levelno), msg)
        now = float(getattr(record, "created", 0.0) or 0.0)
        last = self._last.get(key)
        if last is not None and (now - last) < self.window_seconds:
            return False
        self._last[key] = now
        return True


# Configure root logger
def setup_logging(log_level: str = "INFO"):
    """Setup enterprise-level logging configuration"""
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"

    # Build handlers explicitly so we can attach filters (basicConfig does not
    # guarantee filters run for third-party loggers).
    stream = logging.StreamHandler(sys.stdout)
    fileh = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")

    formatter = logging.Formatter(fmt)
    stream.setFormatter(formatter)
    fileh.setFormatter(formatter)

    # Always provide request_id, even for httpx/uvicorn/etc.
    ctx = RequestContextFilter()

    # Drop identical repeated lines within the window (seconds).
    rl_sec = float(os.getenv("LOG_RATE_LIMIT_SECONDS", "2") or 2)
    rl = RateLimitFilter(rl_sec)

    for h in (stream, fileh):
        h.addFilter(ctx)
        h.addFilter(rl)

    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level)
    root.addHandler(stream)
    root.addHandler(fileh)

    # Reduce noisy third-party libraries (keeps app logs readable).
    quiet = {
        "uvicorn": logging.WARNING,
        "uvicorn.error": logging.WARNING,
        "uvicorn.access": logging.WARNING,
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "urllib3": logging.WARNING,
        "asyncio": logging.WARNING,
        "websockets": logging.WARNING,
        "twisted": logging.WARNING,
    }
    for name, lvl in quiet.items():
        logging.getLogger(name).setLevel(lvl)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with request context support"""
    logger = logging.getLogger(name)
    # Avoid attaching duplicate filters on repeated calls.
    if not any(isinstance(f, RequestContextFilter) for f in logger.filters):
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


_throttled_last: Dict[str, float] = {}


def log_warning_throttled(
    key: str,
    message: str,
    *,
    interval_sec: float = 60.0,
    request_id: Optional[str] = None,
) -> None:
    """Emit a warning at most once per key within interval_sec."""
    import time

    now = time.monotonic()
    last = _throttled_last.get(key)
    if last is not None and (now - last) < max(1.0, interval_sec):
        return
    _throttled_last[key] = now
    log_warning(message, request_id=request_id)
