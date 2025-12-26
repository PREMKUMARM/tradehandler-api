import logging
import os
from datetime import datetime
import json
from typing import Any, Dict

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure the main logger
logger = logging.getLogger("tradehandler_agent")
logger.setLevel(logging.DEBUG)

# File handler for all logs
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "agent.log"))
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Tool execution logger (specifically for inputs/outputs)
tool_logger = logging.getLogger("tool_monitor")
tool_logger.setLevel(logging.DEBUG)
tool_file_handler = logging.FileHandler(os.path.join(LOG_DIR, "tools.log"))
tool_file_handler.setFormatter(formatter)
tool_logger.addHandler(tool_file_handler)

def log_agent_activity(message: str, level: str = "info"):
    """Log general agent activity"""
    if level.lower() == "debug":
        logger.debug(message)
    elif level.lower() == "warning":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    else:
        logger.info(message)

def log_tool_interaction(tool_name: str, inputs: Dict[str, Any], output: Any):
    """Log tool inputs and outputs specifically for debugging"""
    try:
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "inputs": inputs,
            "output_preview": str(output)[:1000] + ("..." if len(str(output)) > 1000 else "")
        }
        tool_logger.debug(f"TOOL_CALL | {json.dumps(interaction, indent=2, default=str)}")
    except Exception as e:
        tool_logger.error(f"Error logging tool interaction: {e}")

def get_logger():
    return logger

