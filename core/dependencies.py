"""
Dependency injection for enterprise-level services
"""
from fastapi import Request, Depends
from typing import Optional

from agent.approval import ApprovalQueue, get_approval_queue
from agent.config import AgentConfig, get_agent_config
from agent.safety import SafetyManager, get_safety_manager
from utils.kite_utils import get_kite_instance
from kiteconnect import KiteConnect


def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, "request_id", "unknown")


def get_approval_queue_dependency() -> ApprovalQueue:
    """Dependency for approval queue"""
    return get_approval_queue()


def get_agent_config_dependency() -> AgentConfig:
    """Dependency for agent configuration"""
    return get_agent_config()


def get_safety_manager_dependency() -> SafetyManager:
    """Dependency for safety manager"""
    return get_safety_manager()


def get_kite_dependency() -> Optional[KiteConnect]:
    """Dependency for Kite Connect instance"""
    try:
        return get_kite_instance()
    except Exception:
        return None

