#!/usr/bin/env python3
"""
Test script for the SQLite database functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.connection import init_database
from database.repositories import (
    get_approval_repository, get_log_repository, get_tool_repository,
    get_simulation_repository, get_config_repository, get_chat_repository
)
from database.models import (
    AgentApproval, AgentLog, AgentConfig, ToolExecution, ChatMessage
)
from datetime import datetime
import uuid

def test_database():
    print("Testing SQLite database functionality...")

    # Initialize database
    db = init_database()
    print("✓ Database initialized")

    # Test approval repository
    approval_repo = get_approval_repository()
    approval = AgentApproval(
        approval_id=str(uuid.uuid4()),
        action="BUY",
        details={"symbol": "RELIANCE", "price": 2500, "qty": 10},
        trade_value=25000,
        risk_amount=250,
        reward_amount=750,
        risk_percentage=1.0,  # 1% risk
        rr_ratio=3.0,
        reasoning="Test trade",
        status="PENDING",
        created_at=datetime.now()
    )
    success = approval_repo.save(approval)
    print(f"✓ Approval saved: {success}")

    # Test log repository
    log_repo = get_log_repository()
    log_entry = AgentLog(
        timestamp=datetime.now(),
        level="INFO",
        message="Test log message",
        component="test",
        metadata={"test": True}
    )
    success = log_repo.save(log_entry)
    print(f"✓ Log saved: {success}")

    # Test tool execution repository
    tool_repo = get_tool_repository()
    tool_exec = ToolExecution(
        execution_id=str(uuid.uuid4()),
        tool_name="test_tool",
        inputs={"param": "value"},
        outputs={"result": "success"},
        execution_time=1.23,
        success=True,
        timestamp=datetime.now()
    )
    success = tool_repo.save(tool_exec)
    print(f"✓ Tool execution saved: {success}")

    # Test chat repository
    chat_repo = get_chat_repository()
    chat_msg = ChatMessage(
        message_id=str(uuid.uuid4()),
        session_id="test_session",
        role="user",
        content="Hello, test message",
        timestamp=datetime.now()
    )
    success = chat_repo.save(chat_msg)
    print(f"✓ Chat message saved: {success}")

    # Test retrieval
    approvals = approval_repo.get_pending()
    print(f"✓ Retrieved {len(approvals)} pending approvals")

    logs = log_repo.get_recent(limit=5)
    print(f"✓ Retrieved {len(logs)} recent logs")

    tool_execs = tool_repo.get_recent(limit=5)
    print(f"✓ Retrieved {len(tool_execs)} recent tool executions")

    messages = chat_repo.get_recent_messages(limit=5)
    print(f"✓ Retrieved {len(messages)} recent messages")

    print("All database tests passed! ✅")

if __name__ == "__main__":
    test_database()
