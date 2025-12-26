#!/usr/bin/env python3
"""
Test script for SQLite database thread-safety
"""
import sys
import os
import threading
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.repositories import get_approval_repository, get_log_repository
from database.models import AgentApproval, AgentLog
from datetime import datetime
import uuid


def worker_thread(thread_id: int):
    """Worker function to test database operations in different threads"""
    print(f"Thread {thread_id}: Starting database operations")

    try:
        # Test approval repository
        approval_repo = get_approval_repository()
        approval = AgentApproval(
            approval_id=f"test-{thread_id}-{uuid.uuid4()}",
            action="BUY",
            details={"symbol": f"TEST{thread_id}", "price": 1000 + thread_id, "qty": 1},
            trade_value=1000 + thread_id,
            risk_amount=10,
            reward_amount=30,
            risk_percentage=1.0,
            rr_ratio=3.0,
            reasoning=f"Thread {thread_id} test trade",
            status="PENDING",
            created_at=datetime.now()
        )
        success = approval_repo.save(approval)
        print(f"Thread {thread_id}: Approval saved: {success}")

        # Test log repository
        log_repo = get_log_repository()
        log_entry = AgentLog(
            timestamp=datetime.now(),
            level="INFO",
            message=f"Test log from thread {thread_id}",
            component="test",
            metadata={"thread_id": thread_id}
        )
        success = log_repo.save(log_entry)
        print(f"Thread {thread_id}: Log saved: {success}")

        # Test retrieval
        approvals = approval_repo.get_pending()
        print(f"Thread {thread_id}: Retrieved {len(approvals)} pending approvals")

        logs = log_repo.get_recent(limit=5)
        print(f"Thread {thread_id}: Retrieved {len(logs)} recent logs")

        print(f"Thread {thread_id}: All operations successful!")

    except Exception as e:
        print(f"Thread {thread_id}: ERROR - {e}")
        raise


def test_threading():
    print("Testing SQLite database thread-safety...")

    # Start multiple threads
    threads = []
    num_threads = 5

    for i in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads completed successfully! ✅")


if __name__ == "__main__":
    test_threading()
