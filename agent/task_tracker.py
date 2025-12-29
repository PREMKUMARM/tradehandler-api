"""
Task tracking for agent executions
"""
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

# In-memory task storage (can be moved to database later)
_active_tasks: Dict[str, Dict[str, Any]] = {}


def create_task(agent_id: str, input_data: Dict[str, Any]) -> str:
    """Create a new task and return task_id"""
    task_id = str(uuid.uuid4())
    _active_tasks[task_id] = {
        "task_id": task_id,
        "agent_id": agent_id,
        "status": "running",
        "input": input_data,
        "output": None,
        "tool_calls": [],
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "duration_seconds": None
    }
    return task_id


def update_task(task_id: str, **kwargs) -> bool:
    """Update task with new data"""
    if task_id not in _active_tasks:
        return False
    
    _active_tasks[task_id].update(kwargs)
    
    # Calculate duration if completed
    if kwargs.get("status") in ["completed", "failed"] and "completed_at" in kwargs:
        started = datetime.fromisoformat(_active_tasks[task_id]["started_at"])
        completed = datetime.fromisoformat(kwargs["completed_at"])
        _active_tasks[task_id]["duration_seconds"] = (completed - started).total_seconds()
    
    return True


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task by ID"""
    return _active_tasks.get(task_id)


def get_tasks(agent_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50) -> list:
    """Get tasks, optionally filtered by agent_id and status"""
    tasks = list(_active_tasks.values())
    
    if agent_id:
        tasks = [t for t in tasks if t.get("agent_id") == agent_id]
    
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    
    # Sort by start time (most recent first)
    tasks.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    
    return tasks[:limit]


def get_active_task_count(agent_id: Optional[str] = None) -> int:
    """Get count of active tasks"""
    tasks = get_tasks(agent_id=agent_id, status="running")
    return len(tasks)

