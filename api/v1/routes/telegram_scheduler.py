"""
Telegram Scheduler API Routes
Provides endpoints for managing scheduled Telegram notifications
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from services.telegram_scheduler import telegram_scheduler, ScheduledTask, ScheduleType, OperationType
from database.repositories_scheduler import get_scheduler_repository
from utils.logger import log_info, log_error, log_warning

router = APIRouter(prefix="/telegram-scheduler", tags=["telegram-scheduler"])


class TaskCreateRequest(BaseModel):
    """Request model for creating a scheduled task"""
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    schedule_type: str = Field(..., description="Schedule type: once, daily, weekly, monthly, interval, cron")
    operation_type: str = Field(..., description="Operation type")
    schedule_config: Dict[str, Any] = Field(..., description="Schedule configuration")
    operation_config: Dict[str, Any] = Field(..., description="Operation parameters")
    enabled: bool = Field(default=True, description="Whether the task is enabled")
    timezone: str = Field(default="UTC", description="Timezone for scheduling")


class TaskUpdateRequest(BaseModel):
    """Request model for updating a scheduled task"""
    schedule_config: Optional[Dict[str, Any]] = Field(None, description="Schedule configuration")
    operation_config: Optional[Dict[str, Any]] = Field(None, description="Operation parameters")
    enabled: Optional[bool] = Field(None, description="Whether the task is enabled")


class TaskResponse(BaseModel):
    """Response model for task creation"""
    success: bool
    message: str
    timestamp: str


@router.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest):
    """Create a new scheduled task"""
    try:
        task_data = {
            'name': request.name,
            'description': request.description,
            'schedule_type': request.schedule_type,
            'operation_type': request.operation_type,
            'schedule_config': request.schedule_config,
            'operation_config': request.operation_config,
            'enabled': request.enabled,
            'timezone': request.timezone
        }
        
        success = await telegram_scheduler.add_task_from_data(task_data)
        
        if success:
            return {
                "success": True,
                "message": "Task created successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create task")
            
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error creating scheduled task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def get_tasks(enabled_only: bool = True):
    """Get all scheduled tasks"""
    try:
        repo = get_scheduler_repository()
        tasks = await repo.get_tasks(enabled_only=enabled_only)
        
        return {
            "success": True,
            "data": tasks,
            "count": len(tasks),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Error getting scheduled tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get a specific scheduled task"""
    try:
        repo = get_scheduler_repository()
        task = await repo.get_task(task_id)
        
        if task:
            return {
                "success": True,
                "data": task,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    except Exception as e:
        log_error(f"Error getting scheduled task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/tasks/{task_id}")
async def update_task(task_id: str, request: TaskUpdateRequest):
    """Update a scheduled task"""
    try:
        repo = get_scheduler_repository()
        
        # Check if task exists
        existing_task = await repo.get_task(task_id)
        if not existing_task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Prepare updates
        updates = {}
        if request.schedule_config is not None:
            updates['schedule_config'] = request.schedule_config
        if request.operation_config is not None:
            updates['operation_config'] = request.operation_config
        if request.enabled is not None:
            updates['enabled'] = request.enabled
        
        success = await repo.update_task(task_id, updates)
        
        if success:
            await telegram_scheduler.refresh_task_from_db(task_id)
            return {
                "success": True,
                "message": "Task updated successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update task")
            
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error updating scheduled task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a scheduled task"""
    try:
        repo = get_scheduler_repository()
        
        # Check if task exists
        existing_task = await repo.get_task(task_id)
        if not existing_task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        success = await repo.delete_task(task_id)
        
        if success:
            telegram_scheduler.remove_task_runtime(task_id)
            return {
                "success": True,
                "message": "Task deleted successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete task")
            
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error deleting scheduled task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/toggle")
async def toggle_task(task_id: str):
    """Enable/disable a scheduled task"""
    try:
        repo = get_scheduler_repository()
        
        # Check if task exists
        existing_task = await repo.get_task(task_id)
        if not existing_task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        success = await repo.toggle_task(task_id)
        
        if success:
            await telegram_scheduler.refresh_task_from_db(task_id)
            updated_task = await repo.get_task(task_id)
            return {
                "success": True,
                "message": f"Task {'enabled' if updated_task['enabled'] else 'disabled'} successfully",
                "data": updated_task,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to toggle task")
            
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error toggling scheduled task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/logs")
async def get_task_logs(task_id: str, limit: int = 50):
    """Get execution logs for a specific task"""
    try:
        repo = get_scheduler_repository()
        logs = await repo.get_execution_logs(task_id=task_id, limit=limit)
        
        return {
            "success": True,
            "data": logs,
            "count": len(logs),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Error getting task logs {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler/start")
async def start_scheduler():
    """Start the Telegram notification scheduler"""
    try:
        await telegram_scheduler.start_scheduler()
        
        return {
            "success": True,
            "message": "Scheduler started successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Error starting scheduler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the Telegram notification scheduler"""
    try:
        await telegram_scheduler.stop_scheduler()
        
        return {
            "success": True,
            "message": "Scheduler stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Error stopping scheduler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status"""
    try:
        tasks = await telegram_scheduler.get_tasks(enabled_only=True)
        
        return {
            "success": True,
            "data": {
                "running": telegram_scheduler.running,
                "total_tasks": len(await telegram_scheduler.get_tasks(enabled_only=False)),
                "enabled_tasks": len(tasks),
                "next_runs": [
                    {
                        "task_id": task.id,
                        "name": task.name,
                        "next_run": task.next_run.isoformat() if task.next_run else None
                    }
                    for task in tasks if task.next_run
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(f"Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule-types")
async def get_schedule_types():
    """Get available schedule types"""
    return {
        "success": True,
        "data": [
            {"value": schedule_type.value, "label": schedule_type.value.replace("_", " ").title()}
            for schedule_type in ScheduleType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/operation-types")
async def get_operation_types():
    """Get available operation types"""
    return {
        "success": True,
        "data": [
            {"value": op_type.value, "label": op_type.value.replace("_", " ").title()}
            for op_type in OperationType
        ],
        "timestamp": datetime.now().isoformat()
    }
