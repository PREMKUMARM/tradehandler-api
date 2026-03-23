"""
Database Repository for Telegram Scheduler
"""

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import uuid

from database.models import SQLAlchemyBase, ScheduledTask, TaskExecutionLog
from utils.logger import log_info, log_error, log_warning


class SchedulerRepository:
    """Repository for scheduled tasks and execution logs"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def create_tables(self):
        """Create scheduler tables if they don't exist"""
        try:
            # Use raw SQL to create tables
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Create ScheduledTask table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    schedule_type TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    schedule_config TEXT NOT NULL,
                    operation_config TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_run TIMESTAMP,
                    next_run TIMESTAMP,
                    run_count INTEGER DEFAULT 0,
                    timezone TEXT DEFAULT 'UTC'
                )
            ''')
            
            # Create TaskExecutionLog table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_execution_logs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success INTEGER DEFAULT 0,
                    error_message TEXT,
                    execution_time REAL,
                    result_data TEXT
                )
            ''')
            
            conn.commit()
            log_info("Scheduler tables created successfully")
            
        except Exception as e:
            log_error(f"Failed to create scheduler tables: {e}")
    
    async def create_task(self, task_data: Dict[str, Any]) -> bool:
        """Create a new scheduled task"""
        try:
            # Get raw connection and create cursor
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            task_id = task_data.get('id') or str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO scheduled_tasks (
                    id, name, description, schedule_type, operation_type, 
                    schedule_config, operation_config, enabled, timezone
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_id,
                task_data.get('name'),
                task_data.get('description'),
                task_data.get('schedule_type'),
                task_data.get('operation_type'),
                json.dumps(task_data.get('schedule_config')),
                json.dumps(task_data.get('operation_config')),
                1 if task_data.get('enabled', True) else 0,
                task_data.get('timezone', 'UTC')
            ))
            
            conn.commit()
            log_info(f"Created scheduled task: {task_data.get('name')}")
            return True
            
        except Exception as e:
            log_error(f"Failed to create scheduled task: {e}")
            try:
                conn.rollback()
            except:
                pass
            return False
    
    async def get_tasks(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get all scheduled tasks"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM scheduled_tasks"
            if enabled_only:
                query += " WHERE enabled = 1"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            tasks = []
            for row in rows:
                tasks.append({
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'schedule_type': row[3],
                    'operation_type': row[4],
                    'schedule_config': json.loads(row[5]) if row[5] else {},
                    'operation_config': json.loads(row[6]) if row[6] else {},
                    'enabled': bool(row[7]),
                    'created_at': row[8],
                    'last_run': row[9],
                    'next_run': row[10],
                    'run_count': row[11],
                    'timezone': row[12]
                })
            
            return tasks
            
        except Exception as e:
            log_error(f"Failed to get scheduled tasks: {e}")
            return []
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific scheduled task"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'schedule_type': row[3],
                    'operation_type': row[4],
                    'schedule_config': json.loads(row[5]) if row[5] else {},
                    'operation_config': json.loads(row[6]) if row[6] else {},
                    'enabled': bool(row[7]),
                    'created_at': row[8],
                    'last_run': row[9],
                    'next_run': row[10],
                    'run_count': row[11],
                    'timezone': row[12]
                }
            return None
            
        except Exception as e:
            log_error(f"Failed to get scheduled task {task_id}: {e}")
            return None
    
    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update a scheduled task"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Check if task exists
            cursor.execute("SELECT id FROM scheduled_tasks WHERE id = ?", (task_id,))
            if not cursor.fetchone():
                log_warning(f"Task {task_id} not found for update")
                return False
            
            # Build update query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key in ['name', 'description', 'schedule_type', 'operation_type']:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
                elif key in ['schedule_config', 'operation_config']:
                    set_clauses.append(f"{key} = ?")
                    values.append(json.dumps(value))
                elif key in ['enabled']:
                    set_clauses.append(f"{key} = ?")
                    values.append(1 if value else 0)
                elif key in ['timezone']:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
            
            if set_clauses:
                query = f"UPDATE scheduled_tasks SET {', '.join(set_clauses)} WHERE id = ?"
                values.append(task_id)
                
                cursor.execute(query, values)
                conn.commit()
                log_info(f"Updated scheduled task: {task_id}")
                return True
            
            return False
            
        except Exception as e:
            log_error(f"Failed to update scheduled task {task_id}: {e}")
            conn.rollback()
            return False
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a scheduled task"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Check if task exists
            cursor.execute("SELECT id FROM scheduled_tasks WHERE id = ?", (task_id,))
            if not cursor.fetchone():
                log_warning(f"Task {task_id} not found for deletion")
                return False
            
            # Delete task
            cursor.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
            conn.commit()
            log_info(f"Deleted scheduled task: {task_id}")
            return True
            
        except Exception as e:
            log_error(f"Failed to delete scheduled task {task_id}: {e}")
            conn.rollback()
            return False
    
    async def toggle_task(self, task_id: str) -> bool:
        """Enable/disable a scheduled task"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Check if task exists
            cursor.execute("SELECT enabled FROM scheduled_tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                log_warning(f"Task {task_id} not found for toggle")
                return False
            
            # Toggle enabled status
            new_status = 0 if row[0] else 1
            cursor.execute("UPDATE scheduled_tasks SET enabled = ? WHERE id = ?", (new_status, task_id))
            conn.commit()
            
            status = "enabled" if new_status else "disabled"
            log_info(f"Toggled task {task_id}: {status}")
            return True
            
        except Exception as e:
            log_error(f"Failed to toggle task {task_id}: {e}")
            conn.rollback()
            return False
    
    async def update_task_runtime_metadata(self, task_id: str, last_run: datetime, 
                                          next_run: datetime, run_count: int) -> bool:
        """Update task runtime metadata after execution"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Update task metadata
            cursor.execute('''
                UPDATE scheduled_tasks 
                SET last_run = ?, next_run = ?, run_count = ?
                WHERE id = ?
            ''', (
                last_run.isoformat() if last_run else None,
                next_run.isoformat() if next_run else None,
                run_count,
                task_id
            ))
            conn.commit()
            
            if cursor.rowcount > 0:
                log_info(f"Updated task {task_id} runtime metadata: run_count={run_count}")
                return True
            else:
                log_warning(f"Task {task_id} not found for runtime metadata update")
                return False
            
        except Exception as e:
            log_error(f"Failed to update task {task_id} runtime metadata: {e}")
            conn.rollback()
            return False
    
    async def log_execution(self, task_id: str, success: bool, execution_time: float = None, 
                       error_message: str = None, result_data: Dict[str, Any] = None) -> bool:
        """Log task execution"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO task_execution_logs (
                    id, task_id, executed_at, success, error_message, execution_time, result_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                task_id,
                datetime.utcnow().isoformat(),
                1 if success else 0,
                error_message,
                execution_time,
                json.dumps(result_data) if result_data else None
            ))
            
            conn.commit()
            log_info(f"Logged execution for task {task_id}: {'SUCCESS' if success else 'FAILED'}")
            return True
            
        except Exception as e:
            log_error(f"Failed to log task execution: {e}")
            conn.rollback()
            return False
    
    async def get_execution_logs(self, task_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get execution logs for tasks"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM task_execution_logs"
            params: List[Any] = []
            if task_id:
                query += " WHERE task_id = ?"
                params.append(task_id)
            query += " ORDER BY executed_at DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, tuple(params))
            
            rows = cursor.fetchall()
            
            logs = []
            for row in rows:
                logs.append({
                    'id': row[0],
                    'task_id': row[1],
                    'executed_at': row[2],
                    'success': bool(row[3]),
                    'error_message': row[4],
                    'execution_time': row[5],
                    'result_data': json.loads(row[6]) if row[6] else None
                })
            
            return logs
            
        except Exception as e:
            log_error(f"Failed to get execution logs: {e}")
            return []


# Dependency injection function
def get_scheduler_repository() -> SchedulerRepository:
    """Get scheduler repository instance"""
    from database.connection import get_database
    db = get_database()
    return SchedulerRepository(db)
