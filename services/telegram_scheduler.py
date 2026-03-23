"""
Telegram Notification Scheduler Service
Allows scheduling of recurring notifications with configurable operations
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import croniter
import aiohttp
import os
import uuid

from utils.logger import log_info, log_error, log_warning, log_debug
from services.telegram_service import telegram_service, TelegramNotification


class ScheduleType(Enum):
    """Types of scheduling patterns"""
    ONCE = "once"           # Run once at specific time
    DAILY = "daily"         # Run daily at specific time
    WEEKLY = "weekly"       # Run weekly on specific day/time
    MONTHLY = "monthly"       # Run monthly on specific day/time
    INTERVAL = "interval"      # Run every X minutes/hours
    CRON = "cron"           # Custom cron expression


class OperationType(Enum):
    """Types of operations that can be scheduled"""
    CUSTOM_MESSAGE = "custom_message"
    FETCH_PRICE = "fetch_price"
    FETCH_NEWS = "fetch_news"
    SYSTEM_STATUS = "system_status"
    MARKET_SUMMARY = "market_summary"
    PORTFOLIO_UPDATE = "portfolio_update"
    STRATEGY_ALERT = "strategy_alert"


@dataclass
class ScheduledTask:
    """Represents a scheduled notification task"""
    id: str
    name: str
    description: str
    schedule_type: ScheduleType
    operation_type: OperationType
    schedule_config: Dict[str, Any]
    operation_config: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    timezone: str = "UTC"


class TelegramScheduler:
    """Telegram Notification Scheduler Service"""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_task = None
        self.timezone = timezone.utc
        self.repo = None  # Will be initialized later
        
    async def add_task_from_data(self, task_data: Dict[str, Any]) -> bool:
        """Add a new scheduled task from data dictionary"""
        try:
            task = ScheduledTask(
                id=str(uuid.uuid4()),
                name=task_data.get('name'),
                description=task_data.get('description'),
                schedule_type=ScheduleType(task_data.get('schedule_type')),
                operation_type=OperationType(task_data.get('operation_type')),
                schedule_config=task_data.get('schedule_config'),
                operation_config=task_data.get('operation_config'),
                enabled=task_data.get('enabled', True),
                timezone=task_data.get('timezone', 'UTC')
            )
            
            # Calculate next run time
            task.next_run = self._calculate_next_run(task)
            
            # Store task
            self.tasks[task.id] = task
            
            # Save to database
            success = await self._save_task_to_database(task)
            
            if success:
                log_info(f"Added scheduled task: {task.name} (ID: {task.id})")
                log_info(f"Next run: {task.next_run}")
            
            return success
            
        except Exception as e:
            log_error(f"Failed to add scheduled task {task_data.get('name')}: {e}")
            return False
    
    async def add_task(self, task: ScheduledTask) -> bool:
        """Add a new scheduled task"""
        try:
            # Calculate next run time
            task.next_run = self._calculate_next_run(task)
            
            # Store in memory
            self.tasks[task.id] = task
            
            # Save to database
            success = await self._save_task_to_database(task)
            
            if success:
                log_info(f"Added scheduled task: {task.name} (ID: {task.id})")
                log_info(f"Next run: {task.next_run}")
            else:
                log_error(f"Failed to save task {task.name} to database")
            
            return success
            
        except Exception as e:
            log_error(f"Failed to add scheduled task {task.name}: {e}")
            return False
    
    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing scheduled task"""
        try:
            if task_id not in self.tasks:
                log_warning(f"Task {task_id} not found for update")
                return False
            
            task = self.tasks[task_id]
            
            # Update task properties
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            # Recalculate next run time if schedule changed
            if 'schedule_config' in updates:
                task.next_run = self._calculate_next_run(task)
            
            log_info(f"Updated scheduled task: {task.name}")
            return True
            
        except Exception as e:
            log_error(f"Failed to update scheduled task {task_id}: {e}")
            return False
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a scheduled task"""
        try:
            if task_id in self.tasks:
                task_name = self.tasks[task_id].name
                del self.tasks[task_id]
                log_info(f"Deleted scheduled task: {task_name} (ID: {task_id})")
                return True
            else:
                log_warning(f"Task {task_id} not found for deletion")
                return False
                
        except Exception as e:
            log_error(f"Failed to delete scheduled task {task_id}: {e}")
            return False
    
    async def get_tasks(self, enabled_only: bool = True) -> List[ScheduledTask]:
        """Get all scheduled tasks"""
        tasks = list(self.tasks.values())
        if enabled_only:
            tasks = [task for task in tasks if task.enabled]
        return tasks
    
    async def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a specific scheduled task"""
        return self.tasks.get(task_id)
    
    async def toggle_task(self, task_id: str) -> bool:
        """Enable/disable a scheduled task"""
        try:
            if task_id not in self.tasks:
                log_warning(f"Task {task_id} not found for toggle")
                return False
            
            task = self.tasks[task_id]
            task.enabled = not task.enabled
            
            # Recalculate next run time
            if task.enabled:
                task.next_run = self._calculate_next_run(task)
            else:
                task.next_run = None
            
            status = "enabled" if task.enabled else "disabled"
            log_info(f"Toggled task {task.name}: {status}")
            return True
            
        except Exception as e:
            log_error(f"Failed to toggle task {task_id}: {e}")
            return False
    
    async def start_scheduler(self):
        """Start the background scheduler"""
        if self.running:
            log_warning("Scheduler is already running")
            return
        
        # Initialize database repository
        from database.repositories_scheduler import get_scheduler_repository
        self.repo = get_scheduler_repository()
        
        # Create database tables if needed
        self.repo.create_tables()
        
        # Load existing tasks from database
        await self._load_tasks_from_database()
        
        self.running = True
        log_info("Starting Telegram notification scheduler")
        
        # Start background task
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
    async def stop_scheduler(self):
        """Stop the background scheduler"""
        if not self.running:
            return
        
        self.running = False
        log_info("Stopping Telegram notification scheduler")
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                current_time = datetime.now(self.timezone)
                
                # Check each task
                for task in self.tasks.values():
                    if not task.enabled:
                        continue
                    
                    if task.next_run and task.next_run <= current_time:
                        await self._execute_task(task)
                
                # Sleep for 1 minute
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                log_info("Scheduler loop cancelled")
                break
            except Exception as e:
                log_error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)
    
    def _task_from_row(self, task_data: Dict[str, Any]) -> ScheduledTask:
        """Build ScheduledTask from repository row dict."""
        def parse_dt(val: Any) -> Optional[datetime]:
            if val is None or val == '':
                return None
            if isinstance(val, datetime):
                return val
            try:
                s = str(val).strip()
                if 'T' not in s and len(s) >= 10 and s[4:5] == '-':
                    s = s.replace(' ', 'T', 1)
                return datetime.fromisoformat(s)
            except (ValueError, TypeError):
                return None

        created = parse_dt(task_data.get('created_at'))
        return ScheduledTask(
            id=task_data['id'],
            name=task_data['name'],
            description=task_data['description'],
            schedule_type=ScheduleType(task_data['schedule_type']),
            operation_type=OperationType(task_data['operation_type']),
            schedule_config=task_data['schedule_config'],
            operation_config=task_data['operation_config'],
            enabled=task_data['enabled'],
            created_at=created if created else datetime.now(),
            last_run=parse_dt(task_data.get('last_run')),
            next_run=parse_dt(task_data.get('next_run')),
            run_count=task_data.get('run_count') or 0,
            timezone=task_data.get('timezone') or 'UTC'
        )

    async def refresh_task_from_db(self, task_id: str) -> None:
        """Sync runtime task map from DB after API update/toggle."""
        try:
            from database.repositories_scheduler import get_scheduler_repository
            repo = self.repo or get_scheduler_repository()
            row = await repo.get_task(task_id)
            if not row:
                self.tasks.pop(task_id, None)
                return
            self.tasks[task_id] = self._task_from_row(row)
        except Exception as e:
            log_error(f"refresh_task_from_db failed for {task_id}: {e}")

    def remove_task_runtime(self, task_id: str) -> None:
        """Remove task from in-memory scheduler after DB delete."""
        self.tasks.pop(task_id, None)

    async def _load_tasks_from_database(self):
        """Load existing tasks from database"""
        try:
            tasks_data = await self.repo.get_tasks(enabled_only=False)
            for task_data in tasks_data:
                task = self._task_from_row(task_data)
                
                # Calculate next_run if it's None and task is enabled
                if task.enabled and task.next_run is None:
                    task.next_run = self._calculate_next_run(task)
                    log_info(f"Calculated next_run for task '{task.name}': {task.next_run}")
                
                self.tasks[task.id] = task
            
            log_info(f"Loaded {len(self.tasks)} tasks from database")
            
        except Exception as e:
            log_error(f"Failed to load tasks from database: {e}")
    
    async def _save_task_to_database(self, task: ScheduledTask) -> bool:
        """Save task to database"""
        try:
            if self.repo is None:
                # If repository not initialized, just store in memory
                log_info(f"Task {task.name} stored in memory (database not available)")
                return True
            
            task_data = {
                'id': task.id,
                'name': task.name,
                'description': task.description,
                'schedule_type': task.schedule_type.value,
                'operation_type': task.operation_type.value,
                'schedule_config': task.schedule_config,
                'operation_config': task.operation_config,
                'enabled': task.enabled,
                'timezone': task.timezone
            }
            
            return await self.repo.create_task(task_data)
            
        except Exception as e:
            log_error(f"Failed to save task {task.name} to database: {e}")
            return False
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task"""
        start_time = datetime.now()
        success = False
        error_message = None
        result_data = None
        
        try:
            log_info(f"Executing scheduled task: {task.name}")
            
            # Execute operation based on type
            result_data = await self._execute_operation(task)
            success = True
            
            log_info(f"Task {task.name} executed successfully")
            
        except Exception as e:
            error_message = str(e)
            log_error(f"Failed to execute task {task.name}: {e}")
        
        finally:
            # Update task metadata
            task.last_run = datetime.now(self.timezone)
            task.run_count += 1
            
            # Calculate next run time
            task.next_run = self._calculate_next_run(task)
            
            # Log execution to database
            if self.repo:
                execution_time = (datetime.now() - start_time).total_seconds()
                try:
                    await self.repo.log_execution(
                        task_id=task.id,
                        success=success,
                        execution_time=execution_time,
                        error_message=error_message,
                        result_data=result_data
                    )
                except Exception as e:
                    log_error(f"Failed to log execution to database: {e}")
                
                # Save updated task metadata to database
                try:
                    await self.repo.update_task_runtime_metadata(
                        task_id=task.id,
                        last_run=task.last_run,
                        next_run=task.next_run,
                        run_count=task.run_count
                    )
                except Exception as e:
                    log_error(f"Failed to update task metadata in database: {e}")
            else:
                log_info(f"Task {task.name} execution logged to memory only (database not available)")
            
            log_info(f"Task {task.name} execution logged. Next run: {task.next_run}")
    
    async def _execute_operation(self, task: ScheduledTask):
        """Execute the specific operation for a task"""
        operation_config = task.operation_config
        
        log_info(f"Executing operation for task '{task.name}': type={task.operation_type}, config={operation_config}")
        
        try:
            if task.operation_type == OperationType.CUSTOM_MESSAGE:
                log_info(f"Sending custom message for task '{task.name}'")
                await self._send_custom_message(task, operation_config)
            
            elif task.operation_type == OperationType.FETCH_PRICE:
                log_info(f"Fetching price data for task '{task.name}'")
                await self._fetch_price_data(task, operation_config)
            
            elif task.operation_type == OperationType.FETCH_NEWS:
                log_info(f"Fetching news data for task '{task.name}'")
                await self._fetch_news_data(task, operation_config)
            
            elif task.operation_type == OperationType.SYSTEM_STATUS:
                log_info(f"Sending system status for task '{task.name}'")
                await self._send_system_status(task, operation_config)
            
            elif task.operation_type == OperationType.MARKET_SUMMARY:
                log_info(f"Sending market summary for task '{task.name}'")
                await self._send_market_summary(task, operation_config)
            
            elif task.operation_type == OperationType.PORTFOLIO_UPDATE:
                log_info(f"Sending portfolio update for task '{task.name}'")
                await self._send_portfolio_update(task, operation_config)
            
            elif task.operation_type == OperationType.STRATEGY_ALERT:
                log_info(f"Sending strategy alert for task '{task.name}'")
                await self._send_strategy_alert(task, operation_config)
            
            else:
                log_warning(f"Unknown operation type for task '{task.name}': {task.operation_type}")
                raise ValueError(f"Unknown operation type: {task.operation_type}")
                
        except Exception as e:
            log_error(f"Operation execution failed for task '{task.name}': {e}")
            raise
    
    async def _send_custom_message(self, task: ScheduledTask, config: Dict[str, Any]):
        """Send custom message"""
        message = config.get('message', 'Scheduled message')
        
        notification = TelegramNotification(
            title=f"Scheduled: {task.name}",
            message=message,
            priority=config.get('priority', 'normal'),
            category="system"
        )
        
        await telegram_service.send_notification(notification)
    
    async def _fetch_price_data(self, task: ScheduledTask, config: Dict[str, Any]):
        """Fetch price data and send to Telegram"""
        try:
            symbol = config.get('symbol', 'NIFTY-50')
            
            # Mock price fetching - replace with actual API call
            # This would integrate with your existing market data APIs
            price_data = {
                'symbol': symbol,
                'price': '19,845.50',  # Mock price
                'change': '+125.30 (+0.64%)',
                'timestamp': datetime.now().isoformat()
            }
            
            message = f"📈 **{symbol} Price Alert**\n\n"
            message += f"**Current Price:** ₹{price_data['price']}\n"
            message += f"**Change:** {price_data['change']}\n"
            message += f"**Time:** {price_data['timestamp']}"
            
            notification = TelegramNotification(
                title=f"Price Alert: {symbol}",
                message=message,
                priority="normal",
                category="trading",
                metadata=price_data
            )
            
            await telegram_service.send_notification(notification)
            
        except Exception as e:
            log_error(f"Failed to fetch price data: {e}")
    
    async def _fetch_news_data(self, task: ScheduledTask, config: Dict[str, Any]):
        """Fetch news data and send to Telegram"""
        try:
            # Mock news fetching - replace with actual news API
            news_items = [
                {
                    'title': 'Market Opens Positive',
                    'summary': 'NIFTY opens 0.5% higher on positive global cues',
                    'time': '2 hours ago'
                },
                {
                    'title': 'Fed Decision Expected',
                    'summary': 'Traders await Fed decision on interest rates',
                    'time': '4 hours ago'
                }
            ]
            
            message = f"📰 **Market News Update**\n\n"
            for news in news_items[:3]:  # Limit to 3 latest items
                message += f"**{news['title']}**\n"
                message += f"{news['summary']}\n"
                message += f"_{news['time']}_\n\n"
            
            notification = TelegramNotification(
                title="Market News Update",
                message=message,
                priority="normal",
                category="system",
                metadata={'news_count': len(news_items)}
            )
            
            await telegram_service.send_notification(notification)
            
        except Exception as e:
            log_error(f"Failed to fetch news data: {e}")
    
    async def _send_system_status(self, task: ScheduledTask, config: Dict[str, Any]):
        """Send system status to Telegram"""
        try:
            # Get system status
            status_data = {
                'server_status': 'Running',
                'uptime': '2 days, 14 hours',
                'active_agents': 3,
                'memory_usage': '45%',
                'disk_space': '78% free'
            }
            
            message = f"🖥️ **System Status Report**\n\n"
            message += f"**Server Status:** {status_data['server_status']}\n"
            message += f"**Uptime:** {status_data['uptime']}\n"
            message += f"**Active Agents:** {status_data['active_agents']}\n"
            message += f"**Memory Usage:** {status_data['memory_usage']}\n"
            message += f"**Disk Space:** {status_data['disk_space']}"
            
            notification = TelegramNotification(
                title="System Status",
                message=message,
                priority="low",
                category="system",
                metadata=status_data
            )
            
            await telegram_service.send_notification(notification)
            
        except Exception as e:
            log_error(f"Failed to send system status: {e}")
    
    async def _send_market_summary(self, task: ScheduledTask, config: Dict[str, Any]):
        """Send market summary to Telegram"""
        try:
            # Mock market summary
            summary = {
                'nifty': {'close': '19,720.15', 'change': '+0.63%'},
                'sensex': {'close': '65,234.56', 'change': '+0.48%'},
                'volume': '₹2,45,678 Cr',
                'advance_decline': '1,245:789',
                'sector_performance': 'IT +2.1%, Banking -0.8%'
            }
            
            message = f"📊 **Daily Market Summary**\n\n"
            message += f"**NIFTY:** ₹{summary['nifty']['close']} ({summary['nifty']['change']})\n"
            message += f"**SENSEX:** {summary['sensex']['close']} ({summary['sensex']['change']})\n"
            message += f"**Volume:** {summary['volume']}\n"
            message += f"**A/D:** {summary['advance_decline']}\n"
            message += f"**Top Sector:** IT {summary['sector_performance'].split(',')[0]}"
            
            notification = TelegramNotification(
                title="Market Summary",
                message=message,
                priority="normal",
                category="trading",
                metadata=summary
            )
            
            await telegram_service.send_notification(notification)
            
        except Exception as e:
            log_error(f"Failed to send market summary: {e}")
    
    async def _send_portfolio_update(self, task: ScheduledTask, config: Dict[str, Any]):
        """Send portfolio update to Telegram"""
        try:
            # Mock portfolio data
            portfolio = {
                'total_value': '₹12,45,678',
                'daily_pnl': '+₹1,234 (+1.0%)',
                'holdings': 15,
                'top_gainer': 'TCS +3.2%',
                'top_loser': 'RIL -2.1%'
            }
            
            message = f"💼 **Portfolio Update**\n\n"
            message += f"**Total Value:** {portfolio['total_value']}\n"
            message += f"**Daily P&L:** {portfolio['daily_pnl']}\n"
            message += f"**Holdings:** {portfolio['holdings']}\n"
            message += f"**Top Gainer:** {portfolio['top_gainer']}\n"
            message += f"**Top Loser:** {portfolio['top_loser']}"
            
            notification = TelegramNotification(
                title="Portfolio Update",
                message=message,
                priority="normal",
                category="trading",
                metadata=portfolio
            )
            
            await telegram_service.send_notification(notification)
            
        except Exception as e:
            log_error(f"Failed to send portfolio update: {e}")
    
    async def _send_strategy_alert(self, task: ScheduledTask, config: Dict[str, Any]):
        """Send strategy alert to Telegram"""
        try:
            strategy_name = config.get('strategy_name', 'Unknown Strategy')
            alert_type = config.get('alert_type', 'info')
            details = config.get('details', {})
            
            message = f"🎯 **Strategy Alert: {strategy_name}**\n\n"
            message += f"**Alert Type:** {alert_type}\n"
            
            if details:
                for key, value in details.items():
                    message += f"**{key.replace('_', ' ').title()}:** {value}\n"
            
            notification = TelegramNotification(
                title=f"Strategy Alert: {strategy_name}",
                message=message,
                priority="high",
                category="trading",
                metadata={'strategy_name': strategy_name, 'alert_type': alert_type}
            )
            
            await telegram_service.send_notification(notification)
            
        except Exception as e:
            log_error(f"Failed to send strategy alert: {e}")
    
    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate next run time for a task"""
        if not task.enabled:
            return None
        
        try:
            current_time = datetime.now(self.timezone)
            schedule_config = task.schedule_config
            
            if task.schedule_type == ScheduleType.ONCE:
                # One-time execution
                run_time = schedule_config.get('run_time')
                if run_time:
                    target_time = datetime.fromisoformat(run_time).replace(tzinfo=self.timezone)
                    if target_time > current_time:
                        return target_time
                return None
            
            elif task.schedule_type == ScheduleType.DAILY:
                # Daily execution
                time_str = schedule_config.get('time', '09:00')
                hour, minute = map(int, time_str.split(':'))
                next_run = current_time.replace(hour=hour, minute=minute)
                if next_run <= current_time:
                    next_run += timedelta(days=1)
                return next_run
            
            elif task.schedule_type == ScheduleType.WEEKLY:
                # Weekly execution
                time_str = schedule_config.get('time', '09:00')
                day = schedule_config.get('day', 'monday')  # monday, tuesday, etc.
                hour, minute = map(int, time_str.split(':'))
                
                # Find next occurrence of the specified day
                days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                target_day = days.index(day.lower())
                
                next_run = current_time.replace(hour=hour, minute=minute)
                while next_run.weekday() != target_day:
                    next_run += timedelta(days=1)
                
                if next_run <= current_time:
                    next_run += timedelta(weeks=1)
                
                return next_run
            
            elif task.schedule_type == ScheduleType.MONTHLY:
                # Monthly execution
                time_str = schedule_config.get('time', '09:00')
                day = schedule_config.get('day', 1)
                hour, minute = map(int, time_str.split(':'))
                
                next_run = current_time.replace(hour=hour, minute=minute, day=day)
                if next_run <= current_time:
                    # Move to next month
                    if current_time.month == 12:
                        next_run = next_run.replace(year=current_time.year + 1, month=1)
                    else:
                        next_run = next_run.replace(month=current_time.month + 1)
                
                return next_run
            
            elif task.schedule_type == ScheduleType.INTERVAL:
                # Interval execution
                minutes = schedule_config.get('minutes', 60)
                next_run = current_time + timedelta(minutes=minutes)
                return next_run
            
            elif task.schedule_type == ScheduleType.CRON:
                # Cron expression
                cron_expr = schedule_config.get('cron', '0 9 * * *')
                cron = croniter.croniter(cron_expr, current_time)
                try:
                    return next(cron)
                except StopIteration:
                    return None
            
            return None
            
        except Exception as e:
            log_error(f"Failed to calculate next run for task {task.name}: {e}")
            return None


# Global scheduler instance
telegram_scheduler = TelegramScheduler()
