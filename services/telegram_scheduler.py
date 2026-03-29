"""
Telegram Notification Scheduler Service
Allows scheduling of recurring notifications with configurable operations
"""

import asyncio
import calendar
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from enum import Enum
import croniter
import aiohttp
import os
import uuid

from utils.logger import log_info, log_error, log_warning, log_debug
from services.telegram_service import telegram_service, TelegramNotification


def _utc_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize datetimes for comparison (SQLite / API often yield naive times)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _zone_for_task(task: "ScheduledTask") -> ZoneInfo:
    tz_name = (task.timezone or "UTC").strip() or "UTC"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        log_warning(f"Invalid timezone '{tz_name}', using UTC")
        return ZoneInfo("UTC")


_TELEGRAM_BODY_MAX = 3800


def _truncate_telegram_body(text: str, max_len: int = _TELEGRAM_BODY_MAX) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 50].rstrip() + "\n\n…(truncated for Telegram)"


def _extract_hybrid_response_text(result: Dict[str, Any]) -> str:
    """Format utils.hybrid_agent.hybrid_agent.process_prompt() result as plain text."""
    if not isinstance(result, dict):
        return str(result)
    if result.get("source") == "error":
        return f"Error: {result.get('error', 'unknown')}"
    src = result.get("source")
    if src == "algofeast":
        inner = result.get("result")
        if isinstance(inner, dict):
            if inner.get("response"):
                return str(inner["response"])
            if inner.get("error"):
                return f"Error: {inner['error']}"
            return json.dumps(inner, default=str)[:2000]
        return str(inner)
    if src == "mcp":
        tool = result.get("tool", "")
        mcp_result = result.get("result", {})
        if isinstance(mcp_result, dict):
            if tool == "place_order":
                oid = mcp_result.get("order_id")
                return f"Order placed. Order ID: {oid}"
            if tool == "get_market_price":
                sym = mcp_result.get("symbol", "")
                price = mcp_result.get("price", "")
                return f"{sym}: ₹{price}"
            if tool == "get_portfolio":
                positions = mcp_result.get("positions", [])
                total_pnl = mcp_result.get("total_pnl", 0)
                if not positions:
                    return "No open positions."
                lines = [f"Portfolio (total P&L: ₹{total_pnl}):"]
                for pos in positions[:40]:
                    symbol = pos.get("symbol", pos.get("tradingsymbol", "?"))
                    qty = pos.get("quantity", 0)
                    pnl = pos.get("pnl", 0)
                    lines.append(f"- {symbol} qty {qty} P&L ₹{pnl}")
                return "\n".join(lines)
            if tool == "get_balance":
                margins = mcp_result.get("balance", {})
                eq = margins.get("equity", {}) if isinstance(margins, dict) else {}
                return f"Available margin: {eq.get('available', margins)}"
            if tool == "cancel_order":
                return f"Order cancelled: {mcp_result.get('order_id')}"
            if tool == "start_strategy":
                return f"Strategy started: {mcp_result.get('strategy_id')}"
            msg = mcp_result.get("message")
            if msg:
                return str(msg)
        return str(mcp_result)
    return json.dumps(result, default=str)[:3000]


def _normalize_scheduler_symbol(symbol: str) -> str:
    """Normalize UI symbols like NIFTY-50 → NIFTY 50 for Kite resolution."""
    s = str(symbol).strip().upper().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    compact = s.replace(" ", "")
    if compact == "NIFTY50":
        return "NIFTY 50"
    if compact == "BANKNIFTY":
        return "NIFTY BANK"
    return s


def _known_index_quote_key(normalized: str) -> Optional[str]:
    """Direct Kite quote keys for common indices (avoids ambiguous instrument search)."""
    u = normalized.upper().strip()
    c = u.replace(" ", "")
    if u == "NIFTY 50" or c == "NIFTY50":
        return "NSE:NIFTY 50"
    if u in ("NIFTY BANK", "BANK NIFTY") or c == "BANKNIFTY":
        return "NSE:NIFTY BANK"
    if u == "SENSEX":
        return "BSE:SENSEX"
    if u in ("INDIA VIX", "INDIAVIX", "VIX"):
        return "NSE:INDIA VIX"
    return None


def _format_last_and_change(quote_data: Dict[str, Any]) -> tuple[bool, str, str]:
    """
    Build display strings from a Kite quote dict.
    Returns (ok, last_price_str, change_str). ok is False if last_price is missing or not usable.
    """
    raw_last = quote_data.get("last_price")
    if raw_last is None:
        return False, "", "last_price missing in quote"
    try:
        last = float(raw_last)
    except (TypeError, ValueError):
        return False, "", "last_price not numeric"
    if last <= 0:
        return False, "", "last_price is zero or negative (live quote unavailable)"

    ohlc = quote_data.get("ohlc") or {}
    try:
        close = float(ohlc.get("close") or 0)
    except (TypeError, ValueError):
        close = 0.0
    last_str = f"{last:,.2f}"
    if close > 0:
        chg = last - close
        pct = (chg / close) * 100.0
        sign = "+" if chg >= 0 else ""
        change_str = f"{sign}{chg:,.2f} ({sign}{pct:.2f}% vs prev. close)"
    else:
        change_str = "previous close not available (cannot compute day change)"
    return True, last_str, change_str


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
    INDICATOR_REPORT = "indicator_report"
    STRATEGY_OVERVIEW = "strategy_overview"
    SCHEDULED_PROMPT = "scheduled_prompt"


def parse_operation_type(raw: Any) -> OperationType:
    """
    Resolve operation_type from API/DB (string). Strips whitespace.
    Prefer this over OperationType(x) so errors list all valid values.
    """
    if isinstance(raw, OperationType):
        return raw
    s = str(raw).strip() if raw is not None else ""
    if not s:
        raise ValueError("operation_type is required")
    for op in OperationType:
        if op.value == s:
            return op
    valid = ", ".join(repr(o.value) for o in OperationType)
    raise ValueError(f"Unknown operation_type {s!r}. Valid: {valid}")


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
                operation_type=parse_operation_type(task_data.get('operation_type')),
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

        except ValueError:
            # Invalid schedule/operation payload — let API return 400 with detail
            raise
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
                current_time = datetime.now(timezone.utc)
                
                # Check each task
                for task in self.tasks.values():
                    if not task.enabled:
                        continue
                    
                    nr = _utc_aware(task.next_run)
                    if nr is not None and nr <= current_time:
                        await self._execute_task(task)
                
                # Poll frequently enough to hit wall-clock schedules (was 60s; missed narrow windows)
                await asyncio.sleep(15)
                
            except asyncio.CancelledError:
                log_info("Scheduler loop cancelled")
                break
            except Exception as e:
                log_error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(15)
    
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
            operation_type=parse_operation_type(task_data['operation_type']),
            schedule_config=task_data['schedule_config'],
            operation_config=task_data['operation_config'],
            enabled=task_data['enabled'],
            created_at=created if created else datetime.now(),
            last_run=_utc_aware(parse_dt(task_data.get('last_run'))),
            next_run=_utc_aware(parse_dt(task_data.get('next_run'))),
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

    async def recalculate_next_run(self, task_id: str) -> None:
        """Recompute next_run from current schedule and persist (after schedule/enabled changes)."""
        try:
            t = self.tasks.get(task_id)
            if not t:
                await self.refresh_task_from_db(task_id)
                t = self.tasks.get(task_id)
            if not t:
                return
            t.next_run = self._calculate_next_run(t)
            if self.repo:
                await self.repo.update_task_runtime_metadata(
                    task_id=t.id,
                    last_run=t.last_run,
                    next_run=t.next_run,
                    run_count=t.run_count,
                )
        except Exception as e:
            log_error(f"recalculate_next_run failed for {task_id}: {e}")

    def remove_task_runtime(self, task_id: str) -> None:
        """Remove task from in-memory scheduler after DB delete."""
        self.tasks.pop(task_id, None)

    def enrich_task_row_for_api(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge live scheduler state into a DB row for list/detail APIs.
        GET /tasks used to return only SQLite columns, so next_run stayed null while
        the in-memory task had the real schedule — UI showed 'Not scheduled'.
        Also registers a DB-only task into self.tasks so the executor can run it.
        """
        tid = row.get("id")
        try:
            if tid and tid not in self.tasks:
                st = self._task_from_row(row)
                if st.enabled and st.next_run is None:
                    st.next_run = self._calculate_next_run(st)
                self.tasks[tid] = st
            if tid and tid in self.tasks:
                t = self.tasks[tid]
                row["next_run"] = t.next_run.isoformat() if t.next_run else None
                row["last_run"] = t.last_run.isoformat() if t.last_run else None
                row["run_count"] = t.run_count
        except Exception as e:
            log_warning(f"enrich_task_row_for_api failed for {tid}: {e}")
        return row

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
                'timezone': task.timezone,
                'next_run': task.next_run,
                'run_count': task.run_count,
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
            task.last_run = datetime.now(timezone.utc)
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

            elif task.operation_type == OperationType.INDICATOR_REPORT:
                log_info(f"Indicator report for task '{task.name}'")
                await self._send_indicator_report(task, operation_config)

            elif task.operation_type == OperationType.STRATEGY_OVERVIEW:
                log_info(f"Strategy overview for task '{task.name}'")
                await self._send_strategy_overview(task, operation_config)

            elif task.operation_type == OperationType.SCHEDULED_PROMPT:
                log_info(f"Scheduled AI prompt for task '{task.name}'")
                await self._send_scheduled_prompt(task, operation_config)
            
            else:
                log_warning(f"Unknown operation type for task '{task.name}': {task.operation_type}")
                raise ValueError(f"Unknown operation type: {task.operation_type}")
                
        except Exception as e:
            log_error(f"Operation execution failed for task '{task.name}': {e}")
            raise

    async def test_run_operation(
        self,
        operation_type: str,
        operation_config: Dict[str, Any],
        label: str = "Test run",
    ) -> None:
        """
        Run the operation once (same code path as a scheduled execution) and send Telegram.
        Does not create or update any persisted task.
        """
        op = parse_operation_type(operation_type)
        name = (label or "Test run").strip()[:200] or "Test run"
        task = ScheduledTask(
            id=f"test-{uuid.uuid4()}",
            name=name,
            description="One-off test run (not saved)",
            schedule_type=ScheduleType.DAILY,
            operation_type=op,
            schedule_config={},
            operation_config=dict(operation_config or {}),
            enabled=True,
            timezone="UTC",
        )
        await self._execute_operation(task)
    
    async def _notify_telegram(self, notification: TelegramNotification) -> None:
        """Send via Telegram and warn once per attempt if delivery is skipped."""
        ok = await telegram_service.send_notification(notification)
        if not ok:
            log_warning(
                "Telegram notification was not sent. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env."
            )

    async def _send_custom_message(self, task: ScheduledTask, config: Dict[str, Any]):
        """Send custom message"""
        message = config.get('message', 'Scheduled message')
        
        notification = TelegramNotification(
            title=f"Scheduled: {task.name}",
            message=message,
            priority=config.get('priority', 'normal'),
            category="system"
        )
        
        await self._notify_telegram(notification)
    
    async def _fetch_price_data(self, task: ScheduledTask, config: Dict[str, Any]):
        """Fetch live last price from Zerodha Kite (same source as the rest of the app)."""
        symbol_raw = config.get("symbol") or "NIFTY-50"
        normalized = _normalize_scheduler_symbol(str(symbol_raw))

        def _fetch_sync() -> tuple:
            from utils.kite_utils import get_kite_instance
            from agent.tools.instrument_resolver import resolve_instrument_name

            kite = get_kite_instance(skip_validation=True)
            quote_key = _known_index_quote_key(normalized)
            if not quote_key:
                info = resolve_instrument_name(normalized, exchange="NSE")
                if not info:
                    info = resolve_instrument_name(normalized, exchange="BSE")
                if not info:
                    raise ValueError(
                        f"Could not resolve symbol '{symbol_raw}'. "
                        "Use NIFTY-50, RELIANCE, or an exact NSE/BSE trading symbol."
                    )
                quote_key = f"{info['exchange']}:{info['tradingsymbol']}"

            quotes = kite.quote([quote_key])
            if quote_key not in quotes:
                raise ValueError(f"No quote returned for {quote_key}")
            return quote_key, quotes[quote_key]

        try:
            quote_key, q = await asyncio.to_thread(_fetch_sync)
        except Exception as e:
            log_error(f"Live price fetch failed for {symbol_raw}: {e}")
            err = TelegramNotification(
                title=f"Price fetch failed: {symbol_raw}",
                message=(
                    f"**Failed to fetch live price** from Zerodha Kite.\n\n"
                    f"**Reason:** {e}\n\n"
                    "Check Kite login / `config/access_token.txt` and that the symbol is valid."
                ),
                priority="high",
                category="trading",
                metadata=None,
            )
            await self._notify_telegram(err)
            return

        ok, price_fmt, change_str = _format_last_and_change(q)
        if not ok:
            log_error(f"Invalid quote payload for {symbol_raw} @ {quote_key}: {change_str}")
            err = TelegramNotification(
                title=f"Price fetch failed: {symbol_raw}",
                message=(
                    f"**Failed to fetch a usable live price** for `{quote_key}`.\n\n"
                    f"**Detail:** {change_str}"
                ),
                priority="high",
                category="trading",
                metadata=None,
            )
            await self._notify_telegram(err)
            return

        message = (
            f"**{symbol_raw}** (Kite: `{quote_key}`)\n\n"
            f"**Last:** ₹{price_fmt}\n"
            f"**Day change:** {change_str}"
        )

        notification = TelegramNotification(
            title=f"Price: {symbol_raw}",
            message=message,
            priority="normal",
            category="trading",
            metadata=None,
        )

        await self._notify_telegram(notification)
    
    async def _fetch_news_data(self, task: ScheduledTask, config: Dict[str, Any]):
        """News: no external feed wired — send an honest status (no mock headlines)."""
        message = (
            "**Scheduled news**\n\n"
            "**No news feed is configured.** This task does not fetch live headlines yet.\n\n"
            "Remove this scheduled operation or integrate a news API later."
        )
        await self._notify_telegram(
            TelegramNotification(
                title="News unavailable",
                message=message,
                priority="normal",
                category="system",
                metadata=None,
            )
        )
    
    async def _send_system_status(self, task: ScheduledTask, config: Dict[str, Any]):
        """Host metrics via psutil (no mock uptime/memory numbers)."""

        def _metrics_sync() -> str:
            try:
                import psutil
                import time as time_mod
                boot = psutil.boot_time()
                up_s = int(time_mod.time() - boot)
                h, rem = divmod(up_s, 3600)
                d, h = divmod(h, 24)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                lines = [
                    "**System status** (this host)",
                    "",
                    f"**Uptime:** {d}d {h}h (approx.)",
                    f"**Memory:** {mem.percent:.1f}% used ({mem.used // (1024**3)} / {mem.total // (1024**3)} GiB)",
                    f"**Disk (/):** {disk.percent:.1f}% used, {disk.free // (1024**3)} GiB free",
                    f"**CPU count:** {psutil.cpu_count(logical=True)}",
                ]
                return "\n".join(lines)
            except Exception as ex:
                return (
                    "**System status**\n\n"
                    f"**Failed to read host metrics:** {ex}\n"
                    "(Install `psutil` or check permissions.)"
                )

        try:
            text = await asyncio.to_thread(_metrics_sync)
            await self._notify_telegram(
                TelegramNotification(
                    title="System status",
                    message=text,
                    priority="low",
                    category="system",
                    metadata=None,
                )
            )
        except Exception as e:
            log_error(f"Failed to send system status: {e}")
            await self._notify_telegram(
                TelegramNotification(
                    title="System status failed",
                    message=f"Could not send system status.\n\n**Reason:** {e}",
                    priority="low",
                    category="system",
                    metadata=None,
                )
            )
    
    async def _send_market_summary(self, task: ScheduledTask, config: Dict[str, Any]):
        """Index snapshot from Kite only (no mock volume/sector lines)."""

        def _sync() -> str:
            from utils.kite_utils import get_kite_instance

            kite = get_kite_instance(skip_validation=True)
            keys = ["NSE:NIFTY 50", "BSE:SENSEX"]
            quotes = kite.quote(keys)
            lines = ["**Market summary** (live Kite quotes)", ""]
            for key, label in zip(keys, ("NIFTY 50", "SENSEX")):
                if key not in quotes:
                    lines.append(f"**{label}:** *Failed to fetch — no quote for `{key}`*")
                    continue
                ok, price, chg = _format_last_and_change(quotes[key])
                if not ok:
                    lines.append(f"**{label}:** *Failed — {chg}*")
                else:
                    lines.append(f"**{label}:** ₹{price}  ({chg})")
            return "\n".join(lines)

        try:
            message = await asyncio.to_thread(_sync)
            await self._notify_telegram(
                TelegramNotification(
                    title="Market summary",
                    message=message,
                    priority="normal",
                    category="trading",
                    metadata=None,
                )
            )
        except Exception as e:
            log_error(f"Failed to send market summary: {e}")
            await self._notify_telegram(
                TelegramNotification(
                    title="Market summary failed",
                    message=(
                        f"**Failed to fetch market summary from Kite.**\n\n"
                        f"**Reason:** {e}"
                    ),
                    priority="normal",
                    category="trading",
                    metadata=None,
                )
            )
    
    async def _send_portfolio_update(self, task: ScheduledTask, config: Dict[str, Any]):
        """Net positions + equity margin summary from Kite (no mock PnL)."""

        def _sync() -> str:
            from utils.kite_utils import get_kite_instance

            kite = get_kite_instance(skip_validation=True)
            pos = kite.positions()
            net = pos.get("net") or []
            lines = ["**Portfolio (Kite)**", ""]
            open_rows = [p for p in net if abs(float(p.get("quantity") or 0)) > 0.0001]
            if not open_rows:
                lines.append("**Net positions:** None (flat).")
            else:
                total_pnl = 0.0
                for p in open_rows[:25]:
                    sym = p.get("tradingsymbol", "?")
                    qty = p.get("quantity", 0)
                    pnl = float(p.get("pnl") or 0)
                    total_pnl += pnl
                    lines.append(f"• {sym}  qty {qty}  P&L ₹{pnl:,.2f}")
                if len(open_rows) > 25:
                    lines.append(f"… and {len(open_rows) - 25} more")
                lines.append("")
                lines.append(f"**Day P&L (net listed):** ₹{total_pnl:,.2f}")
            try:
                margins = kite.margins()
                eq = margins.get("equity", {}) if isinstance(margins, dict) else {}
                if eq:
                    lines.append("")
                    lines.append(f"**Equity available:** ₹{float(eq.get('available', 0) or 0):,.2f}")
            except Exception:
                pass
            return "\n".join(lines)

        try:
            message = await asyncio.to_thread(_sync)
            await self._notify_telegram(
                TelegramNotification(
                    title="Portfolio update",
                    message=message,
                    priority="normal",
                    category="trading",
                    metadata=None,
                )
            )
        except Exception as e:
            log_error(f"Failed to send portfolio update: {e}")
            await self._notify_telegram(
                TelegramNotification(
                    title="Portfolio fetch failed",
                    message=(
                        f"**Failed to load portfolio from Kite.**\n\n"
                        f"**Reason:** {e}"
                    ),
                    priority="normal",
                    category="trading",
                    metadata=None,
                )
            )
    
    async def _send_strategy_alert(self, task: ScheduledTask, config: Dict[str, Any]):
        """Send strategy alert to Telegram"""
        try:
            strategy_name = config.get('strategy_name', 'Unknown Strategy')
            alert_type = config.get('alert_type', 'info')
            details = config.get('details', {})
            
            message = f"**Strategy Alert: {strategy_name}**\n\n"
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
            
            await self._notify_telegram(notification)
            
        except Exception as e:
            log_error(f"Failed to send strategy alert: {e}")

    async def _send_indicator_report(self, task: ScheduledTask, config: Dict[str, Any]):
        """Technical indicator snapshot from Kite historical data."""
        from services.scheduler_indicator_service import build_indicator_report_text, fetch_candles_sync
        from utils.kite_utils import get_kite_instance
        from agent.tools.instrument_resolver import resolve_instrument_name

        symbol_raw = config.get("symbol") or "NIFTY-50"
        interval = (config.get("interval") or "5minute").strip()
        raw_inds = config.get("indicators") or ["rsi_14", "ema_9_21"]
        if isinstance(raw_inds, str):
            indicators = [i.strip() for i in raw_inds.split(",") if i.strip()]
        else:
            indicators = list(raw_inds)

        def _sync() -> tuple:
            normalized = _normalize_scheduler_symbol(str(symbol_raw))
            kite = get_kite_instance(skip_validation=True)
            info = resolve_instrument_name(normalized, exchange="NSE")
            if not info:
                info = resolve_instrument_name(normalized, exchange="BSE")
            if not info:
                raise ValueError(
                    f"Could not resolve symbol '{symbol_raw}'. "
                    "Use a valid NSE/BSE symbol (e.g. NIFTY-50, RELIANCE, SENSEX)."
                )
            quote_key = f"{info['exchange']}:{info['tradingsymbol']}"
            token = info["instrument_token"]
            if token is None:
                raise ValueError(f"Could not resolve instrument token for {symbol_raw}")

            candles = fetch_candles_sync(kite, int(token), interval)
            text = build_indicator_report_text(
                str(symbol_raw), quote_key, interval, candles, indicators
            )
            return text

        try:
            body = await asyncio.to_thread(_sync)
            await self._notify_telegram(
                TelegramNotification(
                    title=f"Indicator report: {task.name}",
                    message=body,
                    priority="normal",
                    category="trading",
                    metadata=None,
                )
            )
        except Exception as e:
            log_error(f"Indicator report failed: {e}")
            await self._notify_telegram(
                TelegramNotification(
                    title=f"Indicator report failed: {task.name}",
                    message=f"**Failed to build indicator report.**\n\n**Reason:** {e}",
                    priority="high",
                    category="trading",
                    metadata=None,
                )
            )

    async def _send_strategy_overview(self, task: ScheduledTask, config: Dict[str, Any]):
        """Summarize registered strategy modules and optional agent config."""
        from services.scheduler_catalog import STRATEGY_DEFINITIONS

        selected = config.get("strategy_ids")
        if isinstance(selected, str):
            selected = [s.strip() for s in selected.split(",") if s.strip()]
        lines = [
            "**Strategy & execution modules**",
            "",
        ]
        for s in STRATEGY_DEFINITIONS:
            if selected and s["id"] not in selected:
                continue
            lines.append(f"**{s['name']}** (`{s['id']}`)")
            lines.append(s["description"])
            if s.get("api"):
                lines.append(f"Reference: `{s['api']}`")
            lines.append("")
        try:
            from agent.config import get_agent_config

            cfg = get_agent_config()
            active = getattr(cfg, "active_strategies", None)
            if active is not None and str(active).strip():
                lines.append(f"**Agent active strategies:** {active}")
            else:
                lines.append("**Agent active strategies:** (none configured)")
        except Exception:
            lines.append("**Agent active strategies:** (not available)")
        message = "\n".join(lines).strip()
        await self._notify_telegram(
            TelegramNotification(
                title=f"Strategy overview: {task.name}",
                message=message,
                priority="low",
                category="system",
                metadata=None,
            )
        )

    async def _send_scheduled_prompt(self, task: ScheduledTask, config: Dict[str, Any]):
        """Run the hybrid AI agent with a user prompt and send the reply to Telegram."""
        prompt_raw = config.get("prompt")
        prompt = str(prompt_raw).strip() if prompt_raw is not None else ""
        if not prompt:
            raise ValueError("operation_config.prompt is required for scheduled_prompt")

        include_ctx = config.get("include_kite_context", True)
        if isinstance(include_ctx, str):
            include_ctx = include_ctx.lower() in ("1", "true", "yes", "on")

        context: Dict[str, Any] = {
            "scheduled_task_id": task.id,
            "scheduled_task_name": task.name,
            "source": "telegram_scheduler",
        }
        if include_ctx:
            try:
                from utils.kite_utils import get_kite_instance

                kite = get_kite_instance(skip_validation=True)
                positions = kite.positions().get("net", [])
                margins = kite.margins()
                context["positions"] = positions
                context["balance"] = margins.get("equity", {}) if isinstance(margins, dict) else {}
            except Exception as ex:
                log_warning(f"Scheduled prompt: Kite context unavailable: {ex}")

        try:
            from utils.hybrid_agent import hybrid_agent

            result = await hybrid_agent.process_prompt(prompt, context)
            body = _extract_hybrid_response_text(result)
            body = _truncate_telegram_body(body)

            prompt_preview = prompt if len(prompt) <= 600 else prompt[:597] + "..."
            message = (
                f"**Prompt**\n{prompt_preview}\n\n"
                f"**Response**\n{body}"
            )
            message = _truncate_telegram_body(message, max_len=_TELEGRAM_BODY_MAX)

            await self._notify_telegram(
                TelegramNotification(
                    title=f"Scheduled prompt: {task.name}",
                    message=message,
                    priority="normal",
                    category="agent",
                    metadata={"task_id": task.id, "operation": "scheduled_prompt"},
                )
            )
        except Exception as e:
            log_error(f"Scheduled prompt failed: {e}")
            await self._notify_telegram(
                TelegramNotification(
                    title=f"Scheduled prompt failed: {task.name}",
                    message=f"**Failed to run AI prompt.**\n\n**Reason:** {e}",
                    priority="high",
                    category="agent",
                    metadata=None,
                )
            )
    
    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate next run time for a task (stored as UTC-aware)."""
        if not task.enabled:
            return None
        
        try:
            tz = _zone_for_task(task)
            now_local = datetime.now(tz)
            now_utc = datetime.now(timezone.utc)
            schedule_config = task.schedule_config
            
            if task.schedule_type == ScheduleType.ONCE:
                # One-time execution
                run_time = schedule_config.get('run_time')
                if not run_time:
                    return None
                raw = datetime.fromisoformat(str(run_time).replace('Z', '+00:00'))
                if raw.tzinfo is None:
                    target_local = raw.replace(tzinfo=tz)
                else:
                    target_local = raw.astimezone(tz)
                target_utc = target_local.astimezone(timezone.utc)
                if target_utc > now_utc:
                    return target_utc
                return None
            
            elif task.schedule_type == ScheduleType.DAILY:
                time_str = schedule_config.get('time', '09:00')
                hour, minute = map(int, time_str.split(':'))
                next_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_local <= now_local:
                    next_local += timedelta(days=1)
                return next_local.astimezone(timezone.utc)
            
            elif task.schedule_type == ScheduleType.WEEKLY:
                time_str = schedule_config.get('time', '09:00')
                day = schedule_config.get('day', 'monday')
                hour, minute = map(int, time_str.split(':'))
                days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                target_day = days.index(day.lower())
                next_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
                while next_local.weekday() != target_day:
                    next_local += timedelta(days=1)
                if next_local <= now_local:
                    next_local += timedelta(weeks=1)
                return next_local.astimezone(timezone.utc)
            
            elif task.schedule_type == ScheduleType.MONTHLY:
                time_str = schedule_config.get('time', '09:00')
                dom = int(schedule_config.get('day', 1))
                hour, minute = map(int, time_str.split(':'))
                y, m = now_local.year, now_local.month
                _, last_day = calendar.monthrange(y, m)
                safe_dom = min(dom, last_day)
                candidate = now_local.replace(
                    day=safe_dom, hour=hour, minute=minute, second=0, microsecond=0
                )
                if candidate <= now_local:
                    if m == 12:
                        y, m = y + 1, 1
                    else:
                        m += 1
                    _, last_day = calendar.monthrange(y, m)
                    safe_dom = min(dom, last_day)
                    candidate = now_local.replace(
                        year=y, month=m, day=safe_dom,
                        hour=hour, minute=minute, second=0, microsecond=0
                    )
                return candidate.astimezone(timezone.utc)
            
            elif task.schedule_type == ScheduleType.INTERVAL:
                minutes = schedule_config.get('minutes', 60)
                return now_utc + timedelta(minutes=minutes)
            
            elif task.schedule_type == ScheduleType.CRON:
                cron_expr = schedule_config.get('cron', '0 9 * * *')
                # croniter uses naive local time if given naive start; anchor in UTC for consistency
                cron = croniter.croniter(cron_expr, now_utc.replace(tzinfo=None))
                try:
                    nxt = cron.get_next(datetime)
                    return nxt.replace(tzinfo=timezone.utc) if nxt.tzinfo is None else nxt.astimezone(timezone.utc)
                except (StopIteration, KeyError, ValueError):
                    return None
            
            return None
            
        except Exception as e:
            log_error(f"Failed to calculate next run for task {task.name}: {e}")
            return None


# Global scheduler instance
telegram_scheduler = TelegramScheduler()
