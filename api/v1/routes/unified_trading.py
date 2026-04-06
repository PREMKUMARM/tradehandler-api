"""
Aggregated dashboard for the Unified Trading UI — single poll for positions, risk, agents, tasks, runs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

router = APIRouter(prefix="/unified-trading", tags=["Unified Trading"])


@router.get("/snapshot")
def get_unified_snapshot() -> Dict[str, Any]:
    """Live snapshot: execution safety, Kite positions/margins, agents + task counts, recent tasks, strategy runs."""
    from agent.agent_registry import get_all_agents
    from agent.graph import get_agent_instance
    from agent.task_tracker import get_active_task_count, get_tasks
    from agent.config import get_agent_config
    from database.connection import get_database
    from services.risk_gate import is_kill_switch_active
    from services.paper_trading import is_paper_mode, paper_trading_env_locks_ui
    from utils.logger import log_warning

    server_time = datetime.now(timezone.utc).isoformat()

    execution = {
        "kill_switch_active": is_kill_switch_active(),
        "paper_trading_mode": is_paper_mode(),
        "paper_trading_env_locks_ui": paper_trading_env_locks_ui(),
    }

    positions: List[Dict[str, Any]] = []
    positions_error: Optional[str] = None
    margins: Dict[str, Any] | None = None

    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance(skip_validation=True)
        raw = kite.positions()
        net = raw.get("net") or []
        for p in net:
            if abs(float(p.get("quantity") or 0)) < 1e-9:
                continue
            positions.append(
                {
                    "tradingsymbol": p.get("tradingsymbol"),
                    "exchange": p.get("exchange"),
                    "quantity": p.get("quantity"),
                    "average_price": p.get("average_price"),
                    "last_price": p.get("last_price"),
                    "pnl": p.get("pnl"),
                    "product": p.get("product"),
                }
            )
        try:
            margins = kite.margins()
        except Exception:
            margins = None
    except Exception as e:
        positions_error = str(e)
        log_warning(f"[UnifiedTrading] snapshot positions: {e}")

    cfg = get_agent_config()
    main_initialized = get_agent_instance() is not None
    agents: List[Dict[str, Any]] = [
        {
            "agent_id": "main_trading_agent",
            "name": "Main Trading Agent",
            "status": "healthy" if main_initialized else "uninitialized",
            "active_tasks": get_active_task_count("main_trading_agent"),
            "model": cfg.agent_model,
            "autonomous_mode": cfg.autonomous_mode,
            "auto_trade_enabled": cfg.is_auto_trade_enabled,
        }
    ]
    for spec in get_all_agents():
        agents.append(
            {
                "agent_id": spec.agent_id,
                "name": spec.name,
                "category": getattr(spec, "category", None) or "",
                "status": "healthy",
                "active_tasks": get_active_task_count(spec.agent_id),
                "model": cfg.agent_model,
                "autonomous_mode": False,
                "auto_trade_enabled": False,
            }
        )

    tasks = get_tasks(limit=40)

    strategy_runs: List[Dict[str, Any]] = []
    try:
        db = get_database()
        conn = db.get_connection()
        cur = conn.execute(
            """
            SELECT id, definition_id, mode, status, started_at, ended_at
            FROM strategy_runs
            ORDER BY started_at DESC
            LIMIT 15
            """
        )
        for r in cur.fetchall():
            strategy_runs.append({k: r[k] for k in r.keys()})
    except Exception as e:
        log_warning(f"[UnifiedTrading] strategy_runs: {e}")

    monitoring: Dict[str, Any] = {}
    try:
        from utils.order_monitor import order_monitor
        from services.paper_order_monitor import paper_order_monitor

        monitoring = {
            "live_order_monitor_running": getattr(order_monitor, "is_running", False),
            "paper_order_monitor_running": getattr(paper_order_monitor, "is_running", False),
        }
    except Exception:
        pass

    config_summary: Dict[str, Any] = {
        "agent_model": cfg.agent_model,
        "autonomous_mode": cfg.autonomous_mode,
        "autonomous_scan_interval_mins": cfg.autonomous_scan_interval_mins,
        "autonomous_target_group": cfg.autonomous_target_group,
        "is_auto_trade_enabled": cfg.is_auto_trade_enabled,
        "max_trades_per_day": cfg.max_trades_per_day,
        "circuit_breaker_enabled": cfg.circuit_breaker_enabled,
    }

    return {
        "data": {
            "server_time": server_time,
            "execution": execution,
            "positions": positions,
            "positions_error": positions_error,
            "margins": margins,
            "agents": agents,
            "tasks": tasks,
            "strategy_runs": strategy_runs,
            "monitoring": monitoring,
            "config_summary": config_summary,
        }
    }
