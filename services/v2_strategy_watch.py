"""
V2 Strategy Watch — server-side deploy with optional full autonomy.

Modes:
  - alert: push/WS when checklist completes; user places manually.
  - autonomous: persists to disk, survives API restart, auto LIMIT+GTT when
    checklist_ready + can_execute (live or paper; retries each poll until placed).

Autonomous is blocked only by kill-switch env/file or V2_WATCH_AUTONOMOUS=0.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, List, Optional
from zoneinfo import ZoneInfo

from agent.ws_manager import broadcast_agent_update
from services import v2_trade_service
from services.push.push_service import push_service
from services.watch_pending_invalidation import (
    is_filled_order_status,
    is_open_order_status,
    pending_entry_invalidated,
)
from utils.logger import log_error, log_info, log_warning
from agent.config import get_agent_config

IST = ZoneInfo("Asia/Kolkata")

_MAX_EVENTS = 40
_DEFAULT_POLL_SEC = 5
_STATE_PATH = Path(os.getenv("V2_WATCH_STATE_FILE", "data/v2_strategy_watch.json"))
_lock = RLock()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def watch_autonomous_globally_disabled() -> bool:
    """Ops-only hard stop via env — normal users enable autonomous from the UI button."""
    return _env_bool("V2_WATCH_DISABLE_AUTONOMOUS", False)


def watch_autonomous_allowed() -> bool:
    """Whether the UI may offer autonomous (blocked only by ops env or kill switch)."""
    if watch_autonomous_globally_disabled():
        return False
    try:
        from services.risk_gate import is_kill_switch_active

        return not is_kill_switch_active()
    except Exception:
        return True


def watch_auto_place_allowed() -> bool:
    return watch_autonomous_allowed()


def _poll_interval() -> float:
    try:
        sec = float(os.getenv("V2_WATCH_POLL_SECONDS", str(_DEFAULT_POLL_SEC)))
        return max(2.0, min(30.0, sec))
    except (TypeError, ValueError):
        return float(_DEFAULT_POLL_SEC)


def _today() -> date:
    return datetime.now(IST).date()


def _today_iso() -> str:
    return _today().isoformat()


@dataclass
class WatchEvent:
    at: str
    kind: str
    message: str
    entry_ready: Optional[bool] = None
    can_place: Optional[bool] = None
    block_reason: Optional[str] = None
    strategy_name: Optional[str] = None
    tradingsymbol: Optional[str] = None
    placed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class WatchConfig:
    direction: str = "AUTO"
    num_lots: int = 1
    risk_percentage: Optional[float] = None
    reward_percentage: Optional[float] = None
    mode: str = "alert"  # alert | autonomous
    auto_place_on_signal: bool = False
    auto_execute_checklist: bool = True
    disarm_after_place: bool = True


def _default_persisted() -> Dict[str, Any]:
    return {
        "armed": False,
        "session_date": _today_iso(),
        "placed_today": False,
        "placed_symbol_today": None,
        "placed_count_today": 0,
        "placed_symbols_today": [],
        "pending_entry_order_id": None,
        "pending_entry_placed_at": None,
        "pending_trade_plan": None,
        "pending_gtt_trigger_id": None,
        "pending_symbol": None,
        "signal_fired_today": False,
        "config": asdict(WatchConfig()),
        "events": [],
        "eval_count": 0,
        "last_entry_ready": None,
        "last_checklist_ready": None,
    }


def _read_persisted() -> Dict[str, Any]:
    if not _STATE_PATH.exists():
        return _default_persisted()
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else _default_persisted()
    except Exception as exc:
        log_warning(f"[V2Watch] state read failed: {exc}")
        return _default_persisted()


def _write_persisted(data: Dict[str, Any]) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        out = copy.deepcopy(data)
        events = out.get("events") or []
        if isinstance(events, list) and len(events) > _MAX_EVENTS:
            out["events"] = events[:_MAX_EVENTS]
        _STATE_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception as exc:
        log_error(f"[V2Watch] state write failed: {exc}")


def _normalize_mode(mode: str, auto_place_on_signal: bool) -> tuple[str, bool]:
    """UI/API mode selects behavior — no separate env flag required to arm."""
    m = (mode or "alert").strip().lower()
    if m in ("auto", "autonomous"):
        return "autonomous", True
    return "alert", False


class V2StrategyWatch:
    def __init__(self) -> None:
        self._armed = False
        self._cfg = WatchConfig()
        self._session_date: Optional[date] = None
        self._last_entry_ready: Optional[bool] = None
        self._last_checklist_ready: Optional[bool] = None
        self._last_can_place = False
        self._last_can_execute = False
        self._last_block_reason: Optional[str] = None
        self._last_trade_plan: Optional[Dict[str, Any]] = None
        self._last_strategy_analysis: Optional[Dict[str, Any]] = None
        self._last_eval_at: Optional[datetime] = None
        self._signal_fired_today = False
        self._placed_today = False
        self._placed_symbol_today: Optional[str] = None
        self._placed_count_today: int = 0
        self._placed_symbols_today: List[str] = []
        self._pending_entry_order_id: Optional[str] = None
        self._pending_entry_placed_at: Optional[str] = None
        self._pending_trade_plan: Optional[Dict[str, Any]] = None
        self._pending_gtt_trigger_id: Optional[str] = None
        self._pending_symbol: Optional[str] = None
        self._eval_count = 0
        self._events: Deque[WatchEvent] = deque(maxlen=_MAX_EVENTS)
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._place_lock: Optional[asyncio.Lock] = None
        self._placing = False
        self._gtt_attach_in_progress = False
        self._last_autonomous_block_reason: Optional[str] = None
        self._last_skip_logged_msg: Optional[str] = None
        self._last_skip_logged_at: Optional[datetime] = None
        self._last_step_statuses: List[Dict[str, Any]] = []
        self._last_market_open = False
        self._last_paper_mode = False
        self._last_kite_connected = False
        self._last_validation: Optional[Dict[str, Any]] = None
        self._last_missing_steps: List[int] = []
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        with _lock:
            data = _read_persisted()
            today = _today_iso()
            if data.get("session_date") != today:
                data["session_date"] = today
                data["placed_today"] = False
                data["placed_symbol_today"] = None
                data["placed_count_today"] = 0
                data["placed_symbols_today"] = []
                data["pending_entry_order_id"] = None
                data["pending_entry_placed_at"] = None
                data["pending_trade_plan"] = None
                data["pending_gtt_trigger_id"] = None
                data["pending_symbol"] = None
                data["signal_fired_today"] = False
                data["last_entry_ready"] = None
                data["last_checklist_ready"] = None
                data["eval_count"] = 0
            cfg_raw = data.get("config") or {}
            self._cfg = WatchConfig(
                direction=str(cfg_raw.get("direction") or "AUTO"),
                num_lots=max(1, min(50, int(cfg_raw.get("num_lots") or 1))),
                risk_percentage=cfg_raw.get("risk_percentage"),
                reward_percentage=cfg_raw.get("reward_percentage"),
                mode=str(cfg_raw.get("mode") or "alert"),
                auto_place_on_signal=bool(cfg_raw.get("auto_place_on_signal")),
                auto_execute_checklist=bool(cfg_raw.get("auto_execute_checklist", True)),
                disarm_after_place=bool(cfg_raw.get("disarm_after_place", True)),
            )
            self._armed = bool(data.get("armed"))
            self._placed_count_today = int(data.get("placed_count_today") or 0)
            syms = data.get("placed_symbols_today") or []
            self._placed_symbols_today = [str(s).upper() for s in syms if s]
            self._placed_today = bool(data.get("placed_today")) or self._placed_count_today > 0
            self._placed_symbol_today = data.get("placed_symbol_today") or (
                self._placed_symbols_today[-1] if self._placed_symbols_today else None
            )
            self._pending_entry_order_id = data.get("pending_entry_order_id")
            self._pending_entry_placed_at = data.get("pending_entry_placed_at")
            self._pending_trade_plan = data.get("pending_trade_plan")
            self._pending_gtt_trigger_id = data.get("pending_gtt_trigger_id")
            self._pending_symbol = data.get("pending_symbol")
            self._signal_fired_today = bool(data.get("signal_fired_today"))
            self._eval_count = int(data.get("eval_count") or 0)
            ler = data.get("last_entry_ready")
            self._last_entry_ready = ler if ler is None else bool(ler)
            lcr = data.get("last_checklist_ready")
            self._last_checklist_ready = lcr if lcr is None else bool(lcr)
            self._session_date = _today()
            for ev in (data.get("events") or [])[:_MAX_EVENTS]:
                if isinstance(ev, dict):
                    fields = {k: ev.get(k) for k in WatchEvent.__dataclass_fields__ if k in ev}
                    if fields.get("at") and fields.get("kind") and fields.get("message"):
                        self._events.append(WatchEvent(**fields))

    def _persist(self) -> None:
        with _lock:
            data = {
                "armed": self._armed,
                "session_date": _today_iso(),
                "placed_today": bool(self._placed_count_today > 0),
                "placed_symbol_today": self._placed_symbol_today,
                "placed_count_today": int(self._placed_count_today),
                "placed_symbols_today": list(self._placed_symbols_today),
                "pending_entry_order_id": self._pending_entry_order_id,
                "pending_entry_placed_at": self._pending_entry_placed_at,
                "pending_trade_plan": self._pending_trade_plan,
                "pending_gtt_trigger_id": self._pending_gtt_trigger_id,
                "pending_symbol": self._pending_symbol,
                "signal_fired_today": self._signal_fired_today,
                "config": asdict(self._cfg),
                "events": [e.to_dict() for e in list(self._events)],
                "eval_count": self._eval_count,
                "last_entry_ready": self._last_entry_ready,
                "last_checklist_ready": self._last_checklist_ready,
            }
        _write_persisted(data)

    @staticmethod
    def _kill_switch_active() -> bool:
        try:
            from services.risk_gate import is_kill_switch_active

            return is_kill_switch_active("nifty")
        except Exception:
            return True

    def _get_place_lock(self) -> asyncio.Lock:
        if self._place_lock is None:
            self._place_lock = asyncio.Lock()
        return self._place_lock

    def _reset_day_if_needed(self) -> None:
        today = _today()
        if self._session_date != today:
            from services.gate_audit import emit_gate_audit_day_summary

            if self._session_date is not None:
                emit_gate_audit_day_summary("nifty50")
            self._session_date = today
            self._signal_fired_today = False
            self._placed_today = False
            self._placed_symbol_today = None
            self._placed_count_today = 0
            self._placed_symbols_today = []
            self._pending_entry_order_id = None
            self._pending_entry_placed_at = None
            self._pending_trade_plan = None
            self._pending_gtt_trigger_id = None
            self._pending_symbol = None
            self._last_entry_ready = None
            self._last_checklist_ready = None
            self._eval_count = 0

    def _max_trades_per_day(self) -> int:
        """Per-day autonomous cap — higher in paper mode for practice runs."""
        if self._last_paper_mode:
            try:
                raw = os.getenv("PAPER_AUTO_MAX_TRADES_PER_DAY", "30").strip()
                return max(1, min(100, int(raw or 30)))
            except Exception:
                return 30
        try:
            raw = os.getenv("V2_WATCH_MAX_TRADES_PER_DAY", "").strip()
            if raw:
                return max(1, min(50, int(raw)))
        except Exception:
            pass
        try:
            return max(1, min(50, int(get_agent_config().max_trades_per_day or 10)))
        except Exception:
            return 10

    def _setup_invalidated(self, plan: Dict[str, Any]) -> tuple[bool, str]:
        """Cancel pending LIMIT when live setup no longer matches the placed entry."""
        try:
            from services.v2_order_guard import min_entry_confirmation_score

            min_score = int(min_entry_confirmation_score() or 65)
        except Exception:
            min_score = 65
        with _lock:
            pending_plan = self._pending_trade_plan
            pending_sym = self._pending_symbol
            paper = self._last_paper_mode
        return pending_entry_invalidated(
            pending_plan=pending_plan,
            pending_symbol=pending_sym,
            current_plan=plan,
            min_score=min_score,
            paper_mode=paper,
        )

    def _order_status(self, order_id: str) -> Optional[str]:
        oid = (order_id or "").strip()
        if not oid:
            return None
        try:
            from utils.kite_utils import get_kite_instance

            kite = get_kite_instance()
            for o in kite.orders() or []:
                if str(o.get("order_id") or "") == oid:
                    return str(o.get("status") or "").upper() or None
        except Exception:
            return None
        return None

    def _order_fill_price(self, order_id: str) -> Optional[float]:
        oid = (order_id or "").strip()
        if not oid:
            return None
        try:
            from utils.kite_utils import get_kite_instance

            kite = get_kite_instance()
            for o in kite.orders() or []:
                if str(o.get("order_id") or "") == oid:
                    avg = o.get("average_price")
                    if avg is not None and float(avg) > 0:
                        return float(avg)
                    px = o.get("price")
                    if px is not None and float(px) > 0:
                        return float(px)
        except Exception:
            return None
        return None

    async def _on_entry_filled(self) -> None:
        with _lock:
            if self._gtt_attach_in_progress:
                return
            entry_id = self._pending_entry_order_id
            if not entry_id:
                return
            self._gtt_attach_in_progress = True
            plan = copy.deepcopy(self._pending_trade_plan) if self._pending_trade_plan else None
            sym = self._pending_symbol
            live_plan = copy.deepcopy(self._last_trade_plan) if self._last_trade_plan else None
            disarm = self._cfg.disarm_after_place
            self._pending_entry_order_id = None
            self._pending_entry_placed_at = None
            self._pending_trade_plan = None

        try:
            from services.v2_order_guard import min_entry_confirmation_score

            min_score = int(min_entry_confirmation_score() or 65)
        except Exception:
            min_score = 65
        invalid, why = pending_entry_invalidated(
            pending_plan=plan,
            pending_symbol=sym,
            current_plan=live_plan,
            min_score=min_score,
            paper_mode=False,
            post_fill=True,
        )
        if invalid:
            await self._abort_stale_fill(
                entry_id=entry_id,
                plan=plan,
                sym=sym,
                reason=why,
                disarm=disarm,
            )
            return

        gtt_id: Optional[str] = None
        gtt_detail = ""
        try:
            fill_px = self._order_fill_price(entry_id or "") if entry_id else None
            executed_plan = plan

            if plan:
                gtt_result = await asyncio.to_thread(
                    v2_trade_service.place_gtt_for_plan,
                    plan,
                    fill_price=fill_px,
                    entry_order_id=entry_id,
                )
                executed_plan = gtt_result.get("trade_plan") or plan
                gtt_id = gtt_result.get("gtt_trigger_id")
                if gtt_id:
                    sl_prem = float(executed_plan.get("stop_loss_premium") or 0)
                    tp_prem = float(executed_plan.get("target_premium") or 0)
                    gtt_detail = f"GTT OCO {gtt_id} · SL ₹{sl_prem:.2f} TP ₹{tp_prem:.2f}"
                else:
                    errs = "; ".join(gtt_result.get("errors") or ["GTT failed"])
                    gtt_detail = f"GTT failed: {errs}"[:180]

                try:
                    from services.risk_gate import record_order_placed

                    qty = int(plan.get("quantity") or 0)
                    px = float(
                        fill_px
                        or plan.get("entry_limit_price")
                        or plan.get("entry_premium")
                        or 0
                    )
                    record_order_placed(max(0.0, qty * px))
                except Exception:
                    pass

            fill_bit = f" @ ₹{fill_px:.2f}" if fill_px else ""
            with _lock:
                self._pending_gtt_trigger_id = str(gtt_id) if gtt_id else None
                self._pending_symbol = None
                if disarm:
                    self._armed = False

            self._push_event(
                WatchEvent(
                    at=datetime.now(IST).isoformat(),
                    kind="auto_gtt_placed" if gtt_id else "auto_gtt_failed",
                    message=(
                        f"Entry {entry_id} filled{fill_bit} — {gtt_detail or 'no exit plan'}"
                        + (" (watch disarmed)" if disarm else "")
                    )[:240],
                    tradingsymbol=sym,
                )
            )
        finally:
            with _lock:
                self._gtt_attach_in_progress = False

    async def _abort_stale_fill(
        self,
        *,
        entry_id: Optional[str],
        plan: Optional[Dict[str, Any]],
        sym: Optional[str],
        reason: str,
        disarm: bool = False,
    ) -> None:
        with _lock:
            if self._gtt_attach_in_progress:
                return
            self._gtt_attach_in_progress = True
            self._pending_entry_order_id = None
            self._pending_entry_placed_at = None
            self._pending_trade_plan = None
        try:
            fill_px = self._order_fill_price(entry_id or "") if entry_id else None
            exit_id = None
            exit_err = None
            if plan and sym and not str(entry_id or "").upper().startswith("PAPER-"):
                qty = int(plan.get("quantity") or 1)
                try:
                    from agent.tools.kite_tools import place_order_tool

                    result = await asyncio.to_thread(
                        place_order_tool.invoke,
                        {
                            "tradingsymbol": sym,
                            "exchange": plan.get("exchange") or "NFO",
                            "transaction_type": "SELL",
                            "quantity": qty,
                            "order_type": "MARKET",
                            "product": plan.get("product") or "MIS",
                            "segment": "nifty",
                        },
                    )
                    if isinstance(result, dict) and result.get("status") == "success":
                        exit_id = result.get("order_id")
                    else:
                        exit_err = (result or {}).get("error") if isinstance(result, dict) else str(result)
                except Exception as exc:
                    exit_err = str(exc)

            fill_bit = f" @ ₹{fill_px:.2f}" if fill_px else ""
            if exit_id:
                detail = f"market exit {exit_id}"
                kind = "auto_stale_abort"
            elif exit_err:
                detail = f"exit failed — {exit_err[:120]}"
                kind = "auto_stale_abort_failed"
            else:
                detail = "no exit placed"
                kind = "auto_stale_abort_failed"

            with _lock:
                self._pending_gtt_trigger_id = None
                self._pending_symbol = None
                if disarm:
                    self._armed = False

            self._push_event(
                WatchEvent(
                    at=datetime.now(IST).isoformat(),
                    kind=kind,
                    message=(
                        f"Stale fill aborted — {reason[:120]}"
                        f" · entry {entry_id}{fill_bit} · {detail}"
                        + (" (watch disarmed)" if disarm else "")
                    )[:240],
                    tradingsymbol=sym,
                )
            )
            log_warning(
                f"[V2Watch] stale fill abort entry={entry_id} sym={sym} reason={reason} exit={exit_id}"
            )
        finally:
            with _lock:
                self._gtt_attach_in_progress = False

    def _cancel_pending(self, *, reason: str, rollback_slot: bool = True) -> None:
        with _lock:
            entry_id = self._pending_entry_order_id
            gtt_id = self._pending_gtt_trigger_id
            sym = self._pending_symbol
            plan = self._pending_trade_plan
            had_entry = bool(entry_id)
            self._pending_entry_order_id = None
            self._pending_entry_placed_at = None
            self._pending_trade_plan = None
            self._pending_gtt_trigger_id = None
            self._pending_symbol = None
            if rollback_slot and had_entry and self._placed_count_today > 0:
                self._placed_count_today -= 1
            self._placed_today = self._placed_count_today > 0
            if not self._placed_today:
                self._placed_symbol_today = None
            elif sym:
                up = sym.upper()
                if up in self._placed_symbols_today and self._placed_count_today <= 0:
                    self._placed_symbols_today = [s for s in self._placed_symbols_today if s != up]
        if rollback_slot and had_entry and plan:
            try:
                from services.risk_gate import rollback_order_reserved

                qty = int(plan.get("quantity") or 0)
                px = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
                rollback_order_reserved(max(0.0, qty * px))
            except Exception:
                pass
        try:
            from agent.tools.kite_tools import cancel_order_tool, delete_gtt_tool

            if entry_id:
                st = (self._order_status(entry_id) or "").upper()
                if is_filled_order_status(st):
                    log_warning(
                        f"[V2Watch] skip cancel — entry {entry_id} already {st}; use stale abort"
                    )
                elif st not in ("CANCELLED", "REJECTED"):
                    cancel_order_tool.invoke({"order_id": entry_id, "variety": "regular"})
            if gtt_id:
                try:
                    delete_gtt_tool.invoke({"trigger_id": int(str(gtt_id))})
                except Exception:
                    pass
        except Exception:
            pass
        self._push_event(
            WatchEvent(
                at=datetime.now(IST).isoformat(),
                kind="auto_cancelled",
                message=f"Cancelled pending entry{f' {entry_id}' if entry_id else ''}"
                f"{f' + GTT {gtt_id}' if gtt_id else ''}: {reason}"[:240],
                tradingsymbol=sym,
            )
        )

    def _push_event(self, ev: WatchEvent) -> None:
        self._events.appendleft(ev)
        self._persist()

    def _skip_log_interval_sec(self) -> float:
        try:
            sec = float(os.getenv("V2_WATCH_SKIP_LOG_SECONDS", "90") or 90)
            return max(15.0, min(600.0, sec))
        except (TypeError, ValueError):
            return 90.0

    def _record_autonomous_skip(
        self,
        message: str,
        plan: Dict[str, Any],
    ) -> None:
        from services.watch_skip_utils import normalize_skip_message

        msg = normalize_skip_message(message)
        sym = str(plan.get("tradingsymbol") or "")
        now = datetime.now(IST)
        should_emit = False
        with _lock:
            self._last_autonomous_block_reason = msg
            same = msg == self._last_skip_logged_msg
            elapsed = (
                (now - self._last_skip_logged_at).total_seconds()
                if self._last_skip_logged_at
                else 9999.0
            )
            if not same or elapsed >= self._skip_log_interval_sec():
                should_emit = True
                self._last_skip_logged_msg = msg
                self._last_skip_logged_at = now
        if should_emit:
            log_info(f"[V2Watch] auto_skipped {sym} — {msg}")
            self._push_event(
                WatchEvent(
                    at=now.isoformat(),
                    kind="auto_skipped",
                    message=msg,
                    tradingsymbol=sym or None,
                    block_reason=msg,
                )
            )

    def arm(
        self,
        *,
        direction: str = "AUTO",
        num_lots: int = 1,
        risk_percentage: Optional[float] = None,
        reward_percentage: Optional[float] = None,
        mode: str = "alert",
        auto_place_on_signal: bool = False,
        auto_execute_checklist: bool = True,
        disarm_after_place: bool = True,
    ) -> Dict[str, Any]:
        mode, auto_place = _normalize_mode(mode, auto_place_on_signal)
        if mode == "autonomous" and watch_autonomous_globally_disabled():
            raise ValueError(
                "Autonomous trading is disabled on this server (V2_WATCH_DISABLE_AUTONOMOUS)"
            )
        if mode == "autonomous" and self._kill_switch_active():
            raise ValueError("Kill switch is ON — release it in Operations → Risk Control first")
        with _lock:
            self._reset_day_if_needed()
            self._cfg = WatchConfig(
                direction=direction.upper() if direction else "AUTO",
                num_lots=max(1, min(50, int(num_lots or 1))),
                risk_percentage=risk_percentage,
                reward_percentage=reward_percentage,
                mode=mode,
                auto_place_on_signal=auto_place,
                auto_execute_checklist=bool(auto_execute_checklist),
                disarm_after_place=bool(disarm_after_place),
            )
            self._armed = True
        auto_suffix = ""
        if mode == "autonomous":
            try:
                from services.v2_order_guard import min_entry_confirmation_score

                auto_suffix = (
                    f" · auto on confirmed entry (score≥{min_entry_confirmation_score()})"
                )
            except Exception:
                auto_suffix = " · auto on confirmed entry"
        self._push_event(
            WatchEvent(
                at=datetime.now(IST).isoformat(),
                kind="armed",
                message=(
                    f"Watch armed ({mode}) — {self._cfg.direction}, {self._cfg.num_lots} lot(s)"
                    + auto_suffix
                ),
            )
        )
        log_info(f"[V2Watch] Armed mode={mode} autonomous={auto_place}")
        self._ensure_loop_task()
        return self.status()

    def reset_placement_counters(self) -> None:
        """Clear daily place slots (e.g. after paper fund reset). Watch stays armed."""
        with _lock:
            self._placed_count_today = 0
            self._placed_today = False
            self._placed_symbol_today = None
            self._placed_symbols_today = []
            self._pending_entry_order_id = None
            self._pending_entry_placed_at = None
            self._pending_trade_plan = None
            self._pending_gtt_trigger_id = None
            self._pending_symbol = None
            self._signal_fired_today = False
        self._persist()
        log_info("[V2Watch] Placement counters reset")

    def disarm(self) -> Dict[str, Any]:
        with _lock:
            was = self._armed
            self._armed = False
        if was:
            self._push_event(
                WatchEvent(
                    at=datetime.now(IST).isoformat(),
                    kind="disarmed",
                    message="Strategy watch stopped",
                )
            )
        else:
            self._persist()
        from services.gate_audit import emit_gate_audit_day_summary

        emit_gate_audit_day_summary("nifty50")
        log_info("[V2Watch] Disarmed")
        return self.status()

    def _clear_runtime_state(self) -> None:
        self._armed = False
        self._cfg = WatchConfig()
        self._session_date = _today()
        self._last_entry_ready = None
        self._last_checklist_ready = None
        self._last_can_place = False
        self._last_can_execute = False
        self._last_block_reason = None
        self._last_trade_plan = None
        self._last_strategy_analysis = None
        self._last_eval_at = None
        self._signal_fired_today = False
        self._placed_today = False
        self._placed_symbol_today = None
        self._placed_count_today = 0
        self._placed_symbols_today = []
        self._pending_entry_order_id = None
        self._pending_entry_placed_at = None
        self._pending_trade_plan = None
        self._pending_gtt_trigger_id = None
        self._pending_symbol = None
        self._eval_count = 0
        self._events.clear()
        self._placing = False
        self._last_autonomous_block_reason = None
        self._last_skip_logged_msg = None
        self._last_skip_logged_at = None
        self._last_step_statuses = []
        self._last_market_open = False
        self._last_paper_mode = False
        self._last_kite_connected = False

    def nuclear_reset(self) -> Dict[str, Any]:
        """Stop watch, wipe persisted state file, and clear in-memory counters/events."""
        with _lock:
            self._clear_runtime_state()
        try:
            if _STATE_PATH.exists():
                _STATE_PATH.unlink()
        except Exception as exc:
            log_error(f"[V2Watch] nuclear reset unlink failed: {exc}")
            _write_persisted(_default_persisted())
        self._push_event(
            WatchEvent(
                at=datetime.now(IST).isoformat(),
                kind="nuclear_reset",
                message="Watch state cleared — event log and daily counters reset",
            )
        )
        log_info("[V2Watch] Nuclear reset")
        return self.status()

    def status(self) -> Dict[str, Any]:
        from services.watch_setup_status import describe_autonomous_setup

        with _lock:
            plan = self._last_trade_plan or {}
            sa = self._last_strategy_analysis or {}
            cfg = self._cfg
            min_score = int(os.getenv("NIFTY_AUTO_MIN_ENTRY_SCORE", "65") or 65)
            setup = describe_autonomous_setup(
                plan,
                min_score=min_score,
                guard_message=self._last_autonomous_block_reason,
            )
            from services.watch_readiness import build_readiness_payload

            from services.segment_balance import is_kite_broker_connected

            base = {
                "armed": self._armed,
                "mode": cfg.mode,
                "autonomous": cfg.mode == "autonomous",
                "direction": cfg.direction,
                "num_lots": cfg.num_lots,
                "auto_place_on_signal": cfg.auto_place_on_signal,
                "autonomous_server_allowed": watch_autonomous_allowed(),
                "autonomous_globally_disabled": watch_autonomous_globally_disabled(),
                "kill_switch_active": self._kill_switch_active(),
                "auto_place_server_allowed": watch_autonomous_allowed(),
                "disarm_after_place": cfg.disarm_after_place,
                "auto_execute_checklist": cfg.auto_execute_checklist,
                "persisted": _STATE_PATH.exists(),
                "poll_interval_sec": _poll_interval(),
                "last_eval_at": self._last_eval_at.isoformat() if self._last_eval_at else None,
                "eval_count": self._eval_count,
                "entry_ready": self._last_entry_ready,
                "checklist_ready": self._last_checklist_ready,
                "can_place": self._last_can_place,
                "can_execute": self._last_can_execute,
                "entry_block_reason": self._last_block_reason,
                "signal_fired_today": self._signal_fired_today,
                "placed_today": self._placed_today,
                "placed_symbol_today": self._placed_symbol_today,
                "strategy_name": plan.get("strategy_name"),
                "tradingsymbol": plan.get("tradingsymbol"),
                "nifty_spot": (plan.get("indicators") or {}).get("nifty_spot"),
                "strategy_candidates": (sa.get("strategies") or [])[:5] if isinstance(sa, dict) else [],
                "min_entry_score": min_score,
                "entry_confirmation_score": setup.get("entry_confirmation_score"),
                "autonomous_eligible": setup.get("autonomous_eligible"),
                "setup_phase": setup.get("setup_phase"),
                "setup_detail": setup.get("setup_detail"),
                "autonomous_block_reason": self._last_autonomous_block_reason,
                "validation": self._last_validation,
                "events": [e.to_dict() for e in list(self._events)],
            }
            extras = build_readiness_payload(
                armed=self._armed,
                autonomous_mode=cfg.mode == "autonomous",
                plan=plan,
                checklist_ready=bool(self._last_checklist_ready),
                entry_ready=self._last_entry_ready,
                can_place=bool(self._last_can_place),
                can_execute=bool(self._last_can_execute),
                autonomous_eligible=bool(setup.get("autonomous_eligible")),
                kill_switch_active=self._kill_switch_active(),
                market_open=v2_trade_service.is_market_session_open(),
                paper_trading_mode=bool(self._last_paper_mode),
                kite_connected=is_kite_broker_connected() or bool(self._last_kite_connected),
                guard_message=self._last_autonomous_block_reason,
                min_entry_score=min_score,
                entry_confirmation_score=setup.get("entry_confirmation_score"),
                pending_entry_order_id=self._pending_entry_order_id,
                step_statuses=self._last_step_statuses,
                segment="nifty50",
                validation=self._last_validation,
                missing_steps=self._last_missing_steps,
            )
            return {**base, **extras}

    def events(self, limit: int = 20) -> List[Dict[str, Any]]:
        with _lock:
            return [e.to_dict() for e in list(self._events)[: max(1, min(limit, _MAX_EVENTS))]]

    def register_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        with _lock:
            armed = self._armed
        if armed:
            log_info("[V2Watch] Restored armed watch from disk — starting loop")
            loop.create_task(self._run_loop())

    def _ensure_loop_task(self) -> None:
        if self._loop and (self._task is None or self._task.done()):
            self._task = self._loop.create_task(self._run_loop())

    async def _run_loop(self) -> None:
        while True:
            with _lock:
                if not self._armed:
                    break
            fire_signal = False
            try_autonomous = False
            preview = None
            plan: Dict[str, Any] = {}
            can_place = False
            try:
                from services.pnl_sync import maybe_sync_pnl_for_watch

                maybe_sync_pnl_for_watch("v2")

                session_open = v2_trade_service.is_market_session_open()

                if session_open or self._last_paper_mode:
                    (
                        fire_signal,
                        try_autonomous,
                        preview,
                        plan,
                        can_place,
                    ) = await asyncio.to_thread(self._evaluate_sync)

                if self._pending_entry_order_id:
                    if str(self._pending_entry_order_id).upper().startswith("PAPER-"):
                        from services.paper_order_guard import is_paper_position_open

                        if not is_paper_position_open(self._pending_entry_order_id):
                            with _lock:
                                self._pending_entry_order_id = None
                                self._pending_entry_placed_at = None
                                self._pending_trade_plan = None
                                self._pending_gtt_trigger_id = None
                                self._pending_symbol = None
                    elif not session_open:
                        self._cancel_pending(reason="Market session closed")
                    else:
                        invalid, why = self._setup_invalidated(plan)
                        status = self._order_status(self._pending_entry_order_id)
                        if invalid:
                            if is_open_order_status(status):
                                self._cancel_pending(reason=why)
                            elif is_filled_order_status(status):
                                with _lock:
                                    entry_id = self._pending_entry_order_id
                                    stale_plan = (
                                        copy.deepcopy(self._pending_trade_plan)
                                        if self._pending_trade_plan
                                        else None
                                    )
                                    stale_sym = self._pending_symbol
                                    disarm = self._cfg.disarm_after_place
                                await self._abort_stale_fill(
                                    entry_id=entry_id,
                                    plan=stale_plan,
                                    sym=stale_sym,
                                    reason=why,
                                    disarm=disarm,
                                )
                        if self._pending_entry_order_id:
                            try:
                                timeout_sec = float(os.getenv("V2_WATCH_ENTRY_TIMEOUT_SEC", "600") or 600)
                                if self._pending_entry_placed_at and timeout_sec > 0:
                                    placed_at = datetime.fromisoformat(str(self._pending_entry_placed_at))
                                    age = (datetime.now(IST) - placed_at).total_seconds()
                                    if age > timeout_sec:
                                        st = self._order_status(self._pending_entry_order_id)
                                        if is_open_order_status(st):
                                            self._cancel_pending(reason=f"Entry timeout ({int(age)}s)")
                            except Exception as exc:
                                log_warning(f"[V2Watch] entry timeout parse failed: {exc}")
                            from services.watch_reconcile import reconcile_pending_watch

                            with _lock:
                                rec = reconcile_pending_watch(
                                    entry_order_id=self._pending_entry_order_id,
                                    gtt_trigger_id=self._pending_gtt_trigger_id,
                                    pending_trade_plan=self._pending_trade_plan,
                                    order_status=self._order_status,
                                )
                            if rec.get("clear_gtt"):
                                with _lock:
                                    self._pending_gtt_trigger_id = None
                            status = self._order_status(self._pending_entry_order_id)
                            if status in ("COMPLETE", "EXECUTED") or rec.get("attach_gtt"):
                                await self._on_entry_filled()
                            elif status in ("CANCELLED", "REJECTED") or rec.get("clear_entry"):
                                self._cancel_pending(
                                    reason=f"Order {status.lower() if status else 'reconciled'}"
                                )

                if session_open or self._last_paper_mode:
                    if fire_signal and preview is not None:
                        await self._on_signal_ready(preview, plan, can_place, try_autonomous)
                    elif try_autonomous and preview is not None:
                        await self._try_auto_place(preview, plan)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_error(f"[V2Watch] loop error: {exc}")
                self._push_event(
                    WatchEvent(
                        at=datetime.now(IST).isoformat(),
                        kind="eval_error",
                        message=str(exc)[:200],
                    )
                )
            await asyncio.sleep(_poll_interval())

    def _should_autonomous_place(self, cfg: WatchConfig) -> bool:
        if watch_autonomous_globally_disabled():
            return False
        return (
            cfg.mode == "autonomous"
            and cfg.auto_place_on_signal
            and self._placed_count_today < self._max_trades_per_day()
            and not self._pending_entry_order_id
        )

    def _evaluate_sync(
        self,
    ) -> tuple[bool, bool, Optional[Dict[str, Any]], Dict[str, Any], bool]:
        with _lock:
            if not self._armed:
                return False, False, None, {}, False
            cfg = self._cfg
            self._reset_day_if_needed()

        from services.watch_execute import resolve_can_execute

        market_open = v2_trade_service.is_market_session_open()
        preview = v2_trade_service.preview_trade(
            completed_steps=None,
            direction=cfg.direction,
            risk_percentage=cfg.risk_percentage,
            reward_percentage=cfg.reward_percentage,
            num_lots=cfg.num_lots,
            auto_execute=cfg.auto_execute_checklist,
        )
        plan = preview.get("trade_plan") or {}
        strategy_analysis = preview.get("strategy_analysis") or {}
        checklist_ready = bool(preview.get("checklist_ready"))
        entry_ready = plan.get("entry_ready") is True
        can_place = bool(preview.get("can_place"))
        can_execute = resolve_can_execute(
            preview, plan, offhours_allowed=v2_trade_service.allow_offhours_v2_place()
        )
        block = plan.get("entry_block_reason")

        fire_signal = False
        try_autonomous = False

        step_rows: List[Dict[str, Any]] = []
        for st in preview.get("step_statuses") or []:
            if isinstance(st, dict):
                step_rows.append(st)
            elif hasattr(st, "model_dump"):
                step_rows.append(st.model_dump())
            elif hasattr(st, "dict"):
                step_rows.append(st.dict())

        ind = (plan or {}).get("indicators") or {}
        from services.segment_balance import is_kite_broker_connected

        kite_connected = is_kite_broker_connected()
        if not kite_connected:
            # Fallback: live quotes in plan imply session data (legacy path)
            kite_connected = ind.get("nifty_spot") is not None or ind.get("option_ltp") is not None

        with _lock:
            self._eval_count += 1
            self._last_eval_at = datetime.now(IST)
            self._last_trade_plan = plan
            self._last_strategy_analysis = strategy_analysis if isinstance(strategy_analysis, dict) else {}
            self._last_can_place = can_place
            self._last_can_execute = can_execute
            self._last_block_reason = block
            self._last_step_statuses = step_rows
            self._last_market_open = market_open
            self._last_paper_mode = bool(preview.get("paper_trading_mode"))
            self._last_kite_connected = kite_connected
            val = preview.get("validation")
            self._last_validation = val if isinstance(val, dict) else None
            self._last_missing_steps = list(preview.get("missing_steps") or [])
            prev_chk = self._last_checklist_ready
            self._last_checklist_ready = checklist_ready
            self._last_entry_ready = entry_ready

            if checklist_ready and not self._signal_fired_today and prev_chk is not True:
                if prev_chk is False or (prev_chk is None and self._eval_count > 1):
                    fire_signal = True
                    self._signal_fired_today = True

            autonomous_armed = self._should_autonomous_place(cfg)
            if (
                autonomous_armed
                and checklist_ready
                and can_execute
                and entry_ready
                and not self._placing
            ):
                from services.v2_order_guard import autonomous_place_allowed

                allowed, guard_msg = autonomous_place_allowed(
                    plan,
                    placed_today=self._placed_count_today >= self._max_trades_per_day(),
                    segment="nifty50",
                )
                if allowed:
                    self._last_autonomous_block_reason = None
                    try_autonomous = True
                else:
                    self._record_autonomous_skip(guard_msg, plan)
            elif autonomous_armed and checklist_ready and not entry_ready:
                reason = plan.get("entry_block_reason") or "Entry not confirmed (entry_ready=false)"
                self._record_autonomous_skip(reason, plan)
            elif autonomous_armed and checklist_ready and entry_ready and not can_execute:
                from services.watch_skip_utils import can_execute_block_errors

                reasons = can_execute_block_errors(preview, plan, segment="nifty50")
                skip_msg = "; ".join(reasons[:2])
                self._record_autonomous_skip(skip_msg, plan)

        from services.gate_audit import record_gate_audit

        record_gate_audit(
            "nifty50",
            plan,
            preview,
            try_autonomous=try_autonomous,
            market_open=market_open,
            can_execute=can_execute,
            entry_ready=entry_ready,
        )

        self._persist()
        return fire_signal, try_autonomous, preview, plan, can_execute

    async def _on_signal_ready(
        self,
        preview: Dict[str, Any],
        plan: Dict[str, Any],
        can_place: bool,
        try_autonomous: bool,
    ) -> None:
        sym = plan.get("tradingsymbol") or "—"
        strat = plan.get("strategy_name") or preview.get("strategy_analysis", {}).get(
            "selected_name", "V2"
        )
        limit_px = plan.get("entry_limit_price") or plan.get("entry_premium")
        paper = bool(preview.get("paper_trading_mode"))
        score = plan.get("entry_confirmation_score")
        with _lock:
            autonomous = self._cfg.mode == "autonomous"
        title = f"V2 checklist complete — {strat}"
        if autonomous and try_autonomous:
            venue = "paper ledger" if paper else "LIMIT+GTT"
            body = f"{sym} — placing via {venue} (score {score})"
        elif autonomous:
            body = (
                f"{sym} checklist OK — "
                f"{plan.get('entry_block_reason') or 'waiting for validation or margin'}"
            )
        else:
            body = f"{sym} LIMIT ₹{limit_px} · confirm in wizard"
            if not can_place:
                body += " (validation or paper mode pending)"

        self._push_event(
            WatchEvent(
                at=datetime.now(IST).isoformat(),
                kind="signal_ready",
                message=body,
                entry_ready=True,
                can_place=can_place,
                block_reason=plan.get("entry_block_reason"),
                strategy_name=strat,
                tradingsymbol=sym,
            )
        )

        payload = {
            "kind": "signal_ready",
            "autonomous": autonomous,
            "strategy_name": strat,
            "tradingsymbol": sym,
            "can_place": can_place,
            "entry_ready": True,
            "trade_plan": plan,
        }
        try:
            await broadcast_agent_update("V2_WATCH_SIGNAL", payload)
        except Exception as exc:
            log_warning(f"[V2Watch] WS broadcast failed: {exc}")

        try:
            await push_service.send_to_user(
                user_id="default",
                title=title,
                body=body[:180],
                data={"type": "v2_watch_signal", "symbol": sym},
            )
        except Exception as exc:
            log_warning(f"[V2Watch] push failed: {exc}")

        log_info(
            f"[V2Watch] Checklist complete {sym} can_execute={can_place} "
            f"paper={bool(preview.get('paper_trading_mode'))} autonomous={try_autonomous}"
        )

        if try_autonomous:
            await self._try_auto_place(preview, plan)

    async def _try_auto_place(self, preview: Dict[str, Any], plan: Dict[str, Any]) -> None:
        async with self._get_place_lock():
            with _lock:
                if self._pending_entry_order_id:
                    return
                at_limit = self._placed_count_today >= self._max_trades_per_day()
                if at_limit or self._placing:
                    return
                cfg = self._cfg
                if not self._should_autonomous_place(cfg):
                    return

                from services.v2_order_guard import autonomous_place_allowed

                allowed, guard_msg = autonomous_place_allowed(
                    plan,
                    placed_today=at_limit,
                    segment="nifty50",
                )
                if not allowed:
                    self._push_event(
                        WatchEvent(
                            at=datetime.now(IST).isoformat(),
                            kind="auto_skipped",
                            message=guard_msg[:240],
                            tradingsymbol=plan.get("tradingsymbol"),
                        )
                    )
                    return
                self._placing = True

            try:
                from services.paper_trading import is_paper_mode_for_segment
                from services.risk_gate import check_order_allowed, is_kill_switch_active

                if is_kill_switch_active("nifty"):
                    self._push_event(
                        WatchEvent(
                            at=datetime.now(IST).isoformat(),
                            kind="auto_skipped",
                            message="Kill switch ON — autonomous place blocked",
                        )
                    )
                    return

                sym = plan.get("tradingsymbol") or ""
                qty = int(plan.get("quantity") or 0)
                entry_px = float(
                    plan.get("entry_limit_price") or plan.get("entry_premium") or 0
                )
                est_value = entry_px * qty
                paper_seg = is_paper_mode_for_segment("nifty50")
                ok, gate_msg = check_order_allowed(
                    "NFO",
                    sym,
                    qty,
                    "BUY",
                    estimated_value_inr=est_value,
                    skip_session_check=paper_seg,
                    segment="nifty50",
                )
                if not ok:
                    self._push_event(
                        WatchEvent(
                            at=datetime.now(IST).isoformat(),
                            kind="auto_skipped",
                            message=f"Risk gate: {gate_msg}",
                            tradingsymbol=sym,
                        )
                    )
                    return

                if paper_seg:
                    log_info("[V2Watch] Autonomous place in paper mode (nifty50)")

                result = await asyncio.to_thread(
                    v2_trade_service.place_trade,
                    completed_steps=None,
                    direction=cfg.direction,
                    risk_percentage=cfg.risk_percentage,
                    reward_percentage=cfg.reward_percentage,
                    num_lots=cfg.num_lots,
                    confirm=True,
                    auto_execute=cfg.auto_execute_checklist,
                    trade_plan_snapshot=plan,
                    defer_gtt_until_fill=True,
                )
                placed = bool(result.get("placed"))
                entry_id = result.get("entry_order_id")
                entry_submitted = bool(entry_id)
                sym = plan.get("tradingsymbol") or sym

                if entry_submitted or placed:
                    entry_limit = float(plan.get("entry_limit_price") or entry_px or 0)
                    fair = float(plan.get("entry_fair_premium") or entry_limit or 0)
                    sl_prem = float(plan.get("stop_loss_premium") or 0)
                    tp_prem = float(plan.get("target_premium") or 0)
                    spot_sl = plan.get("spot_stop_loss")
                    spot_tp = plan.get("spot_target")
                    style = str(plan.get("entry_style") or "")
                    sid = str(plan.get("strategy_id") or "")
                    score = int(plan.get("entry_confirmation_score") or 0)
                    trig = plan.get("entry_spot_trigger")
                    msg = (
                        f"Autonomous entry {entry_id} {sym} @ ₹{entry_limit}"
                        f" · SL ₹{sl_prem:.2f} TP ₹{tp_prem:.2f}"
                        + (
                            f" · spot SL {float(spot_sl):.0f} TP {float(spot_tp):.0f}"
                            if spot_sl is not None and spot_tp is not None
                            else ""
                        )
                        + (
                            f" · fair ₹{fair:.2f} · {sid or 'strategy'} {style}".rstrip()
                            if fair > 0 or sid or style
                            else ""
                        )
                        + (f" · trig {float(trig):.0f}" if trig is not None else "")
                        + (f" · score {score}" if score else "")
                        + (
                            " · GTT after fill"
                            if result.get("gtt_deferred")
                            else (
                                f" · GTT {result.get('gtt_trigger_id')}"
                                if result.get("gtt_trigger_id")
                                else ""
                            )
                        )
                    )
                    with _lock:
                        self._placed_count_today += 1
                        self._placed_today = True
                        self._placed_symbol_today = sym
                        from services.gate_audit import record_gate_audit_placed

                        record_gate_audit_placed("nifty50", sym or "—")
                        if sym:
                            up = sym.upper()
                            if up not in self._placed_symbols_today:
                                self._placed_symbols_today.append(up)
                        is_paper_oid = entry_id and str(entry_id).upper().startswith("PAPER-")
                        if is_paper_oid:
                            self._pending_entry_order_id = str(entry_id)
                            self._pending_entry_placed_at = datetime.now(IST).isoformat()
                            self._pending_gtt_trigger_id = None
                            self._pending_trade_plan = copy.deepcopy(plan)
                        else:
                            self._pending_entry_order_id = str(entry_id) if entry_id else None
                            self._pending_entry_placed_at = (
                                datetime.now(IST).isoformat() if entry_id else None
                            )
                            self._pending_gtt_trigger_id = (
                                str(result.get("gtt_trigger_id"))
                                if result.get("gtt_trigger_id")
                                else None
                            )
                            self._pending_trade_plan = (
                                copy.deepcopy(plan) if result.get("gtt_deferred") else None
                            )
                        self._pending_symbol = sym
                    kind = "auto_placed"
                else:
                    from services.watch_skip_utils import format_place_skip_message

                    errors = result.get("errors") or ["unknown"]
                    detail = "; ".join(str(e).strip() for e in errors if e and str(e).strip())[:200]
                    msg = format_place_skip_message(errors)
                    kind = "auto_skipped"
                    with _lock:
                        self._last_autonomous_block_reason = detail or "unknown"
                    log_info(f"[V2Watch] auto_skipped {sym} — {detail or 'unknown'}")

                self._push_event(
                    WatchEvent(
                        at=datetime.now(IST).isoformat(),
                        kind=kind,
                        message=msg[:240],
                        placed=entry_submitted or placed,
                        tradingsymbol=sym,
                    )
                )
                self._persist()
                await broadcast_agent_update(
                    "V2_WATCH_AUTO_PLACE",
                    {"placed": placed, "entry_submitted": entry_submitted, "result": result},
                )
                await push_service.send_to_user(
                    user_id="default",
                    title="V2 autonomous " + ("placed" if entry_submitted else "skipped"),
                    body=msg[:180],
                )
            except Exception as exc:
                log_error(f"[V2Watch] autonomous place failed: {exc}")
                self._push_event(
                    WatchEvent(
                        at=datetime.now(IST).isoformat(),
                        kind="auto_skipped",
                        message=str(exc)[:200],
                    )
                )
            finally:
                with _lock:
                    self._placing = False


_watch = V2StrategyWatch()


def arm_watch(**kwargs: Any) -> Dict[str, Any]:
    return _watch.arm(**kwargs)


def disarm_watch() -> Dict[str, Any]:
    return _watch.disarm()


def reset_watch_placement_counters() -> None:
    _watch.reset_placement_counters()


def nuclear_reset_watch() -> Dict[str, Any]:
    return _watch.nuclear_reset()


def get_watch_status() -> Dict[str, Any]:
    return _watch.status()


def get_watch_events(limit: int = 20) -> List[Dict[str, Any]]:
    return _watch.events(limit)


def register_v2_strategy_watch(loop: asyncio.AbstractEventLoop) -> None:
    _watch.register_loop(loop)
    st = _watch.status()
    if st.get("armed"):
        log_info(
            f"[V2Watch] Autonomous watch active mode={st.get('mode')} dir={st.get('direction')}"
        )
