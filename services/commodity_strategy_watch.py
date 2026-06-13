"""
Commodity Strategy Watch — CRUDEOILM autonomous with strict entry + no duplicates.

Modes:
  - alert: push/WS when checklist completes; user places manually.
  - autonomous: up to COMMODITY_WATCH_MAX_TRADES_PER_DAY (default 10 live) LIMIT+GTT per day;
    retries each poll until placed or disarmed.

Safeguards:
  - checklist must be fully complete (live auto-execute)
  - autonomous also requires entry confirmation score + guard checks
  - placed_today + open-order + position checks before place
  - in-flight place lock (no concurrent duplicate submits)
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
from typing import Any, Deque, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from agent.ws_manager import broadcast_agent_update
from services import commodity_trade_service
from services.commodity_order_guard import (
    autonomous_place_allowed,
    min_entry_confirmation_score,
)
from services.watch_pending_invalidation import (
    is_filled_order_status,
    is_open_order_status,
    pending_entry_invalidated,
)
from services.push.push_service import push_service
from utils.logger import log_error, log_info, log_warning

from services.commodity_config import IST, commodity_trading_cutoff_label, is_commodity_new_trading_allowed, is_past_commodity_trading_cutoff
from services.commodity_watch_pending import (
    cache_trade_plan,
    get_pending_entry,
    list_mcx_long_positions,
    migrate_pending_entries,
    migrate_trade_plans_by_symbol,
    pending_needing_gtt,
    register_pending_entry,
    remove_pending_entry,
    sync_legacy_pending_fields,
    symbols_with_exit_trail,
    update_pending_gtt,
)

_MAX_EVENTS = 40
_DEFAULT_POLL_SEC = 5
_STATE_PATH = Path(os.getenv("COMMODITY_WATCH_STATE_FILE", "data/commodity_strategy_watch.json"))
_lock = RLock()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def watch_autonomous_globally_disabled() -> bool:
    return _env_bool("COMMODITY_WATCH_DISABLE_AUTONOMOUS", False)


def watch_autonomous_allowed() -> bool:
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
        sec = float(os.getenv("COMMODITY_WATCH_POLL_SECONDS", str(_DEFAULT_POLL_SEC)))
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
    mode: str = "alert"
    auto_place_on_signal: bool = False
    auto_execute_checklist: bool = True
    disarm_after_place: bool = False


def _default_persisted() -> Dict[str, Any]:
    return {
        "armed": False,
        "session_date": _today_iso(),
        "placed_today": False,  # backward-compatible (derived from placed_count_today > 0)
        "placed_symbol_today": None,  # backward-compatible (last traded symbol)
        "placed_count_today": 0,
        "placed_symbols_today": [],
        "pending_entry_order_id": None,
        "pending_entry_placed_at": None,
        "pending_gtt_trigger_id": None,
        "pending_symbol": None,
        "pending_trade_plan": None,
        "pending_entries": {},
        "trade_plans_by_symbol": {},
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
        log_warning(f"[CommodityWatch] state read failed: {exc}")
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
        log_error(f"[CommodityWatch] state write failed: {exc}")


def _normalize_mode(mode: str, auto_place_on_signal: bool) -> tuple[str, bool]:
    m = (mode or "alert").strip().lower()
    if m in ("auto", "autonomous"):
        return "autonomous", True
    return "alert", False


class CommodityStrategyWatch:
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
        self._pending_gtt_trigger_id: Optional[str] = None
        self._pending_symbol: Optional[str] = None
        self._pending_trade_plan: Optional[Dict[str, Any]] = None
        self._pending_entries: Dict[str, Dict[str, Any]] = {}
        self._trade_plans_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._eval_count = 0
        self._events: Deque[WatchEvent] = deque(maxlen=_MAX_EVENTS)
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._place_lock: Optional[asyncio.Lock] = None
        self._placing = False
        self._gtt_attach_in_progress = False
        self._gtt_attach_order_ids: set[str] = set()
        self._last_orphan_gtt_reconcile_at: Optional[datetime] = None
        self._last_autonomous_block_reason: Optional[str] = None
        self._last_skip_logged_msg: Optional[str] = None
        self._last_skip_logged_at: Optional[datetime] = None
        self._last_step_statuses: List[Dict[str, Any]] = []
        self._last_market_open = False
        self._last_paper_mode = False
        self._last_kite_connected = False
        self._last_validation: Optional[Dict[str, Any]] = None
        self._last_missing_steps: List[int] = []
        # After exit/place, require cooldown or entry_ready edge before next autonomous entry.
        self._reentry_armed = True
        self._last_autonomous_placed_at: Optional[datetime] = None
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
                data["pending_gtt_trigger_id"] = None
                data["pending_symbol"] = None
                data["pending_trade_plan"] = None
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
                disarm_after_place=bool(cfg_raw.get("disarm_after_place", False)),
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
            self._pending_gtt_trigger_id = data.get("pending_gtt_trigger_id")
            self._pending_symbol = data.get("pending_symbol")
            self._pending_trade_plan = data.get("pending_trade_plan")
            self._pending_entries = migrate_pending_entries(data)
            self._trade_plans_by_symbol = migrate_trade_plans_by_symbol(data)
            (
                self._pending_entry_order_id,
                self._pending_entry_placed_at,
                self._pending_gtt_trigger_id,
                self._pending_symbol,
                self._pending_trade_plan,
            ) = sync_legacy_pending_fields(self._pending_entries)
            self._signal_fired_today = bool(data.get("signal_fired_today"))
            self._eval_count = int(data.get("eval_count") or 0)
            ler = data.get("last_entry_ready")
            self._last_entry_ready = ler if ler is None else bool(ler)
            lcr = data.get("last_checklist_ready")
            self._last_checklist_ready = lcr if lcr is None else bool(lcr)
            self._reentry_armed = bool(data.get("reentry_armed", True))
            lap = data.get("last_autonomous_placed_at")
            if lap:
                try:
                    self._last_autonomous_placed_at = datetime.fromisoformat(str(lap))
                    if self._last_autonomous_placed_at.tzinfo is None:
                        self._last_autonomous_placed_at = self._last_autonomous_placed_at.replace(
                            tzinfo=IST
                        )
                except Exception:
                    self._last_autonomous_placed_at = None
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
                "pending_gtt_trigger_id": self._pending_gtt_trigger_id,
                "pending_symbol": self._pending_symbol,
                "pending_trade_plan": self._pending_trade_plan,
                "pending_entries": copy.deepcopy(self._pending_entries),
                "trade_plans_by_symbol": copy.deepcopy(self._trade_plans_by_symbol),
                "signal_fired_today": self._signal_fired_today,
                "config": asdict(self._cfg),
                "events": [e.to_dict() for e in list(self._events)],
                "eval_count": self._eval_count,
                "last_entry_ready": self._last_entry_ready,
                "last_checklist_ready": self._last_checklist_ready,
                "reentry_armed": self._reentry_armed,
                "last_autonomous_placed_at": (
                    self._last_autonomous_placed_at.isoformat()
                    if self._last_autonomous_placed_at
                    else None
                ),
            }
        _write_persisted(data)

    @staticmethod
    def _kill_switch_active() -> bool:
        try:
            from services.risk_gate import is_kill_switch_active

            return is_kill_switch_active("commodity")
        except Exception:
            return True

    def _get_place_lock(self) -> asyncio.Lock:
        if self._place_lock is None:
            self._place_lock = asyncio.Lock()
        return self._place_lock

    def _reset_day_if_needed(self) -> None:
        today = _today()
        if self._session_date != today:
            self._session_date = today
            self._signal_fired_today = False
            self._placed_today = False
            self._placed_symbol_today = None
            self._placed_count_today = 0
            self._placed_symbols_today = []
            self._pending_entry_order_id = None
            self._pending_entry_placed_at = None
            self._pending_gtt_trigger_id = None
            self._pending_symbol = None
            self._pending_trade_plan = None
            self._pending_entries = {}
            self._trade_plans_by_symbol = {}
            self._last_entry_ready = None
            self._last_checklist_ready = None
            self._eval_count = 0
            self._reentry_armed = True
            self._last_autonomous_placed_at = None

    def _reentry_cooldown_sec(self) -> float:
        try:
            return max(
                30.0,
                min(900.0, float(os.getenv("COMMODITY_WATCH_REENTRY_COOLDOWN_SEC", "90") or 90)),
            )
        except Exception:
            return 90.0

    def _maybe_rearm_reentry(self, entry_ready: bool) -> None:
        """Allow next trade after cooldown or entry_ready false→true (multi-trade days)."""
        with _lock:
            if self._reentry_armed:
                return
            if not entry_ready:
                return
            if pending_needing_gtt(self._pending_entries) or self._gtt_attach_order_ids:
                return
            if self._placed_count_today >= self._max_trades_per_day():
                return

            if self._last_entry_ready is False and entry_ready:
                self._reentry_armed = True
                return

            if self._last_autonomous_placed_at is None:
                if self._placed_count_today == 0:
                    self._reentry_armed = True
                return

            age = (
                datetime.now(IST) - self._last_autonomous_placed_at.astimezone(IST)
            ).total_seconds()
            if age >= self._reentry_cooldown_sec():
                self._reentry_armed = True

    def _max_trades_per_day(self) -> int:
        """Per-day autonomous cap — higher in paper mode."""
        if self._last_paper_mode:
            try:
                raw = os.getenv("PAPER_AUTO_MAX_TRADES_PER_DAY", "30").strip()
                return max(1, min(100, int(raw or 30)))
            except Exception:
                return 30
        try:
            raw = os.getenv("COMMODITY_WATCH_MAX_TRADES_PER_DAY", "10").strip()
            return max(1, min(50, int(raw or 10)))
        except Exception:
            return 10

    def _should_disarm_watch(self) -> bool:
        """Disarm when daily cap reached, or after first place when disarm_after_place is on."""
        max_t = self._max_trades_per_day()
        if self._placed_count_today >= max_t:
            return True
        if self._cfg.disarm_after_place and self._placed_count_today >= 1:
            return True
        return False

    def _sync_legacy_pending(self) -> None:
        (
            self._pending_entry_order_id,
            self._pending_entry_placed_at,
            self._pending_gtt_trigger_id,
            self._pending_symbol,
            self._pending_trade_plan,
        ) = sync_legacy_pending_fields(self._pending_entries)

    def _register_pending(
        self,
        *,
        entry_id: str,
        sym: str,
        trade_plan: Optional[Dict[str, Any]],
        gtt_trigger_id: Optional[str],
        gtt_deferred: bool,
    ) -> None:
        placed_at = datetime.now(IST).isoformat()
        if trade_plan:
            cache_trade_plan(self._trade_plans_by_symbol, symbol=sym, trade_plan=trade_plan)
        if entry_id:
            register_pending_entry(
                self._pending_entries,
                order_id=entry_id,
                symbol=sym,
                placed_at=placed_at,
                trade_plan=trade_plan if (gtt_deferred or trade_plan) else None,
                gtt_trigger_id=gtt_trigger_id,
            )
            self._sync_legacy_pending()

    def _setup_invalidated(self, plan: Dict[str, Any]) -> tuple[bool, str]:
        """Cancel pending LIMIT when live setup no longer matches the placed entry."""
        with _lock:
            pending_plan = self._pending_trade_plan
            pending_sym = self._pending_symbol
            paper = self._last_paper_mode
        return pending_entry_invalidated(
            pending_plan=pending_plan,
            pending_symbol=pending_sym,
            current_plan=plan,
            min_score=min_entry_confirmation_score(),
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

    async def _on_entry_filled(self, entry_id: Optional[str] = None) -> None:
        oid = ""
        with _lock:
            oid = (entry_id or self._pending_entry_order_id or "").strip()
            if not oid or oid in self._gtt_attach_order_ids:
                return
            pe = get_pending_entry(self._pending_entries, oid)
            if pe and pe.get("gtt_trigger_id"):
                return
            plan = copy.deepcopy(
                (pe or {}).get("trade_plan") or self._pending_trade_plan
            )
            sym = (pe or {}).get("symbol") or self._pending_symbol
            live_plan = copy.deepcopy(self._last_trade_plan) if self._last_trade_plan else None
            disarm = self._should_disarm_watch()

        invalid, why = pending_entry_invalidated(
            pending_plan=plan,
            pending_symbol=sym,
            current_plan=live_plan,
            min_score=min_entry_confirmation_score(),
            paper_mode=False,
        )
        if invalid:
            await self._abort_stale_fill(
                entry_id=oid,
                plan=plan,
                sym=sym,
                reason=why,
                disarm=disarm,
            )
            with _lock:
                remove_pending_entry(self._pending_entries, oid)
                self._sync_legacy_pending()
            return

        with _lock:
            self._gtt_attach_order_ids.add(oid)
            self._gtt_attach_in_progress = True

        gtt_id: Optional[str] = None
        gtt_detail = ""
        try:
            fill_px = self._order_fill_price(oid) if oid else None
            executed_plan = plan

            if plan:
                gtt_result = await asyncio.to_thread(
                    commodity_trade_service.place_gtt_for_plan,
                    plan,
                    fill_price=fill_px,
                    entry_order_id=oid,
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

                    if not str(oid or "").upper().startswith("PAPER-"):
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
                if oid and gtt_id:
                    update_pending_gtt(self._pending_entries, oid, gtt_id)
                elif oid and not gtt_id:
                    log_error(
                        f"[CommodityWatch] GTT attach failed for {oid} {sym}: {gtt_detail}"
                    )
                if gtt_id:
                    remove_pending_entry(self._pending_entries, oid)
                self._sync_legacy_pending()
                if disarm:
                    self._armed = False

            self._push_event(
                WatchEvent(
                    at=datetime.now(IST).isoformat(),
                    kind="auto_gtt_placed" if gtt_id else "auto_gtt_failed",
                    message=(
                        f"Entry {oid} filled{fill_bit} — {gtt_detail or 'no exit plan'}"
                        + (" (watch disarmed — daily cap reached)" if disarm else "")
                    )[:240],
                    tradingsymbol=sym,
                )
            )
            if not gtt_id and oid:
                self._persist()
        finally:
            with _lock:
                if oid:
                    self._gtt_attach_order_ids.discard(oid)
                self._gtt_attach_in_progress = bool(self._gtt_attach_order_ids)

    async def _abort_stale_fill(
        self,
        *,
        entry_id: Optional[str],
        plan: Optional[Dict[str, Any]],
        sym: Optional[str],
        reason: str,
        disarm: bool = False,
    ) -> None:
        """Exit immediately when a LIMIT fills after the setup is no longer valid."""
        with _lock:
            if entry_id:
                remove_pending_entry(self._pending_entries, entry_id)
                self._sync_legacy_pending()
        try:
            fill_px = self._order_fill_price(entry_id or "") if entry_id else None
            exit_id = None
            exit_err = None
            if plan and sym and not str(entry_id or "").upper().startswith("PAPER-"):
                qty = int(plan.get("quantity") or plan.get("kite_qty") or 1)
                try:
                    from agent.tools.kite_tools import place_order_tool

                    result = await asyncio.to_thread(
                        place_order_tool.invoke,
                        {
                            "tradingsymbol": sym,
                            "exchange": plan.get("exchange") or "MCX",
                            "transaction_type": "SELL",
                            "quantity": qty,
                            "order_type": "MARKET",
                            "product": plan.get("product") or "NRML",
                            "segment": "commodity",
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
                self._reentry_armed = True
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
                f"[CommodityWatch] stale fill abort entry={entry_id} sym={sym} reason={reason} exit={exit_id}"
            )
        finally:
            with _lock:
                self._gtt_attach_in_progress = bool(self._gtt_attach_order_ids)

    def _cancel_pending(
        self,
        *,
        reason: str,
        rollback_slot: bool = True,
        entry_order_id: Optional[str] = None,
    ) -> None:
        with _lock:
            oid = (entry_order_id or self._pending_entry_order_id or "").strip()
            pe = get_pending_entry(self._pending_entries, oid) if oid else None
            entry_id = oid or None
            gtt_id = (pe or {}).get("gtt_trigger_id") or self._pending_gtt_trigger_id
            sym = (pe or {}).get("symbol") or self._pending_symbol
            plan = (pe or {}).get("trade_plan") or self._pending_trade_plan
            had_entry = bool(entry_id)
            if oid:
                remove_pending_entry(self._pending_entries, oid)
            self._sync_legacy_pending()
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
                        f"[CommodityWatch] skip cancel — entry {entry_id} already {st}; use stale abort"
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
            sec = float(os.getenv("COMMODITY_WATCH_SKIP_LOG_SECONDS", "90") or 90)
            return max(15.0, min(600.0, sec))
        except (TypeError, ValueError):
            return 90.0

    def _record_autonomous_skip(
        self,
        message: str,
        plan: Dict[str, Any],
    ) -> None:
        """Rate-limited auto_skipped event + log when guard blocks placement."""
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
            log_info(f"[CommodityWatch] auto_skipped {sym} — {msg}")
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
        disarm_after_place: bool = False,
    ) -> Dict[str, Any]:
        mode, auto_place = _normalize_mode(mode, auto_place_on_signal)
        if mode == "autonomous" and watch_autonomous_globally_disabled():
            raise ValueError(
                "Autonomous trading is disabled on this server (COMMODITY_WATCH_DISABLE_AUTONOMOUS)"
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
            self._reentry_armed = True
        self._push_event(
            WatchEvent(
                at=datetime.now(IST).isoformat(),
                kind="armed",
                message=(
                    f"Commodity watch armed ({mode}) — CRUDEOILM · {self._cfg.direction}, "
                    f"{self._cfg.num_lots} lot(s)"
                    + (
                        f" · auto on confirmed entry (score≥{min_entry_confirmation_score()})"
                        if mode == "autonomous"
                        else ""
                    )
                ),
            )
        )
        log_info(f"[CommodityWatch] Armed mode={mode} autonomous={auto_place}")
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
            self._reentry_armed = True
        self._persist()
        log_info("[CommodityWatch] Placement counters reset")

    def on_trading_mode_changed(self, paper_mode: bool) -> None:
        """Paper↔live toggle: clear paper fill state so live autonomous can place on Kite."""
        with _lock:
            self._last_paper_mode = bool(paper_mode)
        self.reset_placement_counters()
        venue = "paper ledger" if paper_mode else "Zerodha (live)"
        self._push_event(
            WatchEvent(
                at=datetime.now(IST).isoformat(),
                kind="mode_changed",
                message=(
                    f"Segment switched to {venue} — daily placement counters cleared; "
                    "re-entry armed for next confirmed setup"
                ),
            )
        )
        log_info(f"[CommodityWatch] Trading mode → {'paper' if paper_mode else 'live'} (counters reset)")

    def eod_shutdown(self, *, reason: str) -> Dict[str, Any]:
        """Cancel pending entry, disarm watch — called at daily cutoff."""
        with _lock:
            had_pending = bool(self._pending_entry_order_id)
        if had_pending:
            self._cancel_pending(reason=reason, rollback_slot=True)
        with _lock:
            was_armed = self._armed
            self._armed = False
            self._reentry_armed = False
        self._push_event(
            WatchEvent(
                at=datetime.now(IST).isoformat(),
                kind="eod_flatten",
                message=reason[:240],
            )
        )
        self._persist()
        log_info(f"[CommodityWatch] EOD shutdown — {reason}")
        return {"disarmed": was_armed or had_pending, "reason": reason}

    def disarm(self) -> Dict[str, Any]:
        with _lock:
            was = self._armed
            self._armed = False
        if was:
            self._push_event(
                WatchEvent(
                    at=datetime.now(IST).isoformat(),
                    kind="disarmed",
                    message="Commodity strategy watch stopped",
                )
            )
        else:
            self._persist()
        log_info("[CommodityWatch] Disarmed")
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
        self._pending_gtt_trigger_id = None
        self._pending_symbol = None
        self._pending_trade_plan = None
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
            log_error(f"[CommodityWatch] nuclear reset unlink failed: {exc}")
            _write_persisted(_default_persisted())
        self._push_event(
            WatchEvent(
                at=datetime.now(IST).isoformat(),
                kind="nuclear_reset",
                message="Commodity watch state cleared — event log and daily counters reset",
            )
        )
        log_info("[CommodityWatch] Nuclear reset")
        return self.status()

    def status(self) -> Dict[str, Any]:
        from services.watch_setup_status import describe_autonomous_setup

        with _lock:
            plan = self._last_trade_plan or {}
            sa = self._last_strategy_analysis or {}
            cfg = self._cfg
            min_score = min_entry_confirmation_score()
            setup = describe_autonomous_setup(
                plan,
                min_score=min_score,
                guard_message=self._last_autonomous_block_reason,
            )
            from services.watch_readiness import build_readiness_payload

            ind = plan.get("indicators") or {}
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
                "min_entry_score": min_entry_confirmation_score(),
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
                "placed_count_today": self._placed_count_today,
                "max_trades_per_day": self._max_trades_per_day(),
                "reentry_armed": self._reentry_armed,
                "reentry_cooldown_sec": int(self._reentry_cooldown_sec()),
                "placed_symbol_today": self._placed_symbol_today,
                "strategy_name": plan.get("strategy_name"),
                "tradingsymbol": plan.get("tradingsymbol"),
                "nifty_spot": ind.get("nifty_spot") or ind.get("crude_spot") or ind.get("spot"),
                "strategy_candidates": (sa.get("strategies") or [])[:5] if isinstance(sa, dict) else [],
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
                market_open=bool(self._last_market_open),
                paper_trading_mode=bool(self._last_paper_mode),
                kite_connected=is_kite_broker_connected() or bool(self._last_kite_connected),
                guard_message=self._last_autonomous_block_reason,
                min_entry_score=min_score,
                entry_confirmation_score=setup.get("entry_confirmation_score"),
                pending_entry_order_id=self._pending_entry_order_id,
                step_statuses=self._last_step_statuses,
                segment="commodity",
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
            log_info("[CommodityWatch] Restored armed watch from disk — starting loop")
            loop.create_task(self._run_loop())

    def _ensure_loop_task(self) -> None:
        if self._loop and (self._task is None or self._task.done()):
            self._task = self._loop.create_task(self._run_loop())

    async def _reconcile_orphan_mcx_gtt(self) -> None:
        """Attach GTT+trail for MCX positions that lost pending state (e.g. second entry overwrote first)."""
        with _lock:
            last = self._last_orphan_gtt_reconcile_at
        if last and (datetime.now(IST) - last.astimezone(IST)).total_seconds() < 60:
            return
        with _lock:
            self._last_orphan_gtt_reconcile_at = datetime.now(IST)

        trails = symbols_with_exit_trail()
        for pos in list_mcx_long_positions():
            sym = str(pos.get("tradingsymbol") or "").upper()
            if not sym or trails.get(sym):
                continue
            with _lock:
                pe_match = next(
                    (
                        pe
                        for pe in self._pending_entries.values()
                        if str(pe.get("symbol") or "").upper() == sym
                        and pe.get("trade_plan")
                        and not pe.get("gtt_trigger_id")
                    ),
                    None,
                )
                plan = (
                    (pe_match or {}).get("trade_plan")
                    or self._trade_plans_by_symbol.get(sym)
                )
                entry_id = (pe_match or {}).get("order_id")
            if not plan:
                log_warning(
                    f"[CommodityWatch] MCX position {sym} has no GTT/trail and no cached plan — manual exit required"
                )
                continue
            if entry_id:
                st = (self._order_status(entry_id) or "").upper()
                if st not in ("COMPLETE", "EXECUTED"):
                    continue
                await self._on_entry_filled(entry_id)
            else:
                log_warning(
                    f"[CommodityWatch] Orphan MCX position {sym} — attempting GTT from cached plan"
                )
                gtt_result = await asyncio.to_thread(
                    commodity_trade_service.place_gtt_for_plan,
                    plan,
                    fill_price=float(pos.get("average_price") or 0) or None,
                )
                if gtt_result.get("gtt_trigger_id"):
                    log_info(
                        f"[CommodityWatch] Orphan GTT attached for {sym}: {gtt_result.get('gtt_trigger_id')}"
                    )
                else:
                    log_error(
                        f"[CommodityWatch] Orphan GTT failed for {sym}: "
                        f"{'; '.join(gtt_result.get('errors') or [])}"
                    )

    async def _tick_all_pending_entries(
        self,
        plan: Dict[str, Any],
        *,
        session_open: bool,
    ) -> None:
        with _lock:
            pending_ids = list(self._pending_entries.keys())
        if not pending_ids and not self._pending_entry_order_id:
            return

        for oid in pending_ids:
            with _lock:
                pe = get_pending_entry(self._pending_entries, oid) or {}
                sym = pe.get("symbol")
                placed_at = pe.get("placed_at")
                trade_plan = pe.get("trade_plan")
                gtt_id = pe.get("gtt_trigger_id")
            if str(oid).upper().startswith("PAPER-"):
                from services.paper_order_guard import is_paper_position_open

                if not is_paper_position_open(oid):
                    with _lock:
                        remove_pending_entry(self._pending_entries, oid)
                        self._sync_legacy_pending()
                        self._reentry_armed = True
                continue
            if not session_open:
                self._cancel_pending(reason="MCX session closed", entry_order_id=oid)
                continue

            invalid, why = pending_entry_invalidated(
                pending_plan=trade_plan,
                pending_symbol=sym,
                current_plan=plan,
                min_score=min_entry_confirmation_score(),
                paper_mode=False,
            )
            if invalid and placed_at:
                try:
                    min_age = float(
                        os.getenv("COMMODITY_WATCH_PENDING_MIN_AGE_SEC", "45") or 45
                    )
                    if min_age > 0:
                        placed_dt = datetime.fromisoformat(str(placed_at))
                        age = (datetime.now(IST) - placed_dt).total_seconds()
                        if age < min_age and "session" not in why.lower():
                            invalid = False
                except Exception:
                    pass
            status = self._order_status(oid)
            if invalid:
                if is_open_order_status(status):
                    self._cancel_pending(reason=why, entry_order_id=oid)
                    continue
                if is_filled_order_status(status):
                    disarm = self._should_disarm_watch()
                    await self._abort_stale_fill(
                        entry_id=oid,
                        plan=trade_plan,
                        sym=sym,
                        reason=why,
                        disarm=disarm,
                    )
                    with _lock:
                        remove_pending_entry(self._pending_entries, oid)
                        self._sync_legacy_pending()
                    continue

            try:
                timeout_sec = float(
                    os.getenv("COMMODITY_WATCH_ENTRY_TIMEOUT_SEC", "900") or 900
                )
                if placed_at and timeout_sec > 0:
                    placed_dt = datetime.fromisoformat(str(placed_at))
                    age = (datetime.now(IST) - placed_dt).total_seconds()
                    if age > timeout_sec:
                        st = self._order_status(oid)
                        if is_open_order_status(st):
                            self._cancel_pending(
                                reason=f"Entry timeout ({int(age)}s)",
                                entry_order_id=oid,
                            )
                            continue
            except Exception as exc:
                log_warning(f"[CommodityWatch] entry timeout parse failed: {exc}")

            from services.watch_reconcile import reconcile_pending_watch

            rec = reconcile_pending_watch(
                entry_order_id=oid,
                gtt_trigger_id=gtt_id,
                pending_trade_plan=trade_plan,
                order_status=self._order_status,
            )
            if rec.get("clear_gtt"):
                with _lock:
                    update_pending_gtt(self._pending_entries, oid, None)
                    self._sync_legacy_pending()
            status = self._order_status(oid)
            if status in ("COMPLETE", "EXECUTED") or rec.get("attach_gtt"):
                await self._on_entry_filled(oid)
            elif status in ("CANCELLED", "REJECTED") or rec.get("clear_entry"):
                self._cancel_pending(
                    reason=f"Order {status.lower() if status else 'reconciled'}",
                    entry_order_id=oid,
                )

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

                maybe_sync_pnl_for_watch("commodity")

                from services.commodity_eod_flatten import maybe_run_commodity_eod_flatten

                eod = maybe_run_commodity_eod_flatten()
                if eod.get("date") and not eod.get("skipped"):
                    break

                from services.paper_trading import is_paper_mode_for_segment

                paper_active = is_paper_mode_for_segment("commodity")
                with _lock:
                    if paper_active:
                        self._last_paper_mode = True

                if is_past_commodity_trading_cutoff() and not paper_active:
                    with _lock:
                        pending_ids = list(self._pending_entries.keys())
                    for oid in pending_ids:
                        self._cancel_pending(
                            reason=(
                                f"Commodity cutoff {commodity_trading_cutoff_label()} IST "
                                "— no further trades today"
                            ),
                            entry_order_id=oid,
                        )
                    # EOD flatten disarms via commodity_eod_shutdown; idle until next session.
                    await asyncio.sleep(_poll_interval())
                    continue

                if not is_commodity_new_trading_allowed() and not paper_active:
                    # Weekend or before MCX open — stay armed and wait for the session window.
                    await asyncio.sleep(_poll_interval())
                    continue

                session_open = commodity_trade_service.is_mcx_session_open()

                if session_open or paper_active or self._last_paper_mode:
                    fire_signal, try_autonomous, preview, plan, can_place = await asyncio.to_thread(
                        self._evaluate_sync
                    )

                if self._pending_entries or self._pending_entry_order_id:
                    await self._tick_all_pending_entries(
                        plan, session_open=session_open or paper_active or self._last_paper_mode
                    )
                if session_open:
                    await self._reconcile_orphan_mcx_gtt()

                if session_open or paper_active or self._last_paper_mode:
                    if fire_signal and preview is not None:
                        await self._on_signal_ready(preview, plan, can_place, try_autonomous)
                    elif try_autonomous and preview is not None:
                        await self._try_auto_place(preview, plan)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_error(f"[CommodityWatch] loop error: {exc}")
                self._push_event(
                    WatchEvent(
                        at=datetime.now(IST).isoformat(),
                        kind="eval_error",
                        message=str(exc)[:200],
                    )
                )
                rate_limited = "too many request" in str(exc).lower()
            else:
                rate_limited = False
            await asyncio.sleep(_poll_interval() + (30.0 if rate_limited else 0.0))

    def _should_autonomous_place(self, cfg: WatchConfig) -> bool:
        if watch_autonomous_globally_disabled():
            return False
        if not is_commodity_new_trading_allowed():
            return False
        with _lock:
            pending_work = pending_needing_gtt(self._pending_entries)
        return (
            cfg.mode == "autonomous"
            and cfg.auto_place_on_signal
            and self._placed_count_today < self._max_trades_per_day()
            and not pending_work
        )

    def _autonomous_entry_allowed(self, entry_ready: bool) -> Tuple[bool, str]:
        """Cooldown or fresh entry edge before next autonomous place (up to daily cap)."""
        self._maybe_rearm_reentry(entry_ready)
        if not entry_ready:
            with _lock:
                self._reentry_armed = True
            return False, "Entry not confirmed (entry_ready=false)"
        with _lock:
            if not self._reentry_armed:
                cd = int(self._reentry_cooldown_sec())
                return (
                    False,
                    f"Re-entry cooldown — wait {cd}s after last trade or until entry resets",
                )
        return True, ""

    def _evaluate_sync(
        self,
    ) -> tuple[bool, bool, Optional[Dict[str, Any]], Dict[str, Any], bool]:
        with _lock:
            if not self._armed:
                return False, False, None, {}, False
            cfg = self._cfg
            self._reset_day_if_needed()

        from services.watch_execute import resolve_can_execute

        market_open = commodity_trade_service.is_mcx_session_open()
        preview = commodity_trade_service.preview_trade(
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
            preview,
            plan,
            offhours_allowed=commodity_trade_service.allow_offhours_commodity_place(),
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
            kite_connected = (
                ind.get("nifty_spot") is not None
                or ind.get("crude_spot") is not None
                or ind.get("spot") is not None
                or ind.get("option_ltp") is not None
            )

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
            aut_entry_ok, aut_entry_msg = self._autonomous_entry_allowed(entry_ready)
            if (
                autonomous_armed
                and checklist_ready
                and can_execute
                and entry_ready
                and aut_entry_ok
                and not self._placing
            ):
                allowed, guard_msg = autonomous_place_allowed(
                    plan,
                    placed_today=self._placed_count_today >= self._max_trades_per_day(),
                    segment="commodity",
                )
                if allowed:
                    self._last_autonomous_block_reason = None
                    try_autonomous = True
                else:
                    self._record_autonomous_skip(guard_msg, plan)
            elif autonomous_armed and checklist_ready and not aut_entry_ok:
                self._record_autonomous_skip(aut_entry_msg, plan)
            elif autonomous_armed and checklist_ready and entry_ready and not can_execute:
                from services.watch_skip_utils import can_execute_block_errors

                reasons = can_execute_block_errors(preview, plan, segment="commodity")
                skip_msg = "; ".join(reasons[:2])
                self._record_autonomous_skip(skip_msg, plan)

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
            "selected_name", "Commodity"
        )
        limit_px = plan.get("entry_limit_price") or plan.get("entry_premium")
        score = plan.get("entry_confirmation_score")
        paper = bool(preview.get("paper_trading_mode"))
        with _lock:
            autonomous = self._cfg.mode == "autonomous"
        title = f"Crude Mini checklist complete — {strat}"
        if autonomous and try_autonomous:
            venue = "paper ledger" if paper else "LIMIT+GTT"
            body = f"{sym} — placing via {venue} (score {score})"
        elif autonomous:
            body = (
                f"{sym} checklist OK — "
                f"{plan.get('entry_block_reason') or 'waiting for margin/validation/guard'}"
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
            await broadcast_agent_update("COMMODITY_WATCH_SIGNAL", payload)
        except Exception as exc:
            log_warning(f"[CommodityWatch] WS broadcast failed: {exc}")

        try:
            await push_service.send_to_user(
                user_id="default",
                title=title,
                body=body[:180],
                data={"type": "commodity_watch_signal", "symbol": sym},
            )
        except Exception as exc:
            log_warning(f"[CommodityWatch] push failed: {exc}")

        log_info(
            f"[CommodityWatch] Checklist complete {sym} can_execute={can_place} "
            f"paper={bool(preview.get('paper_trading_mode'))} auto={try_autonomous} score={score}"
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
                allowed, guard_msg = autonomous_place_allowed(
                    plan,
                    placed_today=at_limit,
                    segment="commodity",
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

                if is_kill_switch_active("commodity"):
                    self._push_event(
                        WatchEvent(
                            at=datetime.now(IST).isoformat(),
                            kind="auto_skipped",
                            message="Kill switch ON — autonomous place blocked",
                        )
                    )
                    return

                sym = plan.get("tradingsymbol") or ""
                qty_lots = int(plan.get("quantity") or plan.get("num_lots") or 1)
                units = int(plan.get("lot_size") or 10)
                entry_px = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
                est_value = entry_px * units * qty_lots
                paper_seg = is_paper_mode_for_segment("commodity")
                ok, gate_msg = check_order_allowed(
                    "MCX",
                    sym,
                    qty_lots,
                    "BUY",
                    estimated_value_inr=est_value,
                    skip_session_check=paper_seg,
                    segment="commodity",
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
                    log_info("[CommodityWatch] Autonomous place in paper mode")

                from services.commodity_indicator_plan import refresh_plan_at_execution

                plan = refresh_plan_at_execution(dict(plan))

                result = await asyncio.to_thread(
                    commodity_trade_service.place_trade,
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
                trade_plan = result.get("trade_plan") or plan

                if entry_submitted or placed:
                    entry_limit = float(trade_plan.get("entry_limit_price") or entry_px or 0)
                    fair = float(trade_plan.get("entry_fair_premium") or entry_limit or 0)
                    sl_prem = float(trade_plan.get("stop_loss_premium") or 0)
                    tp_prem = float(trade_plan.get("target_premium") or 0)
                    spot_sl = trade_plan.get("spot_stop_loss")
                    spot_tp = trade_plan.get("spot_target")
                    style = str(trade_plan.get("entry_style") or "")
                    sid = str(trade_plan.get("strategy_id") or "")
                    score = int(trade_plan.get("entry_confirmation_score") or 0)
                    trig = trade_plan.get("entry_spot_trigger")
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
                                else " · set exit manually if GTT pending"
                            )
                        )
                    )
                    with _lock:
                        self._placed_count_today += 1
                        self._placed_today = True
                        self._placed_symbol_today = sym
                        if sym:
                            up = sym.upper()
                            if up not in self._placed_symbols_today:
                                self._placed_symbols_today.append(up)
                        if entry_id:
                            self._register_pending(
                                entry_id=str(entry_id),
                                sym=sym or "",
                                trade_plan=trade_plan,
                                gtt_trigger_id=result.get("gtt_trigger_id"),
                                gtt_deferred=bool(result.get("gtt_deferred")),
                            )
                        self._reentry_armed = False
                        self._last_autonomous_placed_at = datetime.now(IST)
                        if self._should_disarm_watch():
                            self._armed = False
                    kind = "auto_placed"
                else:
                    from services.watch_skip_utils import format_place_skip_message

                    errors = result.get("errors") or ["unknown"]
                    detail = "; ".join(str(e).strip() for e in errors if e and str(e).strip())[:200]
                    msg = format_place_skip_message(errors)
                    kind = "auto_skipped"
                    with _lock:
                        self._last_autonomous_block_reason = detail or "unknown"
                    log_info(f"[CommodityWatch] auto_skipped {sym} — {detail or 'unknown'}")

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
                    "COMMODITY_WATCH_AUTO_PLACE",
                    {"placed": placed, "entry_submitted": entry_submitted, "result": result},
                )
                await push_service.send_to_user(
                    user_id="default",
                    title="Crude Mini autonomous " + ("placed" if entry_submitted else "skipped"),
                    body=msg[:180],
                )
            except Exception as exc:
                log_error(f"[CommodityWatch] autonomous place failed: {exc}")
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


_watch = CommodityStrategyWatch()


def arm_watch(**kwargs: Any) -> Dict[str, Any]:
    return _watch.arm(**kwargs)


def commodity_eod_shutdown(*, reason: str) -> Dict[str, Any]:
    return _watch.eod_shutdown(reason=reason)


def disarm_watch() -> Dict[str, Any]:
    return _watch.disarm()


def reset_commodity_watch_placement_counters() -> None:
    _watch.reset_placement_counters()


def on_commodity_trading_mode_changed(paper_mode: bool) -> None:
    _watch.on_trading_mode_changed(paper_mode)


def nuclear_reset_watch() -> Dict[str, Any]:
    return _watch.nuclear_reset()


def get_watch_status() -> Dict[str, Any]:
    return _watch.status()


def get_watch_events(limit: int = 20) -> List[Dict[str, Any]]:
    return _watch.events(limit)


def register_commodity_strategy_watch(loop: asyncio.AbstractEventLoop) -> None:
    _watch.register_loop(loop)
    st = _watch.status()
    if st.get("armed"):
        log_info(
            f"[CommodityWatch] Autonomous watch active mode={st.get('mode')} dir={st.get('direction')}"
        )
