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
from utils.logger import log_error, log_info, log_warning

IST = ZoneInfo("Asia/Kolkata")

_MAX_EVENTS = 40
_DEFAULT_POLL_SEC = 5
_STATE_PATH = Path(os.getenv("V2_WATCH_STATE_FILE", "data/v2_strategy_watch.json"))
_lock = RLock()
_place_lock = asyncio.Lock()


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
        "pending_entry_order_id": None,
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
        self._pending_entry_order_id: Optional[str] = None
        self._pending_gtt_trigger_id: Optional[str] = None
        self._pending_symbol: Optional[str] = None
        self._eval_count = 0
        self._events: Deque[WatchEvent] = deque(maxlen=_MAX_EVENTS)
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._placing = False
        self._last_autonomous_block_reason: Optional[str] = None
        self._last_skip_logged_msg: Optional[str] = None
        self._last_skip_logged_at: Optional[datetime] = None
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        with _lock:
            data = _read_persisted()
            today = _today_iso()
            if data.get("session_date") != today:
                data["session_date"] = today
                data["placed_today"] = False
                data["placed_symbol_today"] = None
                data["pending_entry_order_id"] = None
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
            self._placed_today = bool(data.get("placed_today"))
            self._placed_symbol_today = data.get("placed_symbol_today")
            self._pending_entry_order_id = data.get("pending_entry_order_id")
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
                "placed_today": self._placed_today,
                "placed_symbol_today": self._placed_symbol_today,
                "pending_entry_order_id": self._pending_entry_order_id,
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

            return is_kill_switch_active()
        except Exception:
            return False

    def _reset_day_if_needed(self) -> None:
        today = _today()
        if self._session_date != today:
            self._session_date = today
            self._signal_fired_today = False
            self._placed_today = False
            self._placed_symbol_today = None
            self._pending_entry_order_id = None
            self._pending_gtt_trigger_id = None
            self._pending_symbol = None
            self._last_entry_ready = None
            self._last_checklist_ready = None
            self._eval_count = 0

    def _setup_invalidated(self, plan: Dict[str, Any]) -> tuple[bool, str]:
        """B-mode: cancel only when setup invalidates (not time-based)."""
        if not plan:
            return True, "No plan (setup invalidated)"
        if plan.get("entry_ready") is not True:
            return True, str(plan.get("entry_block_reason") or "Entry no longer confirmed")
        try:
            from services.v2_order_guard import min_entry_confirmation_score

            min_score = int(min_entry_confirmation_score() or 65)
        except Exception:
            min_score = 65
        score = int(plan.get("entry_confirmation_score") or 0)
        if score < min_score:
            return True, f"Score dropped to {score} (<{min_score})"
        block = plan.get("entry_block_reason")
        if block:
            return True, str(block)
        return False, ""

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

    def _cancel_pending(self, *, reason: str) -> None:
        entry_id = self._pending_entry_order_id
        gtt_id = self._pending_gtt_trigger_id
        sym = self._pending_symbol
        self._pending_entry_order_id = None
        self._pending_gtt_trigger_id = None
        self._pending_symbol = None
        self._placed_today = False
        self._placed_symbol_today = None
        try:
            from agent.tools.kite_tools import cancel_order_tool, delete_gtt_tool

            if entry_id:
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
        msg = (message or "Autonomous placement blocked").strip()[:240]
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
        log_info("[V2Watch] Disarmed")
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
            return {
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
                "events": [e.to_dict() for e in list(self._events)],
            }

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
            try:
                session_open = v2_trade_service.is_market_session_open()
                if self._pending_entry_order_id:
                    if not session_open:
                        self._cancel_pending(reason="Market session closed")
                    else:
                        status = self._order_status(self._pending_entry_order_id)
                        if status in ("COMPLETE", "EXECUTED"):
                            with _lock:
                                self._pending_entry_order_id = None
                                self._pending_gtt_trigger_id = None
                                self._pending_symbol = None
                                if self._cfg.disarm_after_place:
                                    self._armed = False
                            self._push_event(
                                WatchEvent(
                                    at=datetime.now(IST).isoformat(),
                                    kind="auto_filled",
                                    message="Entry filled — lifecycle complete"
                                    + (" (watch disarmed)" if self._cfg.disarm_after_place else ""),
                                )
                            )
                        elif status in ("CANCELLED", "REJECTED"):
                            self._cancel_pending(reason=f"Order {status.lower()}")

                if session_open:
                    (
                        fire_signal,
                        try_autonomous,
                        preview,
                        plan,
                        can_place,
                    ) = await asyncio.to_thread(self._evaluate_sync)
                    if self._pending_entry_order_id and plan:
                        invalid, why = self._setup_invalidated(plan)
                        if invalid:
                            self._cancel_pending(reason=why)
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
            and not self._placed_today
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

        with _lock:
            self._eval_count += 1
            self._last_eval_at = datetime.now(IST)
            self._last_trade_plan = plan
            self._last_strategy_analysis = strategy_analysis if isinstance(strategy_analysis, dict) else {}
            self._last_can_place = can_place
            self._last_can_execute = can_execute
            self._last_block_reason = block
            prev_chk = self._last_checklist_ready
            self._last_checklist_ready = checklist_ready
            self._last_entry_ready = entry_ready

            if checklist_ready and not self._signal_fired_today and prev_chk is not True:
                if prev_chk is False or (prev_chk is None and self._eval_count > 1):
                    fire_signal = True
                    self._signal_fired_today = True

            autonomous_armed = self._should_autonomous_place(cfg)
            if autonomous_armed and checklist_ready and can_execute and not self._placing:
                from services.v2_order_guard import autonomous_place_allowed

                allowed, guard_msg = autonomous_place_allowed(
                    plan,
                    placed_today=self._placed_today,
                    placed_symbol_today=self._placed_symbol_today,
                )
                if allowed:
                    self._last_autonomous_block_reason = None
                    try_autonomous = True
                else:
                    self._record_autonomous_skip(guard_msg, plan)
            elif autonomous_armed and checklist_ready and not can_execute:
                skip_msg = "Waiting for margin/validation (can_execute=false)"
                if not can_place:
                    skip_msg = "Waiting for margin/validation (can_place=false)"
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
        async with _place_lock:
            with _lock:
                if self._placed_today or self._placing:
                    return
                cfg = self._cfg
                if not self._should_autonomous_place(cfg):
                    return

                from services.v2_order_guard import autonomous_place_allowed

                allowed, guard_msg = autonomous_place_allowed(
                    plan,
                    placed_today=self._placed_today,
                    placed_symbol_today=self._placed_symbol_today,
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
                from services.paper_trading import is_paper_mode
                from services.risk_gate import check_order_allowed, is_kill_switch_active

                if is_kill_switch_active():
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
                ok, gate_msg = check_order_allowed(
                    "NFO",
                    sym,
                    qty,
                    "BUY",
                    estimated_value_inr=est_value,
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

                if is_paper_mode():
                    log_info("[V2Watch] Autonomous place in paper mode")

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
                    )
                    with _lock:
                        self._placed_today = True
                        self._placed_symbol_today = sym
                        # Keep watch alive for lifecycle (cancel if setup invalidates / clear on fill).
                        self._pending_entry_order_id = str(entry_id) if entry_id else None
                        self._pending_gtt_trigger_id = (
                            str(result.get("gtt_trigger_id")) if result.get("gtt_trigger_id") else None
                        )
                        self._pending_symbol = sym
                    kind = "auto_placed"
                else:
                    msg = f"Autonomous skipped: {'; '.join(result.get('errors') or ['unknown'])}"
                    kind = "auto_skipped"

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
