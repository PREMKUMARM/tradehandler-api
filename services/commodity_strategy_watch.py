"""
V2 Strategy Watch — server-side deploy with optional full autonomy.

Modes:
  - alert: push/WS when entry_ready; user places manually.
  - autonomous: persists to disk, survives API restart, auto LIMIT+GTT when
    entry_ready + can_place (retries each poll until placed or disarmed).

Autonomous is blocked only by kill-switch env/file or COMMODITY_WATCH_AUTONOMOUS=0.
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
from services import commodity_trade_service
from services.push.push_service import push_service
from utils.logger import log_error, log_info, log_warning

IST = ZoneInfo("Asia/Kolkata")

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
    """Ops-only hard stop via env — normal users enable autonomous from the UI button."""
    return _env_bool("COMMODITY_WATCH_DISABLE_AUTONOMOUS", False)


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
    mode: str = "alert"  # alert | autonomous
    auto_place_on_signal: bool = False
    auto_execute_checklist: bool = True
    disarm_after_place: bool = True


def _default_persisted() -> Dict[str, Any]:
    return {
        "armed": False,
        "session_date": _today_iso(),
        "placed_today": False,
        "signal_fired_today": False,
        "config": asdict(WatchConfig()),
        "events": [],
        "eval_count": 0,
        "last_entry_ready": None,
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
        self._last_can_place = False
        self._last_block_reason: Optional[str] = None
        self._last_trade_plan: Optional[Dict[str, Any]] = None
        self._last_eval_at: Optional[datetime] = None
        self._signal_fired_today = False
        self._placed_today = False
        self._eval_count = 0
        self._events: Deque[WatchEvent] = deque(maxlen=_MAX_EVENTS)
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        with _lock:
            data = _read_persisted()
            today = _today_iso()
            if data.get("session_date") != today:
                data["session_date"] = today
                data["placed_today"] = False
                data["signal_fired_today"] = False
                data["last_entry_ready"] = None
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
            self._signal_fired_today = bool(data.get("signal_fired_today"))
            self._eval_count = int(data.get("eval_count") or 0)
            ler = data.get("last_entry_ready")
            self._last_entry_ready = ler if ler is None else bool(ler)
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
                "signal_fired_today": self._signal_fired_today,
                "config": asdict(self._cfg),
                "events": [e.to_dict() for e in list(self._events)],
                "eval_count": self._eval_count,
                "last_entry_ready": self._last_entry_ready,
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
            self._last_entry_ready = None
            self._eval_count = 0

    def _push_event(self, ev: WatchEvent) -> None:
        self._events.appendleft(ev)
        self._persist()

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
        self._push_event(
            WatchEvent(
                at=datetime.now(IST).isoformat(),
                kind="armed",
                message=(
                    f"Watch armed ({mode}) — {self._cfg.direction}, {self._cfg.num_lots} lot(s)"
                    + (" · autonomous LIMIT+GTT" if mode == "autonomous" else "")
                ),
            )
        )
        log_info(f"[CommodityWatch] Armed mode={mode} autonomous={auto_place}")
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
        log_info("[CommodityWatch] Disarmed")
        return self.status()

    def status(self) -> Dict[str, Any]:
        with _lock:
            plan = self._last_trade_plan or {}
            cfg = self._cfg
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
                "can_place": self._last_can_place,
                "entry_block_reason": self._last_block_reason,
                "signal_fired_today": self._signal_fired_today,
                "placed_today": self._placed_today,
                "strategy_name": plan.get("strategy_name"),
                "tradingsymbol": plan.get("tradingsymbol"),
                "nifty_spot": (plan.get("indicators") or {}).get("nifty_spot"),
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
            log_info("[CommodityWatch] Restored armed watch from disk — starting loop")
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
                if commodity_trade_service.is_mcx_session_open():
                    (
                        fire_signal,
                        try_autonomous,
                        preview,
                        plan,
                        can_place,
                    ) = await asyncio.to_thread(self._evaluate_sync)
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
            await asyncio.sleep(_poll_interval())

    def _should_autonomous_place(self, cfg: WatchConfig) -> bool:
        if watch_autonomous_globally_disabled():
            return False
        return (
            cfg.mode == "autonomous"
            and cfg.auto_place_on_signal
            and not self._placed_today
        )

    def _evaluate_sync(
        self,
    ) -> tuple[bool, bool, Optional[Dict[str, Any]], Dict[str, Any], bool]:
        with _lock:
            if not self._armed:
                return False, False, None, {}, False
            cfg = self._cfg
            self._reset_day_if_needed()

        preview = commodity_trade_service.preview_trade(
            completed_steps=None,
            direction=cfg.direction,
            risk_percentage=cfg.risk_percentage,
            reward_percentage=cfg.reward_percentage,
            num_lots=cfg.num_lots,
            auto_execute=cfg.auto_execute_checklist,
        )
        plan = preview.get("trade_plan") or {}
        entry_ready = plan.get("entry_ready")
        if entry_ready is None:
            entry_ready = True
        can_place = bool(preview.get("can_place"))
        block = plan.get("entry_block_reason")

        fire_signal = False
        try_autonomous = False

        with _lock:
            self._eval_count += 1
            self._last_eval_at = datetime.now(IST)
            self._last_trade_plan = plan
            self._last_can_place = can_place
            self._last_block_reason = block
            prev = self._last_entry_ready
            self._last_entry_ready = bool(entry_ready)

            if entry_ready and not self._signal_fired_today and prev is not True:
                if prev is False or (prev is None and self._eval_count > 1):
                    fire_signal = True
                    self._signal_fired_today = True

            if (
                self._should_autonomous_place(cfg)
                and entry_ready
                and can_place
            ):
                try_autonomous = True

        self._persist()
        return fire_signal, try_autonomous, preview, plan, can_place

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
        with _lock:
            autonomous = self._cfg.mode == "autonomous"
        title = f"V2 setup ready — {strat}"
        if autonomous and try_autonomous:
            body = f"{sym} — placing LIMIT+GTT autonomously"
        elif autonomous:
            body = f"{sym} setup OK — waiting for risk gate ({plan.get('entry_block_reason') or 'validation'})"
        else:
            body = f"{sym} LIMIT ₹{limit_px} · confirm in wizard"
            if not can_place:
                body += " (risk validation pending)"

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

        log_info(f"[CommodityWatch] Signal ready {sym} can_place={can_place} autonomous={try_autonomous}")

        if try_autonomous:
            await self._try_auto_place(preview, plan)

    async def _try_auto_place(self, preview: Dict[str, Any], plan: Dict[str, Any]) -> None:
        with _lock:
            if self._placed_today:
                return
            cfg = self._cfg
            if not self._should_autonomous_place(cfg):
                return

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
            entry_px = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
            est_value = entry_px * qty
            ok, gate_msg = check_order_allowed(
                "MCX",
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
                log_info("[CommodityWatch] Autonomous place in paper mode")

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
            )
            placed = bool(result.get("placed"))
            msg = (
                f"Autonomous placed {sym} · entry {result.get('entry_order_id')}"
                if placed
                else f"Autonomous skipped: {'; '.join(result.get('errors') or ['unknown'])}"
            )
            with _lock:
                if placed:
                    self._placed_today = True
                    if cfg.disarm_after_place:
                        self._armed = False
            self._push_event(
                WatchEvent(
                    at=datetime.now(IST).isoformat(),
                    kind="auto_placed" if placed else "auto_skipped",
                    message=msg[:240],
                    placed=placed,
                    tradingsymbol=sym,
                )
            )
            self._persist()
            await broadcast_agent_update(
                "COMMODITY_WATCH_AUTO_PLACE",
                {"placed": placed, "autonomous": True, "result": result},
            )
            await push_service.send_to_user(
                user_id="default",
                title="V2 autonomous " + ("placed" if placed else "skipped"),
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


_watch = V2StrategyWatch()


def arm_watch(**kwargs: Any) -> Dict[str, Any]:
    return _watch.arm(**kwargs)


def disarm_watch() -> Dict[str, Any]:
    return _watch.disarm()


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
