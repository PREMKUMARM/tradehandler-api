"""Binance BTCUSDT autonomous watch (poll + optional auto-place)."""
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
from services import crypto_trade_service
from services.crypto_config import DEFAULT_LEVERAGE, SYMBOL
from services.push.push_service import push_service
from services.watch_readiness import build_readiness_payload
from utils.binance_order_utils import cancel_order, get_order_status, place_limit_order, place_stop_market
from utils.logger import log_error, log_info, log_warning

IST = ZoneInfo("Asia/Kolkata")
_MAX_EVENTS = 40
_STATE_PATH = Path(os.getenv("CRYPTO_WATCH_STATE_FILE", "data/crypto_strategy_watch.json"))
_lock = RLock()


@dataclass
class WatchEvent:
    at: str
    kind: str
    message: str
    tradingsymbol: Optional[str] = None
    placed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class WatchConfig:
    direction: str = "AUTO"
    quantity_btc: float = 0.001
    mode: str = "autonomous"
    auto_place_on_signal: bool = True
    auto_execute_checklist: bool = True
    disarm_after_place: bool = True


class CryptoStrategyWatch:
    def __init__(self) -> None:
        self._armed = False
        self._cfg = WatchConfig()
        self._events: Deque[WatchEvent] = deque(maxlen=_MAX_EVENTS)
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._place_lock: Optional[asyncio.Lock] = None
        self._placing = False
        self._pending_entry_order_id: Optional[str] = None
        self._pending_trade_plan: Optional[Dict[str, Any]] = None
        self._placed_today = False
        self._last_eval_at: Optional[datetime] = None
        self._last_plan: Optional[Dict[str, Any]] = None
        self._last_checklist_ready = False
        self._last_entry_ready = False
        self._load()

    def _load(self) -> None:
        if not _STATE_PATH.exists():
            return
        try:
            data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            self._armed = bool(data.get("armed"))
            self._pending_entry_order_id = data.get("pending_entry_order_id")
            self._placed_today = bool(data.get("placed_today"))
            cfg = data.get("config") or {}
            self._cfg = WatchConfig(
                direction=str(cfg.get("direction") or "AUTO"),
                quantity_btc=float(cfg.get("quantity_btc") or 0.001),
                mode=str(cfg.get("mode") or "autonomous"),
                auto_place_on_signal=bool(cfg.get("auto_place_on_signal", True)),
                auto_execute_checklist=bool(cfg.get("auto_execute_checklist", True)),
                disarm_after_place=bool(cfg.get("disarm_after_place", True)),
            )
            for ev in (data.get("events") or [])[:_MAX_EVENTS]:
                if isinstance(ev, dict) and ev.get("at") and ev.get("kind"):
                    self._events.append(WatchEvent(**{k: ev[k] for k in WatchEvent.__dataclass_fields__ if k in ev}))
        except Exception as exc:
            log_warning(f"[CryptoWatch] load failed: {exc}")

    def _persist(self) -> None:
        data = {
            "armed": self._armed,
            "placed_today": self._placed_today,
            "pending_entry_order_id": self._pending_entry_order_id,
            "config": asdict(self._cfg),
            "events": [e.to_dict() for e in list(self._events)],
        }
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _push(self, kind: str, message: str, **kw: Any) -> None:
        self._events.appendleft(WatchEvent(at=datetime.now(IST).isoformat(), kind=kind, message=message[:240], **kw))
        self._persist()

    @staticmethod
    def _kill_switch() -> bool:
        try:
            from services.risk_gate import is_kill_switch_active

            return is_kill_switch_active("crypto")
        except Exception:
            return True

    def arm(
        self,
        *,
        direction: str = "AUTO",
        quantity_btc: float = 0.001,
        mode: str = "autonomous",
        auto_place_on_signal: bool = True,
        auto_execute_checklist: bool = True,
        disarm_after_place: bool = True,
    ) -> Dict[str, Any]:
        with _lock:
            self._cfg = WatchConfig(
                direction=direction,
                quantity_btc=max(0.001, float(quantity_btc or 0.001)),
                mode=mode,
                auto_place_on_signal=auto_place_on_signal,
                auto_execute_checklist=auto_execute_checklist,
                disarm_after_place=disarm_after_place,
            )
            self._armed = True
        self._push("armed", f"Crypto watch armed · {SYMBOL} {DEFAULT_LEVERAGE}x")
        self._ensure_task()
        return self.status()

    def disarm(self) -> Dict[str, Any]:
        with _lock:
            self._armed = False
        self._push("disarmed", "Crypto watch stopped")
        return self.status()

    def nuclear_reset(self) -> Dict[str, Any]:
        with _lock:
            self._armed = False
            self._pending_entry_order_id = None
            self._pending_trade_plan = None
            self._placed_today = False
            self._events.clear()
        if _STATE_PATH.exists():
            _STATE_PATH.unlink(missing_ok=True)
        self._push("nuclear_reset", "Crypto watch state cleared")
        return self.status()

    def status(self) -> Dict[str, Any]:
        with _lock:
            cfg = self._cfg
            plan = self._last_plan or {}
        extras = build_readiness_payload(
            armed=self._armed,
            autonomous_mode=cfg.mode == "autonomous",
            plan=plan,
            checklist_ready=bool(self._last_checklist_ready),
            entry_ready=self._last_entry_ready,
            can_place=bool(self._last_checklist_ready),
            can_execute=bool(self._last_checklist_ready),
            autonomous_eligible=bool(plan.get("entry_ready")),
            kill_switch_active=self._kill_switch(),
            market_open=True,
            paper_trading_mode=False,
            kite_connected=bool(plan.get("indicators", {}).get("connected")),
            guard_message=None,
            min_entry_score=65,
            entry_confirmation_score=plan.get("entry_confirmation_score"),
            pending_entry_order_id=self._pending_entry_order_id,
        )
        return {
            "armed": self._armed,
            "mode": cfg.mode,
            "autonomous": cfg.mode == "autonomous",
            "direction": cfg.direction,
            "quantity_btc": cfg.quantity_btc,
            "leverage": DEFAULT_LEVERAGE,
            "symbol": SYMBOL,
            "kill_switch_active": self._kill_switch(),
            "placed_today": self._placed_today,
            "tradingsymbol": SYMBOL,
            "strategy_name": plan.get("strategy_name"),
            "last_eval_at": self._last_eval_at.isoformat() if self._last_eval_at else None,
            "events": [e.to_dict() for e in list(self._events)],
            **extras,
        }

    def events(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in list(self._events)[:limit]]

    def register_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        if self._armed:
            self._ensure_task()

    def _ensure_task(self) -> None:
        if self._loop and (self._task is None or self._task.done()):
            self._task = self._loop.create_task(self._run_loop())

    def _get_place_lock(self) -> asyncio.Lock:
        if self._place_lock is None:
            self._place_lock = asyncio.Lock()
        return self._place_lock

    async def _on_entry_filled(self) -> None:
        with _lock:
            plan = copy.deepcopy(self._pending_trade_plan) if self._pending_trade_plan else None
            self._pending_entry_order_id = None
            self._pending_trade_plan = None
            disarm = self._cfg.disarm_after_place
        if not plan:
            return
        side = str(plan.get("side") or "LONG").upper()
        exit_side = "SELL" if side == "LONG" else "BUY"
        qty = float(plan.get("quantity") or 0.001)
        sl_px = float(plan.get("stop_loss_premium") or 0)
        tp_px = float(plan.get("target_premium") or 0)
        sl = await asyncio.to_thread(
            place_stop_market,
            symbol=SYMBOL,
            side=exit_side,
            quantity=qty,
            stop_price=sl_px,
        )
        tp = None
        if tp_px > 0:
            tp = await asyncio.to_thread(
                place_limit_order,
                symbol=SYMBOL,
                side=exit_side,
                quantity=qty,
                price=tp_px,
                reduce_only=True,
            )
        msg = (
            f"Entry filled — SL @ ${sl_px:,.0f}"
            + (f" id {sl.get('order_id')}" if sl.get("ok") else "")
            + (f" · TP @ ${tp_px:,.0f}" if tp_px > 0 else "")
            + (f" id {tp.get('order_id')}" if tp and tp.get("ok") else "")
        )
        if disarm:
            with _lock:
                self._armed = False
        self._push("auto_gtt_placed" if sl.get("ok") else "auto_gtt_failed", msg, tradingsymbol=SYMBOL)

    async def _run_loop(self) -> None:
        while True:
            with _lock:
                if not self._armed:
                    break
                cfg = self._cfg
            try:
                if self._pending_entry_order_id:
                    st = await asyncio.to_thread(get_order_status, SYMBOL, self._pending_entry_order_id)
                    if st == "FILLED":
                        await self._on_entry_filled()
                    elif st in ("CANCELED", "REJECTED", "EXPIRED"):
                        with _lock:
                            self._pending_entry_order_id = None
                            self._pending_trade_plan = None
                        self._push("auto_cancelled", f"Entry {st.lower()}")

                preview = await asyncio.to_thread(
                    crypto_trade_service.preview_trade,
                    None,
                    cfg.direction,
                    None,
                    None,
                    cfg.quantity_btc,
                    cfg.auto_execute_checklist,
                )
                plan = preview.get("trade_plan") or {}
                with _lock:
                    self._last_plan = plan
                    self._last_checklist_ready = bool(preview.get("checklist_ready"))
                    self._last_entry_ready = bool(plan.get("entry_ready"))
                    self._last_eval_at = datetime.now(IST)

                if (
                    cfg.mode == "autonomous"
                    and cfg.auto_place_on_signal
                    and not self._pending_entry_order_id
                    and not self._placed_today
                    and not self._kill_switch()
                    and preview.get("can_execute")
                    and plan.get("entry_ready")
                ):
                    await self._try_place(preview, plan)
            except Exception as exc:
                log_error(f"[CryptoWatch] loop: {exc}")
                self._push("eval_error", str(exc)[:200])
            await asyncio.sleep(float(os.getenv("CRYPTO_WATCH_POLL_SEC", "8") or 8))

    async def _try_place(self, preview: Dict[str, Any], plan: Dict[str, Any]) -> None:
        async with self._get_place_lock():
            with _lock:
                if self._placing or self._pending_entry_order_id:
                    return
                self._placing = True
            try:
                result = await asyncio.to_thread(
                    crypto_trade_service.place_trade,
                    None,
                    self._cfg.direction,
                    None,
                    None,
                    self._cfg.quantity_btc,
                    True,
                    self._cfg.auto_execute_checklist,
                    plan,
                    True,
                )
                if result.get("entry_order_id"):
                    with _lock:
                        self._pending_entry_order_id = str(result["entry_order_id"])
                        self._pending_trade_plan = copy.deepcopy(plan)
                        self._placed_today = True
                    self._push("auto_placed", f"Entry {result['entry_order_id']} submitted", placed=True, tradingsymbol=SYMBOL)
                else:
                    err = "; ".join(result.get("errors") or ["skipped"])
                    self._push("auto_skipped", err, tradingsymbol=SYMBOL)
            finally:
                with _lock:
                    self._placing = False


_watch = CryptoStrategyWatch()


def arm_watch(**kwargs: Any) -> Dict[str, Any]:
    return _watch.arm(**kwargs)


def disarm_watch() -> Dict[str, Any]:
    return _watch.disarm()


def nuclear_reset_watch() -> Dict[str, Any]:
    return _watch.nuclear_reset()


def get_watch_status() -> Dict[str, Any]:
    return _watch.status()


def get_watch_events(limit: int = 20) -> List[Dict[str, Any]]:
    return _watch.events(limit)


def register_crypto_strategy_watch(loop: asyncio.AbstractEventLoop) -> None:
    _watch.register_loop(loop)
