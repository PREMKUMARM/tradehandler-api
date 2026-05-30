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
from services.crypto_config import CRYPTO_WATCH_MAX_TRADES_PER_DAY, DEFAULT_LEVERAGE, SYMBOL
from services.crypto_indicator_plan import is_strategy_configured, refresh_exits_at_fill, refresh_plan_at_execution
from services.crypto_order_guard import autonomous_place_allowed, format_pre_place_analysis
from services.push.push_service import push_service
from services.watch_readiness import build_readiness_payload
from services.crypto_trail import compute_crypto_trail_sl, get_crypto_trail_config, sl_changed_enough
from utils.binance_order_utils import (
    cancel_algo_order,
    cancel_order,
    get_order_avg_fill_price,
    get_order_status,
    get_symbol_price,
    place_limit_order,
    place_stop_market,
    signed_request,
)
from utils.logger import log_error, log_info, log_warning

IST = ZoneInfo("Asia/Kolkata")
_MAX_EVENTS = 40
_STATE_PATH = Path(os.getenv("CRYPTO_WATCH_STATE_FILE", "data/crypto_strategy_watch.json"))
_lock = RLock()


def _today() -> date:
    return datetime.now(IST).date()


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
    quantity_btc: Optional[float] = None  # None = auto-size from USDT balance × leverage
    mode: str = "autonomous"
    auto_place_on_signal: bool = True
    auto_execute_checklist: bool = True
    disarm_after_place: bool = False


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
        self._waiting_fill_logged_for: Optional[str] = None
        self._placed_today = False
        self._placed_count_today = 0
        self._session_date: Optional[date] = None
        self._reentry_armed = True
        self._last_autonomous_placed_at: Optional[datetime] = None
        self._last_eval_at: Optional[datetime] = None
        self._last_plan: Optional[Dict[str, Any]] = None
        self._last_strategy_analysis: Dict[str, Any] = {}
        self._last_checklist_ready = False
        self._last_entry_ready = False
        self._last_paper_mode = True
        self._trail_pos: Optional[Dict[str, Any]] = None
        self._entry_cleared_since_last_trade = True
        self._last_consumed_signal_key: Optional[str] = None
        self._load()

    @staticmethod
    def _plan_signal_key(plan: Dict[str, Any]) -> Optional[str]:
        """Unique key for the current 5m reversal signal — prevents re-firing the same bar."""
        ind = plan.get("indicators") if isinstance(plan.get("indicators"), dict) else {}
        side = str(plan.get("side") or "")
        style = str(plan.get("entry_style") or "")
        bar = ind.get("last_5m_bar")
        if isinstance(bar, dict):
            return (
                f"{side}:{style}:"
                f"{float(bar.get('open', 0)):.2f}:"
                f"{float(bar.get('close', 0)):.2f}"
            )
        spot = float(ind.get("signal_spot") or ind.get("btc_spot") or 0)
        if spot > 0 and side:
            return f"{side}:{style}:{spot:.2f}"
        return None

    def _consume_trade_signal(self, plan: Optional[Dict[str, Any]] = None) -> None:
        """Mark current strategy signal as used — require setup to clear before next entry."""
        with _lock:
            self._last_entry_ready = False
            self._reentry_armed = False
            self._entry_cleared_since_last_trade = False
            if plan:
                key = self._plan_signal_key(plan)
                if key:
                    self._last_consumed_signal_key = key
            if self._last_plan and isinstance(self._last_plan, dict):
                stale = dict(self._last_plan)
                stale["entry_ready"] = False
                stale["entry_block_reason"] = "Signal consumed — wait for next setup"
                self._last_plan = stale
        self._persist()

    def reset_placement_counters(self) -> None:
        """Clear daily placement slots (paper fund reset / mode toggle). Watch stays armed."""
        with _lock:
            self._placed_count_today = 0
            self._placed_today = False
            self._clear_pending_entry()
            self._reentry_armed = False
            self._last_autonomous_placed_at = None
            self._last_entry_ready = False
            self._last_checklist_ready = False
            self._last_plan = None
            self._last_strategy_analysis = {}
            self._entry_cleared_since_last_trade = False
            self._last_consumed_signal_key = None
        self._persist()
        log_info("[CryptoWatch] Placement counters reset")

    def on_trading_mode_changed(self, paper_mode: bool) -> None:
        """Paper↔live toggle: drop stale signals so live does not fire on old entry_ready."""
        with _lock:
            self._last_paper_mode = bool(paper_mode)
        self.reset_placement_counters()
        venue = "paper ledger" if paper_mode else "Binance (live)"
        self._push(
            "mode_changed",
            f"Switched to {venue} — watch reset; strategy not configured yet",
            tradingsymbol=SYMBOL,
        )
        log_info(f"[CryptoWatch] Trading mode → {'paper' if paper_mode else 'live'} (signals reset)")

    def _load(self) -> None:
        if not _STATE_PATH.exists():
            return
        try:
            data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            self._armed = bool(data.get("armed"))
            self._pending_entry_order_id = data.get("pending_entry_order_id")
            self._placed_count_today = int(data.get("placed_count_today") or 0)
            if self._placed_count_today == 0 and data.get("placed_today"):
                self._placed_count_today = 1
            self._placed_today = self._placed_count_today > 0
            sd = data.get("session_date")
            if sd:
                try:
                    self._session_date = date.fromisoformat(str(sd))
                except ValueError:
                    self._session_date = None
            self._reentry_armed = bool(data.get("reentry_armed", True))
            tp = data.get("trail_pos")
            self._trail_pos = tp if isinstance(tp, dict) else None
            lap = data.get("last_autonomous_placed_at")
            if lap:
                try:
                    self._last_autonomous_placed_at = datetime.fromisoformat(str(lap))
                    if self._last_autonomous_placed_at.tzinfo is None:
                        self._last_autonomous_placed_at = self._last_autonomous_placed_at.replace(
                            tzinfo=IST
                        )
                except ValueError:
                    self._last_autonomous_placed_at = None
            cfg = data.get("config") or {}
            self._cfg = WatchConfig(
                direction=str(cfg.get("direction") or "AUTO"),
                quantity_btc=float(cfg["quantity_btc"]) if cfg.get("quantity_btc") is not None else None,
                mode=str(cfg.get("mode") or "autonomous"),
                auto_place_on_signal=bool(cfg.get("auto_place_on_signal", True)),
                auto_execute_checklist=bool(cfg.get("auto_execute_checklist", True)),
                disarm_after_place=bool(cfg.get("disarm_after_place", False)),
            )
            for ev in (data.get("events") or [])[:_MAX_EVENTS]:
                if isinstance(ev, dict) and ev.get("at") and ev.get("kind"):
                    self._events.append(WatchEvent(**{k: ev[k] for k in WatchEvent.__dataclass_fields__ if k in ev}))
        except Exception as exc:
            log_warning(f"[CryptoWatch] load failed: {exc}")

    def _persist(self) -> None:
        data = {
            "armed": self._armed,
            "placed_today": bool(self._placed_count_today > 0),
            "placed_count_today": int(self._placed_count_today),
            "session_date": (self._session_date or _today()).isoformat(),
            "reentry_armed": self._reentry_armed,
            "last_autonomous_placed_at": (
                self._last_autonomous_placed_at.isoformat()
                if self._last_autonomous_placed_at
                else None
            ),
            "pending_entry_order_id": self._pending_entry_order_id,
            "trail_pos": self._trail_pos,
            "config": asdict(self._cfg),
            "events": [e.to_dict() for e in list(self._events)],
        }
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _push(self, kind: str, message: str, **kw: Any) -> None:
        self._events.appendleft(WatchEvent(at=datetime.now(IST).isoformat(), kind=kind, message=message[:240], **kw))
        self._persist()

    def _clear_pending_entry(self) -> None:
        with _lock:
            self._pending_entry_order_id = None
            self._pending_trade_plan = None
            self._waiting_fill_logged_for = None

    def _maybe_log_waiting_for_fill(self, order_id: str) -> None:
        oid = str(order_id)
        with _lock:
            if self._waiting_fill_logged_for == oid:
                return
            self._waiting_fill_logged_for = oid
        self._push(
            "waiting_for_fill",
            f"Entry {oid} pending on Binance — waiting for fill",
            tradingsymbol=SYMBOL,
        )

    def _reset_day_if_needed(self) -> None:
        today = _today()
        if self._session_date != today:
            self._session_date = today
            self._placed_today = False
            self._placed_count_today = 0
            self._clear_pending_entry()
            self._reentry_armed = True
            self._last_autonomous_placed_at = None
            self._last_entry_ready = False

    def _max_trades_per_day(self) -> int:
        if self._last_paper_mode:
            try:
                raw = os.getenv("PAPER_AUTO_MAX_TRADES_PER_DAY", str(CRYPTO_WATCH_MAX_TRADES_PER_DAY)).strip()
                return max(1, min(100, int(raw or CRYPTO_WATCH_MAX_TRADES_PER_DAY)))
            except Exception:
                return CRYPTO_WATCH_MAX_TRADES_PER_DAY
        return CRYPTO_WATCH_MAX_TRADES_PER_DAY

    def _reentry_cooldown_sec(self) -> float:
        try:
            return max(
                30.0,
                min(900.0, float(os.getenv("CRYPTO_WATCH_REENTRY_COOLDOWN_SEC", "90") or 90)),
            )
        except Exception:
            return 90.0

    def _maybe_rearm_reentry(self, entry_ready: bool, live: Optional[Dict[str, Any]] = None) -> None:
        with _lock:
            if not entry_ready:
                self._entry_cleared_since_last_trade = True
            if self._reentry_armed:
                return
            if not entry_ready:
                return
            if not self._entry_cleared_since_last_trade:
                return
            if self._pending_entry_order_id:
                return

            if self._last_entry_ready is False and entry_ready:
                self._reentry_armed = True
                return

            if self._last_autonomous_placed_at is None:
                if self._placed_count_today == 0 and self._entry_cleared_since_last_trade:
                    self._reentry_armed = True
                return

            age = (
                datetime.now(IST) - self._last_autonomous_placed_at.astimezone(IST)
            ).total_seconds()
            if age >= self._reentry_cooldown_sec() and self._entry_cleared_since_last_trade:
                self._reentry_armed = True

    def _signal_already_consumed(self, plan: Dict[str, Any]) -> bool:
        key = self._plan_signal_key(plan)
        if not key:
            return False
        with _lock:
            return key == self._last_consumed_signal_key

    def _autonomous_entry_allowed(
        self, entry_ready: bool, live: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str]:
        self._maybe_rearm_reentry(entry_ready, live)
        if not entry_ready:
            with _lock:
                self._reentry_armed = True
            return False, "Entry not confirmed (entry_ready=false)"
        with _lock:
            if not self._reentry_armed:
                cd = int(self._reentry_cooldown_sec())
                if not self._entry_cleared_since_last_trade:
                    return False, "Prior signal still active — wait for entry_ready to clear"
                return (
                    False,
                    f"Re-entry cooldown — wait {cd}s after last trade",
                )
        if not is_strategy_configured():
            return False, "Strategy not configured — autonomous entry disabled"
        return True, ""

    def _should_disarm_watch(self) -> bool:
        max_t = self._max_trades_per_day()
        if self._placed_count_today >= max_t:
            return True
        if self._cfg.disarm_after_place and max_t <= 1:
            return self._placed_count_today >= 1
        return False

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
        quantity_btc: Optional[float] = None,
        mode: str = "autonomous",
        auto_place_on_signal: bool = True,
        auto_execute_checklist: bool = True,
        disarm_after_place: bool = False,
    ) -> Dict[str, Any]:
        with _lock:
            self._cfg = WatchConfig(
                direction=direction,
                quantity_btc=float(quantity_btc) if quantity_btc is not None else None,
                mode=mode,
                auto_place_on_signal=auto_place_on_signal,
                auto_execute_checklist=auto_execute_checklist,
                disarm_after_place=disarm_after_place,
            )
            self._armed = True
            self._reentry_armed = False
            self._last_entry_ready = True
            self._entry_cleared_since_last_trade = False
            self._last_consumed_signal_key = None
        self._push(
            "armed",
            f"Crypto watch armed · {SYMBOL} {DEFAULT_LEVERAGE}x · up to {self._max_trades_per_day()}/day",
        )
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
            self._clear_pending_entry()
            self._placed_today = False
            self._placed_count_today = 0
            self._reentry_armed = True
            self._last_autonomous_placed_at = None
            self._events.clear()
        if _STATE_PATH.exists():
            _STATE_PATH.unlink(missing_ok=True)
        self._push("nuclear_reset", "Crypto watch state cleared")
        return self.status()

    def status(self) -> Dict[str, Any]:
        with _lock:
            cfg = self._cfg
            plan = self._last_plan or {}
            placed_count = self._placed_count_today
            max_trades = self._max_trades_per_day()
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
            paper_trading_mode=bool(self._last_paper_mode),
            kite_connected=bool(plan.get("indicators", {}).get("connected")),
            guard_message=None,
            min_entry_score=65,
            entry_confirmation_score=plan.get("entry_confirmation_score"),
            pending_entry_order_id=self._pending_entry_order_id,
            segment="crypto",
        )
        ind = plan.get("indicators") or {}
        btc_spot = float(ind.get("btc_spot") or plan.get("nifty_spot") or 0)
        return {
            "armed": self._armed,
            "mode": cfg.mode,
            "autonomous": cfg.mode == "autonomous",
            "direction": cfg.direction,
            "quantity_btc": cfg.quantity_btc,
            "leverage": DEFAULT_LEVERAGE,
            "symbol": SYMBOL,
            "kill_switch_active": self._kill_switch(),
            "placed_today": placed_count > 0,
            "placed_count_today": placed_count,
            "max_trades_per_day": max_trades,
            "reentry_armed": self._reentry_armed,
            "strategy_analysis": self._last_strategy_analysis,
            "tradingsymbol": SYMBOL,
            "btc_spot": btc_spot or None,
            "nifty_spot": btc_spot or None,
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
            entry_oid = self._pending_entry_order_id
            disarm = self._should_disarm_watch()
            self._reentry_armed = False
        self._clear_pending_entry()
        self._consume_trade_signal(plan)
        if not plan:
            return
        fill_px = await asyncio.to_thread(get_order_avg_fill_price, SYMBOL, str(entry_oid or ""))
        if not fill_px:
            fill_px = float(plan.get("entry_limit_price") or 0)
        plan = await asyncio.to_thread(refresh_exits_at_fill, plan, fill_price=fill_px)
        side = str(plan.get("side") or "LONG").upper()
        exit_side = "SELL" if side == "LONG" else "BUY"
        qty = float(plan.get("quantity") or 0.001)
        sl_px = float(plan.get("stop_loss_premium") or 0)
        tp_px = float(plan.get("target_premium") or 0)
        trail_on = get_crypto_trail_config().enabled or bool(plan.get("trail_enabled"))
        sl = await asyncio.to_thread(
            place_stop_market,
            symbol=SYMBOL,
            side=exit_side,
            quantity=qty,
            stop_price=sl_px,
        )
        tp = None
        if tp_px > 0 and not trail_on:
            tp = await asyncio.to_thread(
                place_limit_order,
                symbol=SYMBOL,
                side=exit_side,
                quantity=qty,
                price=tp_px,
                reduce_only=True,
            )
        sl_ok = sl.get("ok")
        tp_ok = bool(tp and tp.get("ok"))
        if sl_ok and trail_on:
            with _lock:
                self._trail_pos = {
                    "side": side,
                    "entry": fill_px,
                    "qty": qty,
                    "initial_sl": sl_px,
                    "sl": sl_px,
                    "min_tp": tp_px,
                    "peak": fill_px,
                    "trail_active": False,
                    "sl_algo_id": sl.get("algo_id") or sl.get("order_id"),
                }
        msg = f"Entry filled @ ${fill_px:,.2f} — SL @ ${sl_px:,.0f}"
        if trail_on:
            msg += f" · trail ON (min TP ${tp_px:,.0f} @ R:R ≥ {plan.get('min_rr', 1.5)})"
        if sl_ok:
            msg += f" algo {sl.get('order_id')}"
        else:
            msg += f" FAILED ({sl.get('error') or 'unknown'})"
        if tp_px > 0:
            msg += f" · TP @ ${tp_px:,.0f}"
            if tp_ok:
                msg += f" id {tp.get('order_id')}"
            else:
                msg += " FAILED"
        if disarm:
            with _lock:
                self._armed = False
        kind = "auto_gtt_placed" if sl_ok and (trail_on or tp_px <= 0 or tp_ok) else "auto_gtt_failed"
        if sl_ok and not tp_ok and tp_px > 0 and not trail_on:
            kind = "auto_gtt_partial"
        self._push(kind, msg, tradingsymbol=SYMBOL)

    async def _maybe_trail_open_position(self) -> None:
        with _lock:
            pos = copy.deepcopy(self._trail_pos) if self._trail_pos else None
        if not pos:
            return
        try:
            rows = await asyncio.to_thread(signed_request, "GET", "/fapi/v2/positionRisk", {"symbol": SYMBOL})
            amt = 0.0
            if isinstance(rows, list):
                for row in rows:
                    if str(row.get("symbol") or "").upper() == SYMBOL:
                        amt = abs(float(row.get("positionAmt") or 0))
            if amt < 1e-6:
                with _lock:
                    self._trail_pos = None
                self._consume_trade_signal()
                self._push("trail_closed", "Position flat — trail monitor stopped", tradingsymbol=SYMBOL)
                return
        except Exception as exc:
            log_warning(f"[CryptoWatch] trail position check: {exc}")
            return

        try:
            spot = await asyncio.to_thread(get_symbol_price, SYMBOL)
        except Exception as exc:
            log_warning(f"[CryptoWatch] trail price: {exc}")
            return

        cfg = get_crypto_trail_config()
        new_sl, new_peak, activated, note = compute_crypto_trail_sl(
            side=str(pos["side"]),
            entry=float(pos["entry"]),
            initial_sl=float(pos["initial_sl"]),
            min_tp=float(pos.get("min_tp") or 0),
            peak=float(pos.get("peak") or pos["entry"]),
            ltp=spot,
            current_sl=float(pos["sl"]),
            trail_active=bool(pos.get("trail_active")),
            symbol=SYMBOL,
            cfg=cfg,
        )
        if not sl_changed_enough(float(pos["sl"]), new_sl):
            return

        old_algo = str(pos.get("sl_algo_id") or "")
        if old_algo:
            await asyncio.to_thread(cancel_algo_order, SYMBOL, old_algo)
        exit_side = "SELL" if str(pos["side"]).upper() == "LONG" else "BUY"
        sl_res = await asyncio.to_thread(
            place_stop_market,
            symbol=SYMBOL,
            side=exit_side,
            quantity=float(pos["qty"]),
            stop_price=new_sl,
        )
        if not sl_res.get("ok"):
            self._push("trail_failed", sl_res.get("error") or "SL update failed", tradingsymbol=SYMBOL)
            return

        with _lock:
            if self._trail_pos:
                self._trail_pos["sl"] = new_sl
                self._trail_pos["peak"] = new_peak
                self._trail_pos["trail_active"] = activated or self._trail_pos.get("trail_active")
                self._trail_pos["sl_algo_id"] = sl_res.get("algo_id") or sl_res.get("order_id")
        if note:
            self._push("trail_update", note, tradingsymbol=SYMBOL)

    async def _run_loop(self) -> None:
        while True:
            with _lock:
                armed = self._armed
                has_trail = self._trail_pos is not None
                if not armed and not has_trail:
                    break
                cfg = self._cfg
                self._reset_day_if_needed()
            try:
                await self._maybe_trail_open_position()
                if not armed:
                    await asyncio.sleep(float(os.getenv("CRYPTO_WATCH_POLL_SEC", "8") or 8))
                    continue
                if self._trail_pos:
                    await asyncio.sleep(float(os.getenv("CRYPTO_WATCH_POLL_SEC", "8") or 8))
                    continue
                if self._pending_entry_order_id:
                    pid = str(self._pending_entry_order_id)
                    if pid.upper().startswith("PAPER-"):
                        with _lock:
                            self._reentry_armed = False
                        self._clear_pending_entry()
                    else:
                        st = await asyncio.to_thread(get_order_status, SYMBOL, pid)
                        if st == "FILLED":
                            await self._on_entry_filled()
                        elif st in ("CANCELED", "REJECTED", "EXPIRED"):
                            with _lock:
                                if self._placed_count_today > 0:
                                    self._placed_count_today -= 1
                                self._placed_today = self._placed_count_today > 0
                                self._reentry_armed = False
                                self._last_consumed_signal_key = None
                                self._entry_cleared_since_last_trade = True
                            self._clear_pending_entry()
                            self._push("auto_cancelled", f"Entry {st.lower()}")
                        else:
                            self._maybe_log_waiting_for_fill(pid)

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
                strategy_analysis = preview.get("strategy_analysis") or {}
                with _lock:
                    self._last_plan = plan
                    self._last_strategy_analysis = (
                        strategy_analysis if isinstance(strategy_analysis, dict) else {}
                    )
                    self._last_checklist_ready = bool(preview.get("checklist_ready"))
                    prev_entry = self._last_entry_ready
                    self._last_entry_ready = bool(plan.get("entry_ready"))
                    self._last_paper_mode = bool(preview.get("paper_trading_mode"))
                    self._last_eval_at = datetime.now(IST)
                    entry_ready = self._last_entry_ready

                from services.watch_execute import resolve_can_execute

                can_execute = resolve_can_execute(preview, plan)
                at_limit = self._placed_count_today >= self._max_trades_per_day()
                guard_ok, guard_msg = autonomous_place_allowed(
                    plan,
                    placed_today=at_limit,
                    segment="crypto",
                )
                reentry_ok, reentry_msg = self._autonomous_entry_allowed(
                    entry_ready, plan.get("indicators") if isinstance(plan.get("indicators"), dict) else None
                )
                signal_fresh = not self._signal_already_consumed(plan)
                ready = bool(preview.get("checklist_ready"))
                should_place = (
                    cfg.mode == "autonomous"
                    and cfg.auto_place_on_signal
                    and not self._pending_entry_order_id
                    and not self._trail_pos
                    and not at_limit
                    and not self._kill_switch()
                    and can_execute
                    and ready
                    and entry_ready
                    and signal_fresh
                    and guard_ok
                    and reentry_ok
                )
                if should_place:
                    await self._try_place(preview, plan)
                elif can_execute and ready and entry_ready:
                    pending = self._pending_entry_order_id
                    if pending:
                        pass  # waiting_for_fill logged once in pending poll branch
                    elif not signal_fresh:
                        self._push(
                            "auto_skipped",
                            "Same signal already traded — wait for new setup",
                            tradingsymbol=SYMBOL,
                        )
                    elif not guard_ok and guard_msg:
                        self._push("auto_skipped", guard_msg, tradingsymbol=SYMBOL)
                    elif not reentry_ok and reentry_msg:
                        self._push("auto_skipped", reentry_msg, tradingsymbol=SYMBOL)
                    elif at_limit:
                        self._push(
                            "auto_skipped",
                            f"Daily cap reached ({self._placed_count_today}/{self._max_trades_per_day()})",
                            tradingsymbol=SYMBOL,
                        )
            except Exception as exc:
                log_error(f"[CryptoWatch] loop: {exc}")
                self._push("eval_error", str(exc)[:200])
            await asyncio.sleep(float(os.getenv("CRYPTO_WATCH_POLL_SEC", "8") or 8))

    async def _try_place(self, preview: Dict[str, Any], plan: Dict[str, Any]) -> None:
        async with self._get_place_lock():
            with _lock:
                if self._placing or self._pending_entry_order_id:
                    return
                if self._placed_count_today >= self._max_trades_per_day():
                    return
                self._placing = True
            try:
                plan = await asyncio.to_thread(refresh_plan_at_execution, copy.deepcopy(plan))
                self._push(
                    "pre_place_analysis",
                    format_pre_place_analysis(plan),
                    tradingsymbol=SYMBOL,
                )
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
                    oid = str(result["entry_order_id"])
                    paper_fill = bool(result.get("entry_paper"))
                    with _lock:
                        self._placed_count_today += 1
                        self._placed_today = True
                        self._reentry_armed = False
                        self._last_autonomous_placed_at = datetime.now(IST)
                        if paper_fill:
                            self._pending_entry_order_id = None
                            self._pending_trade_plan = None
                        else:
                            self._pending_entry_order_id = oid
                            self._pending_trade_plan = copy.deepcopy(plan)
                    venue = "paper" if paper_fill else "Binance"
                    n = self._placed_count_today
                    max_t = self._max_trades_per_day()
                    self._push(
                        "auto_placed",
                        f"Entry {oid} on {venue} ({n}/{max_t} today)",
                        placed=True,
                        tradingsymbol=SYMBOL,
                    )
                    self._consume_trade_signal(plan)
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


def reset_crypto_watch_placement_counters() -> None:
    _watch.reset_placement_counters()


def on_crypto_trading_mode_changed(paper_mode: bool) -> None:
    _watch.on_trading_mode_changed(paper_mode)


def get_watch_status() -> Dict[str, Any]:
    return _watch.status()


def get_watch_events(limit: int = 20) -> List[Dict[str, Any]]:
    return _watch.events(limit)


def register_crypto_strategy_watch(loop: asyncio.AbstractEventLoop) -> None:
    _watch.register_loop(loop)
