"""
Poll open positions and apply stepped SL management — T1 → SL at entry, T2 → SL at T1.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from services.exit_trail_store import (
    close_exit_trail,
    get_trail_last_alert_at,
    list_open_exit_trails,
    mark_trail_alert_sent,
    sync_paper_order_levels,
    update_trail_stage,
)
from services.momentum_trail import get_momentum_trail_config
from services.sl_exit_service import modify_sl_exit_order
from services.trail_alert_service import alert_stale_trail, alert_time_stop
from services.trail_ops import check_time_stop, resolve_trail_config
from utils.logger import log_error, log_info, log_warning, log_warning_throttled


class ExitTrailMonitor:
    def __init__(self) -> None:
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._loop())
        log_info("[ExitTrailMonitor] started (exit model from EXIT_MODEL env)")

    async def stop(self) -> None:
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        interval = max(2, int(__import__("os").getenv("MOMENTUM_TRAIL_POLL_SEC", "3") or 3))
        while self.is_running:
            try:
                await asyncio.to_thread(self._tick_sync)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(f"[ExitTrailMonitor] tick error: {e}")
            await asyncio.sleep(interval)

    def _tick_sync(self) -> None:
        base_cfg = get_momentum_trail_config()
        if not base_cfg.enabled:
            return

        trails = list_open_exit_trails()
        if not trails:
            return

        quote_keys: List[str] = []
        key_by_trail: Dict[int, str] = {}
        for t in trails:
            ex = str(t.get("exchange") or "NFO").upper()
            sym = str(t.get("tradingsymbol") or "")
            if not sym:
                continue
            qk = f"{ex}:{sym}"
            key_by_trail[int(t["id"])] = qk
            if qk not in quote_keys:
                quote_keys.append(qk)

        quotes = self._fetch_quotes(quote_keys)
        min_hold = max(0, int(__import__("os").getenv("PAPER_MIN_HOLD_SEC", "15") or 15))
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

        for t in trails:
            tid = int(t["id"])
            qk = key_by_trail.get(tid)
            if not qk or qk not in quotes:
                continue
            ltp = quotes[qk]
            if ltp is None:
                continue

            strategy_id = str(t.get("strategy_id") or "")
            cfg = resolve_trail_config(strategy_id or None)
            entry = float(t.get("entry_price") or 0)
            sl = float(t.get("stop_loss") or 0)
            tp = float(t.get("target") or 0)
            qty = int(t.get("quantity") or 1)
            sym = str(t.get("tradingsymbol") or "")

            trail_active = bool(t.get("trail_active"))

            self._maybe_stale_alert(t, now_ist, cfg)

            time_reason = check_time_stop(t, now=now_ist, trail_active=trail_active, cfg=cfg)
            if time_reason:
                alert_time_stop(sym, time_reason)
                self._close_trail(t, ltp, "time_stop")
                continue

            if not trail_active and min_hold > 0:
                try:
                    updated = t.get("created_at") or t.get("updated_at")
                    if updated:
                        placed = datetime.fromisoformat(str(updated))
                        if placed.tzinfo is None:
                            placed = placed.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
                        age = (now_ist - placed.astimezone(ZoneInfo("Asia/Kolkata"))).total_seconds()
                        if age < min_hold:
                            continue
                except Exception:
                    pass

            if not bool(t.get("paper")) and self._live_position_qty(sym) <= 0:
                close_exit_trail(tid, reason="position_closed")
                continue

            if ltp <= sl and sl > 0:
                self._close_trail(t, ltp, "SL")
                continue

            t1 = float(t.get("initial_target") or tp or 0)
            stage = int(t.get("trail_stage") or 0)
            new_sl = sl
            new_stage = stage
            note = ""

            from services.entry_quality import exit_model, exit_t1_trigger_price

            r_unit = max(0.05, t1 - entry) if t1 > entry else max(0.05, entry - sl)

            if exit_model() == "t1_scalp" and stage < 1 and t1 > 0 and ltp >= exit_t1_trigger_price(t1):
                self._close_trail(t, ltp, "target_t1")
                continue

            desired_stage = 0
            for level in range(1, 20):
                target_px = entry + level * r_unit
                if ltp >= exit_t1_trigger_price(target_px):
                    desired_stage = level
                else:
                    break

            if desired_stage > stage:
                new_stage = desired_stage
                if new_stage == 1:
                    new_sl = entry
                    note = f"T1 ₹{t1:.2f} reached @ {ltp:.2f} — SL → entry ₹{entry:.2f}"
                else:
                    lock_px = entry + (new_stage - 1) * r_unit
                    new_sl = lock_px
                    note = (
                        f"T{new_stage} ₹{entry + new_stage * r_unit:.2f} reached @ {ltp:.2f} "
                        f"— SL → T{new_stage - 1} ₹{lock_px:.2f}"
                    )

            if new_stage > stage and new_sl > sl:
                update_trail_stage(tid, trail_stage=new_stage, stop_loss=new_sl, peak_ltp=ltp)
                sl = new_sl
                if bool(t.get("paper")):
                    paper_oid = str(t.get("paper_order_id") or t.get("entry_order_id") or "")
                    if paper_oid:
                        t2 = float(t.get("target_2") or 0)
                        if t2 <= t1 and entry > sl:
                            t2 = entry + 2 * r_unit
                        sync_paper_order_levels(paper_oid, new_sl, t2)
                else:
                    sl_oid = str(t.get("sl_order_id") or "").strip()
                    if sl_oid:
                        seg = str(t.get("segment") or "nifty50")
                        modify_sl_exit_order(
                            sl_order_id=sl_oid,
                            tradingsymbol=sym,
                            exchange=str(t.get("exchange") or "NFO"),
                            product=str(t.get("product") or "NRML"),
                            quantity=qty,
                            new_sl_premium=new_sl,
                            segment=seg,
                        )
                if note:
                    log_info(f"[ExitTrailMonitor] {sym} {note}")

    def _maybe_stale_alert(
        self, trail: Dict[str, Any], now: datetime, cfg: Any
    ) -> None:
        if cfg.trail_stale_alert_min <= 0:
            return
        tid = int(trail.get("id") or 0)
        last_alert = trail.get("last_alert_at") or get_trail_last_alert_at(tid)
        if last_alert:
            try:
                la = datetime.fromisoformat(str(last_alert))
                if la.tzinfo is None:
                    la = la.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
                if (now.astimezone(ZoneInfo("Asia/Kolkata")) - la.astimezone(ZoneInfo("Asia/Kolkata"))).total_seconds() < 900:
                    return
            except Exception:
                pass
        updated = trail.get("updated_at") or trail.get("created_at")
        if not updated:
            return
        try:
            ts = datetime.fromisoformat(str(updated))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
            stale_min = (now.astimezone(ZoneInfo("Asia/Kolkata")) - ts.astimezone(ZoneInfo("Asia/Kolkata"))).total_seconds() / 60.0
        except Exception:
            return
        if stale_min < cfg.trail_stale_alert_min:
            return
        sym = str(trail.get("tradingsymbol") or "")
        if sym:
            alert_stale_trail(sym, stale_min)
            mark_trail_alert_sent(tid)

    def _live_position_qty(self, tradingsymbol: str) -> int:
        sym = (tradingsymbol or "").strip()
        if not sym:
            return 0
        try:
            from utils.kite_utils import get_kite_instance

            kite = get_kite_instance(skip_validation=True)
            for p in kite.positions().get("net", []) or []:
                if str(p.get("tradingsymbol") or "") == sym:
                    return int(p.get("quantity") or 0)
        except Exception:
            pass
        return 0

    def _fetch_quotes(self, keys: List[str]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        if not keys:
            return out
        try:
            from utils.kite_utils import get_kite_instance

            kite = get_kite_instance(skip_validation=True)
            chunk = 400
            raw: Dict[str, Any] = {}
            for i in range(0, len(keys), chunk):
                raw.update(kite.quote(keys[i : i + chunk]) or {})
            for k in keys:
                row = raw.get(k) or {}
                lp = row.get("last_price")
                out[k] = float(lp) if lp is not None else None
        except Exception as e:
            log_warning_throttled(
                "exit_trail_monitor.quotes",
                f"[ExitTrailMonitor] quotes failed: {e}",
                interval_sec=60.0,
            )
        return out

    def _close_trail(self, trail: Dict[str, Any], ltp: float, reason: str) -> None:
        tid = int(trail["id"])
        if bool(trail.get("paper")):
            self._close_paper_trail(trail, ltp, reason)
        else:
            self._close_live_trail_if_open(trail, ltp, reason)
        close_exit_trail(tid, reason=reason.lower())

    def _close_live_trail_if_open(
        self, trail: Dict[str, Any], ltp: float, reason: str
    ) -> None:
        sym = str(trail.get("tradingsymbol") or "")
        if not sym:
            return
        try:
            from utils.kite_utils import get_kite_instance

            kite = get_kite_instance(skip_validation=True)
            qty = 0
            for p in kite.positions().get("net", []) or []:
                if str(p.get("tradingsymbol") or "") == sym:
                    qty = int(p.get("quantity") or 0)
                    break
            if qty <= 0:
                return

            from agent.tools.kite_tools import place_order_tool

            seg = str(trail.get("segment") or "commodity")
            res = place_order_tool.invoke(
                {
                    "tradingsymbol": sym,
                    "exchange": trail.get("exchange") or "MCX",
                    "transaction_type": "SELL",
                    "quantity": abs(qty),
                    "order_type": "MARKET",
                    "product": trail.get("product") or "NRML",
                    "segment": seg,
                }
            )
            if res.get("status") == "success":
                log_info(
                    f"[ExitTrailMonitor] live exit {sym} reason={reason} order={res.get('order_id')}"
                )
            else:
                log_warning(
                    f"[ExitTrailMonitor] live exit failed {sym}: {res.get('error') or res}"
                )
        except Exception as exc:
            log_warning(f"[ExitTrailMonitor] live exit check failed {sym}: {exc}")

    def _close_paper_trail(self, trail: Dict[str, Any], ltp: float, reason: str) -> None:
        from database.connection import get_database
        from services.paper_trading import paper_place_order

        paper_oid = str(trail.get("paper_order_id") or trail.get("entry_order_id") or "")
        if not paper_oid:
            return

        db = get_database()
        conn = db.get_connection()
        cur = conn.execute(
            "SELECT id, payload FROM paper_orders WHERE order_id = ? AND exit_reason IS NULL",
            (paper_oid,),
        )
        row = cur.fetchone()
        if not row:
            return

        payload = row["payload"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}

        sl = float(trail.get("stop_loss") or ltp)
        exit_price = sl if reason == "SL" else ltp
        exit_side = "SELL" if str(payload.get("transaction_type") or "").upper() == "BUY" else "BUY"
        now = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()

        exit_payload = {
            "paper_exit_leg": True,
            "parent_order_id": paper_oid,
            "exit_reason": reason,
            "tradingsymbol": payload.get("tradingsymbol"),
            "exchange": payload.get("exchange"),
            "transaction_type": exit_side,
            "quantity": payload.get("quantity"),
            "order_type": "MARKET",
            "product": payload.get("product"),
            "paper_fill_price": exit_price,
        }
        try:
            exit_oid = paper_place_order(exit_payload)
        except Exception as e:
            log_error(f"[ExitTrailMonitor] paper exit failed: {e}")
            exit_oid = None

        conn.execute(
            """
            UPDATE paper_orders
            SET exit_reason = ?, exit_price = ?, exit_at = ?, exit_order_id = ?, status = 'CLOSED'
            WHERE id = ?
            """,
            (reason, exit_price, now, exit_oid, int(row["id"])),
        )
        conn.commit()
        log_info(f"[ExitTrailMonitor] paper closed {paper_oid} reason={reason} @ {exit_price}")


exit_trail_monitor = ExitTrailMonitor()
