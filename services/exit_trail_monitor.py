"""
Poll open positions and apply momentum trailing — updates paper SL/TP or live GTT OCO.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from services.exit_trail_store import (
    close_exit_trail,
    list_open_exit_trails,
    sync_paper_order_levels,
    update_exit_trail_levels,
)
from services.momentum_trail import (
    compute_trailed_levels,
    get_momentum_trail_config,
    gtt_triggers_for_levels,
    levels_changed_enough,
)
from services.watch_reconcile import gtt_exists_on_broker
from utils.logger import log_error, log_info, log_warning


class ExitTrailMonitor:
    def __init__(self) -> None:
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self.is_running or not get_momentum_trail_config().enabled:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._loop())
        log_info("[ExitTrailMonitor] started (momentum trail after target)")

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
        cfg = get_momentum_trail_config()
        if not cfg.enabled:
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

            entry = float(t.get("entry_price") or 0)
            sl = float(t.get("stop_loss") or 0)
            tp = float(t.get("target") or 0)
            peak = float(t.get("peak_ltp") or entry)
            trail_active = bool(t.get("trail_active"))

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

            gtt_id = str(t.get("gtt_trigger_id") or "").strip()
            if gtt_id and not bool(t.get("paper")) and not gtt_exists_on_broker(gtt_id):
                close_exit_trail(tid, reason="gtt_triggered")
                continue

            if ltp <= sl and sl > 0:
                self._close_trail(t, ltp, "SL")
                continue

            new_sl, new_tp, new_peak, activated, note = compute_trailed_levels(
                entry=entry,
                peak=peak,
                ltp=ltp,
                current_sl=sl,
                current_tp=tp,
                trail_active=trail_active,
                cfg=cfg,
            )

            if not activated:
                continue

            changed = levels_changed_enough(sl, tp, new_sl, new_tp, cfg=cfg) or not trail_active
            if not changed:
                if new_peak > peak:
                    update_exit_trail_levels(
                        tid, stop_loss=sl, target=tp, peak_ltp=new_peak, trail_active=True
                    )
                continue

            update_exit_trail_levels(
                tid,
                stop_loss=new_sl,
                target=new_tp,
                peak_ltp=new_peak,
                trail_active=True,
            )

            if bool(t.get("paper")):
                paper_oid = str(t.get("paper_order_id") or t.get("entry_order_id") or "")
                if paper_oid:
                    sync_paper_order_levels(paper_oid, new_sl, new_tp)
            elif gtt_id:
                ok = self._modify_live_gtt(t, gtt_id, new_sl, new_tp, ltp)
                if not ok:
                    log_warning(f"[ExitTrailMonitor] GTT modify failed for {gtt_id}")

            if note:
                log_info(f"[ExitTrailMonitor] {t.get('tradingsymbol')} {note}")

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
            log_warning(f"[ExitTrailMonitor] quotes failed: {e}")
        return out

    def _modify_live_gtt(
        self,
        trail: Dict[str, Any],
        gtt_id: str,
        sl_prem: float,
        tp_prem: float,
        ltp: float,
    ) -> bool:
        from agent.tools.kite_tools import modify_gtt_tool

        sym = str(trail.get("tradingsymbol") or "")
        ex = str(trail.get("exchange") or "NFO")
        product = str(trail.get("product") or "NRML")
        qty = int(trail.get("quantity") or 1)
        sl_trigger, tp_trigger, last_price = gtt_triggers_for_levels(
            float(trail.get("entry_price") or ltp),
            sl_prem,
            tp_prem,
            ltp,
        )
        res = modify_gtt_tool.invoke(
            {
                "trigger_id": int(str(gtt_id)),
                "tradingsymbol": sym,
                "exchange": ex,
                "trigger_type": "two-leg",
                "trigger_prices": [sl_trigger, tp_trigger],
                "last_price": last_price,
                "stop_loss_price": sl_prem,
                "target_price": tp_prem,
                "quantity": qty,
                "transaction_type": "SELL",
                "product": product,
            }
        )
        return res.get("status") == "success"

    def _close_trail(self, trail: Dict[str, Any], ltp: float, reason: str) -> None:
        tid = int(trail["id"])
        if bool(trail.get("paper")):
            self._close_paper_trail(trail, ltp, reason)
        close_exit_trail(tid, reason=reason.lower())

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
