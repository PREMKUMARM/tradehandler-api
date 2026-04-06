"""
Polls open paper positions (stop-loss / target / trailing) using Kite quotes
and closes them in the DB (plus optional synthetic exit leg) — no live broker orders.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from utils.logger import log_error, log_info, log_warning


def _parse_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _entry_from_payload(p: Dict[str, Any]) -> Optional[float]:
    for k in ("paper_fill_price", "price"):
        v = _float_or_none(p.get(k))
        if v is not None:
            return v
    return None


class PaperOrderMonitor:
    """Background task: SL / target / trailing for persisted paper orders."""

    def __init__(self) -> None:
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._trailing: Dict[str, Dict[str, float]] = {}

    async def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._loop())
        log_info("[PaperOrderMonitor] started")

    async def stop(self) -> None:
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log_info("[PaperOrderMonitor] stopped")

    async def _loop(self) -> None:
        while self.is_running:
            try:
                await asyncio.to_thread(self._tick_sync)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(f"[PaperOrderMonitor] tick error: {e}")
            await asyncio.sleep(3)

    def _tick_sync(self) -> None:
        from database.connection import get_database
        from services.paper_trading import paper_place_order

        db = get_database()
        conn = db.get_connection()
        cur = conn.execute(
            """
            SELECT id, order_id, payload, stoploss, target, trailing_stoploss
            FROM paper_orders
            WHERE exit_reason IS NULL
              AND (
                stoploss IS NOT NULL
                OR target IS NOT NULL
                OR (trailing_stoploss IS NOT NULL AND stoploss IS NOT NULL)
              )
            """
        )
        rows: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            rows.append({k: r[k] for k in r.keys()})

        if not rows:
            return

        quote_keys: List[str] = []
        seen_q: set = set()
        prepared: List[Dict[str, Any]] = []

        for row in rows:
            p = _parse_payload(row.get("payload"))
            if p.get("paper_exit_leg"):
                continue
            ex = str(p.get("exchange") or "NFO").upper()
            sym = p.get("tradingsymbol")
            if not sym:
                continue
            qk = f"{ex}:{sym}"
            oid = str(row["order_id"])
            tt = str(p.get("transaction_type") or "").upper()
            if tt not in ("BUY", "SELL"):
                continue
            qty = p.get("quantity")
            try:
                qty_i = int(qty) if qty is not None else 0
            except (TypeError, ValueError):
                qty_i = 0
            if qty_i <= 0:
                continue
            entry = _entry_from_payload(p)
            if entry is None:
                continue

            sl = _float_or_none(row.get("stoploss"))
            tgt = _float_or_none(row.get("target"))
            trail_amt = _float_or_none(row.get("trailing_stoploss"))

            if qk not in seen_q:
                seen_q.add(qk)
                quote_keys.append(qk)

            prepared.append(
                {
                    "db_id": int(row["id"]),
                    "order_id": oid,
                    "quote_key": qk,
                    "payload": p,
                    "transaction_type": tt,
                    "quantity": qty_i,
                    "entry": entry,
                    "stoploss": sl,
                    "target": tgt,
                    "trailing_stoploss": trail_amt,
                }
            )

        if not prepared:
            return

        quotes: Dict[str, Any] = {}
        try:
            from utils.kite_utils import get_kite_instance

            kite = get_kite_instance(skip_validation=True)
            chunk = 400
            for i in range(0, len(quote_keys), chunk):
                part = quote_keys[i : i + chunk]
                quotes.update(kite.quote(part))
        except Exception as e:
            log_warning(f"[PaperOrderMonitor] quotes failed: {e}")
            return

        now = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()

        for item in prepared:
            qk = item["quote_key"]
            if qk not in quotes:
                continue
            ltp = quotes[qk].get("last_price")
            if ltp is None:
                continue
            ltp = float(ltp)
            oid = item["order_id"]
            tt = item["transaction_type"]
            sl = item["stoploss"]
            tgt = item["target"]
            trail_amt = item["trailing_stoploss"]

            eff_sl = sl
            if trail_amt and sl:
                st = self._trailing.setdefault(
                    oid, {"highest": ltp, "lowest": ltp, "sl": sl}
                )
                if tt == "BUY":
                    st["highest"] = max(st["highest"], ltp)
                    new_sl = st["highest"] - trail_amt
                    if new_sl > eff_sl:
                        eff_sl = new_sl
                else:
                    st["lowest"] = min(st["lowest"], ltp)
                    new_sl = st["lowest"] + trail_amt
                    if new_sl < eff_sl:
                        eff_sl = new_sl
                st["sl"] = eff_sl

            hit: Optional[str] = None
            if eff_sl is not None:
                if tt == "BUY" and ltp <= eff_sl:
                    hit = "SL"
                elif tt == "SELL" and ltp >= eff_sl:
                    hit = "SL"
            if hit is None and tgt is not None:
                if tt == "BUY" and ltp >= tgt:
                    hit = "TARGET"
                elif tt == "SELL" and ltp <= tgt:
                    hit = "TARGET"

            if not hit:
                continue

            exit_price = ltp
            if hit == "SL" and eff_sl is not None:
                exit_price = eff_sl
            elif hit == "TARGET" and tgt is not None:
                exit_price = tgt

            self._close_position(
                conn,
                paper_place_order,
                item["db_id"],
                oid,
                item["payload"],
                hit,
                exit_price,
                now,
            )
            if oid in self._trailing:
                del self._trailing[oid]

        conn.commit()

    def _close_position(
        self,
        conn,
        paper_place_order_fn,
        db_id: int,
        parent_oid: str,
        payload: Dict[str, Any],
        reason: str,
        exit_price: float,
        exit_at: str,
    ) -> None:
        from services.execution_audit import log_execution_audit

        ex = str(payload.get("exchange") or "NFO").upper()
        sym = payload.get("tradingsymbol", "")
        tt = str(payload.get("transaction_type") or "").upper()
        qty = int(payload.get("quantity") or 0)
        product = str(payload.get("product") or "MIS")
        exit_side = "SELL" if tt == "BUY" else "BUY"

        exit_payload = {
            "paper_exit_leg": True,
            "parent_order_id": parent_oid,
            "exit_reason": reason,
            "tradingsymbol": sym,
            "exchange": ex,
            "transaction_type": exit_side,
            "quantity": qty,
            "order_type": "MARKET",
            "product": product,
            "paper_fill_price": exit_price,
        }
        try:
            exit_oid = paper_place_order_fn(exit_payload)
        except Exception as e:
            log_error(f"[PaperOrderMonitor] exit leg insert failed: {e}")
            exit_oid = None

        conn.execute(
            """
            UPDATE paper_orders
            SET exit_reason = ?, exit_price = ?, exit_at = ?, exit_order_id = ?, status = 'CLOSED'
            WHERE id = ?
            """,
            (reason, exit_price, exit_at, exit_oid, db_id),
        )
        log_info(
            f"[PaperOrderMonitor] closed {parent_oid} reason={reason} exit_price={exit_price}"
        )
        try:
            log_execution_audit(
                "PAPER_EXIT",
                actor="paper_order_monitor",
                exchange=ex,
                tradingsymbol=str(sym),
                payload={"parent": parent_oid, "reason": reason},
                result={"exit_price": exit_price, "exit_order_id": exit_oid},
                paper=True,
            )
        except Exception as e:
            log_warning(f"[PaperOrderMonitor] audit log: {e}")


paper_order_monitor = PaperOrderMonitor()
