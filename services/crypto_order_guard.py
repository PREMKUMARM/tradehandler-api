"""Duplicate-order, position, and entry-quality guards for crypto autonomous place."""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from utils.logger import log_warning


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def min_entry_confirmation_score() -> int:
    return max(40, min(100, _env_int("CRYPTO_AUTO_MIN_ENTRY_SCORE", 65)))


def entry_quality_for_autonomous(plan: Dict[str, Any]) -> Tuple[bool, str]:
    """Autonomous entry gate — requires configured strategy and confirmed entry."""
    from services.crypto_indicator_plan import STRATEGY_PENDING_REASON, is_strategy_configured

    if not is_strategy_configured():
        return False, STRATEGY_PENDING_REASON
    if not plan:
        return False, "No trade plan"
    if plan.get("entry_ready") is not True:
        return False, str(plan.get("entry_block_reason") or "Entry not confirmed by indicators")
    block = plan.get("entry_block_reason")
    if block:
        return False, str(block)

    min_score = min_entry_confirmation_score()
    score = int(plan.get("entry_confirmation_score") or 0)
    if score < min_score:
        return False, (
            f"Confirmation score {score} below minimum {min_score} for autonomous entry"
        )

    style = str(plan.get("entry_style") or "")
    if "blocked" in style.lower() or style == "blocked_wait":
        return False, "Entry style blocked — wait for setup"

    limit_px = float(plan.get("entry_limit_price") or 0)
    if limit_px <= 0:
        return False, "Invalid entry limit price"

    spot = float(plan.get("nifty_spot") or (plan.get("indicators") or {}).get("btc_spot") or 0)
    if spot > 0:
        chase_pct = abs(limit_px - spot) / spot * 100.0
        max_chase = float(os.getenv("CRYPTO_AUTO_MAX_CHASE_PCT", "0.15") or 0.15)
        if chase_pct > max_chase:
            return False, (
                f"Limit ${limit_px:,.2f} chases spot ${spot:,.2f} "
                f"({chase_pct:.2f}% > {max_chase:.2f}%) — wait for pullback"
            )

    return True, "Entry confirmed"


def has_pending_binance_entry(symbol: str) -> Tuple[bool, str]:
    sym = (symbol or "").strip().upper()
    if not sym:
        return False, ""
    try:
        from utils.binance_order_utils import signed_request

        orders = signed_request("GET", "/fapi/v1/openOrders", {"symbol": sym})
        for o in orders or []:
            if str(o.get("reduceOnly") or "").lower() in ("true", "1"):
                continue
            status = str(o.get("status") or "").upper()
            if status not in ("NEW", "PARTIALLY_FILLED"):
                continue
            oid = o.get("orderId")
            side = o.get("side")
            return True, f"Open entry order {oid} ({side}) on {sym}"
        return False, ""
    except Exception as exc:
        log_warning(f"[CryptoGuard] openOrders check failed: {exc}")
        return True, f"Could not verify open orders — blocked: {exc}"


def has_binance_position(symbol: str) -> Tuple[bool, str]:
    sym = (symbol or "").strip().upper()
    if not sym:
        return False, ""
    try:
        from utils.binance_order_utils import signed_request

        rows = signed_request("GET", "/fapi/v2/positionRisk", {"symbol": sym})
        for p in rows or []:
            amt = float(p.get("positionAmt") or 0)
            if abs(amt) > 1e-12:
                return True, (
                    f"Open position on {sym} (amt={amt}) — exit before re-entry"
                )
        return False, ""
    except Exception as exc:
        log_warning(f"[CryptoGuard] position check failed: {exc}")
        return True, f"Could not verify positions — blocked: {exc}"


def format_pre_place_analysis(plan: Dict[str, Any]) -> str:
    side = str(plan.get("side") or "—")
    score = int(plan.get("entry_confirmation_score") or 0)
    style = plan.get("entry_style") or "—"
    entry = float(plan.get("entry_limit_price") or 0)
    sl = float(plan.get("stop_loss_premium") or 0)
    tp = float(plan.get("target_premium") or 0)
    qty = float(plan.get("quantity") or 0)
    return (
        f"{side} score={score} · style {style} · "
        f"LIMIT ${entry:,.2f} · SL ${sl:,.2f} · TP ${tp:,.2f} · qty {qty} BTC"
    )


def autonomous_place_allowed(
    plan: Dict[str, Any],
    *,
    placed_today: bool,
    segment: str = "crypto",
) -> Tuple[bool, str]:
    """Gate autonomous crypto entry — daily cap, quality, no duplicate position/order."""
    if placed_today:
        return False, "Max autonomous crypto trades per day reached"
    if not plan or not plan.get("tradingsymbol"):
        return False, "No trade plan from checklist/strategy"

    try:
        from services.paper_trading import is_paper_mode_for_segment

        if is_paper_mode_for_segment(segment):
            from services.paper_order_guard import paper_autonomous_place_allowed

            return paper_autonomous_place_allowed(
                plan,
                placed_today=placed_today,
                segment=segment,
                entry_quality_check=entry_quality_for_autonomous,
            )
    except Exception:
        pass

    ok, msg = entry_quality_for_autonomous(plan)
    if not ok:
        return False, msg

    sym = str(plan.get("tradingsymbol") or "")
    pending, pend_msg = has_pending_binance_entry(sym)
    if pending:
        return False, pend_msg
    pos, pos_msg = has_binance_position(sym)
    if pos:
        return False, pos_msg
    return True, "OK"
