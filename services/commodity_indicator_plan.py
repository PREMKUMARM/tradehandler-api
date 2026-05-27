"""
V2 — precise entry/exit/SL/size from realtime indicators + live option quotes.
Entry: LIMIT at live premium. Exit control: GTT OCO only (no MARKET exit).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from services.push.option_contract_resolver import (
    estimate_delta_from_spot,
    fetch_option_ltp,
)
from services.commodity_config import COMMODITY_PRODUCT
from services.commodity_instruments import lot_size, resolve_commodity_contract
from services.commodity_live_indicators import recalculate_from_ticker
from services.commodity_entry_pricing import EntryAnalysis, compute_strategy_entry
from services.commodity_strike_pricing import (
    _pick_moneyness,
    refine_spot_levels_from_candles,
)
from utils.kite_order_utils import round_to_tick
from utils.kite_utils import get_kite_instance
from utils.logger import log_warning


def fetch_realtime_indicators() -> Dict[str, Any]:
    """Live indicators from Kite ticker; historical only fills gaps."""
    return recalculate_from_ticker()


def live_option_quote(tradingsymbol: str, exchange: str = "MCX") -> Dict[str, float]:
    """Bid/ask/LTP for precise LIMIT entry."""
    kite = get_kite_instance()
    key = f"{exchange}:{tradingsymbol}"
    row = (kite.quote(key) or {}).get(key, {}) or {}
    depth = row.get("depth") or {}
    bid = float((depth.get("buy") or [{}])[0].get("price") or 0)
    ask = float((depth.get("sell") or [{}])[0].get("price") or 0)
    ltp = float(row.get("last_price") or 0)
    if ltp <= 0:
        o = row.get("ohlc") or {}
        ltp = float(o.get("close") or o.get("open") or 0)
    return {"bid": bid, "ask": ask, "ltp": ltp}


def precise_entry_limit_price(quote: Dict[str, float], transaction_type: str = "BUY") -> float:
    """Legacy: aggressive book price. Prefer compute_strategy_entry for V2 buys."""
    bid, ask, ltp = quote.get("bid", 0), quote.get("ask", 0), quote.get("ltp", 0)
    tx = (transaction_type or "BUY").upper()
    if tx == "BUY":
        px = ask if ask > 0 else ltp
    else:
        px = bid if bid > 0 else ltp
    return round_to_tick(max(0.05, px))


def _apply_entry_analysis_to_plan(
    plan: Dict[str, Any],
    entry: EntryAnalysis,
    *,
    ltp: float,
) -> Dict[str, Any]:
    updated = dict(plan)
    updated["entry_limit_price"] = entry.entry_limit_price
    updated["entry_premium"] = round(float(ltp or entry.fair_premium), 2)
    updated["entry_ready"] = entry.entry_ready
    updated["entry_style"] = entry.entry_style
    updated["entry_fair_premium"] = entry.fair_premium
    updated["entry_confirmation_score"] = entry.confirmation_score
    updated["entry_spot_trigger"] = entry.spot_trigger
    updated["entry_block_reason"] = entry.block_reason
    note = " · ".join(entry.notes) if entry.notes else ""
    if entry.block_reason and not entry.entry_ready:
        updated["note"] = entry.block_reason
    elif note:
        updated["note"] = note
    ind = dict(updated.get("indicators") or {})
    ind["entry_analysis"] = {
        "ready": entry.entry_ready,
        "style": entry.entry_style,
        "score": entry.confirmation_score,
        "spot_trigger": entry.spot_trigger,
        "block_reason": entry.block_reason,
    }
    updated["indicators"] = ind
    return updated


def premium_levels_from_indicators(
    *,
    entry_premium: float,
    spot_entry: float,
    spot_sl: float,
    spot_tgt: float,
    strike: int,
    kind: str,
    vix: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Map indicator-based Nifty levels → option SL/target premiums (long CE/PE, live delta)."""
    delta = estimate_delta_from_spot(spot_entry, strike, kind, vix=vix)
    spot_risk = abs(spot_entry - spot_sl)
    spot_reward = abs(spot_tgt - spot_entry)
    sl_prem = max(0.05, entry_premium - spot_risk * delta)
    tgt_prem = entry_premium + spot_reward * delta
    if sl_prem >= entry_premium:
        sl_prem = max(0.05, entry_premium - 0.05)
    if tgt_prem <= entry_premium:
        tgt_prem = entry_premium + max(0.05, spot_reward * delta)
    return round_to_tick(sl_prem), round_to_tick(tgt_prem), delta


def size_from_risk(
    capital: float,
    risk_pct: float,
    entry_premium: float,
    sl_premium: float,
    lot_size: int,
    num_lots: int,
) -> Tuple[int, int, float]:
    """Return (num_lots, order_quantity, risk_inr). MCX Kite qty is in lots, not barrels."""
    prem_risk = max(0.05, entry_premium - sl_premium)
    max_risk_amt = capital * (risk_pct / 100.0)
    risk_per_lot = prem_risk * lot_size
    max_lots = int(max_risk_amt / risk_per_lot) if risk_per_lot > 0 else 0
    qty_lots = min(num_lots, max_lots) if max_lots >= 1 else num_lots
    qty_lots = max(1, qty_lots)
    order_qty = qty_lots
    risk_inr = risk_per_lot * qty_lots
    return qty_lots, order_qty, risk_inr


def build_indicator_trade_plan(
    *,
    direction: str,
    risk_pct: float,
    reward_pct: float,
    num_lots: int,
    capital: float,
    strategy_analysis: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Full plan: indicator spot levels → strike → live LTP → LIMIT entry + GTT SL/TP premiums → size.
    """
    messages: List[str] = []
    ind = fetch_realtime_indicators()
    spot = ind["nifty_spot"]
    if spot <= 0:
        return None, ["Could not fetch live Crude spot"]

    strategy_id = None
    strategy_name = None
    option_kind = "CE"
    spot_sl = spot_tgt = spot

    if strategy_analysis:
        strategy_id = strategy_analysis.get("selected_id")
        strategy_name = strategy_analysis.get("selected_name")
        option_kind = (strategy_analysis.get("selected_option_kind") or "CE").upper()
        ranked = strategy_analysis.get("strategies") or []
        sel = next((s for s in ranked if s.get("id") == strategy_id), ranked[0] if ranked else {})
        spot = float(sel.get("spot_entry") or spot)
        spot_sl = float(sel.get("spot_stop_loss") or spot)
        spot_tgt = float(sel.get("spot_target") or spot)
    else:
        d = (direction or "AUTO").upper()
        if d in ("CE", "PE"):
            option_kind = d
        elif ind["prev_close"] > 0:
            option_kind = "CE" if spot >= ind["prev_close"] else "PE"
        else:
            option_kind = "CE"
        rr = reward_pct / risk_pct if risk_pct > 0 else 2.0
        risk_pts = max(spot * (risk_pct / 100.0) * 0.35, 15.0)
        if option_kind == "CE":
            spot_sl, spot_tgt = spot - risk_pts, spot + risk_pts * rr
        else:
            spot_sl, spot_tgt = spot + risk_pts, spot - risk_pts * rr

    intra = {
        "pdh": ind.get("pdh"),
        "pdl": ind.get("pdl"),
        "or_high": ind.get("or_high"),
        "or_low": ind.get("or_low"),
        "ema9": ind.get("ema9"),
    }
    sid = strategy_id or "long_atm_directional"
    spot_entry, spot_sl, spot_tgt, level_note = refine_spot_levels_from_candles(
        sid, spot, option_kind, spot_sl, spot_tgt, intra
    )
    moneyness, pattern_tag, m_reason = _pick_moneyness(
        sid, spot_entry, option_kind, spot_sl, spot_tgt, intra
    )

    contract = resolve_commodity_contract(
        spot=spot_entry, kind=option_kind, moneyness=moneyness
    )
    if contract is None:
        return None, ["Could not resolve MCX option contract for live strike"]

    quote = live_option_quote(contract.tradingsymbol, exchange="MCX")
    entry_prem = quote["ltp"]
    if not entry_prem or entry_prem <= 0:
        try:
            kite = get_kite_instance()
            key = f"MCX:{contract.tradingsymbol}"
            row = (kite.quote(key) or {}).get(key, {}) or {}
            entry_prem = float(row.get("last_price") or 0)
        except Exception:
            entry_prem = 0
    if not entry_prem or entry_prem <= 0:
        entry_prem = max(1.0, 0.007 * spot_entry)
        messages.append("Live option LTP unavailable — using estimate")

    sl_prem, tgt_prem, delta = premium_levels_from_indicators(
        entry_premium=float(entry_prem),
        spot_entry=spot_entry,
        spot_sl=spot_sl,
        spot_tgt=spot_tgt,
        strike=contract.strike,
        kind=option_kind,
        vix=ind.get("vix"),
    )

    entry_analysis = compute_strategy_entry(
        strategy_id=sid,
        option_kind=option_kind,
        quote=quote,
        spot=spot_entry,
        strike=contract.strike,
        delta=delta,
        intra={**intra, "last_5m_close": ind.get("last_5m_close")},
        prev_close=float(ind.get("prev_close") or 0),
    )
    entry_limit = entry_analysis.entry_limit_price
    if not entry_analysis.entry_ready:
        messages.append(
            entry_analysis.block_reason or "Entry not confirmed — wait for setup"
        )
    messages.extend(entry_analysis.notes)

    ls = int(contract.lot_size or lot_size())

    qty_lots, quantity, risk_inr = size_from_risk(
        capital, risk_pct, float(entry_prem), sl_prem, ls, num_lots
    )
    reward_inr = max(0.0, (tgt_prem - float(entry_prem)) * ls * qty_lots)
    rr = (reward_inr / risk_inr) if risk_inr > 0 else 0.0

    bb_zone = None
    if ind.get("bb_middle") and ind.get("bb_upper") and ind.get("bb_lower"):
        from services.kite_live_indicators import bollinger_zone

        bb_zone = bollinger_zone(
            spot_entry,
            float(ind["bb_middle"]),
            float(ind["bb_upper"]),
            float(ind["bb_lower"]),
            option_kind,
        ).get("zone")

    indicator_snapshot = {
        **ind,
        "bb_zone": bb_zone,
        "margin": capital,
        "risk_pct": risk_pct,
        "indicator_sources": ind.get("indicator_sources", {}),
        "strategy_id": sid,
        "pattern_tag": pattern_tag,
        "spot_entry": round(spot_entry, 2),
        "spot_stop_loss": round(spot_sl, 2),
        "spot_target": round(spot_tgt, 2),
        "level_note": level_note,
        "option_bid": quote.get("bid"),
        "option_ask": quote.get("ask"),
        "option_ltp": quote.get("ltp"),
    }

    plan = {
        "tradingsymbol": contract.tradingsymbol,
        "exchange": "MCX",
        "option_type": option_kind,
        "strike": int(contract.strike),
        "expiry": contract.expiry.isoformat(),
        "quantity": quantity,
        "lot_size": ls,
        "num_lots": qty_lots,
        "product": COMMODITY_PRODUCT,
        "entry_order_type": "LIMIT",
        "entry_limit_price": entry_limit,
        "exit_order_type": "GTT_OCO",
        "entry_premium": round(float(entry_prem), 2),
        "stop_loss_premium": sl_prem,
        "target_premium": tgt_prem,
        "nifty_spot": round(spot_entry, 2),
        "spot_stop_loss": round(spot_sl, 2),
        "spot_target": round(spot_tgt, 2),
        "risk_inr": round(risk_inr, 2),
        "reward_inr": round(reward_inr, 2),
        "reward_ratio": round(rr, 2),
        "estimated_premium": quote.get("ltp", 0) <= 0,
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "strike_moneyness": moneyness,
        "pattern_tag": pattern_tag,
        "delta_used": round(delta, 3),
        "atm_reference": int(round(spot_entry / 50) * 50),
        "pricing_note": (
            f"{m_reason} · {level_note} · {entry_analysis.entry_style} LIMIT ₹{entry_limit} "
            f"(fair ₹{entry_analysis.fair_premium}) · GTT SL ₹{sl_prem} TP ₹{tgt_prem}"
        ),
        "entry_ready": entry_analysis.entry_ready,
        "entry_style": entry_analysis.entry_style,
        "entry_fair_premium": entry_analysis.fair_premium,
        "entry_confirmation_score": entry_analysis.confirmation_score,
        "entry_spot_trigger": entry_analysis.spot_trigger,
        "entry_block_reason": entry_analysis.block_reason,
        "indicators": indicator_snapshot,
    }
    messages.append(
        f"Indicators: Crude {spot_entry:.0f} | OR {ind.get('or_low')}-{ind.get('or_high')} | "
        f"PDH {ind.get('pdh')} PDL {ind.get('pdl')} | EMA9 {ind.get('ema9')} | VIX {ind.get('vix')}"
    )
    messages.append(
        f"Entry LIMIT ₹{entry_limit} (LTP ₹{entry_prem:.2f}) · {qty_lots} lot(s) × {ls} bbl (Kite qty {quantity}) · "
        f"Risk ₹{risk_inr:.0f} · GTT exit SL ₹{sl_prem} TP ₹{tgt_prem}"
    )
    return plan, messages


def refresh_plan_at_execution(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Re-quote option + refresh indicators from latest ticks before place."""
    live = recalculate_from_ticker()
    sym = plan.get("tradingsymbol")
    if not sym:
        return plan
    quote = live_option_quote(sym)
    entry_prem = quote["ltp"] or plan.get("entry_premium", 0)
    spot_entry = float(live.get("nifty_spot") or plan.get("nifty_spot", 0))
    intra = {
        "pdh": live.get("pdh"),
        "pdl": live.get("pdl"),
        "or_high": live.get("or_high"),
        "or_low": live.get("or_low"),
        "ema9": live.get("ema9"),
    }
    sid = plan.get("strategy_id") or "long_atm_directional"
    kind = plan.get("option_type", "CE")
    spot_sl = float(plan.get("spot_stop_loss", 0))
    spot_tgt = float(plan.get("spot_target", 0))
    spot_entry, spot_sl, spot_tgt, _ = refine_spot_levels_from_candles(
        sid, spot_entry, kind, spot_sl, spot_tgt, intra
    )
    sl_prem, tgt_prem, delta = premium_levels_from_indicators(
        entry_premium=float(entry_prem),
        spot_entry=spot_entry,
        spot_sl=spot_sl,
        spot_tgt=spot_tgt,
        strike=int(plan.get("strike", 0)),
        kind=kind,
        vix=live.get("vix"),
    )
    entry_analysis = compute_strategy_entry(
        strategy_id=sid,
        option_kind=kind,
        quote=quote,
        spot=spot_entry,
        strike=int(plan.get("strike", 0)),
        delta=delta,
        intra={**intra, "last_5m_close": live.get("last_5m_close")},
        prev_close=float(live.get("prev_close") or 0),
    )
    entry_limit = entry_analysis.entry_limit_price
    ind_meta = plan.get("indicators") or {}
    capital = float(ind_meta.get("margin") or 0)
    risk_pct = float(ind_meta.get("risk_pct") or 1.0)
    lot_size = int(plan.get("lot_size") or 75)
    num_lots = int(plan.get("num_lots") or 1)
    if capital > 0:
        qty_lots, quantity, risk_inr = size_from_risk(
            capital,
            risk_pct,
            float(entry_prem),
            sl_prem,
            lot_size,
            num_lots,
        )
    else:
        qty_lots = num_lots
        quantity = plan.get("quantity", num_lots)
        risk_inr = plan.get("risk_inr", 0)

    updated = dict(plan)
    updated.update(
        {
            "entry_limit_price": entry_limit,
            "entry_premium": round(float(entry_prem), 2),
            "stop_loss_premium": sl_prem,
            "target_premium": tgt_prem,
            "delta_used": round(delta, 3),
            "entry_order_type": "LIMIT",
            "exit_order_type": "GTT_OCO",
            "num_lots": qty_lots,
            "quantity": quantity,
            "risk_inr": round(risk_inr, 2) if risk_inr else plan.get("risk_inr"),
            "nifty_spot": round(spot_entry, 2),
            "spot_stop_loss": round(spot_sl, 2),
            "spot_target": round(spot_tgt, 2),
            "entry_ready": entry_analysis.entry_ready,
            "entry_style": entry_analysis.entry_style,
            "entry_fair_premium": entry_analysis.fair_premium,
            "entry_confirmation_score": entry_analysis.confirmation_score,
            "entry_spot_trigger": entry_analysis.spot_trigger,
            "entry_block_reason": entry_analysis.block_reason,
        }
    )
    ind = dict(plan.get("indicators") or {})
    ind.update(live)
    ind.update(
        {
            "option_bid": quote.get("bid"),
            "option_ask": quote.get("ask"),
            "option_ltp": quote.get("ltp"),
            "refreshed_at_execution": True,
        }
    )
    updated["indicators"] = ind
    return updated


def gtt_triggers_from_plan(plan: Dict[str, Any]) -> Tuple[float, float, float]:
    """OCO trigger prices and last_price for GTT placement."""
    sl_prem = float(plan["stop_loss_premium"])
    tgt_prem = float(plan["target_premium"])
    last_price = float(plan.get("entry_premium") or plan.get("entry_limit_price") or 0)
    # Zerodha constraint: trigger must be sufficiently away from last_price
    # (observed rejection: "difference should be more than 0.25%").
    min_gap = max(0.05, last_price * 0.0026) if last_price > 0 else 0.05

    sl_trigger = round_to_tick(sl_prem * 1.002)
    tp_trigger = round_to_tick(tgt_prem * 0.998)

    # For long options exits:
    # - SL trigger should be below last_price with enough gap
    # - TP trigger should be above last_price with enough gap
    if last_price > 0:
        if last_price - sl_trigger < min_gap:
            sl_trigger = round_to_tick(max(0.05, last_price - min_gap))
        if tp_trigger - last_price < min_gap:
            tp_trigger = round_to_tick(last_price + min_gap)

    # Keep triggers on correct sides of intended prices (avoid nonsensical inversions)
    if sl_trigger >= last_price and last_price > 0:
        sl_trigger = round_to_tick(max(0.05, last_price - min_gap))
    if tp_trigger <= last_price and last_price > 0:
        tp_trigger = round_to_tick(last_price + min_gap)
    return sl_trigger, tp_trigger, last_price
