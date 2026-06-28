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
from services.commodity_instruments import future_token
from services.commodity_live_indicators import recalculate_from_ticker
from services.commodity_entry_pricing import EntryAnalysis, compute_strategy_entry
from services.commodity_strike_pricing import (
    _pick_moneyness,
    refine_spot_levels_from_candles,
)
from utils.kite_order_utils import merge_quote_with_circuit, round_to_tick, validate_buy_limit_price
from utils.kite_utils import get_kite_instance
from utils.logger import log_warning


def fetch_realtime_indicators() -> Dict[str, Any]:
    """Live indicators from Kite ticker; historical only fills gaps."""
    return recalculate_from_ticker()


def live_option_quote(tradingsymbol: str, exchange: str = "MCX") -> Dict[str, float]:
    """Bid/ask/LTP for precise LIMIT entry."""
    try:
        kite = get_kite_instance()
        key = f"{exchange}:{tradingsymbol}"
        row = (kite.quote(key) or {}).get(key, {}) or {}
    except Exception as exc:
        log_warning(f"[Commodity] quote {tradingsymbol}: {exc}")
        return {"bid": 0.0, "ask": 0.0, "ltp": 0.0}
    depth = row.get("depth") or {}
    bid = float((depth.get("buy") or [{}])[0].get("price") or 0)
    ask = float((depth.get("sell") or [{}])[0].get("price") or 0)
    ltp = float(row.get("last_price") or 0)
    if ltp <= 0:
        o = row.get("ohlc") or {}
        ltp = float(o.get("close") or o.get("open") or 0)
    base = {"bid": bid, "ask": ask, "ltp": ltp}
    return merge_quote_with_circuit(base, row)


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


def _last_closed_5m_candle_high_low(token: int) -> tuple[Optional[float], Optional[float]]:
    """Return (high, low) of the last closed 5m candle for the underlying future."""
    try:
        kite = get_kite_instance()
        # Today's 5m candles; last row can be the forming candle => use second last when available.
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        IST = ZoneInfo("Asia/Kolkata")
        now = datetime.now(IST)
        from_dt = datetime.combine(now.date(), datetime.min.time()).replace(tzinfo=IST)
        rows = kite.historical_data(
            token,
            from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d %H:%M:%S"),
            "5minute",
            continuous=False,
            oi=False,
        ) or []
        if not rows:
            return None, None
        bar = rows[-2] if len(rows) >= 2 else rows[-1]
        hi = float(bar.get("high") or 0)
        lo = float(bar.get("low") or 0)
        if hi <= 0 or lo <= 0:
            return None, None
        return hi, lo
    except Exception:
        return None, None


def _spot_levels_from_last_5m(
    *,
    spot_entry: float,
    kind: str,
    reward_ratio: float,
    token: int,
) -> tuple[Optional[float], Optional[float], str]:
    """
    Candle-structure exits:
    - CE: SL below last closed 5m low; target = entry + R*(entry-SL)
    - PE: SL above last closed 5m high; target = entry - R*(SL-entry)
    """
    hi, lo = _last_closed_5m_candle_high_low(token)
    if hi is None or lo is None:
        return None, None, ""
    rng = max(1.0, hi - lo)
    buf = max(6.0, rng * 0.15)
    k = (kind or "CE").upper()
    rr = max(0.5, min(5.0, float(reward_ratio or 2.0)))
    if k == "CE":
        sl = lo - buf
        tgt = spot_entry + (spot_entry - sl) * rr
        return sl, tgt, f"5m candle SL/TP: low {lo:.1f} − buf {buf:.1f}"
    sl = hi + buf
    tgt = spot_entry - (sl - spot_entry) * rr
    return sl, tgt, f"5m candle SL/TP: high {hi:.1f} + buf {buf:.1f}"


def _normalize_long_option_exits(
    entry_premium: float,
    sl_prem: float,
    tgt_prem: float,
    *,
    min_gap: float = 0.05,
    risk_pct: float = 1.0,
    reward_pct: float = 2.0,
    min_rr: Optional[float] = None,
) -> Tuple[float, float]:
    """Long CE/PE: SL below entry, target above entry (exit SELL limits)."""
    from services.premium_exit_policy import enforce_min_premium_exits

    entry = float(entry_premium)
    sl = float(sl_prem)
    tp = float(tgt_prem)
    gap = max(min_gap, entry * 0.01)
    if sl >= entry:
        sl = max(0.05, entry - gap)
    if tp <= entry:
        tp = entry + gap
    if sl >= tp:
        sl = max(0.05, entry - gap)
        tp = entry + gap
    sl, tp = enforce_min_premium_exits(
        entry,
        sl,
        tp,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        min_rr=min_rr,
    )
    return round_to_tick(sl), round_to_tick(tp)


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
    sl_prem, tgt_prem = _normalize_long_option_exits(entry_premium, sl_prem, tgt_prem)
    return round_to_tick(sl_prem), round_to_tick(tgt_prem), delta


MCX_MAX_ORDER_QTY = 50


def resolve_mcx_qty_cap(max_qty_cap: int) -> int:
    """Ceiling for Kite qty: live fixed qty, explicit cap (>1), or auto (paper/live)."""
    try:
        from services.commodity_config import live_fixed_order_qty
        from services.paper_trading import is_paper_mode_for_segment

        fixed = live_fixed_order_qty()
        if fixed is not None and not is_paper_mode_for_segment("commodity"):
            return fixed
    except Exception:
        pass
    if int(max_qty_cap or 0) > 1:
        return int(max_qty_cap)
    return MCX_MAX_ORDER_QTY


def size_from_risk(
    capital: float,
    risk_pct: float,
    reward_pct: float,
    entry_premium: float,
    sl_premium: float,
    target_premium: float,
    lot_size: int,
    max_qty_cap: int = 1,
) -> Tuple[int, int, float, Dict[str, Any]]:
    """
    Return (kite_qty, kite_qty, risk_inr) for MCX options.

    Qty is derived from entry vs SL (risk per unit), available capital, and the
    configured risk/reward policy. ``max_qty_cap`` from settings is a ceiling only
    when > 1; the default of 1 means auto-size (up to MCX_MAX_ORDER_QTY).
    """
    ls = max(1, int(lot_size))
    entry = max(0.05, float(entry_premium))
    sl = float(sl_premium)
    tgt = float(target_premium)
    prem_risk = max(0.05, entry - sl)
    prem_reward = max(0.05, tgt - entry)

    cap = resolve_mcx_qty_cap(max_qty_cap)

    max_risk_amt = max(0.0, float(capital) * (float(risk_pct) / 100.0))
    risk_per_qty = prem_risk * ls
    max_from_risk = int(max_risk_amt / risk_per_qty) if risk_per_qty > 0 else 0

    premium_per_qty = entry * ls
    max_from_capital = (
        int(float(capital) / premium_per_qty) if premium_per_qty > 0 else 0
    )

    limits = [x for x in (max_from_risk, max_from_capital) if x > 0]
    qty = min(limits) if limits else 1

    # Premium R:R vs entry policy (1:1 default); scale down only if TP is tighter than policy.
    from services.premium_exit_policy import entry_initial_rr

    rr_policy = entry_initial_rr()
    prem_rr = prem_reward / prem_risk if prem_risk > 0 else rr_policy
    if prem_rr > 0 and prem_rr < rr_policy * 0.85 and max_from_risk > 1:
        qty = max(1, int(qty * (prem_rr / rr_policy)))

    qty = max(1, min(qty, cap))
    risk_inr = risk_per_qty * qty
    sizing = {
        "capital_inr": round(float(capital), 2),
        "risk_pct": float(risk_pct),
        "max_risk_inr": round(max_risk_amt, 2),
        "premium_risk_per_barrel": round(prem_risk, 2),
        "risk_inr_per_kite_qty": round(risk_per_qty, 2),
        "max_qty_from_risk": max_from_risk,
        "max_qty_from_capital": max_from_capital,
        "contract_units_per_qty": ls,
        "kite_qty": qty,
    }
    return qty, qty, risk_inr, sizing


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
        from services.premium_exit_policy import entry_initial_rr

        strategy_rr = reward_pct / risk_pct if risk_pct > 0 else 2.0
        rr_entry = entry_initial_rr()
        risk_pts = max(spot * (risk_pct / 100.0) * 0.35, 15.0)
        if option_kind == "CE":
            spot_sl, spot_tgt = spot - risk_pts, spot + risk_pts * rr_entry
        else:
            spot_sl, spot_tgt = spot + risk_pts, spot - risk_pts * rr_entry

    intra = {
        "pdh": ind.get("pdh"),
        "pdl": ind.get("pdl"),
        "or_high": ind.get("or_high"),
        "or_low": ind.get("or_low"),
        "ema9": ind.get("ema9"),
        "day_open": ind.get("day_open"),
        "crude_spot": spot,
    }
    from datetime import datetime
    from zoneinfo import ZoneInfo

    _now = datetime.now(ZoneInfo("Asia/Kolkata"))
    intra["session_minutes"] = _now.hour * 60 + _now.minute
    sid = strategy_id or "long_atm_directional"
    spot_entry, spot_sl, spot_tgt, level_note = refine_spot_levels_from_candles(
        sid, spot, option_kind, spot_sl, spot_tgt, intra
    )

    # Prefer candle-structure exits (last closed 5m high/low) over heuristics.
    from services.premium_exit_policy import entry_initial_rr

    rr_struct = entry_initial_rr()
    strategy_rr = reward_pct / risk_pct if risk_pct > 0 else 2.0
    sl5, tgt5, candle_note = _spot_levels_from_last_5m(
        spot_entry=spot_entry,
        kind=option_kind,
        reward_ratio=rr_struct,
        token=future_token(),
    )
    if sl5 is not None and tgt5 is not None:
        spot_sl, spot_tgt = float(sl5), float(tgt5)
        if candle_note:
            level_note = (level_note + " · " + candle_note).strip(" ·")
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

    from services.kite_live_indicators import get_option_bollinger_snapshot
    from services.option_contract_indicators import (
        merge_option_bb_into_intra,
        resolve_long_buy_exit_levels,
    )

    opt_bb = get_option_bollinger_snapshot(contract.tradingsymbol, "MCX")
    intra_bb = merge_option_bb_into_intra(intra, opt_bb, contract.tradingsymbol)
    intra_bb["contract_ltp"] = float(entry_prem)
    intra_bb["underlying_spot"] = spot_entry
    rr_ratio = entry_initial_rr()
    sl_prem, tgt_prem, spot_sl, spot_tgt, delta, exit_note = resolve_long_buy_exit_levels(
        strategy_id=sid,
        entry_premium=float(entry_prem),
        option_kind=option_kind,
        intra_bb=intra_bb,
        underlying_spot=spot_entry,
        underlying_sl=spot_sl,
        underlying_tgt=spot_tgt,
        strike=contract.strike,
        vix=ind.get("vix"),
        reward_ratio=rr_ratio,
        normalize_exits=_normalize_long_option_exits,
    )
    if exit_note:
        level_note = (level_note + " · " + exit_note).strip(" ·")

    entry_analysis = compute_strategy_entry(
        strategy_id=sid,
        option_kind=option_kind,
        quote=quote,
        spot=spot_entry,
        strike=contract.strike,
        delta=delta,
        intra=intra_bb,
        prev_close=float(ind.get("prev_close") or 0),
    )
    entry_limit = entry_analysis.entry_limit_price
    entry_limit, limit_ok, limit_msg = validate_buy_limit_price(entry_limit, quote=quote)
    if not limit_ok:
        entry_analysis = EntryAnalysis(
            entry_ready=False,
            entry_limit_price=entry_limit,
            fair_premium=entry_analysis.fair_premium,
            entry_style="circuit_blocked",
            spot_trigger=entry_analysis.spot_trigger,
            confirmation_score=entry_analysis.confirmation_score,
            notes=list(entry_analysis.notes) + ([limit_msg] if limit_msg else []),
            block_reason=limit_msg,
        )
    elif limit_msg:
        entry_analysis.notes.append(limit_msg)
    if not entry_analysis.entry_ready:
        messages.append(
            entry_analysis.block_reason or "Entry not confirmed — wait for setup"
        )
    messages.extend(entry_analysis.notes)

    ls = int(contract.lot_size or lot_size())

    qty_lots, quantity, risk_inr, qty_sizing = size_from_risk(
        capital,
        risk_pct,
        reward_pct,
        float(entry_prem),
        sl_prem,
        tgt_prem,
        ls,
        num_lots,
    )
    reward_inr = max(0.0, (tgt_prem - float(entry_prem)) * ls * qty_lots)
    rr = (reward_inr / risk_inr) if risk_inr > 0 else 0.0

    bb_zone = None
    bb_mid = intra_bb.get("bb_middle")
    bb_up = intra_bb.get("bb_upper")
    bb_lo = intra_bb.get("bb_lower")
    if bb_mid is not None and bb_up is not None and bb_lo is not None:
        from services.kite_live_indicators import bollinger_zone

        bb_zone = bollinger_zone(
            float(entry_prem),
            float(bb_mid),
            float(bb_up),
            float(bb_lo),
            option_kind,
        ).get("zone")

    indicator_snapshot = {
        **ind,
        "bb_lower": bb_lo,
        "bb_middle": bb_mid,
        "bb_upper": bb_up,
        "bb_zone": bb_zone,
        "margin": capital,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "indicator_sources": intra_bb.get("indicator_sources") or {},
        "strategy_id": sid,
        "pattern_tag": pattern_tag,
        "underlying_spot": round(spot_entry, 2),
        "spot_entry": round(float(entry_prem), 2),
        "spot_stop_loss": round(spot_sl, 2),
        "spot_target": round(spot_tgt, 2),
        "level_note": level_note,
        "option_bid": quote.get("bid"),
        "option_ask": quote.get("ask"),
        "option_ltp": quote.get("ltp") or opt_bb.get("option_ltp"),
        "bb_on_contract": contract.tradingsymbol,
        "contract_ltp": round(float(entry_prem), 2),
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
        "max_qty_cap": num_lots,
        "qty_sizing": qty_sizing,
        "product": COMMODITY_PRODUCT,
        "entry_order_type": "LIMIT",
        "entry_limit_price": entry_limit,
        "exit_order_type": "SL_STEPPED",
        "entry_premium": round(float(entry_prem), 2),
        "stop_loss_premium": sl_prem,
        "target_premium": tgt_prem,
        "nifty_spot": round(spot_entry, 2),
        "spot_stop_loss": round(spot_sl, 2),
        "spot_target": round(spot_tgt, 2),
        "risk_inr": round(risk_inr, 2),
        "reward_inr": round(reward_inr, 2),
        "reward_ratio": round(rr, 2),
        "strategy_reward_ratio": round(strategy_rr, 2),
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
    prem_risk = max(0.05, float(entry_prem) - sl_prem)
    messages.append(
        f"Entry LIMIT ₹{entry_limit} (LTP ₹{entry_prem:.2f}) · Kite qty {quantity} "
        f"(₹{prem_risk:.2f} risk/bbl × {ls} bbl, {risk_pct:.1f}% cap) · "
        f"Risk ₹{risk_inr:.0f} · R:R {reward_pct:.0f}:{risk_pct:.0f} · GTT SL ₹{sl_prem} TP ₹{tgt_prem}"
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
    ind_meta = plan.get("indicators") or {}
    risk_pct = float(ind_meta.get("risk_pct") or 1.0)
    from services.premium_exit_policy import entry_initial_rr

    rr_struct = entry_initial_rr()
    sl5, tgt5, _ = _spot_levels_from_last_5m(
        spot_entry=spot_entry,
        kind=kind,
        reward_ratio=rr_struct,
        token=future_token(),
    )
    if sl5 is not None and tgt5 is not None:
        spot_sl, spot_tgt = float(sl5), float(tgt5)
    entry_analysis = compute_strategy_entry(
        strategy_id=sid,
        option_kind=kind,
        quote=quote,
        spot=spot_entry,
        strike=int(plan.get("strike", 0)),
        delta=estimate_delta_from_spot(
            spot_entry, int(plan.get("strike", 0)), kind, vix=live.get("vix")
        ),
        intra={**intra, "last_5m_close": live.get("last_5m_close")},
        prev_close=float(live.get("prev_close") or 0),
    )
    entry_limit = entry_analysis.entry_limit_price
    entry_limit, limit_ok, limit_msg = validate_buy_limit_price(entry_limit, quote=quote)
    if not limit_ok:
        entry_analysis = EntryAnalysis(
            entry_ready=False,
            entry_limit_price=entry_limit,
            fair_premium=entry_analysis.fair_premium,
            entry_style="circuit_blocked",
            spot_trigger=entry_analysis.spot_trigger,
            confirmation_score=entry_analysis.confirmation_score,
            notes=list(entry_analysis.notes) + ([limit_msg] if limit_msg else []),
            block_reason=limit_msg,
        )
    elif limit_msg:
        entry_analysis.notes.append(limit_msg)
    from services.kite_live_indicators import get_option_bollinger_snapshot
    from services.option_contract_indicators import (
        merge_option_bb_into_intra,
        resolve_long_buy_exit_levels,
    )

    opt_bb = get_option_bollinger_snapshot(sym, "MCX")
    intra_bb = merge_option_bb_into_intra(intra, opt_bb, sym)
    exit_anchor = float(entry_limit or entry_prem or 0)
    intra_bb["contract_ltp"] = exit_anchor
    intra_bb["underlying_spot"] = spot_entry
    sl_prem, tgt_prem, spot_sl, spot_tgt, delta, _ = resolve_long_buy_exit_levels(
        strategy_id=sid,
        entry_premium=exit_anchor,
        option_kind=kind,
        intra_bb=intra_bb,
        underlying_spot=spot_entry,
        underlying_sl=spot_sl,
        underlying_tgt=spot_tgt,
        strike=int(plan.get("strike", 0)),
        vix=live.get("vix"),
        reward_ratio=rr_struct,
        normalize_exits=_normalize_long_option_exits,
    )
    capital = float(ind_meta.get("margin") or 0)
    lot_size = int(plan.get("lot_size") or 75)
    max_qty_cap = int(plan.get("max_qty_cap") or 1)
    reward_pct = float(ind_meta.get("reward_pct") or plan.get("reward_pct") or 2.0)
    if capital > 0:
        qty_lots, quantity, risk_inr, qty_sizing = size_from_risk(
            capital,
            risk_pct,
            reward_pct,
            float(entry_prem),
            sl_prem,
            tgt_prem,
            lot_size,
            max_qty_cap,
        )
    else:
        qty_lots = int(plan.get("num_lots") or plan.get("quantity") or 1)
        quantity = qty_lots
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
            "exit_order_type": "SL_STEPPED",
            "num_lots": qty_lots,
            "quantity": quantity,
            "risk_inr": round(risk_inr, 2) if risk_inr else plan.get("risk_inr"),
            "reward_inr": round(
                max(0.0, (tgt_prem - float(entry_prem)) * lot_size * qty_lots), 2
            ),
            "qty_sizing": qty_sizing,
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
    ind.update(
        {
            "pdh": live.get("pdh"),
            "pdl": live.get("pdl"),
            "or_high": live.get("or_high"),
            "or_low": live.get("or_low"),
            "ema9": live.get("ema9"),
            "underlying_spot": round(spot_entry, 2),
            "bb_lower": intra_bb.get("bb_lower"),
            "bb_middle": intra_bb.get("bb_middle"),
            "bb_upper": intra_bb.get("bb_upper"),
            "bb_on_contract": sym,
            "contract_ltp": round(exit_anchor, 2),
            "option_bid": quote.get("bid"),
            "option_ask": quote.get("ask"),
            "option_ltp": quote.get("ltp"),
            "indicator_sources": intra_bb.get("indicator_sources") or {},
            "refreshed_at_execution": True,
        }
    )
    updated["indicators"] = ind
    from services.premium_exit_policy import enforce_plan_exits

    return enforce_plan_exits(updated, entry=exit_anchor)


def gtt_triggers_from_plan(plan: Dict[str, Any]) -> Tuple[float, float, float]:
    """OCO trigger prices and last_price for GTT placement."""
    from services.trading_agents.gtt_agent import gtt_triggers_from_plan as _gtt

    return _gtt(plan)
