"""
V2 — precise entry/exit/SL/size from realtime indicators + live option quotes.
Entry: LIMIT at live premium. Exit control: GTT OCO only (no MARKET exit).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from services.push.option_contract_resolver import (
    OptionContract,
    estimate_delta_from_spot,
    fetch_option_ltp,
)
from services.sensex_option_chain import resolve_sensex_contract
from services.sensex_live_indicators import get_sensex_bundle_for_v2, get_vix_snapshot, recalculate_from_ticker
from services.sensex_constants import SENSEX_BFO_PRODUCT
from services.sensex_entry_pricing import EntryAnalysis, compute_strategy_entry
from services.sensex_constants import sensex_max_lots_per_trade
from services.sensex_strategy_analysis import STRATEGY_ID as TWENTY_RUPEES_ID
from services.option_contract_indicators import (
    merge_option_bb_into_intra,
    order_exit_levels_from_contract_bb,
)
from services.sensex_strike_pricing import (
    _pick_moneyness,
    refine_spot_levels_from_candles,
)
from utils.kite_order_utils import merge_quote_with_circuit, round_to_tick, validate_buy_limit_price
from utils.kite_utils import get_kite_instance
from utils.logger import log_warning


def fetch_realtime_indicators() -> Dict[str, Any]:
    """Live indicators from Kite ticker; historical only fills gaps."""
    return recalculate_from_ticker()


def live_option_quote(tradingsymbol: str, exchange: str = "BFO") -> Dict[str, float]:
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
    """Map indicator-based Sensex levels → option SL/target premiums (long CE/PE, live delta)."""
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
    """Return (num_lots, quantity, risk_inr) from risk rule and live premium risk."""
    prem_risk = max(0.05, entry_premium - sl_premium)
    max_risk_amt = capital * (risk_pct / 100.0)
    max_lots = int(max_risk_amt / (prem_risk * lot_size)) if prem_risk > 0 else 1
    qty_lots = min(num_lots, max(1, max_lots))
    quantity = qty_lots * lot_size
    risk_inr = prem_risk * quantity
    return qty_lots, quantity, risk_inr


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
        return None, ["Could not fetch live Sensex spot"]

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
        anchor_strike = sel.get("anchor_strike")
    else:
        anchor_strike = None
        sel = {}
        d = (direction or "AUTO").upper()
        if d in ("CE", "PE"):
            option_kind = d
        elif ind["prev_close"] > 0:
            option_kind = "CE" if spot >= ind["prev_close"] else "PE"
        else:
            option_kind = "CE"
        rr = reward_pct / risk_pct if risk_pct > 0 else 2.0
        from services.premium_exit_policy import entry_initial_rr

        risk_pts = max(spot * (risk_pct / 100.0) * 0.35, 15.0)
        rr_entry = entry_initial_rr()
        if option_kind == "CE":
            spot_sl, spot_tgt = spot - risk_pts, spot + risk_pts * rr_entry
        else:
            spot_sl, spot_tgt = spot + risk_pts, spot - risk_pts * rr_entry

    sid = strategy_id or TWENTY_RUPEES_ID
    # Index structure only — BB for order prices comes from the selected contract chart.
    nifty_spot = float(ind["nifty_spot"])
    intra = {
        "pdh": ind.get("pdh"),
        "pdl": ind.get("pdl"),
        "or_high": ind.get("or_high"),
        "or_low": ind.get("or_low"),
        "ema9": ind.get("ema9"),
        "day_high": ind.get("day_high"),
        "day_low": ind.get("day_low"),
        "day_open": ind.get("day_open"),
        "prev_close": ind.get("prev_close"),
        "indicator_sources": ind.get("indicator_sources") or {},
        "anchor_strike": anchor_strike,
        "nifty_spot": nifty_spot,
        "oi_baseline_ready": bool(
            sid == TWENTY_RUPEES_ID and anchor_strike
        ),
    }
    spot_entry = nifty_spot
    if strategy_analysis and sid == TWENTY_RUPEES_ID:
        prem = float(sel.get("entry_premium") or 0)
        if prem > 0:
            intra["contract_ltp"] = prem
            intra["option_ltp"] = prem
    if strategy_analysis:
        cand_entry = float(sel.get("spot_entry") or nifty_spot)
        if cand_entry > 5000:
            spot_entry = cand_entry
    level_note = ""
    spot_entry, spot_sl, spot_tgt, level_note = refine_spot_levels_from_candles(
        sid, spot_entry, option_kind, spot_sl, spot_tgt, intra
    )

    moneyness, pattern_tag, m_reason = _pick_moneyness(
        sid, spot_entry, option_kind, spot_sl, spot_tgt, intra
    )

    contract: Optional[OptionContract] = None
    kite = get_kite_instance()
    pick_strike = int(anchor_strike or round(nifty_spot / 100) * 100)
    contract = resolve_sensex_contract(kite, strike=pick_strike, kind=option_kind)
    if contract is None:
        return None, ["Could not resolve Sensex BFO option contract for live strike"]

    quote = live_option_quote(contract.tradingsymbol)
    entry_prem = quote["ltp"] or fetch_option_ltp(contract)
    if not entry_prem or entry_prem <= 0:
        entry_prem = max(1.0, 0.007 * spot_entry)
        messages.append("Live option LTP unavailable — using estimate")

    from services.sensex_live_indicators import get_option_bollinger_snapshot

    opt_bb = get_option_bollinger_snapshot(contract.tradingsymbol, "BFO")
    intra_bb = merge_option_bb_into_intra(intra, opt_bb, contract.tradingsymbol)
    intra_bb["contract_ltp"] = float(entry_prem)
    intra_bb["option_ltp"] = quote.get("ltp") or opt_bb.get("option_ltp")
    if opt_bb.get("bb_middle") is None:
        messages.append(
            f"Option 5m BB not ready on {contract.tradingsymbol} — "
            "need 20×5m bars (order SL/TP uses fallback until loaded)"
        )

    from services.premium_exit_policy import entry_initial_rr

    rr_ratio = entry_initial_rr()
    strategy_rr = reward_pct / risk_pct if risk_pct > 0 else 2.0
    from services.option_contract_indicators import resolve_long_buy_exit_levels

    if sid == TWENTY_RUPEES_ID:
        from services.sensex_strategy_analysis import FIXED_SL_INR

        sl_prem = round_to_tick(max(0.05, float(entry_prem) - FIXED_SL_INR))
        tgt_prem = round_to_tick(float(entry_prem) + FIXED_SL_INR)
        spot_sl = sl_prem
        spot_tgt = tgt_prem
        delta = estimate_delta_from_spot(nifty_spot, contract.strike, option_kind, vix=ind.get("vix"))
        exit_note = "20rupees-strategy: ₹10 SL, 1:1 target; trail after fill; no new entries after 15:00 IST"
    else:
        sl_prem, tgt_prem, spot_sl, spot_tgt, delta, exit_note = resolve_long_buy_exit_levels(
            strategy_id=sid,
            entry_premium=float(entry_prem),
            option_kind=option_kind,
            intra_bb=intra_bb,
            underlying_spot=nifty_spot,
            underlying_sl=spot_sl,
            underlying_tgt=spot_tgt,
            strike=contract.strike,
            vix=ind.get("vix"),
            reward_ratio=rr_ratio,
        )
    if exit_note:
        level_note = (level_note + " · " + exit_note).strip(" ·")

    entry_analysis = compute_strategy_entry(
        strategy_id=sid,
        option_kind=option_kind,
        quote=quote,
        spot=nifty_spot,
        strike=contract.strike,
        delta=delta,
        intra=intra_bb,
        prev_close=float(ind.get("prev_close") or 0),
    )
    entry_limit = entry_analysis.entry_limit_price
    if not entry_analysis.entry_ready:
        messages.append(
            entry_analysis.block_reason or "Entry not confirmed — wait for setup"
        )
    messages.extend(entry_analysis.notes)

    lot_size = 20
    try:
        for row in kite.instruments("BFO"):
            if row.get("name") == "SENSEX" and row.get("instrument_type") == option_kind:
                lot_size = int(row.get("lot_size") or 20)
                break
    except Exception:
        pass

    lot_cap = min(sensex_max_lots_per_trade(), max(1, num_lots))
    if sid == TWENTY_RUPEES_ID or capital > 0:
        qty_lots, quantity, risk_inr = size_from_risk(
            capital, risk_pct, float(entry_prem), sl_prem, lot_size, lot_cap
        )
    else:
        qty_lots = num_lots
        quantity = num_lots * lot_size
        prem_risk = max(0.05, float(entry_prem) - sl_prem)
        risk_inr = prem_risk * quantity
    reward_inr = max(0.0, (tgt_prem - float(entry_prem)) * quantity)
    rr = (reward_inr / risk_inr) if risk_inr > 0 else 0.0

    bb_zone = None
    bb_mid = intra_bb.get("bb_middle")
    bb_up = intra_bb.get("bb_upper")
    bb_lo = intra_bb.get("bb_lower")
    if bb_mid is not None and bb_up is not None and bb_lo is not None:
        from services.sensex_live_indicators import bollinger_zone

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
        "indicator_sources": intra_bb.get("indicator_sources") or {},
        "strategy_id": sid,
        "pattern_tag": pattern_tag,
        "spot_entry": round(float(entry_prem), 2),
        "spot_stop_loss": round(spot_sl, 2),
        "spot_target": round(spot_tgt, 2),
        "level_note": level_note,
        "option_bid": quote.get("bid"),
        "option_ask": quote.get("ask"),
        "option_ltp": quote.get("ltp") or opt_bb.get("option_ltp"),
        "nifty_spot": round(nifty_spot, 2),
        "bb_on_contract": contract.tradingsymbol,
        "contract_ltp": round(float(entry_prem), 2),
    }

    plan = {
        "tradingsymbol": contract.tradingsymbol,
        "exchange": "BFO",
        "option_type": option_kind,
        "strike": int(contract.strike),
        "expiry": contract.expiry.isoformat(),
        "quantity": quantity,
        "lot_size": lot_size,
        "num_lots": qty_lots,
        "product": SENSEX_BFO_PRODUCT,
        "entry_order_type": "LIMIT",
        "entry_limit_price": entry_limit,
        "exit_order_type": "GTT_OCO",
        "entry_premium": round(float(entry_prem), 2),
        "stop_loss_premium": sl_prem,
        "target_premium": tgt_prem,
        "nifty_spot": round(float(ind.get("nifty_spot") or spot), 2),
        "spot_stop_loss": round(spot_sl, 2),
        "spot_target": round(spot_tgt, 2),
        "bb_on_contract": contract.tradingsymbol,
        "risk_inr": round(risk_inr, 2),
        "reward_inr": round(reward_inr, 2),
        "reward_ratio": round(rr, 2),
        "strategy_reward_ratio": round(strategy_rr, 2),
        "estimated_premium": quote.get("ltp", 0) <= 0,
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "strike_moneyness": moneyness,
        "pattern_tag": pattern_tag,
        "anchor_strike": anchor_strike,
        "oi_change": sel.get("oi_change") if strategy_analysis and sid == TWENTY_RUPEES_ID else None,
        "delta_used": round(delta, 3),
        "atm_reference": int(round(nifty_spot / 100) * 100),
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
        f"Indicators: Sensex {nifty_spot:.0f} | OR {ind.get('or_low')}-{ind.get('or_high')} | "
        f"PDH {ind.get('pdh')} PDL {ind.get('pdl')} | EMA9 {ind.get('ema9')} | VIX {ind.get('vix')}"
    )
    messages.append(
        f"Entry LIMIT ₹{entry_limit} (LTP ₹{entry_prem:.2f}) · {qty_lots} lot(s) × {lot_size} = {quantity} qty · "
        f"Risk ₹{risk_inr:.0f} · GTT exit SL ₹{sl_prem} TP ₹{tgt_prem}"
        + (" (20rupees 1:1 + trail)" if sid == TWENTY_RUPEES_ID else f" (from {contract.tradingsymbol} 5m BB)")
    )
    return plan, messages


def refresh_plan_at_execution(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Re-quote option + refresh contract BB for entry / SL / TP before place."""
    live = recalculate_from_ticker()
    sym = plan.get("tradingsymbol")
    if not sym:
        return plan
    quote = live_option_quote(sym)
    entry_prem = float(quote["ltp"] or plan.get("entry_premium", 0) or 0)
    nifty_spot = float(live.get("nifty_spot") or plan.get("nifty_spot", 0))
    intra = {
        "pdh": live.get("pdh"),
        "pdl": live.get("pdl"),
        "or_high": live.get("or_high"),
        "or_low": live.get("or_low"),
        "ema9": live.get("ema9"),
        "nifty_spot": nifty_spot,
    }
    sid = plan.get("strategy_id") or TWENTY_RUPEES_ID
    kind = plan.get("option_type", "CE")
    from services.sensex_live_indicators import get_option_bollinger_snapshot

    opt_bb = get_option_bollinger_snapshot(sym, "BFO")
    intra_bb = merge_option_bb_into_intra(intra, opt_bb, sym)
    intra_bb["contract_ltp"] = entry_prem
    intra_bb["option_ltp"] = quote.get("ltp") or opt_bb.get("option_ltp")

    spot_sl = float(plan.get("spot_stop_loss", 0))
    spot_tgt = float(plan.get("spot_target", 0))
    spot_entry, spot_sl, spot_tgt, _ = refine_spot_levels_from_candles(
        sid, nifty_spot, kind, spot_sl, spot_tgt, intra
    )
    ind_meta = plan.get("indicators") or {}
    risk_pct = float(ind_meta.get("risk_pct") or 1.0)
    from services.premium_exit_policy import entry_initial_rr

    rr_ratio = entry_initial_rr()
    from services.option_contract_indicators import resolve_long_buy_exit_levels

    sl_prem, tgt_prem, spot_sl, spot_tgt, delta, _ = resolve_long_buy_exit_levels(
        strategy_id=sid,
        entry_premium=entry_prem,
        option_kind=kind,
        intra_bb=intra_bb,
        underlying_spot=nifty_spot,
        underlying_sl=spot_sl,
        underlying_tgt=spot_tgt,
        strike=int(plan.get("strike", 0)),
        vix=live.get("vix"),
        reward_ratio=rr_ratio,
    )
    entry_analysis = compute_strategy_entry(
        strategy_id=sid,
        option_kind=kind,
        quote=quote,
        spot=nifty_spot,
        strike=int(plan.get("strike", 0)),
        delta=delta,
        intra=intra_bb,
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
    ind_meta = plan.get("indicators") or {}
    capital = float(ind_meta.get("margin") or 0)
    risk_pct = float(ind_meta.get("risk_pct") or 1.0)
    lot_size = int(plan.get("lot_size") or 20)
    num_lots = int(plan.get("num_lots") or 1)
    lot_cap = min(sensex_max_lots_per_trade(), max(1, num_lots))
    if capital > 0:
        qty_lots, quantity, risk_inr = size_from_risk(
            capital,
            risk_pct,
            float(entry_prem),
            sl_prem,
            lot_size,
            lot_cap,
        )
    else:
        qty_lots = num_lots
        quantity = plan.get("quantity", lot_size * num_lots)
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
            "nifty_spot": round(nifty_spot, 2),
            "spot_stop_loss": round(spot_sl, 2),
            "spot_target": round(spot_tgt, 2),
            "bb_on_contract": sym,
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
            "nifty_spot": round(nifty_spot, 2),
            "bb_lower": intra_bb.get("bb_lower"),
            "bb_middle": intra_bb.get("bb_middle"),
            "bb_upper": intra_bb.get("bb_upper"),
            "bb_on_contract": sym,
            "contract_ltp": round(entry_prem, 2),
            "option_bid": quote.get("bid"),
            "option_ask": quote.get("ask"),
            "option_ltp": quote.get("ltp"),
            "indicator_sources": intra_bb.get("indicator_sources") or {},
            "refreshed_at_execution": True,
        }
    )
    updated["indicators"] = ind
    from services.premium_exit_policy import enforce_plan_exits

    exit_anchor = float(
        updated.get("entry_limit_price") or updated.get("entry_premium") or 0
    )
    return enforce_plan_exits(updated, entry=exit_anchor)


def gtt_triggers_from_plan(plan: Dict[str, Any]) -> Tuple[float, float, float]:
    """OCO trigger prices and last_price for GTT placement."""
    from services.trading_agents.gtt_agent import gtt_triggers_from_plan as _gtt

    return _gtt(plan)
