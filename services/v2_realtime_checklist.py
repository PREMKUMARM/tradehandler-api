"""
V2 — live checklist: every step uses fresh Kite ticker (Nifty) + quotes + chain OI.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from schemas.v2_trading import ChecklistStepStatus
from services.nifty_option_chain import build_nifty_options_universe, nifty50_index_token
from services.v2_strategy_analysis import analyze_fno_strategies
from utils.kite_utils import get_kite_instance
from utils.logger import log_warning

IST = ZoneInfo("Asia/Kolkata")
NIFTY_TOKEN = 256265


def _now_ist() -> datetime:
    return datetime.now(IST)


def _ts() -> str:
    return _now_ist().strftime("%H:%M:%S IST")


def _step(
    index: int,
    title: str,
    ok: bool,
    message: str,
    output: Optional[str] = None,
) -> ChecklistStepStatus:
    return ChecklistStepStatus(
        index=index,
        title=title,
        completed=ok,
        server_ok=ok,
        message=message,
        output=(f"[{_ts()}] {output}" if output else f"Updated {_ts()}"),
    )


def _nifty_live(kite) -> Dict[str, Any]:
    """Prefer Kite ticker cache; fallback to REST quote."""
    spot = 0.0
    ohlc: Dict[str, float] = {}
    source = "quote"
    try:
        from utils.kite_websocket_ticker import get_kite_ticker_instance

        ticker = get_kite_ticker_instance()
        if ticker:
            tick = ticker.get_latest_tick(NIFTY_TOKEN)
            if tick and tick.get("last_price"):
                spot = float(tick["last_price"])
                raw_ohlc = tick.get("ohlc") or {}
                ohlc = {
                    "open": float(raw_ohlc.get("open") or spot),
                    "high": float(raw_ohlc.get("high") or spot),
                    "low": float(raw_ohlc.get("low") or spot),
                    "close": float(raw_ohlc.get("close") or spot),
                }
                source = "kite_ticker"
    except Exception as exc:
        log_warning(f"[V2 realtime] ticker read: {exc}")

    if spot <= 0:
        q = kite.quote("NSE:NIFTY 50").get("NSE:NIFTY 50", {}) or {}
        spot = float(q.get("last_price") or 0)
        o = q.get("ohlc") or {}
        ohlc = {
            "open": float(o.get("open") or spot),
            "high": float(o.get("high") or spot),
            "low": float(o.get("low") or spot),
            "close": float(o.get("close") or spot),
        }
        source = "kite_quote"

    return {"spot": spot, "ohlc": ohlc, "source": source}


def _vix_live(kite) -> Dict[str, Any]:
    try:
        q = kite.quote("NSE:INDIA VIX").get("NSE:INDIA VIX", {}) or {}
        ltp = float(q.get("last_price") or 0)
        prev = float((q.get("ohlc") or {}).get("close") or ltp)
        chg = ((ltp - prev) / prev * 100) if prev else 0
        return {"ltp": ltp, "prev": prev, "chg_pct": chg}
    except Exception:
        return {"ltp": 0.0, "prev": 0.0, "chg_pct": 0.0}


def _intraday_context(kite, token: int) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "pdh": None,
        "pdl": None,
        "or_high": None,
        "or_low": None,
        "ema9": None,
        "last_5m_close": None,
    }
    now = _now_ist()
    try:
        daily = kite.historical_data(
            token,
            (now.date() - timedelta(days=10)).strftime("%Y-%m-%d"),
            now.strftime("%Y-%m-%d"),
            "day",
            continuous=False,
            oi=False,
        )
        if daily and len(daily) >= 2:
            prev = daily[-2]
            ctx["pdh"] = float(prev.get("high") or 0)
            ctx["pdl"] = float(prev.get("low") or 0)
    except Exception as exc:
        log_warning(f"[V2 realtime] daily: {exc}")

    try:
        from_dt = datetime.combine(now.date(), datetime.min.time()).replace(tzinfo=IST)
        c15 = kite.historical_data(
            token,
            from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d %H:%M:%S"),
            "15minute",
            continuous=False,
            oi=False,
        )
        or_bars = [
            c
            for c in (c15 or [])
            if c.get("date")
            and 9 * 60 + 15
            <= c["date"].astimezone(IST).hour * 60 + c["date"].astimezone(IST).minute
            < 9 * 60 + 30
        ]
        if or_bars:
            ctx["or_high"] = max(float(c["high"]) for c in or_bars)
            ctx["or_low"] = min(float(c["low"]) for c in or_bars)

        c5 = kite.historical_data(
            token,
            from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d %H:%M:%S"),
            "5minute",
            continuous=False,
            oi=False,
        )
        if c5:
            closes = [float(c["close"]) for c in c5[-12:]]
            ctx["last_5m_close"] = closes[-1] if closes else None
            if len(closes) >= 9:
                k = 2 / (9 + 1)
                ema = closes[0]
                for px in closes[1:]:
                    ema = px * k + ema * (1 - k)
                ctx["ema9"] = ema
    except Exception as exc:
        log_warning(f"[V2 realtime] intraday: {exc}")
    return ctx


def _chain_live(kite, spot: float, universe: List[Dict[str, Any]]) -> Dict[str, Any]:
    """ATM ±5 strikes, nearest expiry — live OI, LTP, spread."""
    if spot <= 0 or not universe:
        return {}
    atm = int(round(spot / 50) * 50)
    expiries = sorted({u["expiry"] for u in universe if u.get("expiry")})
    if not expiries:
        return {}
    expiry = expiries[0]
    if hasattr(expiry, "isoformat"):
        expiry_key = expiry
    else:
        expiry_key = expiry

    rows = [
        u
        for u in universe
        if u.get("expiry") == expiry_key
        and abs(int(u.get("strike") or 0) - atm) <= 250
    ]
    keys = [f"NFO:{r['tradingsymbol']}" for r in rows if r.get("tradingsymbol")]
    quotes: Dict[str, Any] = {}
    if keys:
        try:
            quotes = kite.quote(keys) or {}
        except Exception as exc:
            log_warning(f"[V2 realtime] chain quote: {exc}")

    ce_oi = pe_oi = 0.0
    max_oi_strike = atm
    max_oi_val = 0.0
    atm_ce = atm_pe = None

    for r in rows:
        key = f"NFO:{r['tradingsymbol']}"
        qd = quotes.get(key, {}) or {}
        oi = float(qd.get("oi") or 0)
        ltp = float(qd.get("last_price") or 0)
        depth = qd.get("depth") or {}
        bid = float((depth.get("buy") or [{}])[0].get("price") or 0) if depth.get("buy") else 0
        ask = float((depth.get("sell") or [{}])[0].get("price") or 0) if depth.get("sell") else 0
        spread_pct = ((ask - bid) / ltp * 100) if ltp > 0 and ask > bid else 0
        strike = int(r.get("strike") or 0)
        kind = r.get("instrument_type")
        if kind == "CE":
            ce_oi += oi
            if strike == atm:
                atm_ce = {"ltp": ltp, "oi": oi, "spread_pct": round(spread_pct, 2)}
        else:
            pe_oi += oi
            if strike == atm:
                atm_pe = {"ltp": ltp, "oi": oi, "spread_pct": round(spread_pct, 2)}
        if oi > max_oi_val:
            max_oi_val = oi
            max_oi_strike = strike

    pcr = (pe_oi / ce_oi) if ce_oi > 0 else 0.0
    exp_str = expiry_key.isoformat() if hasattr(expiry_key, "isoformat") else str(expiry_key)
    return {
        "expiry": exp_str,
        "atm": atm,
        "pcr": round(pcr, 2),
        "ce_oi_total": int(ce_oi),
        "pe_oi_total": int(pe_oi),
        "max_pain_strike": max_oi_strike,
        "atm_ce": atm_ce,
        "atm_pe": atm_pe,
    }


def run_realtime_checklist(
    direction: str,
    margin: float,
    market_open: bool,
    risk_pct: float,
    reward_pct: float,
    num_lots: int,
    capital: float,
) -> Dict[str, Any]:
    """
    Execute all 12 checklist steps with live Kite data.
    Returns step_statuses, strategy_analysis, trade_plan, missing, checklist_ready.
    """
    from services.v2_trade_service import (
        STEP_TITLES,
        _validate_trade_plan,
        build_trade_plan,
    )

    kite = get_kite_instance()
    token = nifty50_index_token(kite)
    nifty = _nifty_live(kite)
    spot = nifty["spot"]
    ohlc = nifty["ohlc"]
    prev_close = float(ohlc.get("close") or spot)
    vix = _vix_live(kite)
    intra = _intraday_context(kite, token)
    universe = build_nifty_options_universe(kite)
    chain = _chain_live(kite, spot, universe)

    strategy_analysis = analyze_fno_strategies(direction_pref=direction, margin=margin)
    trade_plan = None
    plan_msgs: List[str] = []
    validation = None
    if spot > 0:
        plan, plan_msgs = build_trade_plan(
            direction,
            risk_pct,
            reward_pct,
            num_lots,
            capital,
            strategy_analysis=strategy_analysis,
        )
        trade_plan = plan
        if trade_plan:
            validation = _validate_trade_plan(trade_plan, capital, risk_pct, reward_pct)

    missing: List[int] = []
    statuses: List[ChecklistStepStatus] = []
    opt = (trade_plan or {}).get("option_type") or strategy_analysis.get("selected_option_kind") or "CE"

    for i, title in enumerate(STEP_TITLES):
        if i == 0:
            ok = margin > 0 and spot > 0
            if not ok:
                missing.append(i)
            src = nifty.get("source", "quote")
            out = f"Margin ₹{margin:,.0f} · Nifty {spot:.2f} via {src}"
            statuses.append(_step(i, title, ok, "Session live", out))
        elif i == 1:
            gap = ((ohlc.get("open", spot) - prev_close) / prev_close * 100) if prev_close else 0
            bias = "CE" if spot >= prev_close else "PE"
            if direction.upper() in ("CE", "PE"):
                bias = direction.upper()
            ok = spot > 0
            or_h, or_l = intra.get("or_high"), intra.get("or_low")
            pdh, pdl = intra.get("pdh"), intra.get("pdl")
            parts = [
                f"Spot {spot:.2f} vs prior {prev_close:.2f} → {bias}",
                f"Day {ohlc.get('low', spot):.0f}–{ohlc.get('high', spot):.0f}",
                f"Gap {gap:+.2f}%",
            ]
            if or_h and or_l:
                parts.append(f"OR {or_l:.0f}–{or_h:.0f}")
            if pdh and pdl:
                parts.append(f"PDH {pdh:.0f} PDL {pdl:.0f}")
            statuses.append(_step(i, title, ok, "Hypothesis (live)", " · ".join(parts)))
        elif i == 2:
            ok = vix["ltp"] > 0
            iv_note = "elevated" if vix["ltp"] > 18 else "moderate" if vix["ltp"] > 12 else "low"
            out = f"VIX {vix['ltp']:.2f} ({vix['chg_pct']:+.1f}% vs prior close) — IV {iv_note}"
            statuses.append(_step(i, title, ok, "VIX live", out))
        elif i == 3:
            ok = bool(strategy_analysis.get("selected_id"))
            out = strategy_analysis.get("output_summary", "")
            statuses.append(
                _step(
                    i,
                    title,
                    ok,
                    f"Best: {strategy_analysis.get('selected_name')} ({strategy_analysis.get('selected_score')}/100)",
                    out,
                )
            )
        elif i == 4:
            ok = bool(chain)
            if chain:
                ce = chain.get("atm_ce") or {}
                pe = chain.get("atm_pe") or {}
                out = (
                    f"Expiry {chain.get('expiry')} · ATM {chain.get('atm')} · PCR {chain.get('pcr')} · "
                    f"CE OI {chain.get('ce_oi_total'):,} PE OI {chain.get('pe_oi_total'):,} · "
                    f"Max-OI strike {chain.get('max_pain_strike')} · "
                    f"ATM CE ₹{ce.get('ltp', 0)} spread {ce.get('spread_pct', 0)}% · "
                    f"ATM PE ₹{pe.get('ltp', 0)} spread {pe.get('spread_pct', 0)}%"
                )
            else:
                out = "Chain quote unavailable"
                ok = False
            statuses.append(_step(i, title, ok, "Option chain (live OI)", out))
        elif i == 5:
            ok = bool(trade_plan)
            out = f"Expiry {trade_plan.get('expiry')}" if trade_plan else "—"
            statuses.append(_step(i, title, ok, "Expiry (nearest liquid)", out))
        elif i == 6:
            ok = bool(trade_plan)
            spread = ""
            if chain and chain.get("atm_ce"):
                spread = f" · ATM CE spread {chain['atm_ce'].get('spread_pct')}%"
            money = trade_plan.get("strike_moneyness", "ATM") if trade_plan else "ATM"
            atm_ref = trade_plan.get("atm_reference", chain.get("atm", "")) if trade_plan else ""
            delta_u = trade_plan.get("delta_used", "") if trade_plan else ""
            out = (
                f"{money} strike {trade_plan.get('strike')} {opt} "
                f"(live ATM ref {atm_ref}, δ {delta_u}){spread}"
                if trade_plan
                else "—"
            )
            tag = trade_plan.get("pattern_tag", "") if trade_plan else ""
            if tag:
                out += f" · pattern {tag}"
            statuses.append(_step(i, title, ok, "Strike (live moneyness)", out))
        elif i == 7:
            ok = bool(trade_plan)
            est = " (estimated)" if trade_plan and trade_plan.get("estimated_premium") else ""
            out = (
                f"Entry ₹{trade_plan.get('entry_premium')}{est} · Risk ₹{trade_plan.get('risk_inr')} · "
                f"Reward ₹{trade_plan.get('reward_inr')}"
                if trade_plan
                else "—"
            )
            statuses.append(_step(i, title, ok, "Premium & max loss (live)", out))
        elif i == 8:
            ok = bool(trade_plan)
            out = (
                f"GTT SL ₹{trade_plan.get('stop_loss_premium')} TP ₹{trade_plan.get('target_premium')} · "
                f"Nifty {trade_plan.get('spot_stop_loss')} → {trade_plan.get('spot_target')}"
                if trade_plan
                else "—"
            )
            statuses.append(_step(i, title, ok, "Exit levels (live)", out))
        elif i == 9:
            ok = bool(trade_plan)
            out = (
                f"{trade_plan.get('num_lots')} lots × {trade_plan.get('lot_size')} = {trade_plan.get('quantity')} qty"
                if trade_plan
                else "—"
            )
            statuses.append(_step(i, title, ok, "Size (live margin)", out))
        elif i == 10:
            ok = bool(trade_plan)
            spread_ok = True
            if chain.get("atm_ce") and chain["atm_ce"].get("spread_pct", 99) > 3:
                spread_ok = False
            out = f"MIS · MARKET entry · GTT OCO · spread OK={spread_ok}"
            statuses.append(_step(i, title, ok and spread_ok, "Product & order type", out))
        elif i == 11:
            ok = bool(trade_plan)
            ltp = trade_plan.get("entry_premium") if trade_plan else 0
            if trade_plan and trade_plan.get("tradingsymbol"):
                try:
                    sym = trade_plan["tradingsymbol"]
                    q = kite.quote(f"NFO:{sym}").get(f"NFO:{sym}", {}) or {}
                    ltp = float(q.get("last_price") or ltp)
                except Exception:
                    pass
            out = (
                f"BUY {trade_plan.get('quantity')} {trade_plan.get('tradingsymbol')} @ ₹{ltp:.2f}"
                if trade_plan
                else "—"
            )
            statuses.append(
                _step(
                    i,
                    title,
                    ok,
                    "Ticket (live LTP)" if market_open else "Ticket (live quote, market closed)",
                    out,
                )
            )

    if 0 not in missing and margin <= 0:
        missing.append(0)
    if not trade_plan:
        for idx in range(4, 12):
            if idx not in missing:
                missing.append(idx)

    checklist_ready = len(missing) == 0 and spot > 0 and bool(trade_plan)

    return {
        "step_statuses": [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in statuses],
        "missing_steps": sorted(set(missing)),
        "checklist_ready": checklist_ready,
        "strategy_analysis": strategy_analysis,
        "trade_plan": trade_plan,
        "validation": validation,
        "market_open": market_open,
        "data_source": nifty.get("source"),
        "nifty_spot": spot,
    }
