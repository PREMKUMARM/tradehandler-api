"""
V2 — live checklist: every step uses fresh Kite ticker (Nifty) + quotes + chain OI.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# 0-based; must match tradehandler commodity.component.ts preBuyChecklist[].dependsOn
COMMODITY_STEP_DEPENDS_ON: List[List[int]] = [
    [],  # 0 session
    [],  # 1 hypothesis
    [0, 1],  # 2 strategy
    [0, 1, 2],  # 3 chain
    [1, 3],  # 4 expiry
    [3, 4],  # 5 strike
    [4, 5],  # 6 premium
    [1, 2],  # 7 exit
    [6, 7],  # 8 size
    [5, 6, 7, 8],  # 9 product
    [9],  # 10 final
]


def step_indices_for_analysis(step_index: int) -> List[int]:
    """This step plus its declared prerequisite steps (live realtime)."""
    if step_index < 0 or step_index >= len(COMMODITY_STEP_DEPENDS_ON):
        raise ValueError(f"Invalid checklist step index: {step_index}")
    deps = COMMODITY_STEP_DEPENDS_ON[step_index]
    return sorted(set(deps + [step_index]))

from schemas.commodity_trading import ChecklistStepStatus
from services.commodity_instruments import build_crude_options_universe, future_token
from services.commodity_strategy_analysis import analyze_commodity_strategies
from utils.kite_utils import get_kite_instance
from utils.logger import log_warning

IST = ZoneInfo("Asia/Kolkata")


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


def _underlying_live(kite) -> Dict[str, Any]:
    """Ticker-first Nifty snapshot (Kite app–aligned day OHLC)."""
    from services.commodity_live_indicators import recalculate_from_ticker

    bundle = recalculate_from_ticker()
    spot = float(bundle.get("nifty_spot") or 0)
    return {
        "spot": spot,
        "ohlc": {
            "open": float(bundle.get("day_open") or spot),
            "high": float(bundle.get("day_high") or spot),
            "low": float(bundle.get("day_low") or spot),
            "close": float(bundle.get("prev_close") or spot),
        },
        "source": bundle.get("spot_source", "kite_quote"),
    }


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
        log_warning(f"[Commodity realtime] daily: {exc}")

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
        log_warning(f"[Commodity realtime] intraday: {exc}")
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
    keys = [f"MCX:{r['tradingsymbol']}" for r in rows if r.get("tradingsymbol")]
    quotes: Dict[str, Any] = {}
    if keys:
        try:
            quotes = kite.quote(keys) or {}
        except Exception as exc:
            log_warning(f"[Commodity realtime] chain quote: {exc}")

    ce_oi = pe_oi = 0.0
    max_oi_strike = atm
    max_oi_val = 0.0
    atm_ce = atm_pe = None

    for r in rows:
        key = f"MCX:{r['tradingsymbol']}"
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


@dataclass
class ChecklistContext:
    direction: str
    margin: float
    market_open: bool
    live: Dict[str, Any]
    spot: float
    prev_close: float
    ohlc: Dict[str, Any]
    vix: Dict[str, float]
    intra: Dict[str, Any]
    src_map: Dict[str, Any]
    kite: Any
    chain: Dict[str, Any]
    strategy_analysis: Dict[str, Any]
    trade_plan: Optional[Dict[str, Any]]
    validation: Optional[Dict[str, Any]]


def _build_checklist_context(
    direction: str,
    margin: float,
    market_open: bool,
    risk_pct: float,
    reward_pct: float,
    num_lots: int,
    capital: float,
) -> ChecklistContext:
    from services.commodity_trade_service import _validate_trade_plan, build_trade_plan
    from services.commodity_live_indicators import recalculate_from_ticker

    live = recalculate_from_ticker()
    spot = float(live.get("nifty_spot") or 0)
    prev_close = float(live.get("prev_close") or spot)
    ohlc = {
        "open": live.get("day_open"),
        "high": live.get("day_high"),
        "low": live.get("day_low"),
        "close": prev_close,
    }
    vix_ltp = float(live.get("vix") or 0)
    vix: Dict[str, float] = {"ltp": vix_ltp, "chg_pct": 0.0}
    if prev_close > 0 and vix_ltp:
        try:
            vq = (get_kite_instance().quote(["NSE:INDIA VIX"]) or {}).get("NSE:INDIA VIX", {})
            vpc = float((vq.get("ohlc") or {}).get("close") or vix_ltp)
            vix["chg_pct"] = ((vix_ltp - vpc) / vpc * 100) if vpc else 0
        except Exception:
            pass
    src_map = live.get("indicator_sources") or {}
    kite = get_kite_instance()
    universe = build_crude_options_universe(kite)
    chain = _chain_live(kite, spot, universe)
    strategy_analysis = analyze_commodity_strategies(direction_pref=direction, margin=margin)
    trade_plan = None
    validation = None
    if spot > 0:
        plan, _ = build_trade_plan(
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
    return ChecklistContext(
        direction=direction,
        margin=margin,
        market_open=market_open,
        live=live,
        spot=spot,
        prev_close=prev_close,
        ohlc=ohlc,
        vix=vix,
        intra=live,
        src_map=src_map,
        kite=kite,
        chain=chain,
        strategy_analysis=strategy_analysis,
        trade_plan=trade_plan,
        validation=validation,
    )


def _status_for_step(i: int, title: str, ctx: ChecklistContext) -> ChecklistStepStatus:
    direction = ctx.direction
    margin = ctx.margin
    market_open = ctx.market_open
    live = ctx.live
    spot = ctx.spot
    prev_close = ctx.prev_close
    ohlc = ctx.ohlc
    vix = ctx.vix
    intra = ctx.intra
    chain = ctx.chain
    strategy_analysis = ctx.strategy_analysis
    trade_plan = ctx.trade_plan
    kite = ctx.kite
    opt = (trade_plan or {}).get("option_type") or strategy_analysis.get("selected_option_kind") or "CE"

    src_map = ctx.src_map

    if i == 0:
        ok = margin > 0 and spot > 0
        src = live.get("spot_source", "quote")
        ticks = live.get("tick_count_today", 0)
        out = f"Margin ₹{margin:,.0f} · Crude {spot:.2f} via {src} ({ticks} ticks today)"
        return _step(i, title, ok, "Session live", out)
    if i == 1:
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
        ema_src = src_map.get("ema9", "?")
        parts.append(f"EMA9 {intra.get('ema9')} ({ema_src})")
        return _step(i, title, ok, "Hypothesis (ticker+hist)", " · ".join(parts))
    if i == 2:
        ok = bool(strategy_analysis.get("selected_id"))
        out = strategy_analysis.get("output_summary", "")
        return _step(
            i,
            title,
            ok,
            f"Best: {strategy_analysis.get('selected_name')} ({strategy_analysis.get('selected_score')}/100)",
            out,
        )
    if i == 3:
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
        return _step(i, title, ok, "Option chain (live OI)", out)
    if i == 4:
        ok = bool(trade_plan)
        out = f"Expiry {trade_plan.get('expiry')} (CRUDEOIL26JUN)" if trade_plan else "—"
        return _step(i, title, ok, "Expiry locked", out)
    if i == 5:
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
        return _step(i, title, ok, "Strike (live moneyness)", out)
    if i == 6:
        ok = bool(trade_plan)
        if not trade_plan:
            return _step(i, title, False, "No trade plan", "—")
        est = " (estimated)" if trade_plan.get("estimated_premium") else ""
        lim = trade_plan.get("entry_limit_price")
        ready = trade_plan.get("entry_ready")
        style = trade_plan.get("entry_style") or ""
        trig = trade_plan.get("entry_spot_trigger")
        ready_tag = "confirmed" if ready else "wait"
        trig_s = f" · trigger {trig}" if trig else ""
        ind = trade_plan.get("indicators") or {}
        bb_s = ""
        if ind.get("bb_middle"):
            bb_s = (
                f" · BB {ind.get('bb_zone', '?')} "
                f"(L {ind.get('bb_lower'):.0f} M {ind.get('bb_middle'):.0f} U {ind.get('bb_upper'):.0f})"
            )
        out = (
            f"{ready_tag} · {style} LIMIT ₹{lim} · LTP ₹{trade_plan.get('entry_premium')}{est}{trig_s}{bb_s} · "
            f"Risk ₹{trade_plan.get('risk_inr')} · Reward ₹{trade_plan.get('reward_inr')}"
            if trade_plan
            else "—"
        )
        msg = "Entry priced from indicators" if ready else (
            trade_plan.get("entry_block_reason") or "Entry not confirmed"
        )
        return _step(i, title, ok and ready, msg, out)
    if i == 7:
        ok = bool(trade_plan)
        out = (
            f"GTT SL ₹{trade_plan.get('stop_loss_premium')} TP ₹{trade_plan.get('target_premium')} · "
            f"Crude {trade_plan.get('spot_stop_loss')} → {trade_plan.get('spot_target')}"
            if trade_plan
            else "—"
        )
        return _step(i, title, ok, "Exit levels (live)", out)
    if i == 8:
        ok = bool(trade_plan)
        out = (
            f"{trade_plan.get('num_lots')} lots × {trade_plan.get('lot_size')} = {trade_plan.get('quantity')} qty"
            if trade_plan
            else "—"
        )
        return _step(i, title, ok, "Size (live margin)", out)
    if i == 9:
        ok = bool(trade_plan)
        spread_ok = True
        if chain.get("atm_ce") and chain["atm_ce"].get("spread_pct", 99) > 3:
            spread_ok = False
        out = (
            f"NRML · LIMIT entry ₹{trade_plan.get('entry_limit_price')} · GTT OCO exit only · "
            f"spread OK={spread_ok}"
            if trade_plan
            else "—"
        )
        return _step(i, title, ok and spread_ok, "LIMIT entry + GTT exit", out)
    if i == 10:
        ok = bool(trade_plan)
        ltp = trade_plan.get("entry_premium") if trade_plan else 0
        if trade_plan and trade_plan.get("tradingsymbol"):
            try:
                sym = trade_plan["tradingsymbol"]
                q = kite.quote(f"MCX:{sym}").get(f"MCX:{sym}", {}) or {}
                ltp = float(q.get("last_price") or ltp)
            except Exception:
                pass
        out = (
            f"BUY {trade_plan.get('quantity')} {trade_plan.get('tradingsymbol')} @ ₹{ltp:.2f}"
            if trade_plan
            else "—"
        )
        return _step(
            i,
            title,
            ok,
            "Ticket (live LTP)" if market_open else "Ticket (live quote, market closed)",
            out,
        )
    raise ValueError(f"Unknown checklist step index: {i}")


def run_realtime_checklist(
    direction: str,
    margin: float,
    market_open: bool,
    risk_pct: float,
    reward_pct: float,
    num_lots: int,
    capital: float,
    only_steps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Execute checklist steps with live Kite data.
    If only_steps is set, return statuses for those indices only (plus shared plan/analysis).
    """
    from services.commodity_trade_service import STEP_TITLES

    ctx = _build_checklist_context(
        direction, margin, market_open, risk_pct, reward_pct, num_lots, capital
    )
    emit = only_steps if only_steps is not None else list(range(len(STEP_TITLES)))
    emit = sorted(set(emit))

    statuses: List[ChecklistStepStatus] = []
    missing: List[int] = []
    for i in emit:
        if i < 0 or i >= len(STEP_TITLES):
            continue
        st = _status_for_step(i, STEP_TITLES[i], ctx)
        statuses.append(st)
        if not st.server_ok:
            missing.append(i)

    if 0 in emit and 0 not in missing and ctx.margin <= 0:
        missing.append(0)
    if not ctx.trade_plan:
        for idx in range(4, 12):
            if idx in emit and idx not in missing:
                missing.append(idx)

    needs_plan = any(s >= 4 for s in emit)
    checklist_ready = (
        len(missing) == 0
        and ctx.spot > 0
        and (bool(ctx.trade_plan) if needs_plan else True)
    )

    return {
        "step_statuses": [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in statuses],
        "missing_steps": sorted(set(missing)),
        "checklist_ready": checklist_ready,
        "strategy_analysis": ctx.strategy_analysis,
        "trade_plan": ctx.trade_plan,
        "validation": ctx.validation,
        "market_open": market_open,
        "data_source": ctx.live.get("spot_source"),
        "indicator_sources": ctx.src_map,
        "nifty_spot": ctx.spot,
    }
