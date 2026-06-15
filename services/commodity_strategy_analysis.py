"""
V2 — Score top Indian Nifty F&O intraday strategies using prior checklist context.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from utils.kite_utils import get_kite_instance
from utils.logger import log_warning

IST = ZoneInfo("Asia/Kolkata")

# Widely used profitable Nifty F&O intraday frameworks (single-leg long option execution).
STRATEGY_IDS = (
    "long_atm_directional",
    "orb_15m_breakout",
    "pdh_pdl_breakout",
    "ema_pullback_continuation",
)


@dataclass
class MarketContext:
    nifty_ltp: float = 0.0
    prev_close: float = 0.0
    day_open: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    margin: float = 0.0
    direction_pref: str = "AUTO"
    vix_ltp: Optional[float] = None
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    ema9: Optional[float] = None
    minutes: int = 0
    is_weekday: bool = True


@dataclass
class StrategyCandidate:
    id: str
    name: str
    description: str
    score: int
    fit: str  # excellent | good | fair | poor
    option_kind: str  # CE | PE
    spot_entry: float
    spot_stop_loss: float
    spot_target: float
    rr_ratio: float
    reasons: List[str]
    warnings: List[str]
    strike_moneyness: str = "ATM"
    pattern_tag: str = ""


def _fetch_market_context(direction_pref: str, margin: float) -> MarketContext:
    ctx = MarketContext(margin=margin, direction_pref=(direction_pref or "AUTO").upper())
    now = datetime.now(IST)
    ctx.minutes = now.hour * 60 + now.minute
    ctx.is_weekday = now.weekday() <= 4  # Mon–Fri

    try:
        from services.commodity_live_indicators import recalculate_from_ticker

        live = recalculate_from_ticker()
        ctx.nifty_ltp = float(live.get("nifty_spot") or 0)
        ctx.prev_close = float(live.get("prev_close") or ctx.nifty_ltp)
        ctx.day_open = float(live.get("day_open") or ctx.nifty_ltp)
        ctx.day_high = float(live.get("day_high") or ctx.nifty_ltp)
        ctx.day_low = float(live.get("day_low") or ctx.nifty_ltp)
        if live.get("vix"):
            ctx.vix_ltp = float(live["vix"])
        if live.get("pdh"):
            ctx.pdh = float(live["pdh"])
        if live.get("pdl"):
            ctx.pdl = float(live["pdl"])
        if live.get("or_high"):
            ctx.or_high = float(live["or_high"])
        if live.get("or_low"):
            ctx.or_low = float(live["or_low"])
        if live.get("ema9"):
            ctx.ema9 = float(live["ema9"])
    except Exception as exc:
        log_warning(f"[Commodity strategy] live indicators failed: {exc}")
        return ctx

    return ctx


def _fit_label(score: int) -> str:
    if score >= 75:
        return "excellent"
    if score >= 55:
        return "good"
    if score >= 35:
        return "fair"
    return "poor"


def _resolve_kind(ctx: MarketContext, bias: str) -> str:
    if bias in ("CE", "PE"):
        return bias
    if ctx.direction_pref in ("CE", "PE"):
        return ctx.direction_pref
    if ctx.prev_close > 0:
        return "CE" if ctx.nifty_ltp >= ctx.prev_close else "PE"
    return "CE"


def _score_long_atm(ctx: MarketContext) -> StrategyCandidate:
    kind = _resolve_kind(ctx, "AUTO")
    risk_pts = max(15.0, ctx.nifty_ltp * 0.0035)
    rr = 2.0
    if kind == "CE":
        sl, tgt = ctx.nifty_ltp - risk_pts, ctx.nifty_ltp + risk_pts * rr
    else:
        sl, tgt = ctx.nifty_ltp + risk_pts, ctx.nifty_ltp - risk_pts * rr

    score = 50
    reasons: List[str] = []
    warnings: List[str] = []

    if ctx.nifty_ltp > 0 and ctx.margin > 5000:
        score += 15
        reasons.append(f"Margin ₹{ctx.margin:,.0f} supports ATM buy")
    gap_pct = abs(ctx.day_open - ctx.prev_close) / ctx.prev_close * 100 if ctx.prev_close else 0
    if gap_pct < 0.8:
        score += 10
        reasons.append("No extreme gap — trend read reliable")
    else:
        warnings.append(f"Gap {gap_pct:.1f}% — wait for ORB/PDH clarity")

    if ctx.vix_ltp:
        if 12 <= ctx.vix_ltp <= 22:
            score += 15
            reasons.append(f"VIX {ctx.vix_ltp:.1f} in buy-zone (not too cheap/expensive)")
        elif ctx.vix_ltp > 24:
            score -= 10
            warnings.append(f"VIX {ctx.vix_ltp:.1f} elevated — premium expensive")

    if (kind == "CE" and ctx.nifty_ltp >= ctx.prev_close) or (
        kind == "PE" and ctx.nifty_ltp < ctx.prev_close
    ):
        score += 10
        reasons.append(f"Spot vs prior close favours {kind}")

    return StrategyCandidate(
        id="long_atm_directional",
        name="Long ATM directional",
        description="Buy ATM CE/PE in direction of session bias; GTT on premium.",
        score=max(0, min(100, score)),
        fit=_fit_label(score),
        option_kind=kind,
        spot_entry=ctx.nifty_ltp,
        spot_stop_loss=sl,
        spot_target=tgt,
        rr_ratio=rr,
        reasons=reasons,
        warnings=warnings,
    )


def _score_orb(ctx: MarketContext) -> StrategyCandidate:
    kind = "CE"
    sl, tgt, entry = ctx.nifty_ltp, ctx.nifty_ltp - 30, ctx.nifty_ltp
    rr = 1.5
    score = 25
    reasons: List[str] = []
    warnings: List[str] = []

    if ctx.or_high is None or ctx.or_low is None:
        warnings.append("Opening range not built yet (need 9:15–9:30 15m candles)")
        return StrategyCandidate(
            id="orb_15m_breakout",
            name="15m Opening Range Breakout",
            description="Break of 9:15–9:30 IST range with OR as stop reference.",
            score=score,
            fit=_fit_label(score),
            option_kind=kind,
            spot_entry=entry,
            spot_stop_loss=sl,
            spot_target=tgt,
            rr_ratio=rr,
            reasons=reasons,
            warnings=warnings,
        )

    or_range = ctx.or_high - ctx.or_low
    entry = ctx.nifty_ltp
    if entry > ctx.or_high:
        kind = "CE"
        sl = ctx.or_low - 2
        risk = max(1.0, entry - sl)
        tgt = entry + rr * risk
        score += 35
        reasons.append(f"Price above OR high {ctx.or_high:.0f} — long breakout")
    elif entry < ctx.or_low:
        kind = "PE"
        sl = ctx.or_high + 2
        risk = max(1.0, sl - entry)
        tgt = entry - rr * risk
        score += 35
        reasons.append(f"Price below OR low {ctx.or_low:.0f} — short breakout")
    else:
        warnings.append(f"Inside OR ({ctx.or_low:.0f}–{ctx.or_high:.0f}) — no breakout yet")
        kind = _resolve_kind(ctx, "AUTO")
        if kind == "CE":
            sl, tgt = ctx.or_low - 2, ctx.or_high + or_range * 0.5
        else:
            sl, tgt = ctx.or_high + 2, ctx.or_low - or_range * 0.5

    if 25 <= or_range <= 180:
        score += 25
        reasons.append(f"OR range {or_range:.0f} pts in tradeable band")
    else:
        warnings.append(f"OR range {or_range:.0f} pts outside 25–180 sweet spot")

    if ctx.minutes > 9 * 60 + 30:
        score += 10
    if ctx.minutes < 13 * 60 + 30:
        score += 10
        reasons.append("Within ORB window (before 13:30 cutoff)")
    else:
        warnings.append("Past ORB cutoff — lower edge")

    return StrategyCandidate(
        id="orb_15m_breakout",
        name="15m Opening Range Breakout",
        description="Nifty ORB: break 9:15–9:30 range, SL opposite side of range.",
        score=max(0, min(100, score)),
        fit=_fit_label(score),
        option_kind=kind,
        spot_entry=entry,
        spot_stop_loss=sl,
        spot_target=tgt,
        rr_ratio=rr,
        reasons=reasons,
        warnings=warnings,
    )


def _score_pdh_pdl(ctx: MarketContext) -> StrategyCandidate:
    kind = _resolve_kind(ctx, "AUTO")
    entry = ctx.nifty_ltp
    sl, tgt = entry - 20, entry + 30
    rr = 1.5
    score = 30
    reasons: List[str] = []
    warnings: List[str] = []

    if not ctx.pdh or not ctx.pdl:
        warnings.append("PDH/PDL unavailable — need prior day daily candle")
        return StrategyCandidate(
            id="pdh_pdl_breakout",
            name="PDH / PDL breakout",
            description="Break prior day high/low with structure stop.",
            score=score,
            fit=_fit_label(score),
            option_kind=kind,
            spot_entry=entry,
            spot_stop_loss=sl,
            spot_target=tgt,
            rr_ratio=rr,
            reasons=reasons,
            warnings=warnings,
        )

    buf = 3.0
    if entry > ctx.pdh:
        kind = "CE"
        sl = ctx.pdh - buf
        risk = max(1.0, entry - sl)
        tgt = entry + rr * risk
        score += 40
        reasons.append(f"Above PDH {ctx.pdh:.0f} — bullish day structure")
    elif entry < ctx.pdl:
        kind = "PE"
        sl = ctx.pdl + buf
        risk = max(1.0, sl - entry)
        tgt = entry - rr * risk
        score += 40
        reasons.append(f"Below PDL {ctx.pdl:.0f} — bearish day structure")
    else:
        warnings.append(f"Between PDL {ctx.pdl:.0f} and PDH {ctx.pdh:.0f} — wait for break")
        mid = (ctx.pdh + ctx.pdl) / 2
        if entry >= mid:
            kind, sl, tgt = "CE", ctx.pdl, ctx.pdh + (ctx.pdh - ctx.pdl) * 0.5
        else:
            kind, sl, tgt = "PE", ctx.pdh, ctx.pdl - (ctx.pdh - ctx.pdl) * 0.5

    if ctx.minutes >= 9 * 60 + 30:
        score += 15
        reasons.append("After 9:30 warm-up — PDH/PDL valid")
    if ctx.pdh - ctx.pdl > 50:
        score += 10
        reasons.append("Prior day range wide enough for breakout follow-through")

    return StrategyCandidate(
        id="pdh_pdl_breakout",
        name="PDH / PDL breakout",
        description="Trade break of previous day high/low; SL at broken level.",
        score=max(0, min(100, score)),
        fit=_fit_label(score),
        option_kind=kind,
        spot_entry=entry,
        spot_stop_loss=sl,
        spot_target=tgt,
        rr_ratio=rr,
        reasons=reasons,
        warnings=warnings,
    )


def _score_ema_pullback(ctx: MarketContext) -> StrategyCandidate:
    kind = _resolve_kind(ctx, "AUTO")
    entry = ctx.nifty_ltp
    risk_pts = max(12.0, entry * 0.0025)
    rr = 1.8
    if kind == "CE":
        sl, tgt = entry - risk_pts, entry + risk_pts * rr
    else:
        sl, tgt = entry + risk_pts, entry - risk_pts * rr

    score = 35
    reasons: List[str] = []
    warnings: List[str] = []

    if ctx.ema9 is None:
        warnings.append("9 EMA not computed — need 5m history")
        return StrategyCandidate(
            id="ema_pullback_continuation",
            name="9 EMA pullback continuation",
            description="Trend day: buy CE/PE after pullback to 9 EMA on 5m.",
            score=score,
            fit=_fit_label(score),
            option_kind=kind,
            spot_entry=entry,
            spot_stop_loss=sl,
            spot_target=tgt,
            rr_ratio=rr,
            reasons=reasons,
            warnings=warnings,
        )

    dist = entry - ctx.ema9
    if entry > ctx.ema9 and entry > ctx.prev_close:
        kind = "CE"
        sl = min(ctx.ema9 - 5, entry - risk_pts)
        risk = max(1.0, entry - sl)
        tgt = entry + rr * risk
        if 0 <= dist <= 25:
            score += 35
            reasons.append(f"Uptrend — pullback near 9 EMA ({ctx.ema9:.0f})")
        elif dist > 25:
            score += 15
            reasons.append("Uptrend but extended above EMA — chase risk")
            warnings.append("Far above 9 EMA — wait for pullback")
    elif entry < ctx.ema9 and entry < ctx.prev_close:
        kind = "PE"
        sl = max(ctx.ema9 + 5, entry + risk_pts)
        risk = max(1.0, sl - entry)
        tgt = entry - rr * risk
        if 0 >= dist >= -25:
            score += 35
            reasons.append(f"Downtrend — pullback near 9 EMA ({ctx.ema9:.0f})")
        elif dist < -25:
            score += 15
            warnings.append("Far below 9 EMA — wait for pullback")
    else:
        warnings.append("No clear trend vs EMA — mixed session")

    if 10 * 60 + 15 <= ctx.minutes <= 14 * 60 + 30:
        score += 15
        reasons.append("Prime session window for pullback entries")

    return StrategyCandidate(
        id="ema_pullback_continuation",
        name="9 EMA pullback continuation",
        description="Intraday trend: enter after hold/pullback to 9 EMA (5m).",
        score=max(0, min(100, score)),
        fit=_fit_label(score),
        option_kind=kind,
        spot_entry=entry,
        spot_stop_loss=sl,
        spot_target=tgt,
        rr_ratio=rr,
        reasons=reasons,
        warnings=warnings,
    )


def analyze_commodity_strategies(
    direction_pref: str = "AUTO",
    margin: float = 0.0,
    hypothesis_note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rank top 4 Nifty F&O intraday strategies using session + prior-step context.
    """
    ctx = _fetch_market_context(direction_pref, margin)
    candidates = [
        _score_long_atm(ctx),
        _score_orb(ctx),
        _score_pdh_pdl(ctx),
        _score_ema_pullback(ctx),
    ]
    try:
        from services.commodity_live_indicators import recalculate_from_ticker

        intra = recalculate_from_ticker()
    except Exception as exc:
        log_warning(f"[Commodity strategy] intraday for strike pick: {exc}")
        intra = {}

    from services.commodity_strike_pricing import _pick_moneyness

    for c in candidates:
        m, pt, reason = _pick_moneyness(
            c.id, ctx.nifty_ltp, c.option_kind, c.spot_stop_loss, c.spot_target, intra
        )
        c.strike_moneyness = m
        c.pattern_tag = pt
        if reason:
            c.reasons = list(c.reasons) + [f"Strike {m}: {reason}"]

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected = ranked[0]

    context_summary = {
        "nifty_spot": round(ctx.nifty_ltp, 2),
        "prev_close": round(ctx.prev_close, 2),
        "margin": round(ctx.margin, 2),
        "vix": round(ctx.vix_ltp, 2) if ctx.vix_ltp else None,
        "pdh": round(ctx.pdh, 2) if ctx.pdh else None,
        "pdl": round(ctx.pdl, 2) if ctx.pdl else None,
        "or_high": round(ctx.or_high, 2) if ctx.or_high else None,
        "or_low": round(ctx.or_low, 2) if ctx.or_low else None,
        "ema9": round(ctx.ema9, 2) if ctx.ema9 else None,
        "direction_pref": ctx.direction_pref,
        "hypothesis_note": hypothesis_note,
    }

    def _to_dict(c: StrategyCandidate) -> Dict[str, Any]:
        return {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "score": c.score,
            "fit": c.fit,
            "option_kind": c.option_kind,
            "spot_entry": round(c.spot_entry, 2),
            "spot_stop_loss": round(c.spot_stop_loss, 2),
            "spot_target": round(c.spot_target, 2),
            "rr_ratio": c.rr_ratio,
            "reasons": c.reasons,
            "warnings": c.warnings,
            "strike_moneyness": c.strike_moneyness,
            "pattern_tag": c.pattern_tag,
        }

    output_lines = [
        f"Selected: {selected.name} (score {selected.score}/100, {selected.fit})",
        f"Leg: BUY Crude {selected.option_kind} · SL {selected.spot_stop_loss:.0f} · Tgt {selected.spot_target:.0f}",
    ]
    if selected.reasons:
        output_lines.append("Why: " + "; ".join(selected.reasons[:3]))

    return {
        "selected_id": selected.id,
        "selected_name": selected.name,
        "selected_score": selected.score,
        "selected_fit": selected.fit,
        "selected_option_kind": selected.option_kind,
        "context": context_summary,
        "strategies": [_to_dict(c) for c in ranked],
        "output_summary": " | ".join(output_lines),
    }
