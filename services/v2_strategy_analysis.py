"""
V2 — Nifty50 paper/live strategy: simple 5m Bollinger Bands (20, 2σ) mean reversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from utils.logger import log_warning

IST = ZoneInfo("Asia/Kolkata")

STRATEGY_ID = "bb_5m_mean_reversion"
STRATEGY_NAME = "5m Bollinger Bands"
STRATEGY_DESC = (
    "Buy ATM options on 5m BB pullbacks: CE at lower/middle band, PE at upper/middle band."
)

STRATEGY_IDS = (STRATEGY_ID,)


@dataclass
class MarketContext:
    nifty_ltp: float = 0.0
    prev_close: float = 0.0
    day_open: float = 0.0
    margin: float = 0.0
    direction_pref: str = "AUTO"
    vix_ltp: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_zone: str = ""
    minutes: int = 0
    is_weekday: bool = True


@dataclass
class StrategyCandidate:
    id: str
    name: str
    description: str
    score: int
    fit: str
    option_kind: str
    spot_entry: float
    spot_stop_loss: float
    spot_target: float
    rr_ratio: float
    reasons: List[str]
    warnings: List[str]
    strike_moneyness: str = "ATM"
    pattern_tag: str = "bb_5m"
    anchor_strike: Optional[int] = None
    oi_change: Optional[float] = None


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


def _fetch_market_context(direction_pref: str, margin: float) -> MarketContext:
    ctx = MarketContext(margin=margin, direction_pref=(direction_pref or "AUTO").upper())
    now = datetime.now(IST)
    ctx.minutes = now.hour * 60 + now.minute
    ctx.is_weekday = now.weekday() <= 4  # Mon–Fri

    try:
        from services.kite_live_indicators import recalculate_from_ticker

        live = recalculate_from_ticker()
        ctx.nifty_ltp = float(live.get("nifty_spot") or 0)
        ctx.prev_close = float(live.get("prev_close") or ctx.nifty_ltp)
        ctx.day_open = float(live.get("day_open") or ctx.nifty_ltp)
        if live.get("vix"):
            ctx.vix_ltp = float(live["vix"])
    except Exception as exc:
        log_warning(f"[V2 strategy] live indicators failed: {exc}")

    return ctx


def _score_bb_5m(ctx: MarketContext) -> StrategyCandidate:
    kind = _resolve_kind(ctx, "AUTO")
    nifty = ctx.nifty_ltp
    rr = 1.5
    score = 20
    reasons: List[str] = []
    warnings: List[str] = []

    if nifty <= 0:
        warnings.append("Nifty spot unavailable — connect Kite ticker")
        return StrategyCandidate(
            id=STRATEGY_ID,
            name=STRATEGY_NAME,
            description=STRATEGY_DESC,
            score=score,
            fit=_fit_label(score),
            option_kind=kind,
            spot_entry=nifty,
            spot_stop_loss=nifty - 30,
            spot_target=nifty + 45,
            rr_ratio=rr,
            reasons=reasons,
            warnings=warnings,
        )

    from services.kite_live_indicators import bollinger_zone, get_option_bollinger_snapshot
    from services.push.option_contract_resolver import resolve_nifty_contract

    contract = resolve_nifty_contract(spot=nifty, kind=kind, moneyness="ATM")
    if contract is None:
        warnings.append("Could not resolve ATM option for contract BB")
        return StrategyCandidate(
            id=STRATEGY_ID,
            name=STRATEGY_NAME,
            description=STRATEGY_DESC,
            score=score,
            fit=_fit_label(score),
            option_kind=kind,
            spot_entry=nifty,
            spot_stop_loss=nifty - 30,
            spot_target=nifty + 45,
            rr_ratio=rr,
            reasons=reasons,
            warnings=warnings,
        )

    opt_bb = get_option_bollinger_snapshot(contract.tradingsymbol, "NFO")
    entry = float(opt_bb.get("option_ltp") or 0)
    mid = opt_bb.get("bb_middle")
    upper = opt_bb.get("bb_upper")
    lower = opt_bb.get("bb_lower")

    if entry <= 0 or mid is None or upper is None or lower is None:
        warnings.append(
            f"5m BB not ready on {contract.tradingsymbol} — need 20×5m bars on contract chart"
        )
        risk = max(0.5, (entry or 50) * 0.12)
        if kind == "CE":
            sl, tgt = (entry or 50) - risk, (entry or 50) + risk * rr
        else:
            sl, tgt = (entry or 50) + risk, (entry or 50) - risk * rr
        return StrategyCandidate(
            id=STRATEGY_ID,
            name=STRATEGY_NAME,
            description=STRATEGY_DESC,
            score=score,
            fit=_fit_label(score),
            option_kind=kind,
            spot_entry=nifty,
            spot_stop_loss=sl,
            spot_target=tgt,
            rr_ratio=rr,
            reasons=reasons,
            warnings=warnings,
        )

    ctx.bb_middle = float(mid)
    ctx.bb_upper = float(upper)
    ctx.bb_lower = float(lower)

    bb = bollinger_zone(entry, ctx.bb_middle, ctx.bb_upper, ctx.bb_lower, kind)
    zone = bb["zone"]
    width = ctx.bb_upper - ctx.bb_lower
    buf = max(0.5, width * 0.04)

    if kind == "CE":
        sl = ctx.bb_lower - buf
        risk = max(0.05, entry - sl)
        tgt = ctx.bb_middle + rr * risk * 0.85
        if zone == "lower":
            score = 88
            reasons.append(f"CE: contract BB lower touch (₹{ctx.bb_lower:.2f})")
        elif zone == "middle":
            score = 78
            reasons.append(f"CE: contract BB middle (₹{ctx.bb_middle:.2f})")
        elif zone == "between":
            score = 62
            reasons.append("CE: LTP between contract bands — patient limit")
        elif bb["extended"]:
            score = 25
            warnings.append("CE blocked: LTP at upper band — wait for pullback")
        else:
            score = 45
            warnings.append(bb["wait_msg"])
    else:
        sl = ctx.bb_upper + buf
        risk = max(0.05, sl - entry)
        tgt = ctx.bb_middle - rr * risk * 0.85
        if zone == "upper":
            score = 88
            reasons.append(f"PE: contract BB upper touch (₹{ctx.bb_upper:.2f})")
        elif zone == "middle":
            score = 78
            reasons.append(f"PE: contract BB middle (₹{ctx.bb_middle:.2f})")
        elif zone == "between":
            score = 62
            reasons.append("PE: LTP between contract bands — patient limit")
        elif bb["extended"]:
            score = 25
            warnings.append("PE blocked: LTP at lower band — wait for rally")
        else:
            score = 45
            warnings.append(bb["wait_msg"])

    if ctx.margin > 5000:
        score += 8
        reasons.append(f"Margin ₹{ctx.margin:,.0f} OK for ATM buy")
    if ctx.vix_ltp and 12 <= ctx.vix_ltp <= 24:
        score += 5
        reasons.append(f"VIX {ctx.vix_ltp:.1f} in tradeable range")

    reasons.append(
        f"{contract.tradingsymbol} BB L ₹{ctx.bb_lower:.2f} M ₹{ctx.bb_middle:.2f} "
        f"U ₹{ctx.bb_upper:.2f} · LTP ₹{entry:.2f} · zone {zone}"
    )

    return StrategyCandidate(
        id=STRATEGY_ID,
        name=STRATEGY_NAME,
        description=STRATEGY_DESC,
        score=max(0, min(100, score)),
        fit=_fit_label(score),
        option_kind=kind,
        spot_entry=nifty,
        spot_stop_loss=sl,
        spot_target=tgt,
        rr_ratio=rr,
        reasons=reasons,
        warnings=warnings,
        strike_moneyness="ATM",
        pattern_tag=f"bb_{zone}",
    )


def analyze_fno_strategies(
    direction_pref: str = "AUTO",
    margin: float = 0.0,
    hypothesis_note: Optional[str] = None,
    chain_oi: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Single 5m Bollinger Bands strategy for Nifty50 (paper debug + live)."""
    del chain_oi  # OI ranking not used for BB-only strategy
    ctx = _fetch_market_context(direction_pref, margin)
    selected = _score_bb_5m(ctx)

    try:
        from services.kite_live_indicators import recalculate_from_ticker

        intra = recalculate_from_ticker()
    except Exception as exc:
        log_warning(f"[V2 strategy] intraday for strike pick: {exc}")
        intra = {}

    from services.v2_strike_pricing import _pick_moneyness

    m, pt, reason = _pick_moneyness(
        selected.id,
        ctx.nifty_ltp,
        selected.option_kind,
        selected.spot_stop_loss,
        selected.spot_target,
        intra,
    )
    selected.strike_moneyness = m
    selected.pattern_tag = pt
    if reason:
        selected.reasons = list(selected.reasons) + [f"Strike {m}: {reason}"]

    ranked = [selected]

    context_summary = {
        "nifty_spot": round(ctx.nifty_ltp, 2),
        "prev_close": round(ctx.prev_close, 2),
        "margin": round(ctx.margin, 2),
        "vix": round(ctx.vix_ltp, 2) if ctx.vix_ltp else None,
        "bb_middle": round(ctx.bb_middle, 2) if ctx.bb_middle else None,
        "bb_upper": round(ctx.bb_upper, 2) if ctx.bb_upper else None,
        "bb_lower": round(ctx.bb_lower, 2) if ctx.bb_lower else None,
        "bb_zone": ctx.bb_zone or None,
        "direction_pref": ctx.direction_pref,
        "hypothesis_note": hypothesis_note,
        "strategy_mode": "bb_5m_only",
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
            "anchor_strike": c.anchor_strike,
            "oi_change": c.oi_change,
        }

    output_lines = [
        f"Selected: {selected.name} (score {selected.score}/100, {selected.fit})",
        f"Leg: BUY Nifty {selected.option_kind} · BB zone {ctx.bb_zone or '—'}",
        f"SL {selected.spot_stop_loss:.0f} · Tgt {selected.spot_target:.0f}",
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
