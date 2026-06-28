"""
Sensex — single custom strategy: 20rupees-strategy.

Enter when ATM or highest-OI strike premium is in ₹17–₹23.
Fixed 1 lot, ₹10 stop-loss, 1:1 initial target; trailing handled by momentum trail.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from services.sensex_constants import (
    is_past_sensex_entry_cutoff,
    sensex_entry_cutoff_label,
    sensex_entry_cutoff_message,
    sensex_entry_scan_start_minutes,
    sensex_gap_direction_kind,
    sensex_gap_pct,
)
from services.sensex_run_params import SensexRunParams
from services.sensex_strike_selection import (
    moneyness_label,
    pick_smart_from_chain,
    PREMIUM_BAND_HIGH,
    PREMIUM_BAND_LOW,
)
from utils.logger import log_warning

IST = ZoneInfo("Asia/Kolkata")

STRATEGY_ID = "20rupees_strategy"
STRATEGY_NAME = "20rupees-strategy"


def _strategy_exit_blurb() -> str:
    from services.entry_quality import exit_model

    if exit_model() == "t1_scalp":
        return "full exit at 1R target (T1 scalp — locks profit, no breakeven trail)"
    return "1:1 target then stepped trail (T1→entry, T2→T1, …)"


STRATEGY_DESC = (
    "Buy Sensex option when premium closes ₹17–₹23 on 5m (from 14:00 IST). "
    "AUTO: leg from index day direction (CE up / PE down vs open); monitors ATM±2. "
    f"Fixed initial SL at ₹9 premium; 1R = entry − SL; {_strategy_exit_blurb()}. "
    "Size lots from risk % at SL. "
    f"No new entries after {sensex_entry_cutoff_label()} IST."
)

STRATEGY_IDS = (STRATEGY_ID,)

FIXED_SL_PREMIUM = 9.0  # absolute initial stop-loss option premium (trail adjusts after fill)
FIXED_SL_INR = FIXED_SL_PREMIUM  # legacy export name
FIXED_LOTS = 1


@dataclass
class MarketContext:
    nifty_ltp: float = 0.0
    prev_close: float = 0.0
    day_open: float = 0.0
    margin: float = 0.0
    direction_pref: str = "AUTO"
    vix_ltp: Optional[float] = None
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
    pattern_tag: str = "20rupees"
    anchor_strike: Optional[int] = None
    oi_change: Optional[float] = None
    entry_premium: Optional[float] = None
    stop_loss_premium: Optional[float] = None
    target_premium: Optional[float] = None
    strike_source: str = ""
    tradingsymbol: Optional[str] = None


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
    from services.entry_quality import auto_entry_kind

    kind = auto_entry_kind(ctx.day_open or ctx.nifty_ltp, ctx.nifty_ltp, ctx.prev_close)
    return kind or "CE"


def _in_premium_band(
    ltp: float,
    band_low: float = PREMIUM_BAND_LOW,
    band_high: float = PREMIUM_BAND_HIGH,
) -> bool:
    return band_low <= ltp <= band_high


def _band_score(
    ltp: float,
    band_low: float = PREMIUM_BAND_LOW,
    band_high: float = PREMIUM_BAND_HIGH,
) -> int:
    if not _in_premium_band(ltp, band_low, band_high):
        return 25
    center = (band_low + band_high) / 2.0
    return max(72, min(95, int(92 - abs(ltp - center) * 4)))


def _highest_oi_row(ranked: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not ranked:
        return None
    return max(ranked, key=lambda row: float(row.get("oi") or 0))


def _pick_auto_strike_from_chain(
    chain_oi: Dict[str, Any],
    spot: float,
    *,
    prev_close: float = 0.0,
    day_open: float = 0.0,
    band_low: float = PREMIUM_BAND_LOW,
    band_high: float = PREMIUM_BAND_HIGH,
    segment: str = "sensex",
) -> Tuple[Optional[int], float, str, Optional[str], Optional[float], str]:
    """AUTO: day-direction leg + smart OI strike in premium band."""
    from services.entry_quality import auto_entry_kind, entry_day_aligned_ok
    from services.index_atm import true_atm_from_spot

    day_open = day_open or spot
    kind = auto_entry_kind(day_open, spot, prev_close)
    atm = true_atm_from_spot(spot, segment=segment)
    if not kind or not entry_day_aligned_ok(kind=kind, index_open=day_open, spot=spot):
        return atm, 0.0, "ATM", None, None, kind or "CE"
    picked = pick_smart_from_chain(
        chain_oi,
        spot,
        kinds=(kind,),
        band_low=band_low,
        band_high=band_high,
        prev_close=prev_close,
        segment=segment,
    )
    if not picked:
        return atm, 0.0, "ATM", None, None, kind
    money = moneyness_label(picked.strike, atm, picked.offset)
    return picked.strike, picked.ltp, money, picked.symbol, picked.oi, picked.kind


def _pick_strike_from_chain(
    chain_oi: Dict[str, Any],
    kind: str,
    spot: float,
    prev_close: float = 0.0,
    band_low: float = PREMIUM_BAND_LOW,
    band_high: float = PREMIUM_BAND_HIGH,
    segment: str = "sensex",
) -> Tuple[Optional[int], float, str, Optional[str], Optional[float]]:
    """CE/PE: smart strike in premium band on the chosen leg."""
    from services.index_atm import true_atm_from_spot

    picked = pick_smart_from_chain(
        chain_oi,
        spot,
        kinds=(kind.upper(),),
        band_low=band_low,
        band_high=band_high,
        prev_close=prev_close,
        segment=segment,
    )
    atm = int(chain_oi.get("atm") or true_atm_from_spot(spot, segment=segment))
    if picked:
        money = moneyness_label(picked.strike, atm, picked.offset)
        return picked.strike, picked.ltp, money, picked.symbol, picked.oi

    # Fallback: ATM quote on leg
    atm_row = (chain_oi.get("atm_ce") or {}) if kind == "CE" else (chain_oi.get("atm_pe") or {})
    atm_ltp = float(atm_row.get("ltp") or 0)
    return atm if atm > 0 else None, atm_ltp, "ATM", None, None


def _fetch_market_context(
    direction_pref: str, margin: float, segment: str = "sensex"
) -> MarketContext:
    ctx = MarketContext(margin=margin, direction_pref=(direction_pref or "AUTO").upper())
    now = datetime.now(IST)
    ctx.minutes = now.hour * 60 + now.minute
    ctx.is_weekday = now.weekday() <= 4  # Mon–Fri

    seg = (segment or "sensex").strip().lower()
    try:
        if seg in ("nifty50", "nifty", "nfo"):
            from services.kite_live_indicators import recalculate_from_ticker

            live = recalculate_from_ticker()
        else:
            from services.sensex_live_indicators import recalculate_from_ticker

            live = recalculate_from_ticker()
        ctx.nifty_ltp = float(live.get("nifty_spot") or 0)
        ctx.prev_close = float(live.get("prev_close") or ctx.nifty_ltp)
        ctx.day_open = float(live.get("day_open") or ctx.nifty_ltp)
        if live.get("vix"):
            ctx.vix_ltp = float(live["vix"])
    except Exception as exc:
        log_warning(f"[{seg} strategy] live indicators failed: {exc}")

    return ctx


def _score_20rupees(
    ctx: MarketContext,
    chain_oi: Optional[Dict[str, Any]],
    run_params: Optional[SensexRunParams] = None,
    *,
    segment: str = "sensex",
) -> StrategyCandidate:
    rp = run_params or SensexRunParams.defaults()
    band_low = rp.entry_band_low
    band_high = rp.entry_band_high
    sl_prem_fixed = rp.sl_inr
    chain = chain_oi or {}
    spot = ctx.nifty_ltp
    index_label = "Nifty" if (segment or "").lower() in ("nifty50", "nifty", "nfo") else "Sensex"
    if spot <= 0:
        spot = float(chain.get("atm") or 0)
    rr = 1.0
    reasons: List[str] = []
    warnings: List[str] = []

    if spot <= 0 and not chain:
        return StrategyCandidate(
            id=STRATEGY_ID,
            name=STRATEGY_NAME,
            description=STRATEGY_DESC,
            score=15,
            fit=_fit_label(15),
            option_kind="CE",
            spot_entry=0,
            spot_stop_loss=0,
            spot_target=0,
            rr_ratio=rr,
            reasons=reasons,
            warnings=[f"{index_label} spot unavailable — connect Kite ticker"],
        )

    if ctx.direction_pref == "AUTO":
        strike, ltp, moneyness, sym, oi, kind = _pick_auto_strike_from_chain(
            chain,
            spot,
            prev_close=ctx.prev_close,
            day_open=ctx.day_open,
            band_low=band_low,
            band_high=band_high,
            segment=segment,
        )
        gap_pct = sensex_gap_pct(ctx.day_open, ctx.prev_close)
        if ctx.prev_close > 0:
            reasons.append(
                f"AUTO gap → {kind} (open vs prev close {gap_pct:+.2f}%)"
            )
    else:
        kind = ctx.direction_pref
        strike, ltp, moneyness, sym, oi = _pick_strike_from_chain(
            chain,
            kind,
            spot,
            prev_close=ctx.prev_close,
            band_low=band_low,
            band_high=band_high,
            segment=segment,
        )

    if strike is None or ltp <= 0:
        warnings.append("Option chain LTP unavailable — refresh after 9:20 AM")
        return StrategyCandidate(
            id=STRATEGY_ID,
            name=STRATEGY_NAME,
            description=STRATEGY_DESC,
            score=20,
            fit=_fit_label(20),
            option_kind=kind,
            spot_entry=spot,
            spot_stop_loss=0,
            spot_target=0,
            rr_ratio=rr,
            reasons=reasons,
            warnings=warnings,
            anchor_strike=strike,
        )

    entry_prem = round(ltp, 2)
    sl_prem = round(sl_prem_fixed, 2)
    r_dist = max(0.05, entry_prem - sl_prem)
    tgt_prem = round(entry_prem + r_dist, 2)
    score = _band_score(entry_prem, band_low, band_high)

    if _in_premium_band(entry_prem, band_low, band_high):
        reasons.append(
            f"{kind} {moneyness} strike {strike}: premium ₹{entry_prem:.2f} in "
            f"₹{band_low:.0f}–{band_high:.0f} band"
        )
        reasons.append(
            f"Plan: size to risk % of capital · SL ₹{sl_prem:.2f} · target ₹{tgt_prem:.2f} (1:1)"
        )
    elif entry_prem > band_high:
        warnings.append(
            f"Premium ₹{entry_prem:.2f} above band — wait for ₹{band_high:.0f}–{band_low:.0f}"
        )
        score = min(score, 38)
    else:
        warnings.append(
            f"Premium ₹{entry_prem:.2f} below band — wait for ₹{band_high:.0f}–{band_low:.0f}"
        )
        score = min(score, 32)

    if moneyness in ("MAX_OI", "SMART_OI") and oi:
        reasons.append(f"Smart OI {kind} strike {strike} (OI {oi:,.0f}, near ATM)")
    elif moneyness == "ATM":
        reasons.append(f"ATM {kind} strike {strike}")

    if ctx.margin > 5000:
        score += 5
        reasons.append(f"Margin ₹{ctx.margin:,.0f} OK for 1-lot entry")
    if ctx.vix_ltp and 12 <= ctx.vix_ltp <= 28:
        score += 3

    scan_start = rp.scan_start_minutes()
    scan_end = rp.scan_end_minutes()
    if ctx.minutes > 0 and ctx.minutes < scan_start:
        warnings.append(
            f"Before {scan_start // 60:02d}:{scan_start % 60:02d} IST — wait for afternoon 5m close in band"
        )
        score = min(score, 28)
        pattern = "20rupees_open_bar_skip"
    elif ctx.minutes > 0 and ctx.minutes >= scan_end:
        warnings.append(
            f"No new {index_label} entries after {rp.entry_cutoff_label()} IST "
            f"(avoid last-minute gamma and settlement decay)"
        )
        score = min(score, 25)
        pattern = "20rupees_cutoff"
    elif is_past_sensex_entry_cutoff() and ctx.minutes <= 0:
        warnings.append(sensex_entry_cutoff_message())
        score = min(score, 25)
        pattern = "20rupees_cutoff"
    elif _in_premium_band(entry_prem, band_low, band_high):
        pattern = "20rupees_ready"
    else:
        pattern = "20rupees_wait"

    return StrategyCandidate(
        id=STRATEGY_ID,
        name=STRATEGY_NAME,
        description=STRATEGY_DESC,
        score=max(0, min(100, score)),
        fit=_fit_label(score),
        option_kind=kind,
        spot_entry=spot,
        spot_stop_loss=sl_prem,
        spot_target=tgt_prem,
        rr_ratio=rr,
        reasons=reasons,
        warnings=warnings,
        strike_moneyness=moneyness,
        pattern_tag=pattern,
        anchor_strike=int(strike),
        oi_change=oi,
        entry_premium=entry_prem,
        stop_loss_premium=sl_prem,
        target_premium=tgt_prem,
        strike_source=moneyness,
        tradingsymbol=sym,
    )


def analyze_fno_strategies(
    direction_pref: str = "AUTO",
    margin: float = 0.0,
    hypothesis_note: Optional[str] = None,
    chain_oi: Optional[Dict[str, Any]] = None,
    run_params: Optional[SensexRunParams] = None,
    *,
    segment: str = "sensex",
) -> Dict[str, Any]:
    """Single 20rupees-strategy (paper + live)."""
    rp = run_params or SensexRunParams.from_mapping({"direction": direction_pref})
    ctx = _fetch_market_context(rp.direction, margin, segment=segment)
    selected = _score_20rupees(ctx, chain_oi, rp, segment=segment)
    index_label = "Nifty" if (segment or "").lower() in ("nifty50", "nifty", "nfo") else "Sensex"

    ranked = [selected]

    context_summary = {
        "nifty_spot": round(ctx.nifty_ltp, 2),
        "prev_close": round(ctx.prev_close, 2),
        "margin": round(ctx.margin, 2),
        "vix": round(ctx.vix_ltp, 2) if ctx.vix_ltp else None,
        "direction_pref": ctx.direction_pref,
        "hypothesis_note": hypothesis_note,
        "strategy_mode": "20rupees_only",
        "premium_band": [rp.entry_band_low, rp.entry_band_high],
        "fixed_sl_premium": rp.sl_inr,
        "fixed_sl_inr": rp.sl_inr,
        "fixed_lots": FIXED_LOTS,
        "entry_cutoff_ist": rp.entry_cutoff_label(),
        "entry_scan_start_ist": rp.entry_scan_start_ist,
        "entry_scan_end_ist": rp.entry_scan_end_ist,
        "capital": rp.capital,
        "risk_pct": rp.risk_pct,
        "entry_allowed": not (
            (ctx.minutes > 0 and ctx.minutes >= rp.scan_end_minutes())
            or is_past_sensex_entry_cutoff()
        ),
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
            "entry_premium": c.entry_premium,
            "stop_loss_premium": c.stop_loss_premium,
            "target_premium": c.target_premium,
            "strike_source": c.strike_source,
            "tradingsymbol": c.tradingsymbol,
            "fixed_lots": FIXED_LOTS,
        }

    band = f"₹{rp.entry_band_low:.0f}–{rp.entry_band_high:.0f}"
    output_lines = [
        f"Selected: {selected.name} (score {selected.score}/100, {selected.fit})",
        f"Leg: BUY {index_label} {selected.option_kind} · premium band {band}",
    ]
    if selected.entry_premium:
        output_lines.append(
            f"LTP ₹{selected.entry_premium:.2f} · 1 lot · SL ₹{selected.stop_loss_premium:.2f} · "
            f"Tgt ₹{selected.target_premium:.2f}"
        )
    if selected.reasons:
        output_lines.append("Why: " + "; ".join(selected.reasons[:2]))
    if selected.warnings:
        output_lines.append("Wait: " + selected.warnings[0])

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
