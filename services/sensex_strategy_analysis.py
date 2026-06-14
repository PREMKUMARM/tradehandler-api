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
STRATEGY_DESC = (
    "Buy Sensex option when premium closes ₹17–₹23 on 5m (from 14:00 IST). "
    "AUTO: PE on gap-down, CE on gap-up/flat; monitors 11 strikes around ATM (ATM±5); ATM from index LTP. "
    "Fixed initial SL at ₹9 premium; 1R = entry − SL; 1:1 target then stepped trail. "
    "Size lots from risk % ÷ entry premium (not SL distance). "
    "No new entries after 3:00 PM IST."
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
    return sensex_gap_direction_kind(ctx.day_open or ctx.nifty_ltp, ctx.prev_close)


def _in_premium_band(ltp: float) -> bool:
    return PREMIUM_BAND_LOW <= ltp <= PREMIUM_BAND_HIGH


def _band_score(ltp: float) -> int:
    if not _in_premium_band(ltp):
        return 25
    center = (PREMIUM_BAND_LOW + PREMIUM_BAND_HIGH) / 2.0
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
) -> Tuple[Optional[int], float, str, Optional[str], Optional[float], str]:
    """AUTO: gap direction leg + smart OI strike in premium band."""
    kind = sensex_gap_direction_kind(day_open or spot, prev_close)
    picked = pick_smart_from_chain(
        chain_oi,
        spot,
        kinds=(kind,),
        band_low=PREMIUM_BAND_LOW,
        band_high=PREMIUM_BAND_HIGH,
        prev_close=prev_close,
    )
    if not picked:
        atm = int(chain_oi.get("atm") or round(spot / 100) * 100)
        return atm, 0.0, "ATM", None, None, kind
    atm = int(chain_oi.get("atm") or round(spot / 100) * 100)
    money = moneyness_label(picked.strike, atm, picked.offset)
    return picked.strike, picked.ltp, money, picked.symbol, picked.oi, picked.kind


def _pick_strike_from_chain(
    chain_oi: Dict[str, Any],
    kind: str,
    spot: float,
    prev_close: float = 0.0,
) -> Tuple[Optional[int], float, str, Optional[str], Optional[float]]:
    """CE/PE: smart strike in premium band on the chosen leg."""
    picked = pick_smart_from_chain(
        chain_oi,
        spot,
        kinds=(kind.upper(),),
        band_low=PREMIUM_BAND_LOW,
        band_high=PREMIUM_BAND_HIGH,
        prev_close=prev_close,
    )
    atm = int(chain_oi.get("atm") or round(spot / 100) * 100)
    if picked:
        money = moneyness_label(picked.strike, atm, picked.offset)
        return picked.strike, picked.ltp, money, picked.symbol, picked.oi

    # Fallback: ATM quote on leg
    atm_row = (chain_oi.get("atm_ce") or {}) if kind == "CE" else (chain_oi.get("atm_pe") or {})
    atm_ltp = float(atm_row.get("ltp") or 0)
    return atm if atm > 0 else None, atm_ltp, "ATM", None, None


def _fetch_market_context(direction_pref: str, margin: float) -> MarketContext:
    ctx = MarketContext(margin=margin, direction_pref=(direction_pref or "AUTO").upper())
    now = datetime.now(IST)
    ctx.minutes = now.hour * 60 + now.minute
    ctx.is_weekday = 1 <= now.weekday() <= 5

    try:
        from services.sensex_live_indicators import recalculate_from_ticker

        live = recalculate_from_ticker()
        ctx.nifty_ltp = float(live.get("nifty_spot") or 0)
        ctx.prev_close = float(live.get("prev_close") or ctx.nifty_ltp)
        ctx.day_open = float(live.get("day_open") or ctx.nifty_ltp)
        if live.get("vix"):
            ctx.vix_ltp = float(live["vix"])
    except Exception as exc:
        log_warning(f"[Sensex strategy] live indicators failed: {exc}")

    return ctx


def _score_20rupees(ctx: MarketContext, chain_oi: Optional[Dict[str, Any]]) -> StrategyCandidate:
    chain = chain_oi or {}
    spot = ctx.nifty_ltp
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
            warnings=["Sensex spot unavailable — connect Kite ticker"],
        )

    if ctx.direction_pref == "AUTO":
        strike, ltp, moneyness, sym, oi, kind = _pick_auto_strike_from_chain(
            chain, spot, prev_close=ctx.prev_close, day_open=ctx.day_open
        )
        gap_pct = sensex_gap_pct(ctx.day_open, ctx.prev_close)
        if ctx.prev_close > 0:
            reasons.append(
                f"AUTO gap → {kind} (open vs prev close {gap_pct:+.2f}%)"
            )
    else:
        kind = ctx.direction_pref
        strike, ltp, moneyness, sym, oi = _pick_strike_from_chain(
            chain, kind, spot, prev_close=ctx.prev_close
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
    sl_prem = round(FIXED_SL_PREMIUM, 2)
    r_dist = max(0.05, entry_prem - sl_prem)
    tgt_prem = round(entry_prem + r_dist, 2)
    score = _band_score(entry_prem)

    if _in_premium_band(entry_prem):
        reasons.append(
            f"{kind} {moneyness} strike {strike}: premium ₹{entry_prem:.2f} in "
            f"₹{PREMIUM_BAND_LOW:.0f}–{PREMIUM_BAND_HIGH:.0f} band"
        )
        reasons.append(
            f"Plan: size to risk % of capital · SL ₹{sl_prem:.2f} · target ₹{tgt_prem:.2f} (1:1)"
        )
    elif entry_prem > PREMIUM_BAND_HIGH:
        warnings.append(
            f"Premium ₹{entry_prem:.2f} above band — wait for ₹{PREMIUM_BAND_HIGH:.0f}–{PREMIUM_BAND_LOW:.0f}"
        )
        score = min(score, 38)
    else:
        warnings.append(
            f"Premium ₹{entry_prem:.2f} below band — wait for ₹{PREMIUM_BAND_HIGH:.0f}–{PREMIUM_BAND_LOW:.0f}"
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

    elif ctx.minutes > 0 and ctx.minutes < sensex_entry_scan_start_minutes():
        scan = sensex_entry_scan_start_minutes()
        warnings.append(
            f"Before {scan // 60:02d}:{scan % 60:02d} IST — wait for afternoon 5m close in band"
        )
        score = min(score, 28)
        pattern = "20rupees_open_bar_skip"
    elif is_past_sensex_entry_cutoff():
        warnings.append(sensex_entry_cutoff_message())
        score = min(score, 25)
        pattern = "20rupees_cutoff"
    elif _in_premium_band(entry_prem):
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
) -> Dict[str, Any]:
    """Single 20rupees-strategy for Sensex (paper + live)."""
    ctx = _fetch_market_context(direction_pref, margin)
    selected = _score_20rupees(ctx, chain_oi)

    ranked = [selected]

    context_summary = {
        "nifty_spot": round(ctx.nifty_ltp, 2),
        "prev_close": round(ctx.prev_close, 2),
        "margin": round(ctx.margin, 2),
        "vix": round(ctx.vix_ltp, 2) if ctx.vix_ltp else None,
        "direction_pref": ctx.direction_pref,
        "hypothesis_note": hypothesis_note,
        "strategy_mode": "20rupees_only",
        "premium_band": [PREMIUM_BAND_LOW, PREMIUM_BAND_HIGH],
        "fixed_sl_premium": FIXED_SL_PREMIUM,
        "fixed_sl_inr": FIXED_SL_PREMIUM,
        "fixed_lots": FIXED_LOTS,
        "entry_cutoff_ist": sensex_entry_cutoff_label(),
        "entry_allowed": not is_past_sensex_entry_cutoff(),
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

    band = f"₹{PREMIUM_BAND_LOW:.0f}–{PREMIUM_BAND_HIGH:.0f}"
    output_lines = [
        f"Selected: {selected.name} (score {selected.score}/100, {selected.fit})",
        f"Leg: BUY Sensex {selected.option_kind} · premium band {band}",
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
