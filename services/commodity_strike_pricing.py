"""
V2 — dynamic strike (ATM/OTM) and premium from live spot, candles, and patterns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.push.option_contract_resolver import (
    OptionLegEstimate,
    estimate_delta_from_spot,
    estimate_option_leg,
    true_atm_strike,
)
from services.commodity_realtime_checklist import _intraday_context, _underlying_live
from utils.kite_utils import get_kite_instance
from utils.logger import log_warning


@dataclass
class StrikeProfile:
    moneyness: str  # ATM | OTM1 | OTM2 | ITM1
    pattern_tag: str
    reason: str
    spot_entry: float
    spot_stop_loss: float
    spot_target: float


def _pick_moneyness(
    strategy_id: str,
    spot: float,
    kind: str,
    spot_sl: float,
    spot_tgt: float,
    intra: Dict[str, Any],
) -> Tuple[str, str, str]:
    """Return (moneyness, pattern_tag, reason) from strategy + live structure."""
    or_h, or_l = intra.get("or_high"), intra.get("or_low")
    pdh, pdl = intra.get("pdh"), intra.get("pdl")
    ema9 = intra.get("ema9")
    risk_pts = abs(spot - spot_sl)
    reward_pts = abs(spot_tgt - spot)

    if strategy_id == "long_atm_directional":
        return "ATM", "directional", "Session bias — ATM for delta ~0.5 and liquidity"

    if strategy_id == "orb_15m_breakout":
        if or_h and or_l:
            or_range = or_h - or_l
            if kind == "CE" and spot > or_h:
                ext = spot - or_h
                if or_range > 0 and ext > 0.28 * or_range:
                    return (
                        "OTM1",
                        "orb_extension",
                        f"ORB CE break +{ext:.0f} pts past OR high — OTM1",
                    )
                return "ATM", "orb_breakout", "ORB CE breakout — ATM"
            if kind == "PE" and spot < or_l:
                ext = or_l - spot
                if or_range > 0 and ext > 0.28 * or_range:
                    return (
                        "OTM1",
                        "orb_extension",
                        f"ORB PE break +{ext:.0f} pts past OR low — OTM1",
                    )
                return "ATM", "orb_breakout", "ORB PE breakout — ATM"
        return "ATM", "orb_wait", "Inside/warming OR — ATM default"

    if strategy_id == "pdh_pdl_breakout":
        buf = 3.0
        if kind == "CE" and pdh and spot > pdh + buf:
            ext_pct = (spot - pdh) / spot * 100
            if ext_pct > 0.12:
                return "OTM1", "pdh_extended", f"Above PDH {pdh:.0f} by {ext_pct:.2f}% — OTM1"
            return "ATM", "pdh_break", f"Fresh break above PDH {pdh:.0f} — ATM"
        if kind == "PE" and pdl and spot < pdl - buf:
            ext_pct = (pdl - spot) / spot * 100
            if ext_pct > 0.12:
                return "OTM1", "pdl_extended", f"Below PDL {pdl:.0f} by {ext_pct:.2f}% — OTM1"
            return "ATM", "pdl_break", f"Fresh break below PDL {pdl:.0f} — ATM"
        return "ATM", "pdh_pdl_range", "Between levels — ATM until break confirms"

    if strategy_id == "ema_pullback_continuation":
        if ema9 is not None:
            dist = abs(spot - ema9)
            if dist <= 22:
                return "ATM", "ema_pullback", f"Pullback within {dist:.0f} pts of 9 EMA — ATM"
            if dist <= 45:
                return "OTM1", "ema_extended", f"Extended {dist:.0f} pts from 9 EMA — OTM1"
            return "OTM2", "ema_chase", "Far from 9 EMA — cheaper OTM2 (higher risk)"
        return "ATM", "ema_unknown", "9 EMA unavailable — ATM"

    # Fallback: use R:R — wide target → slightly OTM
    if reward_pts > 1.8 * risk_pts and risk_pts > 0:
        return "OTM1", "rr_otm", "Reward > 1.8× risk — OTM1 for cost efficiency"
    return "ATM", "default", "Default ATM"


def refine_spot_levels_from_candles(
    strategy_id: str,
    spot: float,
    kind: str,
    spot_sl: float,
    spot_tgt: float,
    intra: Dict[str, Any],
) -> Tuple[float, float, float, str]:
    """
    Re-anchor entry/SL/target to latest candle structure (OR, PDH/PDL, EMA).
    Returns (entry, sl, tgt, note).
    """
    entry = spot
    sl, tgt = spot_sl, spot_tgt
    notes: List[str] = []
    or_h, or_l = intra.get("or_high"), intra.get("or_low")
    pdh, pdl = intra.get("pdh"), intra.get("pdl")
    ema9 = intra.get("ema9")
    rr = 1.5
    if abs(spot_tgt - spot) > 0 and abs(spot_sl - spot) > 0:
        rr = abs(spot_tgt - spot) / abs(spot_sl - spot)

    if strategy_id == "orb_15m_breakout" and or_h and or_l:
        if kind == "CE" and spot > or_h:
            sl = or_l - 2
            risk = max(1.0, entry - sl)
            tgt = entry + rr * risk
            notes.append(f"ORB SL below OR low {or_l:.0f}")
        elif kind == "PE" and spot < or_l:
            sl = or_h + 2
            risk = max(1.0, sl - entry)
            tgt = entry - rr * risk
            notes.append(f"ORB SL above OR high {or_h:.0f}")

    elif strategy_id == "pdh_pdl_breakout" and pdh and pdl:
        buf = 3.0
        if kind == "CE" and spot > pdh:
            sl = pdh - buf
            risk = max(1.0, entry - sl)
            tgt = entry + rr * risk
            notes.append(f"PDH break SL {sl:.0f}")
        elif kind == "PE" and spot < pdl:
            sl = pdl + buf
            risk = max(1.0, sl - entry)
            tgt = entry - rr * risk
            notes.append(f"PDL break SL {sl:.0f}")

    elif strategy_id == "ema_pullback_continuation" and ema9 is not None:
        risk_pts = max(12.0, entry * 0.0025)
        if kind == "CE":
            sl = min(ema9 - 5, entry - risk_pts)
            risk = max(1.0, entry - sl)
            tgt = entry + rr * risk
            notes.append(f"EMA pullback SL near {ema9:.0f}")
        else:
            sl = max(ema9 + 5, entry + risk_pts)
            risk = max(1.0, sl - entry)
            tgt = entry - rr * risk
            notes.append(f"EMA pullback SL near {ema9:.0f}")

    elif strategy_id == "long_atm_directional":
        risk_pts = max(15.0, entry * 0.0035)
        if kind == "CE":
            sl, tgt = entry - risk_pts, entry + risk_pts * rr
        else:
            sl, tgt = entry + risk_pts, entry - risk_pts * rr
        notes.append("Directional SL from live spot % risk")

    note = "; ".join(notes) if notes else "Levels from strategy score"
    return entry, sl, tgt, note


def build_dynamic_option_leg(
    *,
    strategy_id: Optional[str],
    spot_entry: float,
    spot_stop_loss: float,
    spot_target: float,
    kind: str,
    lot_size: int,
    num_lots: int,
    intra: Optional[Dict[str, Any]] = None,
    vix: Optional[float] = None,
    atm_step: int = 50,
) -> Tuple[OptionLegEstimate, StrikeProfile]:
    """Pick ATM/OTM from pattern, refine spot levels, price with live delta + LTP."""
    kite = get_kite_instance()
    if intra is None:
        try:
            from services.commodity_instruments import future_token

            token = future_token(kite)
            intra = _intraday_context(kite, token)
        except Exception as exc:
            log_warning(f"[Commodity strike] intraday: {exc}")
            intra = {}

    nifty = _underlying_live(kite)
    live_spot = float(nifty.get("spot") or spot_entry)
    if live_spot > 0:
        spot_entry = live_spot

    sid = strategy_id or "long_atm_directional"
    entry, sl, tgt, level_note = refine_spot_levels_from_candles(
        sid, spot_entry, kind, spot_stop_loss, spot_target, intra
    )

    moneyness, pattern_tag, m_reason = _pick_moneyness(sid, entry, kind, sl, tgt, intra)
    leg = estimate_option_leg(
        spot_entry=entry,
        spot_stop_loss=sl,
        spot_target=tgt,
        kind=kind,
        delta=0.5,
        lot_size=lot_size,
        num_lots=num_lots,
        atm_step=atm_step,
        moneyness=moneyness,
        vix=vix,
        use_live_delta=True,
    )

    atm = true_atm_strike(entry, atm_step)
    strike_note = ""
    if leg.contract:
        delta_live = estimate_delta_from_spot(entry, leg.contract.strike, kind, vix=vix)
        otm_label = "ATM"
        if leg.contract.strike != atm:
            if kind == "CE":
                otm_label = "OTM" if leg.contract.strike > atm else "ITM"
            else:
                otm_label = "OTM" if leg.contract.strike < atm else "ITM"
        strike_note = (
            f"{otm_label} strike {leg.contract.strike} (ATM ref {atm}) · "
            f"δ {delta_live:.2f} · {m_reason} · {level_note}"
        )
    else:
        strike_note = f"{m_reason} · {level_note}"

    profile = StrikeProfile(
        moneyness=moneyness,
        pattern_tag=pattern_tag,
        reason=strike_note,
        spot_entry=entry,
        spot_stop_loss=sl,
        spot_target=tgt,
    )
    if leg.note:
        leg = OptionLegEstimate(
            contract=leg.contract,
            entry_premium=leg.entry_premium,
            sl_premium=leg.sl_premium,
            target_premium=leg.target_premium,
            premium_risk=leg.premium_risk,
            premium_reward=leg.premium_reward,
            risk_inr=leg.risk_inr,
            reward_inr=leg.reward_inr,
            delta_used=leg.delta_used,
            estimated=leg.estimated,
            note=strike_note if not leg.estimated else f"{leg.note}; {strike_note}",
        )
    else:
        leg = OptionLegEstimate(
            contract=leg.contract,
            entry_premium=leg.entry_premium,
            sl_premium=leg.sl_premium,
            target_premium=leg.target_premium,
            premium_risk=leg.premium_risk,
            premium_reward=leg.premium_reward,
            risk_inr=leg.risk_inr,
            reward_inr=leg.reward_inr,
            delta_used=leg.delta_used,
            estimated=leg.estimated,
            note=strike_note,
        )
    return leg, profile
