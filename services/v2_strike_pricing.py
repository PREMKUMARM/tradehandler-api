"""
V2 — 20rupees-strategy strike selection (smart OI in premium band).
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
from services.v2_realtime_checklist import _intraday_context, _nifty_live, _vix_live
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
    """20rupees-strategy: smart OI strike in ₹17–₹23 near ATM."""
    del spot, kind, spot_sl, spot_tgt, intra
    return (
        "SMART_OI",
        "20rupees",
        "Smart OI pick in ₹17–₹23 near ATM · size to risk % · scan window 14:00–14:45 IST",
    )


def refine_spot_levels_from_candles(
    strategy_id: str,
    spot: float,
    kind: str,
    spot_sl: float,
    spot_tgt: float,
    intra: Dict[str, Any],
) -> Tuple[float, float, float, str]:
    """20rupees uses premium-based SL/target from strategy analysis — spot levels are reference only."""
    del strategy_id, kind, intra
    return spot, spot_sl, spot_tgt, "20rupees-strategy: premium SL ₹9 · 1:1 target"


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
            from services.nifty_option_chain import nifty50_index_token

            token = nifty50_index_token(kite)
            intra = _intraday_context(kite, token)
        except Exception as exc:
            log_warning(f"[V2 strike] intraday: {exc}")
            intra = {}

    nifty = _nifty_live(kite)
    live_spot = float(nifty.get("spot") or spot_entry)
    if live_spot > 0:
        spot_entry = live_spot

    sid = strategy_id or "20rupees_strategy"
    entry, sl, tgt, level_note = refine_spot_levels_from_candles(
        sid, spot_entry, kind, spot_stop_loss, spot_target, intra
    )

    moneyness, pattern_tag, m_reason = _pick_moneyness(sid, entry, kind, sl, tgt, intra)
    if vix is None:
        vix = _vix_live(kite).get("ltp")

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
