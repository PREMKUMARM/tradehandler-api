"""
Option-contract indicators for order pricing — BB(20,2) on the traded symbol's 5m chart.
Entry / SL / target premiums are derived from contract LTP vs contract bands, not Nifty index.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from utils.kite_order_utils import round_to_tick


def merge_option_bb_into_intra(
    intra: Dict[str, Any],
    opt_bb: Dict[str, Any],
    tradingsymbol: str,
) -> Dict[str, Any]:
    out = dict(intra)
    if opt_bb.get("bb_middle") is None:
        return out
    out["bb_lower"] = opt_bb["bb_lower"]
    out["bb_middle"] = opt_bb["bb_middle"]
    out["bb_upper"] = opt_bb["bb_upper"]
    out["last_5m_close"] = opt_bb.get("last_5m_close")
    out["indicator_sources"] = {
        **(intra.get("indicator_sources") or {}),
        **(opt_bb.get("indicator_sources") or {}),
    }
    out["bb_on_contract"] = tradingsymbol
    return out


def contract_bb_is_active(intra: Dict[str, Any]) -> bool:
    src = (intra.get("indicator_sources") or {}).get("bb_middle", "")
    return src == "kite_historical_5m_option" or bool(intra.get("bb_on_contract"))


def contract_price_for_bb(spot: float, intra: Dict[str, Any]) -> float:
    """LTP used for BB zone checks — contract premium when BB is on the option."""
    if contract_bb_is_active(intra):
        ltp = intra.get("contract_ltp") or intra.get("option_ltp")
        if ltp is not None:
            try:
                v = float(ltp)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    return float(spot)


def order_exit_levels_from_contract_bb(
    entry_premium: float,
    option_kind: str,
    intra: Dict[str, Any],
    *,
    reward_ratio: float = 1.5,
) -> Tuple[float, float, float, float, str]:
    """
    GTT stop-loss and target (option premium) from contract 5m Bollinger bands.
    Returns (sl_premium, tgt_premium, raw_sl, raw_tgt, note).
    """
    prem = float(entry_premium)
    kind = (option_kind or "CE").upper()
    lower, mid, upper = intra.get("bb_lower"), intra.get("bb_middle"), intra.get("bb_upper")
    sym = intra.get("bb_on_contract") or "contract"

    if lower is None or mid is None or upper is None:
        risk = max(0.5, prem * 0.12)
        rr = reward_ratio
        sl, tgt = prem - risk, prem + risk * rr
        return (
            round_to_tick(max(0.05, sl)),
            round_to_tick(max(0.05, tgt)),
            sl,
            tgt,
            f"{sym}: BB not ready — premium % risk fallback",
        )

    width = float(upper) - float(lower)
    buf = max(0.5, width * 0.04)
    rr = reward_ratio
    # Long-only (buy CE / buy PE): loss when premium falls, profit when premium rises.
    sl = float(lower) - buf
    risk = max(0.05, prem - sl)
    tgt = prem + rr * risk * 0.85

    sl_p = round_to_tick(max(0.05, sl))
    tgt_p = round_to_tick(max(0.05, tgt))
    if tgt_p <= prem:
        tgt_p = round_to_tick(prem + max(0.5, risk * 0.5))
    note = f"{sym}: 5m contract BB — SL below lower band, TP above entry"
    return sl_p, tgt_p, sl, tgt, note


# Underlying ORB/PDH/EMA signals; exits mapped to bought-option premiums.
STRUCTURE_STRATEGIES = frozenset(
    {
        "orb_15m_breakout",
        "pdh_pdl_breakout",
        "ema_pullback_continuation",
        "long_atm_directional",
        "green_bar_sentinel_2nd_oi",
    }
)
COMMODITY_STRUCTURE_STRATEGIES = STRUCTURE_STRATEGIES


def resolve_long_buy_exit_levels(
    *,
    strategy_id: str,
    entry_premium: float,
    option_kind: str,
    intra_bb: Dict[str, Any],
    underlying_spot: float,
    underlying_sl: float,
    underlying_tgt: float,
    strike: int,
    vix: Optional[float] = None,
    reward_ratio: float = 1.5,
    normalize_exits: Optional[
        Callable[[float, float, float], Tuple[float, float]]
    ] = None,
) -> Tuple[float, float, float, float, float, str]:
    """
    Long-only option exits: structure strategies use underlying levels → premium (delta);
    BB / default strategies use contract 5m Bollinger on the option chart.
    """
    from services.push.option_contract_resolver import estimate_delta_from_spot

    sid = strategy_id or "bb_5m_mean_reversion"
    prem = float(entry_premium)

    if sid in STRUCTURE_STRATEGIES:
        from services.push.option_contract_resolver import estimate_delta_from_spot

        delta = estimate_delta_from_spot(
            float(underlying_spot),
            int(strike),
            option_kind,
            vix=vix,
        )
        spot_risk = abs(float(underlying_spot) - float(underlying_sl))
        spot_reward = abs(float(underlying_tgt) - float(underlying_spot))
        sl_prem = max(0.05, prem - spot_risk * delta)
        tgt_prem = prem + spot_reward * delta
        prem_risk = max(0.05, prem - sl_prem)
        min_reward = prem_risk * max(1.0, float(reward_ratio or 1.0))
        if tgt_prem - prem < min_reward:
            tgt_prem = prem + min_reward
        sl_prem = round_to_tick(sl_prem)
        tgt_prem = round_to_tick(tgt_prem)
        if normalize_exits:
            sl_prem, tgt_prem = normalize_exits(
                prem, sl_prem, tgt_prem, min_rr=reward_ratio
            )
        note = (
            f"{sid}: underlying SL {underlying_sl:.0f} → TP {underlying_tgt:.0f} "
            f"mapped to premium (δ)"
        )
        return sl_prem, tgt_prem, underlying_sl, underlying_tgt, delta, note

    sl_prem, tgt_prem, raw_sl, raw_tgt, note = order_exit_levels_from_contract_bb(
        prem,
        option_kind,
        intra_bb,
        reward_ratio=reward_ratio,
    )
    if normalize_exits:
        sl_prem, tgt_prem = normalize_exits(
            prem, sl_prem, tgt_prem, min_rr=reward_ratio
        )
    delta = estimate_delta_from_spot(
        float(underlying_spot),
        int(strike),
        option_kind,
        vix=vix,
    )
    return sl_prem, tgt_prem, raw_sl, raw_tgt, delta, note
