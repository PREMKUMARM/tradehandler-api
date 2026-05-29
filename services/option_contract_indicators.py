"""
Option-contract indicators for order pricing — BB(20,2) on the traded symbol's 5m chart.
Entry / SL / target premiums are derived from contract LTP vs contract bands, not Nifty index.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

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
        if kind == "CE":
            sl, tgt = prem - risk, prem + risk * rr
        else:
            sl, tgt = prem + risk, prem - risk * rr
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
    if kind == "CE":
        sl = float(lower) - buf
        risk = max(0.05, prem - sl)
        tgt = float(mid) + rr * risk * 0.85
    else:
        sl = float(upper) + buf
        risk = max(0.05, sl - prem)
        tgt = float(mid) - rr * risk * 0.85

    sl_p = round_to_tick(max(0.05, sl))
    tgt_p = round_to_tick(max(0.05, tgt))
    note = f"{sym}: 5m BB SL beyond band, target toward middle"
    return sl_p, tgt_p, sl, tgt, note
