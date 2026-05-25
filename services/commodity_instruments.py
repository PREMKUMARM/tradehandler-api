"""Resolve MCX CRUDEOIL26JUN future + option contracts from Kite instruments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any, Dict, List, Optional

from services.commodity_config import (
    DEFAULT_LOT_SIZE,
    EXCHANGE,
    FUTURE_SYMBOL,
    OPTION_PREFIX,
    STRIKE_STEP,
)
from utils.kite_utils import get_kite_instance
from utils.logger import log_warning


@lru_cache(maxsize=1)
def _mcx_rows() -> List[Dict[str, Any]]:
    kite = get_kite_instance()
    return list(kite.instruments(EXCHANGE) or [])


def resolve_future() -> Dict[str, Any]:
    for row in _mcx_rows():
        if (
            str(row.get("tradingsymbol") or "") == FUTURE_SYMBOL
            and str(row.get("instrument_type") or "").upper() == "FUT"
        ):
            return row
    for row in _mcx_rows():
        sym = str(row.get("tradingsymbol") or "")
        if sym == FUTURE_SYMBOL or sym.startswith(FUTURE_SYMBOL):
            if str(row.get("instrument_type") or "").upper() == "FUT":
                return row
    raise ValueError(f"MCX future {FUTURE_SYMBOL} not found — refresh instruments cache")


def future_token() -> int:
    return int(resolve_future()["instrument_token"])


def lot_size() -> int:
    try:
        return int(resolve_future().get("lot_size") or DEFAULT_LOT_SIZE)
    except Exception:
        return DEFAULT_LOT_SIZE


def list_option_rows(kind: Optional[str] = None) -> List[Dict[str, Any]]:
    prefix = OPTION_PREFIX
    out = []
    for row in _mcx_rows():
        it = str(row.get("instrument_type") or "").upper()
        sym = str(row.get("tradingsymbol") or "")
        if not sym.startswith(prefix):
            continue
        if it not in ("CE", "PE"):
            continue
        if kind and it != kind.upper():
            continue
        out.append(row)
    return out


def _strike_from_row(row: Dict[str, Any]) -> int:
    """MCX rows sometimes have strike=0; parse from tradingsymbol (e.g. CRUDEOIL26JUN8700PE)."""
    try:
        s = int(float(row.get("strike") or 0))
        if s >= 1000:
            return s
    except (TypeError, ValueError):
        pass
    sym = str(row.get("tradingsymbol") or "")
    if sym.startswith(OPTION_PREFIX):
        rest = sym[len(OPTION_PREFIX) :]
        for suffix in ("CE", "PE"):
            if rest.endswith(suffix):
                rest = rest[: -len(suffix)]
                try:
                    return int(rest)
                except ValueError:
                    break
    return 0


def pick_option_tradingsymbol(strike: int, kind: str) -> str:
    k = kind.upper()
    rows = list_option_rows(k)
    if not rows:
        raise ValueError(f"No MCX {k} options for {OPTION_PREFIX}")
    near = [
        r
        for r in rows
        if _strike_from_row(r) > 0 and abs(_strike_from_row(r) - strike) <= max(500, strike * 0.15)
    ]
    scan = near if near else rows
    best = None
    best_dist = 10**9
    for row in scan:
        s = _strike_from_row(row)
        if s <= 0:
            continue
        dist = abs(s - strike)
        if dist < best_dist:
            best_dist = dist
            best = row
    if not best:
        raise ValueError(f"No strike near {strike} for {OPTION_PREFIX} {k}")
    return str(best["tradingsymbol"])


def atm_strike(spot: float) -> int:
    return int(round(spot / STRIKE_STEP) * STRIKE_STEP)


def build_crude_options_universe(kite) -> List[Dict[str, Any]]:
    """MCX CRUDEOIL options for chain / strike resolution."""
    out: List[Dict[str, Any]] = []
    try:
        for inst in list_option_rows():
            out.append(
                {
                    "strike": inst.get("strike"),
                    "instrument_type": inst.get("instrument_type"),
                    "expiry": inst.get("expiry"),
                    "tradingsymbol": inst.get("tradingsymbol"),
                    "instrument_token": inst.get("instrument_token"),
                }
            )
    except Exception as exc:
        log_warning(f"[Commodity instruments] universe: {exc}")
    return out


def strike_for_moneyness(
    spot: float,
    kind: str,
    moneyness: str,
    step: int = STRIKE_STEP,
) -> int:
    atm = atm_strike(spot)
    k = (kind or "CE").upper()
    m = (moneyness or "ATM").upper()
    if m == "ATM":
        return atm
    if m == "OTM1":
        return atm + step if k == "CE" else atm - step
    if m == "OTM2":
        return atm + 2 * step if k == "CE" else atm - 2 * step
    if m == "ITM1":
        return atm - step if k == "CE" else atm + step
    return atm


@dataclass
class CommodityOptionContract:
    tradingsymbol: str
    strike: int
    expiry: date
    instrument_token: int
    lot_size: int
    instrument_type: str


def resolve_commodity_contract(
    *,
    spot: float,
    kind: str,
    moneyness: str = "ATM",
) -> Optional[CommodityOptionContract]:
    k = (kind or "").upper()
    if k not in ("CE", "PE"):
        return None
    target = strike_for_moneyness(spot, k, moneyness)
    expected_sym = f"{OPTION_PREFIX}{target}{k}"
    for row in list_option_rows(k):
        if str(row.get("tradingsymbol") or "") == expected_sym:
            exp = row.get("expiry")
            if hasattr(exp, "date"):
                exp = exp.date()
            elif not isinstance(exp, date):
                exp = date.today()
            ls = int(row.get("lot_size") or DEFAULT_LOT_SIZE)
            if ls < DEFAULT_LOT_SIZE:
                ls = DEFAULT_LOT_SIZE
            return CommodityOptionContract(
                tradingsymbol=expected_sym,
                strike=target,
                expiry=exp,
                instrument_token=int(row.get("instrument_token") or 0),
                lot_size=ls,
                instrument_type=k,
            )
    rows = list_option_rows(k)
    if not rows:
        return None
    # Ignore far OTM series (e.g. strike 3750 when spot ~8700).
    near = [
        r
        for r in rows
        if _strike_from_row(r) > 0 and abs(_strike_from_row(r) - spot) <= max(500, spot * 0.15)
    ]
    if near:
        rows = near
    best_row = None
    best_dist = 10**9
    for row in rows:
        s = _strike_from_row(row)
        if s <= 0:
            continue
        dist = abs(s - target)
        if dist < best_dist:
            best_dist = dist
            best_row = row
    if not best_row:
        return None
    exp = best_row.get("expiry")
    if hasattr(exp, "date"):
        exp = exp.date()
    elif not isinstance(exp, date):
        exp = date.today()
    ls = int(best_row.get("lot_size") or DEFAULT_LOT_SIZE)
    if ls < DEFAULT_LOT_SIZE:
        ls = DEFAULT_LOT_SIZE
    resolved_strike = _strike_from_row(best_row) or target
    return CommodityOptionContract(
        tradingsymbol=str(best_row["tradingsymbol"]),
        strike=int(resolved_strike),
        expiry=exp,
        instrument_token=int(best_row.get("instrument_token") or 0),
        lot_size=ls,
        instrument_type=k,
    )
