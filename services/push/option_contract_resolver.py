"""
NIFTY ATM option contract resolver + live LTP fetcher.

Used by intraday push-notification strategies (ORB, 9-EMA pullback, PDH/PDL
breakout, ...) to attach a *real* CE/PE contract and its live premium to each
signal so the notification can show actual entry / SL / target premiums (₹),
not just spot levels.

Behavior:
  * Reads the daily-refreshed instruments cache (`data/instruments/instruments.csv`)
    to filter NIFTY weekly options.
  * Caches the parsed NIFTY F&O subset (today's expiry) in-memory per day.
  * Picks the nearest expiry on/after today as the "current weekly".
  * Picks the strike nearest to spot (rounded to `atm_step`, default 50).
  * `fetch_option_ltp(...)` calls `kite.quote("NFO:<symbol>")` for live LTP.

All functions are safe to call from threads / async — they only hold a short
in-memory lock during cache population.
"""
from __future__ import annotations

import csv
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.kite_utils import get_kite_instance
from utils.logger import log_error, log_info, log_warning


CSV_FILE = Path("data/instruments/instruments.csv")


@dataclass(frozen=True)
class OptionContract:
    tradingsymbol: str
    exchange: str
    instrument_token: int
    strike: int
    expiry: date
    instrument_type: str  # 'CE' | 'PE'

    def quote_key(self) -> str:
        return f"{self.exchange}:{self.tradingsymbol}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["expiry"] = self.expiry.isoformat()
        return d


# -------------------------- in-memory cache --------------------------

_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {
    "as_of": None,            # date the cache was built for
    "expiry": None,           # date of the chosen "current weekly" expiry
    "by_kind_strike": {},     # {('CE', 22050): OptionContract, ('PE', 22050): ...}
    "available_strikes": [],  # sorted ints
}


def _parse_date(s: Any) -> Optional[date]:
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    if not isinstance(s, str) or not s:
        return None
    # Try ISO date first, then ISO datetime
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(s.strip())
    except Exception:
        return None


def _parse_int(s: Any) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _build_cache_for_today(*, name: str = "NIFTY") -> bool:
    """
    Parse the instruments CSV, filter NIFTY CE/PE rows, pick the nearest
    expiry on/after today, and build a {(kind, strike) -> contract} map.
    Returns True if cache is populated (i.e., a usable expiry was found).
    """
    today = date.today()
    if not CSV_FILE.exists():
        log_warning(f"[OptResolver] instruments CSV missing at {CSV_FILE}")
        return False

    candidates: List[Dict[str, Any]] = []
    try:
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("exchange") or "").upper() != "NFO":
                    continue
                if (row.get("name") or "").upper() != name.upper():
                    continue
                kind = (row.get("instrument_type") or "").upper()
                if kind not in ("CE", "PE"):
                    continue
                expiry = _parse_date(row.get("expiry"))
                if expiry is None or expiry < today:
                    continue
                strike = _parse_int(row.get("strike"))
                tok = _parse_int(row.get("instrument_token"))
                tsym = (row.get("tradingsymbol") or "").strip()
                if not tsym or strike is None or tok is None:
                    continue
                candidates.append(
                    {
                        "kind": kind,
                        "strike": int(strike),
                        "expiry": expiry,
                        "tradingsymbol": tsym,
                        "instrument_token": tok,
                    }
                )
    except Exception as e:  # noqa: BLE001
        log_error(f"[OptResolver] failed to read CSV: {e}")
        return False

    if not candidates:
        log_warning("[OptResolver] no NIFTY CE/PE candidates found (cache empty?)")
        return False

    nearest_expiry = min(c["expiry"] for c in candidates)
    selected = [c for c in candidates if c["expiry"] == nearest_expiry]

    by_kind_strike: Dict[Tuple[str, int], OptionContract] = {}
    strikes: set = set()
    for c in selected:
        contract = OptionContract(
            tradingsymbol=c["tradingsymbol"],
            exchange="NFO",
            instrument_token=int(c["instrument_token"]),
            strike=int(c["strike"]),
            expiry=c["expiry"],
            instrument_type=c["kind"],
        )
        by_kind_strike[(c["kind"], int(c["strike"]))] = contract
        strikes.add(int(c["strike"]))

    _cache["as_of"] = today
    _cache["expiry"] = nearest_expiry
    _cache["by_kind_strike"] = by_kind_strike
    _cache["available_strikes"] = sorted(strikes)
    log_info(
        f"[OptResolver] cache built: name={name} expiry={nearest_expiry} "
        f"strikes={len(strikes)} contracts={len(by_kind_strike)}"
    )
    return True


def _ensure_cache(*, name: str = "NIFTY") -> bool:
    today = date.today()
    with _cache_lock:
        if _cache.get("as_of") == today and _cache.get("by_kind_strike"):
            return True
        return _build_cache_for_today(name=name)


def _nearest_strike(target: int, available: List[int]) -> Optional[int]:
    if not available:
        return None
    return min(available, key=lambda s: abs(s - target))


def true_atm_strike(spot: float, atm_step: int = 50, available: Optional[List[int]] = None) -> int:
    """Strike closest to live spot (may differ from round(spot/step)*step)."""
    if available:
        return int(min(available, key=lambda s: abs(s - spot)))
    return int(round(float(spot) / max(1, atm_step)) * atm_step)


def strike_for_moneyness(
    spot: float,
    kind: str,
    moneyness: str,
    atm_step: int = 50,
    available: Optional[List[int]] = None,
) -> int:
    """
    Map moneyness label to strike. CE: OTM = higher strike; PE: OTM = lower strike.
    Labels: ATM, ITM1, OTM1, OTM2
    """
    kind = (kind or "").upper()
    atm = true_atm_strike(spot, atm_step, available)
    step = max(1, int(atm_step))
    m = (moneyness or "ATM").upper()
    if m == "ATM":
        return atm
    if kind == "CE":
        if m == "OTM1":
            return atm + step
        if m == "OTM2":
            return atm + 2 * step
        if m == "ITM1":
            return atm - step
    elif kind == "PE":
        if m == "OTM1":
            return atm - step
        if m == "OTM2":
            return atm - 2 * step
        if m == "ITM1":
            return atm + step
    return atm


def estimate_delta_from_spot(
    spot: float,
    strike: int,
    kind: str,
    *,
    vix: Optional[float] = None,
) -> float:
    """Rough live delta from spot–strike distance (Nifty weekly)."""
    if spot <= 0 or strike <= 0:
        return 0.5
    kind = (kind or "CE").upper()
    # CE: ITM when strike < spot; PE: ITM when strike > spot
    moneyness_pts = (float(spot) - float(strike)) if kind == "CE" else (float(strike) - float(spot))
    scale = max(float(spot) * 0.018, 45.0)
    delta = 0.5 + moneyness_pts / scale
    if vix and vix > 22:
        delta *= 0.92
    elif vix and vix < 12:
        delta *= 1.05
    return min(0.88, max(0.12, delta))


# -------------------------- public API --------------------------


def resolve_nifty_contract(
    *,
    spot: float,
    kind: str,
    moneyness: str = "ATM",
    atm_step: int = 50,
    name: str = "NIFTY",
) -> Optional[OptionContract]:
    """Resolve weekly contract for ATM / OTM1 / OTM2 / ITM1 relative to live spot."""
    kind = (kind or "").upper()
    if kind not in ("CE", "PE"):
        return None
    if not _ensure_cache(name=name):
        return None

    strikes: List[int] = list(_cache.get("available_strikes") or [])
    by_ks: Dict[Tuple[str, int], OptionContract] = _cache.get("by_kind_strike") or {}
    target_strike = strike_for_moneyness(spot, kind, moneyness, atm_step, strikes)
    c = by_ks.get((kind, target_strike))
    if c:
        return c
    same_kind_strikes = [s for s in strikes if (kind, s) in by_ks]
    nearest = _nearest_strike(target_strike, same_kind_strikes)
    if nearest is None:
        return None
    return by_ks.get((kind, nearest))


def resolve_nifty_atm_contract(
    *,
    spot: float,
    kind: str,
    atm_step: int = 50,
    name: str = "NIFTY",
) -> Optional[OptionContract]:
    """
    Pick the current-week ATM NIFTY contract closest to `spot`.
    `kind` must be 'CE' or 'PE'. Returns None if cache can't be built or no
    suitable strike is available.
    """
    kind = (kind or "").upper()
    if kind not in ("CE", "PE"):
        return None
    if not _ensure_cache(name=name):
        return None

    return resolve_nifty_contract(
        spot=spot, kind=kind, moneyness="ATM", atm_step=atm_step, name=name
    )


def resolve_nifty_contract_at_strike(
    *,
    strike: int,
    kind: str,
    name: str = "NIFTY",
) -> Optional[OptionContract]:
    """Resolve weekly contract at an exact strike (OI Sentinel anchor)."""
    kind = (kind or "").upper()
    if kind not in ("CE", "PE") or strike <= 0:
        return None
    if not _ensure_cache(name=name):
        return None
    by_ks: Dict[Tuple[str, int], OptionContract] = _cache.get("by_kind_strike") or {}
    c = by_ks.get((kind, int(strike)))
    if c:
        return c
    strikes: List[int] = list(_cache.get("available_strikes") or [])
    nearest = _nearest_strike(int(strike), strikes)
    if nearest is None:
        return None
    return by_ks.get((kind, nearest))


def fetch_option_ltp(contract: OptionContract) -> Optional[float]:
    """Fetch the live last_price for `contract` via kite.quote()."""
    try:
        kite = get_kite_instance()
        key = contract.quote_key()
        q = kite.quote(key) or {}
        row = q.get(key) or {}
        lp = row.get("last_price")
        if lp is None:
            ohlc = row.get("ohlc") or {}
            lp = ohlc.get("close") or ohlc.get("open")
        return float(lp) if lp is not None else None
    except Exception as e:  # noqa: BLE001
        log_warning(f"[OptResolver] quote failed for {contract.tradingsymbol}: {e}")
        return None


@dataclass(frozen=True)
class OptionLegEstimate:
    contract: Optional[OptionContract]
    entry_premium: Optional[float]
    sl_premium: Optional[float]
    target_premium: Optional[float]
    premium_risk: Optional[float]       # entry - SL  (₹/unit)
    premium_reward: Optional[float]     # target - entry (₹/unit)
    risk_inr: Optional[float]           # × lot_size × num_lots
    reward_inr: Optional[float]
    delta_used: float
    estimated: bool                     # True if entry_premium is a fallback estimate
    note: Optional[str] = None

    def to_payload(self) -> Dict[str, str]:
        d: Dict[str, str] = {
            "delta_used": f"{self.delta_used:.2f}",
            "premium_estimated": "1" if self.estimated else "0",
        }
        if self.contract:
            d["tradingsymbol"] = self.contract.tradingsymbol
            d["exchange"] = self.contract.exchange
            d["instrument_token"] = str(self.contract.instrument_token)
            d["strike"] = str(self.contract.strike)
            d["expiry"] = self.contract.expiry.isoformat()
            d["option_kind"] = self.contract.instrument_type
        if self.entry_premium is not None:
            d["entry_premium"] = f"{self.entry_premium:.2f}"
        if self.sl_premium is not None:
            d["sl_premium"] = f"{self.sl_premium:.2f}"
        if self.target_premium is not None:
            d["target_premium"] = f"{self.target_premium:.2f}"
        if self.premium_risk is not None:
            d["premium_risk"] = f"{self.premium_risk:.2f}"
        if self.premium_reward is not None:
            d["premium_reward"] = f"{self.premium_reward:.2f}"
        if self.risk_inr is not None:
            d["risk_inr"] = f"{self.risk_inr:.2f}"
        if self.reward_inr is not None:
            d["reward_inr"] = f"{self.reward_inr:.2f}"
        if self.note:
            d["premium_note"] = self.note
        return d


def estimate_option_leg(
    *,
    spot_entry: float,
    spot_stop_loss: float,
    spot_target: float,
    kind: str,
    delta: float,
    lot_size: int,
    num_lots: int,
    atm_step: int = 50,
    name: str = "NIFTY",
    moneyness: str = "ATM",
    vix: Optional[float] = None,
    use_live_delta: bool = True,
) -> OptionLegEstimate:
    """
    Resolve contract by moneyness (ATM/OTM1/…), fetch live LTP, and project
    SL/target premium using live delta when `use_live_delta` is True.
    """
    spot_risk = abs(spot_entry - spot_stop_loss)
    spot_reward = abs(spot_target - spot_entry)
    qty = max(1, int(lot_size)) * max(1, int(num_lots))

    contract = resolve_nifty_contract(
        spot=spot_entry, kind=kind, moneyness=moneyness, atm_step=atm_step, name=name
    )

    if contract is not None and use_live_delta:
        delta = estimate_delta_from_spot(spot_entry, contract.strike, kind, vix=vix)
    else:
        delta = min(0.99, max(0.05, float(delta)))

    entry_prem: Optional[float] = None
    estimated = False
    note: Optional[str] = None

    if contract is not None:
        entry_prem = fetch_option_ltp(contract)
        if entry_prem is None:
            estimated = True
            entry_prem = max(1.0, 0.007 * float(spot_entry))
            note = "live premium unavailable; entry estimated at ~0.7% × spot"
    else:
        estimated = True
        entry_prem = max(1.0, 0.007 * float(spot_entry))
        note = f"{moneyness} contract not resolvable (instruments cache); premium estimated"

    sl_prem = max(0.05, entry_prem - spot_risk * delta)
    target_prem = entry_prem + spot_reward * delta
    prem_risk = entry_prem - sl_prem
    prem_reward = target_prem - entry_prem
    risk_inr = prem_risk * qty
    reward_inr = prem_reward * qty

    return OptionLegEstimate(
        contract=contract,
        entry_premium=entry_prem,
        sl_premium=sl_prem,
        target_premium=target_prem,
        premium_risk=prem_risk,
        premium_reward=prem_reward,
        risk_inr=risk_inr,
        reward_inr=reward_inr,
        delta_used=delta,
        estimated=estimated,
        note=note,
    )


def cache_state() -> Dict[str, Any]:
    """For debugging / UI."""
    with _cache_lock:
        return {
            "as_of": _cache["as_of"].isoformat() if _cache.get("as_of") else None,
            "expiry": _cache["expiry"].isoformat() if _cache.get("expiry") else None,
            "strikes": len(_cache.get("available_strikes") or []),
            "contracts": len(_cache.get("by_kind_strike") or {}),
        }
