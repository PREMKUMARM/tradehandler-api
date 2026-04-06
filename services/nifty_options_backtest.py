"""
Nifty options backtest using Kite historical data.

For each session day, uses the **front weekly expiry that was listed on/after that day**
(contracts that have since expired when you backtest past dates). If OHLC for that token is
unavailable, tries subsequent expiries for the same ATM strike until data returns.

PnL uses a simplified return model — see `pnl_model` in the response.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from simulation.helpers import find_option
from services.nifty_option_chain import build_nifty_options_universe, nifty50_index_token
from strategies.runner import run_strategy_on_candles
from utils.logger import log_error, log_info

_IST = ZoneInfo("Asia/Kolkata")


def _candle_dt_ist(candle: dict) -> Optional[datetime]:
    """Kite may return tz-aware UTC or naive IST; normalize to Asia/Kolkata."""
    t = candle.get("date")
    if not isinstance(t, datetime):
        return None
    if t.tzinfo is not None:
        return t.astimezone(_IST)
    return t.replace(tzinfo=_IST)


def _filter_session(candles: List[dict]) -> List[dict]:
    out = []
    for c in candles or []:
        dt_ist = _candle_dt_ist(c)
        if not dt_ist:
            continue
        tt = dt_ist.time()
        if time(9, 15) <= tt <= time(15, 30):
            out.append(c)
    return out


def _pick_first_candle(candles: List[dict]) -> Optional[dict]:
    for c in candles:
        dt_ist = _candle_dt_ist(c)
        if not dt_ist:
            continue
        tt = dt_ist.time()
        if time(9, 15) <= tt < time(9, 20):
            return c
    return candles[0] if candles else None


def _expiries_on_or_after(universe: List[dict], trade_date: date) -> List[date]:
    s = {o["expiry"] for o in universe if o.get("expiry") and o["expiry"] >= trade_date}
    return sorted(s)


def _choose_expiry_for_day(expiries: List[date], contract_selection: str) -> Optional[date]:
    if not expiries:
        return None
    if contract_selection == "next_week" and len(expiries) >= 2:
        return expiries[1]
    return expiries[0]


def _ranked_ce_for_history(
    universe: List[dict],
    trade_date: date,
    strike: int,
    preferred_expiry: date,
    max_candidates: int = 12,
) -> List[dict]:
    """CE instruments at strike, expiries >= trade_date; preferred_expiry tried first, then later expiries."""
    cands: List[dict] = []
    for o in universe:
        if (
            o.get("strike") == strike
            and o.get("instrument_type") == "CE"
            and o.get("expiry")
            and o["expiry"] >= trade_date
            and o.get("instrument_token")
        ):
            cands.append(o)

    def sort_key(o: dict) -> Tuple[int, date, str]:
        exp = o["expiry"]
        primary = 0 if exp == preferred_expiry else 1
        return (primary, exp, o.get("tradingsymbol") or "")

    cands.sort(key=sort_key)
    out: List[dict] = []
    seen_exp: set = set()
    for o in cands:
        e = o["expiry"]
        if e in seen_exp:
            continue
        seen_exp.add(e)
        out.append(o)
        if len(out) >= max_candidates:
            break
    return out


def _pe_same_expiry(universe: List[dict], strike: int, expiry_d: date) -> Optional[dict]:
    for o in universe:
        if (
            o.get("strike") == strike
            and o.get("instrument_type") == "PE"
            and o.get("expiry") == expiry_d
            and o.get("instrument_token")
        ):
            return o
    return None


async def run_nifty50_options_backtest_async(
    kite,
    start_date: date,
    end_date: date,
    strategy_type: str,
    fund: float,
    risk_pct: float,
    reward_pct: float,
    contract_selection: str = "front_week",
) -> Dict[str, Any]:
    trades: List[Dict[str, Any]] = []
    wins = losses = 0
    total_pnl = 0.0

    d = start_date
    universe_full = await asyncio.to_thread(build_nifty_options_universe, kite)
    idx_token = await asyncio.to_thread(nifty50_index_token, kite)

    diag: Dict[str, int] = {
        "weekend_days_skipped": 0,
        "no_index_minute_bars": 0,
        "no_first_candle": 0,
        "invalid_index_close": 0,
        "no_expiry_in_master": 0,
        "strategy_no_signal": 0,
        "days_with_trade": 0,
    }

    while d <= end_date:
        if d.weekday() >= 5:
            diag["weekend_days_skipped"] += 1
            d += timedelta(days=1)
            continue

        try:
            from_dt = d
            to_dt = d + timedelta(days=1)
            raw = await asyncio.to_thread(kite.historical_data, idx_token, from_dt, to_dt, "minute")
            candles = _filter_session(list(raw or []))
            if len(candles) < 5:
                diag["no_index_minute_bars"] += 1
                d += timedelta(days=1)
                continue

            first_candle = _pick_first_candle(candles)
            if not first_candle:
                diag["no_first_candle"] += 1
                d += timedelta(days=1)
                continue

            nifty_price = float(candles[-1].get("close") or 0)
            if nifty_price <= 0:
                diag["invalid_index_close"] += 1
                d += timedelta(days=1)
                continue

            strike = round(nifty_price / 50) * 50

            expiries = _expiries_on_or_after(universe_full, d)
            fe = _choose_expiry_for_day(expiries, contract_selection)
            if not fe:
                diag["no_expiry_in_master"] += 1
                log_error(f"[NiftyBacktest] {d}: no NIFTY expiry in instrument master on/after this date")
                d += timedelta(days=1)
                continue

            nifty_options = [o for o in universe_full if o.get("expiry") == fe]
            atm_ce = find_option(nifty_options, strike, "CE", d)
            atm_pe = find_option(nifty_options, strike, "PE", d)
            if not atm_ce or not atm_pe:
                atm_ce = {
                    "tradingsymbol": f"NIFTY-CE-PROXY-{strike}",
                    "strike": strike,
                    "instrument_type": "CE",
                    "expiry": fe,
                }
                atm_pe = {
                    "tradingsymbol": f"NIFTY-PE-PROXY-{strike}",
                    "strike": strike,
                    "instrument_type": "PE",
                    "expiry": fe,
                }

            sym_candles = candles
            oi_note = "index"
            ce_used: Optional[dict] = None
            ranked = _ranked_ce_for_history(universe_full, d, strike, fe)
            for cand in ranked:
                try:
                    oc = await asyncio.to_thread(
                        kite.historical_data, int(cand["instrument_token"]), from_dt, to_dt, "minute"
                    )
                    oc = _filter_session(list(oc or []))
                    if len(oc) >= 5:
                        sym_candles = oc
                        oi_note = "option"
                        ce_used = cand
                        atm_ce = cand
                        pe_alt = _pe_same_expiry(universe_full, strike, cand["expiry"])
                        if pe_alt:
                            atm_pe = pe_alt
                        break
                except Exception:
                    continue

            date_str = d.strftime("%Y-%m-%d")
            result = await run_strategy_on_candles(
                kite,
                strategy_type,
                sym_candles,
                first_candle,
                nifty_price,
                strike,
                atm_ce,
                atm_pe,
                date_str,
                nifty_options,
                d,
            )

            if not result:
                diag["strategy_no_signal"] += 1
            if result:
                diag["days_with_trade"] += 1
                exit_src = sym_candles if (oi_note == "option" and sym_candles) else candles
                exit_bar = exit_src[-1] if exit_src else {}
                exit_i = float(exit_bar.get("close") or 0)
                entry_i = float(result.get("entry_price") or 0)
                trend = result.get("trend") or "BULLISH"
                if entry_i > 0 and exit_i > 0:
                    move = (exit_i - entry_i) / entry_i if trend == "BULLISH" else (entry_i - exit_i) / entry_i
                    pnl = move * float(fund) * (float(risk_pct) / 100.0)
                else:
                    pnl = 0.0

                total_pnl += pnl
                if pnl >= 0:
                    wins += 1
                else:
                    losses += 1

                opt_trade = result.get("option_to_trade") or {}
                sym = opt_trade.get("tradingsymbol")
                pnl_r = round(pnl, 2)
                opt_type = result.get("option_type") or ("CE" if trend == "BULLISH" else "PE")
                trades.append(
                    {
                        "date": date_str,
                        "nifty_price": round(nifty_price, 2),
                        "trend": trend,
                        "strike": strike,
                        "symbol": sym,
                        "tradingsymbol": sym,
                        "option_type": opt_type,
                        "option_expiry": str(fe),
                        "historical_ce_symbol": ce_used.get("tradingsymbol") if ce_used else None,
                        "contract_selection": contract_selection,
                        "entry_price": entry_i,
                        "exit_price": exit_i,
                        "pnl": pnl_r,
                        "pnl_estimated": pnl_r,
                        "status": "WIN" if pnl_r >= 0 else "LOSS",
                        "reason": result.get("reason"),
                        "candle_source": oi_note,
                    }
                )
        except Exception as e:
            log_error(f"[NiftyBacktest] day {d}: {e}")

        d += timedelta(days=1)

    n = len(trades)
    warnings: List[str] = []
    if n == 0:
        warnings.append(
            "No trades generated. Check day_diagnostics: empty index candles usually means the date range "
            "has no Kite data (future dates, holidays, or API limits). strategy_no_signal means data existed "
            "but the strategy did not fire each day."
        )
    if diag.get("no_index_minute_bars", 0) > 0:
        warnings.append(
            f"{diag['no_index_minute_bars']} weekdays had fewer than 5 Nifty 50 minute bars after IST session filter."
        )

    backtest_results = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "strategy_type": strategy_type,
        "fund": fund,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "contract_selection": contract_selection,
        "total_trading_days_computed": n,
        "day_diagnostics": diag,
        "warnings": warnings,
        "trades": trades,
        "statistics": {
            "total_trades": n,
            "winning_trades": wins,
            "losing_trades": losses,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(100.0 * wins / n, 2) if n else 0.0,
        },
        "pnl_model": "simplified_return_risk_scaled",
        "disclaimer": (
            "Uses NIFTY index + option minute candles from Kite when available. "
            "Each day targets the front listed weekly expiry on/after that date (since-expired contracts for past dates). "
            "Fills, rolls, and brokerage are not exchange-accurate."
        ),
    }
    log_info(f"[NiftyBacktest] completed {n} signal days for {strategy_type} ({contract_selection})")
    return {"data": backtest_results}
