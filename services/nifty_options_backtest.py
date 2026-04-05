"""
P0: Nifty options-oriented backtest (real Kite historical calls).
Uses nearest weekly expiry >= trade date; prefers option OHLC when contract exists, else index proxy.
PnL is a simplified return model — see `pnl_model` in response (not a substitute for exchange fills).
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, time
from typing import Any, Dict, List, Optional

from simulation.helpers import find_option
from services.nifty_option_chain import build_nifty_options_universe, nifty50_index_token
from strategies.runner import run_strategy_on_candles
from utils.logger import log_info, log_error


def _filter_session(candles: List[dict]) -> List[dict]:
    out = []
    for c in candles or []:
        t = c.get("date")
        if isinstance(t, datetime):
            tt = t.time()
            if time(9, 15) <= tt <= time(15, 30):
                out.append(c)
    return out


def _pick_first_candle(candles: List[dict]) -> Optional[dict]:
    for c in candles:
        t = c.get("date")
        if isinstance(t, datetime):
            tt = t.time()
            if time(9, 15) <= tt < time(9, 20):
                return c
    return candles[0] if candles else None


async def run_nifty50_options_backtest_async(
    kite,
    start_date: date,
    end_date: date,
    strategy_type: str,
    fund: float,
    risk_pct: float,
    reward_pct: float,
) -> Dict[str, Any]:
    trades: List[Dict[str, Any]] = []
    wins = losses = 0
    total_pnl = 0.0

    d = start_date
    universe_full = await asyncio.to_thread(build_nifty_options_universe, kite)
    idx_token = await asyncio.to_thread(nifty50_index_token, kite)

    while d <= end_date:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue

        try:
            from_dt = d
            to_dt = d + timedelta(days=1)
            raw = await asyncio.to_thread(kite.historical_data, idx_token, from_dt, to_dt, "minute")
            candles = _filter_session(list(raw or []))
            if len(candles) < 5:
                d += timedelta(days=1)
                continue

            first_candle = _pick_first_candle(candles)
            if not first_candle:
                d += timedelta(days=1)
                continue

            nifty_price = float(candles[-1].get("close") or 0)
            if nifty_price <= 0:
                d += timedelta(days=1)
                continue

            strike = round(nifty_price / 50) * 50
            nifty_options = [o for o in universe_full if o.get("expiry") and o["expiry"] >= d]

            atm_ce = find_option(nifty_options, strike, "CE", d)
            atm_pe = find_option(nifty_options, strike, "PE", d)
            if not atm_ce or not atm_pe:
                atm_ce = {
                    "tradingsymbol": f"NIFTY-CE-PROXY-{strike}",
                    "strike": strike,
                    "instrument_type": "CE",
                    "expiry": d,
                }
                atm_pe = {
                    "tradingsymbol": f"NIFTY-PE-PROXY-{strike}",
                    "strike": strike,
                    "instrument_type": "PE",
                    "expiry": d,
                }

            sym_candles = candles
            oi_note = "index"
            if atm_ce.get("instrument_token"):
                try:
                    oc = await asyncio.to_thread(
                        kite.historical_data, int(atm_ce["instrument_token"]), from_dt, to_dt, "minute"
                    )
                    oc = _filter_session(list(oc or []))
                    if len(oc) >= 5:
                        sym_candles = oc
                        oi_note = "option"
                except Exception:
                    pass

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

            if result:
                entry_i = float(result.get("entry_price") or 0)
                exit_i = float(candles[-1].get("close") or 0)
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

                trades.append(
                    {
                        "date": date_str,
                        "trend": trend,
                        "strike": strike,
                        "symbol": (result.get("option_to_trade") or {}).get("tradingsymbol"),
                        "entry_price": entry_i,
                        "exit_price": exit_i,
                        "pnl_estimated": round(pnl, 2),
                        "reason": result.get("reason"),
                        "candle_source": oi_note,
                    }
                )
        except Exception as e:
            log_error(f"[NiftyBacktest] day {d}: {e}")

        d += timedelta(days=1)

    n = len(trades)
    backtest_results = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "strategy_type": strategy_type,
        "fund": fund,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "total_trading_days_computed": n,
        "trades": trades,
        "statistics": {
            "total_trades": n,
            "winning_trades": wins,
            "losing_trades": losses,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(100.0 * wins / n, 2) if n else 0.0,
        },
        "pnl_model": "simplified_return_risk_scaled",
        "disclaimer": "Estimates use historical Kite data; fills, costs, and weekly contract rolls are not exchange-accurate. Validate before live deployment.",
    }
    log_info(f"[NiftyBacktest] completed {n} signal days for {strategy_type}")
    return {"data": backtest_results}
