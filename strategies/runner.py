"""
Strategy runner - executes strategies on candle data
"""
from .implementations import (
    strategy_915_candle_break,
    strategy_mean_reversion_bollinger,
    strategy_momentum_breakout,
    strategy_support_resistance_breakout,
    strategy_long_straddle,
    strategy_long_strangle,
    strategy_bull_call_spread,
    strategy_bear_put_spread,
    strategy_iron_condor,
    strategy_macd_crossover,
    strategy_rsi_reversal,
    strategy_ema_cross
)


async def run_strategy_on_candles(kite, strategy_type, trading_candles, first_candle, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date):
    """Unified function to run any strategy on a set of candles"""
    
    if strategy_type == "915_candle_break":
        return strategy_915_candle_break(kite, trading_candles, first_candle, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "mean_reversion":
        return strategy_mean_reversion_bollinger(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "momentum_breakout":
        return strategy_momentum_breakout(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "support_resistance":
        return strategy_support_resistance_breakout(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "long_straddle":
        return strategy_long_straddle(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "long_strangle":
        return strategy_long_strangle(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "bull_call_spread":
        return strategy_bull_call_spread(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "bear_put_spread":
        return strategy_bear_put_spread(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "iron_condor":
        return strategy_iron_condor(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "macd_crossover":
        return strategy_macd_crossover(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "rsi_reversal":
        return strategy_rsi_reversal(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "ema_cross":
        return strategy_ema_cross(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    return None





