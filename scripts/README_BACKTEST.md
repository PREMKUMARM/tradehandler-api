# Binance Futures Backtest Script

A standalone Python script to backtest the VWAP strategy on Binance Futures.

## Usage

### Option 1: Using config.json (Recommended)

1. Edit `config.json` with your parameters:
```json
{
    "startDateStr": "2025-12-03",
    "endDateStr": "2026-01-03",
    "timeframe": "5minute",
    "symbols": [
        "1000PEPEUSDT",
        "ETHUSDT",
        "XRPUSDT",
        "SOLUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "MATICUSDT"
    ]
}
```

2. Run the script:
```bash
cd tradehandler-api/scripts
python backtest_binance_futures.py
```

### Option 2: Using command line arguments

```bash
python backtest_binance_futures.py --config custom_config.json
```

### Option 3: Using default parameters

The script will use default parameters if no config file is found.

## Parameters

- **startDateStr**: Start date in YYYY-MM-DD format
- **endDateStr**: End date in YYYY-MM-DD format
- **timeframe**: One of: `1minute`, `5minute`, `15minute`, `30minute`, `60minute`, `4hour`, `day`
- **symbols**: Array of Binance Futures symbols (e.g., `["ETHUSDT", "SOLUSDT"]`)

## Output

The script will:

1. **Print progress** for each symbol as it processes
2. **Display results** for each symbol:
   - Total signals found
   - Total profit/loss
   - Win rate
   - Number of profitable vs losing trades
3. **Print summary** at the end:
   - Total instruments tested
   - Total signals across all symbols
   - Total profit/loss
   - Average profit per signal
   - Number of profitable vs losing instruments
4. **Save results** to a JSON file: `backtest_results_{start_date}_{end_date}.json`

## Output File Structure

```json
{
  "config": { ... },
  "summary": {
    "total_instruments": 8,
    "total_signals": 45,
    "total_profit": 1234.56,
    "avg_profit_per_signal": 27.43,
    "profitable_instruments": 5,
    "loss_instruments": 3,
    "test_period": {
      "start_date": "2025-12-03",
      "end_date": "2026-01-03",
      "trading_days": 31
    }
  },
  "results": [
    {
      "instrument": "ETHUSDT",
      "total_signals": 5,
      "total_profit": 234.56,
      "avg_profit": 46.91,
      "win_rate": 60.0,
      "profitable_signals": 3,
      "loss_signals": 2,
      "orders": [ ... ]
    },
    ...
  ]
}
```

## Strategy Details

The backtest uses the same VWAP strategy logic as the live system:

- **Entry Signal**: BUY after "Three Black Crows" pattern followed by bullish reversal pattern
- **Entry Conditions**:
  - Previous candle must be "Three Black Crows"
  - Current candle must be green and above VWAP
  - Current candle must match: Dragonfly Doji, Piercing Pattern, Inverted Hammer, or Long White Candle
  - Price within 2% of VWAP
  - RSI between 30-65 with specific conditions
  - MACD bullish (MACD > Signal)
  - Volume confirmation
  
- **Exit Conditions**:
  - Stop-loss: 1.5% below entry
  - Trailing stop: 1.0% from highest price after entry
  - Early exit: If price drops 0.5% below entry after 2+ candles
  - End of period: If no exit condition is met

## Requirements

- Python 3.8+
- pandas
- numpy
- httpx
- All dependencies from `requirements.txt`

## Example

```bash
cd tradehandler-api/scripts
python backtest_binance_futures.py
```

Output:
```
============================================================
BINANCE FUTURES BACKTEST
============================================================
Start Date: 2025-12-03
End Date: 2026-01-03
Timeframe: 5minute (5m)
Symbols: 1000PEPEUSDT, ETHUSDT, XRPUSDT, SOLUSDT, ADAUSDT, DOGEUSDT, MATICUSDT
Total Symbols: 8
============================================================

============================================================
Processing ETHUSDT...
============================================================
Fetching historical data for ETHUSDT...
Fetched 8928 candles for ETHUSDT
Calculating VWAP...
Calculating RSI...
Calculating MACD...
Detecting candlestick patterns...
Generating trading signals...
Calculating P&L for signals...

ETHUSDT Results:
  Total Signals: 5
  Total Profit: $234.56
  Win Rate: 60.00%
  Profitable: 3, Losses: 2

...

============================================================
BACKTEST SUMMARY
============================================================
Total Instruments: 8
Total Signals: 45
Total Profit: $1234.56
Avg Profit per Signal: $27.43
Profitable Instruments: 5
Loss Instruments: 3
Test Period: 2025-12-03 to 2026-01-03 (31 days)
============================================================

Results saved to: backtest_results_2025-12-03_2026-01-03.json

Backtest completed!
```

