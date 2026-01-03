"""
Test strategy on multiple time periods to validate consistency
"""
import json
import asyncio
import sys
import os
from backtest_binance_futures import backtest_symbol

async def test_period(start_date: str, end_date: str, period_name: str):
    """Test a specific time period"""
    print(f"\n{'='*70}")
    print(f"TESTING PERIOD: {period_name}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"{'='*70}\n")
    
    # Update config
    config = {
        "startDateStr": start_date,
        "endDateStr": end_date,
        "timeframe": "5minute",
        "symbols": ["1000PEPEUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "MATICUSDT"]
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Run backtest
    binance_interval = "5m"
    config_data = config
    
    results = []
    total_signals = 0
    total_profit = 0.0
    
    for symbol in config_data['symbols']:
        try:
            result = await backtest_symbol(symbol, config_data['startDateStr'], config_data['endDateStr'], binance_interval)
            if result.get('total_signals', 0) > 0:
                results.append(result)
                total_signals += result['total_signals']
                total_profit += result['total_profit']
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PERIOD SUMMARY: {period_name}")
    print(f"{'='*70}")
    print(f"Total Signals: {total_signals}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Avg Profit per Signal: ${total_profit/total_signals:.2f}" if total_signals > 0 else "Avg Profit: $0.00")
    
    profitable_count = sum(1 for r in results if r.get('total_profit', 0) > 0)
    print(f"Profitable Symbols: {profitable_count}/{len(results)}")
    print()
    
    return {
        'period': period_name,
        'start_date': start_date,
        'end_date': end_date,
        'total_signals': total_signals,
        'total_profit': total_profit,
        'avg_profit_per_signal': total_profit/total_signals if total_signals > 0 else 0,
        'profitable_symbols': profitable_count,
        'total_symbols': len(results)
    }

async def main():
    """Test multiple periods"""
    periods = [
        {'start': '2025-12-03', 'end': '2026-01-03', 'name': 'Dec 2025 - Jan 2026'},
        {'start': '2025-11-01', 'end': '2025-11-30', 'name': 'November 2025'},
        {'start': '2025-10-01', 'end': '2025-10-31', 'name': 'October 2025'},
        {'start': '2025-09-01', 'end': '2025-09-30', 'name': 'September 2025'},
    ]
    
    all_results = []
    
    for period in periods:
        result = await test_period(period['start'], period['end'], period['name'])
        all_results.append(result)
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL CONSISTENCY ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"{'Period':<25} {'Signals':<10} {'Profit':<15} {'Avg/Signal':<12} {'Profitable':<12}")
    print('-'*70)
    
    total_all_signals = 0
    total_all_profit = 0.0
    
    for result in all_results:
        total_all_signals += result['total_signals']
        total_all_profit += result['total_profit']
        print(f"{result['period']:<25} {result['total_signals']:<10} ${result['total_profit']:>12.2f}  ${result['avg_profit_per_signal']:>10.2f}  {result['profitable_symbols']}/{result['total_symbols']}")
    
    print('-'*70)
    print(f"{'TOTAL':<25} {total_all_signals:<10} ${total_all_profit:>12.2f}  ${total_all_profit/total_all_signals:>10.2f}" if total_all_signals > 0 else f"{'TOTAL':<25} {total_all_signals:<10} ${total_all_profit:>12.2f}  $0.00")
    print()
    
    # Consistency metrics
    profits = [r['total_profit'] for r in all_results]
    avg_profit = sum(profits) / len(profits) if profits else 0
    profit_std = (sum((p - avg_profit)**2 for p in profits) / len(profits))**0.5 if len(profits) > 1 else 0
    
    print(f"Average Profit per Period: ${avg_profit:.2f}")
    print(f"Profit Std Deviation: ${profit_std:.2f}")
    print(f"Consistency Score: {'High' if profit_std < abs(avg_profit) * 0.5 else 'Medium' if profit_std < abs(avg_profit) else 'Low'}")
    print()
    
    profitable_periods = sum(1 for r in all_results if r['total_profit'] > 0)
    print(f"Profitable Periods: {profitable_periods}/{len(all_results)} ({profitable_periods/len(all_results)*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())

