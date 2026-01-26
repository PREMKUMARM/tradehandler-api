# TradingView-Style Chart Replay Implementation Guide

## Overview
This document outlines the technical implementation details for creating a TradingView-style chart replay feature with indicators and multiple timeframes, using Python (FastAPI) backend and Angular frontend.

## Backend Implementation (Python/FastAPI)

### 1. Data Aggregation for Multiple Timeframes

**Key Concept**: Aggregate 1-minute data to higher timeframes using pandas resampling.

```python
import pandas as pd
from datetime import datetime

def aggregate_to_timeframe(candles_1m, timeframe_minutes):
    """
    Aggregate 1-minute candles to specified timeframe.
    
    Args:
        candles_1m: List of 1-minute candle dictionaries
        timeframe_minutes: Target timeframe in minutes (5, 15, 30, 60)
    
    Returns:
        List of aggregated candles
    """
    if not candles_1m:
        return []
    
    # Convert to DataFrame
    df = pd.DataFrame(candles_1m)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Resample to target timeframe
    # OHLC aggregation rules:
    # - Open: First value in period
    # - High: Maximum value in period
    # - Low: Minimum value in period
    # - Close: Last value in period
    # - Volume: Sum of volumes in period
    resampled = df.resample(f'{timeframe_minutes}T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Convert back to list of dictionaries
    aggregated = []
    for idx, row in resampled.iterrows():
        aggregated.append({
            'date': idx,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        })
    
    return aggregated
```

### 2. Indicator Calculation Best Practices

**RSI (Relative Strength Index) - Wilder's Smoothing Method**:
```python
import numpy as np

def calculate_rsi_wilders(closes, period=14):
    """
    Calculate RSI using Wilder's Smoothing Method (matches TradingView).
    
    Formula:
    - First Average Gain/Loss = Simple Average of first 'period' gains/losses
    - Subsequent Average = (Previous Average * (period-1) + Current) / period
    - RS = Average Gain / Average Loss
    - RSI = 100 - (100 / (1 + RS))
    """
    if len(closes) < period + 1:
        return [np.nan] * len(closes)
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gains = np.full(len(gains), np.nan)
    avg_losses = np.full(len(losses), np.nan)
    
    # First average: Simple average
    avg_gains[period - 1] = np.mean(gains[:period])
    avg_losses[period - 1] = np.mean(losses[:period])
    
    # Wilder's smoothing for remaining values
    for i in range(period, len(gains)):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period
    
    # Calculate RS and RSI
    rs = np.divide(avg_gains, avg_losses, out=np.full_like(avg_gains, np.nan), where=(avg_losses != 0))
    rsi = 100 - (100 / (1 + rs))
    
    # Return with first value as NaN (no delta available)
    return [np.nan] * period + rsi[period - 1:].tolist()
```

**Bollinger Bands - Population Standard Deviation**:
```python
def calculate_bollinger_bands(closes, period=20, num_std=2):
    """
    Calculate Bollinger Bands (matches TradingView default).
    
    Uses Population Standard Deviation (ddof=0) not Sample (ddof=1).
    """
    df = pd.DataFrame({'close': closes})
    df['sma'] = df['close'].rolling(window=period, min_periods=1).mean()
    # Population std (ddof=0) - matches TradingView
    df['std'] = df['close'].rolling(window=period, min_periods=1).std(ddof=0)
    df['upper'] = df['sma'] + (df['std'] * num_std)
    df['lower'] = df['sma'] - (df['std'] * num_std)
    
    return df['upper'].tolist(), df['sma'].tolist(), df['lower'].tolist()
```

**Pivot Points - Traditional Method**:
```python
def calculate_pivot_points(high, low, close):
    """
    Calculate Traditional Pivot Points (matches TradingView).
    
    Formulas:
    - Pivot = (High + Low + Close) / 3
    - R1 = 2 * Pivot - Low
    - R2 = Pivot + (High - Low)
    - S1 = 2 * Pivot - High
    - S2 = Pivot - (High - Low)
    """
    pivot = (high + low + close) / 3
    return {
        'pivot': float(pivot),
        'r1': float(2 * pivot - low),
        'r2': float(pivot + (high - low)),
        'r3': float(high + 2 * (pivot - low)),
        's1': float(2 * pivot - high),
        's2': float(pivot - (high - low)),
        's3': float(low - 2 * (high - pivot))
    }
```

### 3. API Endpoint Structure

```python
@app.get("/simulation/chart-data")
async def get_simulation_chart_data(
    timeframe: str = "1m",
    indicators: str = "",
    current_index: int = None
):
    """
    Get chart data with indicators for replay.
    
    Args:
        timeframe: 1m, 5m, 15m, 30m, 1h
        indicators: Comma-separated list (e.g., "rsi,bollinger,pivot")
        current_index: Current position in simulation (for replay)
    
    Returns:
        {
            "candles": [...],  # OHLC data with timestamps
            "indicators": {
                "rsi": [...],  # RSI values with timestamps
                "bollinger": [...],  # BB data with timestamps
                "pivot": {...}  # Pivot points (horizontal lines)
            },
            "current_index": int,
            "total_candles": int
        }
    """
    # 1. Get base candles (1-minute)
    candles_1m = get_historical_candles()
    
    # 2. Slice to current_index for replay
    if current_index is not None:
        candles_1m = candles_1m[:current_index + 1]
    
    # 3. Aggregate to requested timeframe
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}[timeframe]
    aggregated = aggregate_to_timeframe(candles_1m, tf_minutes)
    
    # 4. Calculate indicators on aggregated data
    indicator_data = {}
    indicator_list = [i.strip() for i in indicators.split(",") if i.strip()]
    
    closes = [c["close"] for c in aggregated]
    highs = [c["high"] for c in aggregated]
    lows = [c["low"] for c in aggregated]
    
    if "rsi" in indicator_list:
        rsi_values = calculate_rsi_wilders(closes, period=14)
        indicator_data["rsi"] = [
            {"time": int(c["date"].timestamp()), "value": rsi}
            for c, rsi in zip(aggregated, rsi_values)
            if not pd.isna(rsi)
        ]
    
    if "bollinger" in indicator_list:
        upper, middle, lower = calculate_bollinger_bands(closes, period=20, num_std=2)
        indicator_data["bollinger"] = [
            {
                "time": int(c["date"].timestamp()),
                "upper": u, "middle": m, "lower": l
            }
            for c, u, m, l in zip(aggregated, upper, middle, lower)
            if not (pd.isna(u) or pd.isna(m) or pd.isna(l))
        ]
    
    if "pivot" in indicator_list:
        # Use previous day's H/L/C for pivot calculation
        prev_high = max(highs[:20]) if len(highs) >= 20 else max(highs)
        prev_low = min(lows[:20]) if len(lows) >= 20 else min(lows)
        prev_close = closes[0]
        
        pivot_points = calculate_pivot_points(prev_high, prev_low, prev_close)
        first_time = int(aggregated[0]["date"].timestamp())
        last_time = int(aggregated[-1]["date"].timestamp())
        
        indicator_data["pivot"] = {
            "time_start": first_time,
            "time_end": last_time,
            **pivot_points
        }
    
    # 5. Format candles with timestamps
    chart_candles = [
        {
            "time": int(c["date"].timestamp()),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": int(c.get("volume", 0))
        }
        for c in aggregated
    ]
    
    return {
        "data": {
            "candles": chart_candles,
            "indicators": indicator_data,
            "current_index": current_index or len(candles_1m) - 1,
            "total_candles": len(candles_1m)
        }
    }
```

## Frontend Implementation (Angular)

### 1. ApexCharts Configuration for Replay

**Key Points**:
- Use `updateOptions()` instead of `updateSeries()` for smooth updates
- Sort data by timestamp before rendering
- Use proper datetime formatting for x-axis
- Configure multiple y-axes for indicators

```typescript
export class SimulationReplayComponent {
  chartOptions: Partial<ChartOptions> = {
    series: [{
      name: 'Nifty 50',
      type: 'candlestick',
      data: []
    }],
    chart: {
      type: 'candlestick',
      height: 600,
      animations: {
        enabled: true,
        easing: 'easeinout',
        speed: 800
      },
      toolbar: {
        show: true,
        tools: {
          download: true,
          selection: true,
          zoom: true,
          zoomin: true,
          zoomout: true,
          pan: true,
          reset: true
        }
      }
    },
    xaxis: {
      type: 'datetime',
      labels: {
        datetimeUTC: false,
        format: 'HH:mm'
      },
      tooltip: {
        enabled: true
      }
    },
    yaxis: {
      title: { text: 'Price' },
      opposite: false,
      tooltip: { enabled: true }
    },
    tooltip: {
      shared: true,
      intersect: false,
      y: {
        formatter: (val: number, opts: any) => {
          if (opts?.seriesIndex === 0) {
            // Candlestick tooltip
            const candle = opts.w.globals.seriesCandleO[opts.dataPointIndex];
            if (candle && candle.length === 4) {
              return `O: ${candle[0].toFixed(2)}<br/>H: ${candle[1].toFixed(2)}<br/>L: ${candle[2].toFixed(2)}<br/>C: ${candle[3].toFixed(2)}`;
            }
          }
          return val ? val.toFixed(2) : '0.00';
        }
      }
    }
  };
}
```

### 2. Chart Update Method (Smooth Replay)

```typescript
updateChart(data: any): void {
  if (!data?.candles || data.candles.length === 0) return;
  
  // 1. Convert and sort candles by timestamp
  const chartData = data.candles
    .map((candle: any) => {
      const time = typeof candle.time === 'number' 
        ? (candle.time < 946684800000 ? candle.time * 1000 : candle.time)
        : new Date(candle.time).getTime();
      
      return {
        x: time,
        y: [
          parseFloat(candle.open) || 0,
          parseFloat(candle.high) || 0,
          parseFloat(candle.low) || 0,
          parseFloat(candle.close) || 0
        ]
      };
    })
    .sort((a: any, b: any) => a.x - b.x); // Critical: Sort by timestamp
  
  // 2. Build series array
  const series: any[] = [{
    name: 'Nifty 50',
    type: 'candlestick',
    data: chartData
  }];
  
  const yAxes: any[] = [{
    title: { text: 'Price' },
    opposite: false,
    tooltip: { enabled: true }
  }];
  
  // 3. Add indicators
  if (data.indicators?.rsi && this.selectedIndicators.includes('rsi')) {
    const rsiData = data.indicators.rsi
      .map((rsi: any) => ({
        x: typeof rsi.time === 'number' 
          ? (rsi.time < 946684800000 ? rsi.time * 1000 : rsi.time)
          : new Date(rsi.time).getTime(),
        y: parseFloat(rsi.value) || 50
      }))
      .sort((a: any, b: any) => a.x - b.x);
    
    series.push({
      name: 'RSI',
      type: 'line',
      data: rsiData,
      color: '#ff5722',
      yAxisIndex: 1,
      strokeWidth: 2
    });
    
    yAxes.push({
      title: { text: 'RSI' },
      opposite: true,
      min: 0,
      max: 100,
      tickAmount: 5
    });
  }
  
  if (data.indicators?.bollinger && this.selectedIndicators.includes('bollinger')) {
    const bbData = data.indicators.bollinger;
    const upperData = bbData.map(bb => ({
      x: typeof bb.time === 'number' 
        ? (bb.time < 946684800000 ? bb.time * 1000 : bb.time)
        : new Date(bb.time).getTime(),
      y: parseFloat(bb.upper) || 0
    })).sort((a, b) => a.x - b.x);
    
    const middleData = bbData.map(bb => ({
      x: typeof bb.time === 'number' 
        ? (bb.time < 946684800000 ? bb.time * 1000 : bb.time)
        : new Date(bb.time).getTime(),
      y: parseFloat(bb.middle) || 0
    })).sort((a, b) => a.x - b.x);
    
    const lowerData = bbData.map(bb => ({
      x: typeof bb.time === 'number' 
        ? (bb.time < 946684800000 ? bb.time * 1000 : bb.time)
        : new Date(bb.time).getTime(),
      y: parseFloat(bb.lower) || 0
    })).sort((a, b) => a.x - b.x);
    
    series.push(
      { name: 'BB Upper', type: 'line', data: upperData, color: '#ff9800', yAxisIndex: 0, strokeWidth: 2 },
      { name: 'BB Middle', type: 'line', data: middleData, color: '#2196f3', yAxisIndex: 0, strokeWidth: 2 },
      { name: 'BB Lower', type: 'line', data: lowerData, color: '#ff9800', yAxisIndex: 0, strokeWidth: 2 }
    );
  }
  
  if (data.indicators?.pivot && this.selectedIndicators.includes('pivot')) {
    const pivot = data.indicators.pivot;
    const timeStart = typeof pivot.time_start === 'number' 
      ? (pivot.time_start < 946684800000 ? pivot.time_start * 1000 : pivot.time_start)
      : new Date(pivot.time_start).getTime();
    const timeEnd = typeof pivot.time_end === 'number' 
      ? (pivot.time_end < 946684800000 ? pivot.time_end * 1000 : pivot.time_end)
      : new Date(pivot.time_end).getTime();
    
    // Create horizontal lines
    const pivotLines = [
      { name: 'Pivot', data: [{ x: timeStart, y: pivot.pivot }, { x: timeEnd, y: pivot.pivot }], color: '#9c27b0', strokeWidth: 2 },
      { name: 'R1', data: [{ x: timeStart, y: pivot.r1 }, { x: timeEnd, y: pivot.r1 }], color: '#f44336', strokeWidth: 1 },
      { name: 'R2', data: [{ x: timeStart, y: pivot.r2 }, { x: timeEnd, y: pivot.r2 }], color: '#f44336', strokeWidth: 1 },
      { name: 'S1', data: [{ x: timeStart, y: pivot.s1 }, { x: timeEnd, y: pivot.s1 }], color: '#4caf50', strokeWidth: 1 },
      { name: 'S2', data: [{ x: timeStart, y: pivot.s2 }, { x: timeEnd, y: pivot.s2 }], color: '#4caf50', strokeWidth: 1 }
    ];
    
    pivotLines.forEach(line => {
      series.push({ ...line, type: 'line', yAxisIndex: 0 });
    });
  }
  
  // 4. Update chart smoothly
  setTimeout(() => {
    if (this.chart?.chart) {
      this.chart.updateOptions({
        series: series,
        yaxis: yAxes.length === 1 ? yAxes[0] : yAxes,
        xaxis: {
          type: 'datetime',
          labels: { datetimeUTC: false, format: 'HH:mm' }
        }
      }, false, true, false); // animate, updateSyncedCharts, updateAxis
    } else {
      this.chartOptions.series = series;
      this.chartOptions.yaxis = yAxes.length === 1 ? yAxes[0] : yAxes;
      this._changeDetectorRef.markForCheck();
    }
  }, 0);
}
```

### 3. Replay Control Implementation

```typescript
export class SimulationReplayComponent {
  isPlaying: boolean = false;
  replaySpeed: number = 1; // 1x, 5x, 10x, etc.
  currentIndex: number = 0;
  totalCandles: number = 0;
  
  private replayInterval: any;
  
  startReplay(): void {
    if (this.isPlaying) return;
    
    this.isPlaying = true;
    const intervalMs = 1000 / this.replaySpeed; // Adjust based on speed
    
    this.replayInterval = setInterval(() => {
      if (this.currentIndex < this.totalCandles - 1) {
        this.currentIndex++;
        this.fetchChartData();
      } else {
        this.pauseReplay();
      }
    }, intervalMs);
  }
  
  pauseReplay(): void {
    this.isPlaying = false;
    if (this.replayInterval) {
      clearInterval(this.replayInterval);
      this.replayInterval = null;
    }
  }
  
  setReplaySpeed(speed: number): void {
    this.replaySpeed = speed;
    if (this.isPlaying) {
      this.pauseReplay();
      this.startReplay();
    }
  }
  
  fetchChartData(): void {
    this._liveApi.getSimulationChartData(
      this.selectedTimeframe,
      this.selectedIndicators.join(','),
      this.currentIndex
    ).subscribe(response => {
      if (response.data) {
        this.updateChart(response.data);
      }
    });
  }
}
```

## Best Practices

### 1. Performance Optimization

**Backend**:
- Cache indicator calculations
- Use efficient pandas operations
- Limit data sent per request (pagination)
- Use async/await for I/O operations

**Frontend**:
- Debounce chart updates (don't update on every poll)
- Use `updateOptions()` with `animate: false` for rapid updates
- Virtual scrolling for large datasets
- Lazy load indicator data

### 2. Data Synchronization

- Always sort data by timestamp before rendering
- Ensure indicator timestamps match candle timestamps
- Use consistent timestamp format (Unix seconds or milliseconds)
- Handle timezone conversions properly

### 3. Error Handling

```typescript
updateChart(data: any): void {
  try {
    if (!data?.candles || data.candles.length === 0) {
      console.warn('No candle data available');
      return;
    }
    
    // Validate data structure
    const validCandles = data.candles.filter(c => 
      c.time && 
      typeof c.open === 'number' && 
      typeof c.high === 'number' &&
      typeof c.low === 'number' &&
      typeof c.close === 'number'
    );
    
    if (validCandles.length === 0) {
      console.error('No valid candles found');
      return;
    }
    
    // ... rest of update logic
  } catch (error) {
    console.error('Error updating chart:', error);
  }
}
```

### 4. Memory Management

- Clear old chart data when switching timeframes
- Limit indicator history (e.g., last 1000 candles)
- Use WeakMap for temporary data structures
- Dispose subscriptions properly

## Testing Checklist

- [ ] Candles render in correct chronological order
- [ ] Indicators align with candle timestamps
- [ ] Multiple timeframes aggregate correctly
- [ ] Replay controls work smoothly
- [ ] Indicator values match TradingView calculations
- [ ] Chart updates without flashing
- [ ] Performance is acceptable with 1000+ candles
- [ ] Multiple indicators can be displayed simultaneously
- [ ] Y-axes scale correctly for different indicators

