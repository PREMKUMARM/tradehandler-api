"""
Binance Commentary Service
Manages historical and real-time commentary generation
"""
import asyncio
from typing import Dict, List, Optional, Set
from collections import deque
from datetime import datetime
import time

from utils.binance_commentary import get_commentary_generator
from utils.binance_client import fetch_klines
from utils.binance_signals import analyze_symbol_for_signals
from core.config import get_settings

def get_binance_symbols() -> list:
    """Get Binance symbols from environment configuration"""
    settings = get_settings()
    symbols = settings.binance_symbols
    if isinstance(symbols, str):
        # Handle comma-separated string
        symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif isinstance(symbols, list):
        # Ensure all symbols are uppercase
        symbols = [s.upper() if isinstance(s, str) else str(s).upper() for s in symbols]
    else:
        # Fallback to default
        symbols = ["1000PEPEUSDT"]
    return symbols


class CommentaryService:
    """Service to manage commentary generation and storage"""
    
    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self.commentary_queue: deque = deque(maxlen=max_messages)
        self.commentary_generator = get_commentary_generator()
        self.last_candle_timestamps: Dict[str, int] = {}  # {symbol: last_5min_timestamp}
        self.is_initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize_historical_commentary(self):
        """Generate commentary from last 100 candles for all symbols"""
        if self.is_initialized:
            return
        
        print("[Commentary Service] Initializing historical commentary...")
        symbols = get_binance_symbols()
        
        try:
            for symbol in symbols:
                await self._generate_historical_for_symbol(symbol)
            
            self.is_initialized = True
            print(f"[Commentary Service] Historical commentary initialized with {len(self.commentary_queue)} messages")
        except Exception as e:
            print(f"[Commentary Service] Error initializing historical commentary: {e}")
            import traceback
            traceback.print_exc()
    
    async def _generate_historical_for_symbol(self, symbol: str):
        """Generate commentary for last 100 candles of a symbol"""
        try:
            # Fetch last 100 candles (5-minute intervals)
            klines = await fetch_klines(symbol, '5m', limit=100)
            
            if not klines or len(klines) < 2:
                print(f"[Commentary Service] Not enough data for {symbol}")
                return
            
            # Process each candle chronologically, analyzing up to that point
            previous_data = None
            
            # We'll analyze in sliding windows to get accurate indicators for each candle
            for i in range(1, len(klines)):  # Start from 1 to compare with previous
                # Get candles up to current index (need at least 30 for indicators)
                window_start = max(0, i - 99)  # Use up to 100 candles for analysis
                window_klines = klines[window_start:i+1]
                
                if len(window_klines) < 30:
                    # Not enough data for accurate analysis, skip
                    continue
                
                # Analyze signals for this window
                signal_data = analyze_symbol_for_signals(symbol, window_klines)
                validation_checks = signal_data.get('validation_checks', {})
                indicators = validation_checks.get('indicators', {})
                
                # Get current candle
                candle = klines[i]
                prev_candle = klines[i-1] if i > 0 else None
                
                # Get candle timestamp - use the actual close_time from the candle
                # This is the historical timestamp when the candle closed
                candle_timestamp_ms = candle.get('close_time') or candle.get('timestamp') or candle.get('open_time', 0)
                if not candle_timestamp_ms:
                    # Fallback: calculate from index if no timestamp available
                    print(f"[Commentary Service] Warning: No timestamp found for candle {i} of {symbol}")
                    continue
                
                # Use the actual historical timestamp (don't round it)
                # This is the actual time when the candle closed, which could be hours/days ago
                candle_5min_timestamp = candle_timestamp_ms
                
                # Debug: Log first and last candle timestamps to verify
                if i == 1:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(candle_timestamp_ms / 1000)
                    print(f"[Commentary Service] First historical candle for {symbol}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                if i == len(klines) - 1:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(candle_timestamp_ms / 1000)
                    print(f"[Commentary Service] Last historical candle for {symbol}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Build current data
                current_data = {
                    'price': float(candle.get('close', 0)),
                    'high': float(candle.get('high', 0)),
                    'low': float(candle.get('low', 0)),
                    'volume': float(candle.get('volume', 0)),
                    'timestamp': candle_5min_timestamp,
                    'candle_pattern': signal_data.get('candle_pattern', ''),
                    'signal': signal_data.get('signal'),
                    'signal_priority': signal_data.get('signal_priority'),
                    'signal_reason': signal_data.get('signal_reason'),
                    'vwap': indicators.get('vwap'),
                    'rsi': indicators.get('rsi'),
                    'buy_conditions_met': validation_checks.get('buy_conditions_met', 0),
                    'buy_conditions_total': validation_checks.get('buy_conditions_total', 8),
                    'sell_conditions_met': validation_checks.get('sell_conditions_met', 0),
                    'sell_conditions_total': validation_checks.get('sell_conditions_total', 8)
                }
                
                # Generate commentary comparing with previous candle
                messages = self.commentary_generator.generate_commentary(
                    symbol,
                    current_data,
                    previous_data
                )
                
                # Add messages to queue
                async with self._lock:
                    for msg in messages:
                        self.commentary_queue.append(msg)
                
                previous_data = current_data.copy()
                self.last_candle_timestamps[symbol] = candle_5min_timestamp
            
            print(f"[Commentary Service] Generated {len(list(self.commentary_queue))} historical commentary messages for {symbol}")
            
        except Exception as e:
            print(f"[Commentary Service] Error generating historical for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    async def process_new_candle(self, symbol: str, ticker_data: Dict, signal_data: Dict):
        """Process a new candle and generate commentary"""
        try:
            # Get current 5-minute candle timestamp
            current_timestamp_ms = ticker_data.get('timestamp', 0)
            current_5min_timestamp = (current_timestamp_ms // 300000) * 300000
            
            # Check if this is a new candle
            last_5min_timestamp = self.last_candle_timestamps.get(symbol, 0)
            is_new_candle = (current_5min_timestamp > last_5min_timestamp)
            
            if not is_new_candle:
                return []  # Not a new candle, no commentary
            
            # Build current data
            validation_checks = signal_data.get('validation_checks', {})
            indicators = validation_checks.get('indicators', {})
            
            current_data = {
                'price': ticker_data.get('price', 0),
                'high': ticker_data.get('high_24h', 0),
                'low': ticker_data.get('low_24h', 0),
                'volume': ticker_data.get('volume_24h', 0),
                'timestamp': current_5min_timestamp,
                'candle_pattern': signal_data.get('candle_pattern', ''),
                'signal': signal_data.get('signal'),
                'signal_priority': signal_data.get('signal_priority'),
                'signal_reason': signal_data.get('signal_reason'),
                'vwap': indicators.get('vwap'),
                'rsi': indicators.get('rsi'),
                'buy_conditions_met': validation_checks.get('buy_conditions_met', 0),
                'buy_conditions_total': validation_checks.get('buy_conditions_total', 8),
                'sell_conditions_met': validation_checks.get('sell_conditions_met', 0),
                'sell_conditions_total': validation_checks.get('sell_conditions_total', 8)
            }
            
            # Get previous state from generator
            previous_data = self.commentary_generator.previous_states.get(symbol.upper(), {})
            
            # Generate commentary
            messages = self.commentary_generator.generate_commentary(
                symbol,
                current_data,
                previous_data
            )
            
            # Add to queue
            async with self._lock:
                for msg in messages:
                    self.commentary_queue.append(msg)
            
            # Update timestamp
            self.last_candle_timestamps[symbol] = current_5min_timestamp
            
            return messages
            
        except Exception as e:
            print(f"[Commentary Service] Error processing new candle for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_historical_commentary(self, limit: Optional[int] = None) -> List[Dict]:
        """Get historical commentary messages"""
        async def _get():
            async with self._lock:
                messages = list(self.commentary_queue)
                if limit:
                    return messages[-limit:]
                return messages
        
        # Since this is called from sync context, we need to handle it differently
        # For now, return a copy of the queue
        return list(self.commentary_queue)[-limit:] if limit else list(self.commentary_queue)
    
    def get_recent_commentary(self, count: int = 100) -> List[Dict]:
        """Get most recent commentary messages"""
        return list(self.commentary_queue)[-count:]


# Global commentary service instance
_commentary_service: Optional[CommentaryService] = None

def get_commentary_service() -> CommentaryService:
    """Get the global commentary service instance"""
    global _commentary_service
    if _commentary_service is None:
        _commentary_service = CommentaryService(max_messages=100)
    return _commentary_service

async def initialize_commentary_service():
    """Initialize the commentary service with historical data"""
    service = get_commentary_service()
    await service.initialize_historical_commentary()
    # Start background task to monitor for new candles
    asyncio.create_task(monitor_new_candles())


async def monitor_new_candles():
    """Background task to continuously monitor for new candles and generate commentary"""
    from utils.binance_websocket_ticker import start_global_ticker_listener, get_all_latest_tickers
    from utils.binance_client import fetch_klines
    from utils.binance_signals import analyze_symbol_for_signals
    import time
    
    service = get_commentary_service()
    symbols = get_binance_symbols()
    
    # Start global ticker listener
    await start_global_ticker_listener(symbols_filter=set(symbols))
    
    # Track last signal update time
    last_signal_update = {}
    signal_cache = {}
    SIGNAL_UPDATE_INTERVAL = 30  # Update signals every 30 seconds
    
    print("[Commentary Service] Started background monitoring for new candles")
    
    while True:
        try:
            # Get latest ticker data
            all_tickers = get_all_latest_tickers()
            current_time = time.time()
            
            # Check which symbols need signal updates
            symbols_to_update = set()
            for symbol in symbols:
                should_update = (
                    symbol not in last_signal_update or 
                    (current_time - last_signal_update[symbol]) >= SIGNAL_UPDATE_INTERVAL
                )
                if should_update:
                    symbols_to_update.add(symbol)
            
            # Update signals for symbols that need it
            if symbols_to_update:
                for symbol in symbols_to_update:
                    try:
                        klines = await fetch_klines(symbol, '5m', limit=200)
                        signal_data = analyze_symbol_for_signals(symbol, klines)
                        signal_cache[symbol] = signal_data
                        last_signal_update[symbol] = current_time
                    except Exception as e:
                        print(f"[Commentary Service] Error updating signals for {symbol}: {e}")
            
            # Process new candles for all symbols
            for symbol in symbols:
                ticker_data = all_tickers.get(symbol)
                if not ticker_data:
                    continue
                
                signal_data = signal_cache.get(symbol, {})
                await service.process_new_candle(symbol, ticker_data, signal_data)
            
            # Wait 10 seconds before next check
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"[Commentary Service] Error in monitor_new_candles: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(10)  # Wait before retry

