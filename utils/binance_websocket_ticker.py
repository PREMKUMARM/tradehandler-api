"""
Binance Futures WebSocket ticker listener for real-time price updates
Uses official Binance Futures WebSocket API: wss://fstream.binance.com/ws/!ticker@arr
"""
import asyncio
import json
import websockets
from typing import Dict, Callable, Optional, Set
from datetime import datetime

BINANCE_FUTURES_WS_URL = "wss://fstream.binance.com/ws/!ticker@arr"


class BinanceFuturesTickerListener:
    """
    WebSocket listener for Binance Futures 24hr ticker stream
    
    Streams all symbols ticker updates in real-time
    """
    
    def __init__(self, symbols_filter: Optional[Set[str]] = None, callback: Optional[Callable] = None):
        """
        Initialize the ticker listener
        
        Args:
            symbols_filter: Optional set of symbols to filter (e.g., {"ETHUSDT", "SOLUSDT"}).
                          If None, listens to all symbols.
            callback: Optional callback function to handle ticker updates.
                     Signature: callback(symbol: str, ticker_data: dict)
        """
        self.symbols_filter = symbols_filter
        self.callback = callback
        self.websocket = None
        self.is_running = False
        self.reconnect_delay = 5
        self._latest_tickers = {}  # Cache latest ticker data
        
    async def connect(self):
        """Connect to Binance Futures WebSocket"""
        try:
            self.websocket = await websockets.connect(
                BINANCE_FUTURES_WS_URL,
                ping_interval=20,
                ping_timeout=10
            )
            print(f"[Binance WS Ticker] Connected to {BINANCE_FUTURES_WS_URL}")
            return True
        except Exception as e:
            print(f"[Binance WS Ticker] Connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.is_running = False
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None
        print("[Binance WS Ticker] Disconnected")
    
    def get_latest_ticker(self, symbol: str) -> Optional[Dict]:
        """Get latest ticker data for a symbol"""
        return self._latest_tickers.get(symbol.upper())
    
    def get_all_latest_tickers(self) -> Dict[str, Dict]:
        """Get all latest ticker data"""
        return self._latest_tickers.copy()
    
    async def listen(self):
        """
        Start listening to ticker stream
        
        This method runs indefinitely until disconnect() is called
        """
        self.is_running = True
        
        while self.is_running:
            try:
                if not self.websocket or self.websocket.closed:
                    if not await self.connect():
                        await asyncio.sleep(self.reconnect_delay)
                        continue
                
                # Listen for messages
                async for message in self.websocket:
                    if not self.is_running:
                        break
                    
                    try:
                        # Parse ticker array
                        tickers = json.loads(message)
                        
                        if not isinstance(tickers, list):
                            continue
                        
                        # Process each ticker update
                        for ticker in tickers:
                            symbol = ticker.get('s', '').upper()  # 's' = symbol
                            
                            # Filter symbols if specified
                            if self.symbols_filter and symbol not in self.symbols_filter:
                                continue
                            
                            # Format ticker data (Binance WebSocket ticker format)
                            # Reference: https://binance-docs.github.io/apidocs/futures/en/#24hr-ticker-price-change-statistics-streams
                            ticker_data = {
                                "symbol": symbol,
                                "price": float(ticker.get('c', 0)),  # 'c' = last price
                                "high_24h": float(ticker.get('h', 0)),  # 'h' = high price
                                "low_24h": float(ticker.get('l', 0)),  # 'l' = low price
                                "volume_24h": float(ticker.get('v', 0)),  # 'v' = volume
                                "quote_volume_24h": float(ticker.get('q', 0)),  # 'q' = quote volume
                                "open_price": float(ticker.get('o', 0)),  # 'o' = open price
                                "prev_close_price": float(ticker.get('x', 0)),  # 'x' = prev close price
                                "bid_price": float(ticker.get('b', 0)),  # 'b' = best bid price
                                "ask_price": float(ticker.get('a', 0)),  # 'a' = best ask price
                                "count": int(ticker.get('n', 0)),  # 'n' = number of trades (24h)
                                "timestamp": int(ticker.get('E', 0))  # 'E' = event time
                            }
                            
                            # Update cache
                            self._latest_tickers[symbol] = ticker_data
                            
                            # Call callback if provided
                            if self.callback:
                                try:
                                    await self.callback(symbol, ticker_data) if asyncio.iscoroutinefunction(self.callback) else self.callback(symbol, ticker_data)
                                except Exception as e:
                                    print(f"[Binance WS Ticker] Callback error for {symbol}: {e}")
                    
                    except json.JSONDecodeError as e:
                        print(f"[Binance WS Ticker] JSON decode error: {e}")
                    except Exception as e:
                        print(f"[Binance WS Ticker] Error processing message: {e}")
                        import traceback
                        traceback.print_exc()
            
            except websockets.exceptions.ConnectionClosed:
                print("[Binance WS Ticker] Connection closed, attempting to reconnect...")
                self.websocket = None
                if self.is_running:
                    await asyncio.sleep(self.reconnect_delay)
            except Exception as e:
                print(f"[Binance WS Ticker] Error in listen loop: {e}")
                import traceback
                traceback.print_exc()
                self.websocket = None
                if self.is_running:
                    await asyncio.sleep(self.reconnect_delay)


# Global listener instance
_global_ticker_listener: Optional[BinanceFuturesTickerListener] = None
_listener_task: Optional[asyncio.Task] = None


async def start_global_ticker_listener(symbols_filter: Optional[Set[str]] = None):
    """
    Start a global ticker listener that runs in the background
    
    Args:
        symbols_filter: Optional set of symbols to filter
    """
    global _global_ticker_listener, _listener_task
    
    if _global_ticker_listener and _global_ticker_listener.is_running:
        print("[Binance WS Ticker] Global listener already running")
        return
    
    _global_ticker_listener = BinanceFuturesTickerListener(symbols_filter=symbols_filter)
    _listener_task = asyncio.create_task(_global_ticker_listener.listen())
    print("[Binance WS Ticker] Global listener started")
    
    return _global_ticker_listener


async def stop_global_ticker_listener():
    """Stop the global ticker listener"""
    global _global_ticker_listener, _listener_task
    
    if _global_ticker_listener:
        await _global_ticker_listener.disconnect()
        _global_ticker_listener = None
    
    if _listener_task:
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
        _listener_task = None
    
    print("[Binance WS Ticker] Global listener stopped")


def get_latest_ticker(symbol: str) -> Optional[Dict]:
    """Get latest ticker data for a symbol from global listener"""
    global _global_ticker_listener
    if _global_ticker_listener:
        return _global_ticker_listener.get_latest_ticker(symbol)
    return None


def get_all_latest_tickers() -> Dict[str, Dict]:
    """Get all latest ticker data from global listener"""
    global _global_ticker_listener
    if _global_ticker_listener:
        return _global_ticker_listener.get_all_latest_tickers()
    return {}

