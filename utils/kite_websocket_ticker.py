"""
Kite Connect WebSocket ticker listener for real-time market data
Uses KiteTicker from kiteconnect library to maintain persistent connection
Automatically starts when market opens and stops when market closes
"""
import threading
import time
from datetime import datetime, time as dt_time
from typing import Dict, Optional, List, Callable, Set
from kiteconnect import KiteTicker
from kiteconnect.exceptions import KiteException

from utils.kite_utils import get_kite_api_key, get_access_token
from utils.logger import log_info, log_error, log_warning, log_debug
from agent.config import get_agent_config
from agent.ws_manager import add_agent_log


class KiteTickerListener:
    """
    WebSocket ticker listener for Kite Connect
    Maintains persistent connection and receives real-time ticker updates
    """
    
    def __init__(self, instrument_tokens: Optional[List[int]] = None, instrument_names: Optional[List[str]] = None, callback: Optional[Callable] = None):
        """
        Initialize the Kite ticker listener
        
        Args:
            instrument_tokens: List of instrument tokens to subscribe to (optional).
            instrument_names: List of instrument names to subscribe to (optional).
                            Will be resolved to tokens using instruments API.
                            Examples: ["NIFTY 50", "NIFTY BANK", "SENSEX"]
            callback: Optional callback function to handle ticker updates.
                     Signature: callback(ticks: List[Dict])
        """
        self.api_key = get_kite_api_key()
        self.access_token = get_access_token()
        self.callback = callback
        self.kws = None
        self.is_connected = False
        self.is_running = False
        self.reconnect_delay = 5
        self._latest_ticks = {}  # Cache latest tick data: {instrument_token: tick_data}
        self._lock = threading.RLock()  # Reentrant lock for thread-safe operations (prevents deadlocks)
        
        # Resolve instrument names to tokens if provided
        resolved_tokens = []
        if instrument_tokens:
            resolved_tokens.extend(instrument_tokens)
        
        if instrument_names:
            try:
                from utils.kite_utils import get_kite_instance
                kite = get_kite_instance()
                
                # Use cached instruments to avoid rate limiting
                nse_instruments = []
                bse_instruments = []
                
                try:
                    # Try to get from cache first (from instrument_resolver)
                    from agent.tools.instrument_resolver import _instruments_cache
                    if "NSE" in _instruments_cache:
                        nse_instruments = _instruments_cache["NSE"]
                        log_debug("[Kite Ticker] Using cached NSE instruments")
                    if "BSE" in _instruments_cache:
                        bse_instruments = _instruments_cache["BSE"]
                        log_debug("[Kite Ticker] Using cached BSE instruments")
                except:
                    pass
                
                # Fetch only if not in cache (and only once for all instruments)
                if not nse_instruments:
                    try:
                        nse_instruments = kite.instruments("NSE")
                        log_debug("[Kite Ticker] Fetched NSE instruments (not cached)")
                    except Exception as e:
                        log_error(f"[Kite Ticker] Error fetching NSE instruments: {e}")
                        nse_instruments = []
                
                if not bse_instruments:
                    try:
                        bse_instruments = kite.instruments("BSE")
                        log_debug("[Kite Ticker] Fetched BSE instruments (not cached)")
                    except Exception as e:
                        log_error(f"[Kite Ticker] Error fetching BSE instruments: {e}")
                        bse_instruments = []
                
                # Resolve all names from the fetched/cached lists
                for name in instrument_names:
                    token = None
                    name_upper = name.upper().strip()
                    
                    if name_upper == "SENSEX":
                        # For SENSEX, search in BSE instruments
                        for inst in bse_instruments:
                            inst_name = inst.get("name", "").upper()
                            inst_type = inst.get("instrument_type", "").upper()
                            if inst_name == "SENSEX" and inst_type == "INDEX":
                                token = inst.get("instrument_token")
                                log_info(f"[Kite Ticker] Found SENSEX index: token {token}")
                                break
                    elif name_upper in ["NIFTY 50", "NIFTY BANK", "NIFTY BANKNIFTY", "BANKNIFTY"]:
                        # For NSE indices, search in NSE instruments
                        search_name = "NIFTY 50" if name_upper == "NIFTY 50" else "NIFTY BANK"
                        for inst in nse_instruments:
                            inst_name = inst.get("name", "").upper()
                            inst_tradingsymbol = inst.get("tradingsymbol", "").upper()
                            inst_type = inst.get("instrument_type", "").upper()
                            if inst_type == "INDEX" and (inst_name == search_name or inst_tradingsymbol == search_name):
                                token = inst.get("instrument_token")
                                log_info(f"[Kite Ticker] Found {name} in NSE: token {token}")
                                break
                    else:
                        # For other instruments, search both exchanges
                        for inst in nse_instruments + bse_instruments:
                            inst_name = inst.get("name", "").upper()
                            inst_tradingsymbol = inst.get("tradingsymbol", "").upper()
                            if inst_name == name_upper or inst_tradingsymbol == name_upper:
                                token = inst.get("instrument_token")
                                log_info(f"[Kite Ticker] Found {name}: token {token}")
                                break
                    
                    if token:
                        resolved_tokens.append(token)
                        log_info(f"[Kite Ticker] Resolved '{name}' to token {token}")
                    else:
                        log_warning(f"[Kite Ticker] Could not resolve instrument name '{name}' to token")
            except Exception as e:
                log_error(f"[Kite Ticker] Error resolving instrument names: {e}")
                import traceback
                traceback.print_exc()
        
        # Default instruments if none provided
        if not resolved_tokens:
            # Default to major indices: NIFTY 50, BANKNIFTY, SENSEX
            default_names = ["NIFTY 50", "NIFTY BANK", "SENSEX"]
            try:
                from utils.kite_utils import get_kite_instance
                kite = get_kite_instance()
                
                # Use cached instruments to avoid rate limiting
                nse_instruments = []
                bse_instruments = []
                
                try:
                    # Try to get from cache first (from instrument_resolver)
                    from agent.tools.instrument_resolver import _instruments_cache
                    if "NSE" in _instruments_cache:
                        nse_instruments = _instruments_cache["NSE"]
                        log_debug("[Kite Ticker] Using cached NSE instruments for defaults")
                    if "BSE" in _instruments_cache:
                        bse_instruments = _instruments_cache["BSE"]
                        log_debug("[Kite Ticker] Using cached BSE instruments for defaults")
                except:
                    pass
                
                # Fetch only if not in cache (and only once for all instruments)
                if not nse_instruments:
                    try:
                        nse_instruments = kite.instruments("NSE")
                        log_debug("[Kite Ticker] Fetched NSE instruments for defaults (not cached)")
                    except Exception as e:
                        log_error(f"[Kite Ticker] Error fetching NSE instruments: {e}")
                        nse_instruments = []
                
                if not bse_instruments:
                    try:
                        bse_instruments = kite.instruments("BSE")
                        log_debug("[Kite Ticker] Fetched BSE instruments for defaults (not cached)")
                    except Exception as e:
                        log_error(f"[Kite Ticker] Error fetching BSE instruments: {e}")
                        bse_instruments = []
                
                # Resolve all instruments from cached/fetched lists
                for name in default_names:
                    token = None
                    name_upper = name.upper().strip()
                    
                    if name_upper == "SENSEX":
                        # For SENSEX, search in BSE instruments
                        for inst in bse_instruments:
                            inst_name = inst.get("name", "").upper()
                            inst_type = inst.get("instrument_type", "").upper()
                            if inst_name == "SENSEX" and inst_type == "INDEX":
                                token = inst.get("instrument_token")
                                log_info(f"[Kite Ticker] Found SENSEX index in BSE: token {token}")
                                break
                    elif name_upper in ["NIFTY 50", "NIFTY BANK"]:
                        # For NSE indices, search in NSE instruments
                        search_name = name_upper
                        for inst in nse_instruments:
                            inst_name = inst.get("name", "").upper()
                            inst_tradingsymbol = inst.get("tradingsymbol", "").upper()
                            inst_type = inst.get("instrument_type", "").upper()
                            if inst_type == "INDEX" and (inst_name == search_name or inst_tradingsymbol == search_name):
                                token = inst.get("instrument_token")
                                log_info(f"[Kite Ticker] Found {name} in NSE: token {token}")
                                break
                    
                    if token:
                        resolved_tokens.append(token)
                        log_info(f"[Kite Ticker] Resolved default '{name}' to token {token}")
                    else:
                        log_warning(f"[Kite Ticker] Could not resolve default instrument '{name}'")
                
                # If still no tokens resolved, use fallback
                if not resolved_tokens:
                    log_warning("[Kite Ticker] No instruments resolved, using fallback hardcoded tokens")
                    resolved_tokens = [256265, 260105, 265]
                    
            except Exception as e:
                log_error(f"[Kite Ticker] Error resolving default instruments: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to hardcoded tokens if resolution fails
                resolved_tokens = [256265, 260105, 265]
                log_warning("[Kite Ticker] Using fallback hardcoded tokens")
        
        self.instrument_tokens = resolved_tokens
        
    def _on_ticks(self, ws, ticks):
        """
        Callback when ticks are received from Kite
        This runs in KiteTicker's thread, so we need to be thread-safe
        
        Args:
            ws: WebSocket instance
            ticks: List of tick data dictionaries
        """
        try:
            # Quick update to cache (thread-safe with lock)
            with self._lock:
                # Update cache with latest ticks
                for tick in ticks:
                    instrument_token = tick.get('instrument_token')
                    if instrument_token:
                        self._latest_ticks[instrument_token] = tick
            
            # Call user callback if provided (outside lock to avoid blocking)
            if self.callback:
                try:
                    # Run callback in a way that won't block
                    self.callback(ticks)
                except Exception as e:
                    log_error(f"[Kite Ticker] Callback error: {e}")
            
            # Minimal logging to avoid blocking (only log occasionally)
            # Don't log every tick to avoid log spam and blocking
                
        except Exception as e:
            log_error(f"[Kite Ticker] Error processing ticks: {e}")
    
    def _on_connect(self, ws, response):
        """Callback when WebSocket connects (runs in KiteTicker thread)"""
        try:
            self.is_connected = True
            log_info("[Kite Ticker] WebSocket connected successfully")
            # Use thread-safe logging (add_agent_log might not be thread-safe, so just use log_info)
            # add_agent_log("Kite ticker WebSocket connected", "info", "system")
            
            # Subscribe to instruments
            if self.instrument_tokens:
                try:
                    ws.subscribe(self.instrument_tokens)
                    ws.set_mode(ws.MODE_FULL, self.instrument_tokens)  # MODE_FULL for all tick data
                    log_info(f"[Kite Ticker] Subscribed to {len(self.instrument_tokens)} instrument(s): {self.instrument_tokens}")
                except Exception as e:
                    log_error(f"[Kite Ticker] Error subscribing: {e}")
        except Exception as e:
            log_error(f"[Kite Ticker] Error in on_connect callback: {e}")
    
    def _on_close(self, ws, code, reason):
        """Callback when WebSocket closes (runs in KiteTicker thread)"""
        try:
            self.is_connected = False
            self.is_running = False
            log_warning(f"[Kite Ticker] WebSocket closed. Code: {code}, Reason: {reason}")
        except Exception as e:
            log_error(f"[Kite Ticker] Error in on_close callback: {e}")
    
    def _on_error(self, ws, code, reason):
        """Callback when WebSocket error occurs (runs in KiteTicker thread)"""
        try:
            log_error(f"[Kite Ticker] WebSocket error. Code: {code}, Reason: {reason}")
            self.is_connected = False
        except Exception as e:
            log_error(f"[Kite Ticker] Error in on_error callback: {e}")
    
    def _on_reconnect(self, ws, attempts_count):
        """Callback when WebSocket reconnects (runs in KiteTicker thread)"""
        try:
            log_info(f"[Kite Ticker] Reconnecting... Attempt {attempts_count}")
        except Exception as e:
            log_error(f"[Kite Ticker] Error in on_reconnect callback: {e}")
    
    def _on_noreconnect(self, ws):
        """Callback when WebSocket cannot reconnect (runs in KiteTicker thread)"""
        try:
            log_error("[Kite Ticker] Reconnection failed. Will retry later.")
            self.is_connected = False
            self.is_running = False
        except Exception as e:
            log_error(f"[Kite Ticker] Error in on_noreconnect callback: {e}")
    
    def connect(self) -> bool:
        """
        Connect to Kite WebSocket ticker (non-blocking)
        
        Returns:
            True if connection initiated successfully, False otherwise
        """
        if not self.api_key or not self.access_token:
            log_error("[Kite Ticker] API key or access token not found")
            return False
        
        # Don't reconnect if already running
        if self.is_running and self.kws:
            log_warning("[Kite Ticker] Already running, skipping connect")
            return True
        
        try:
            # Create KiteTicker instance
            self.kws = KiteTicker(self.api_key, self.access_token)
            
            # Set callbacks (make sure they're thread-safe and don't block)
            self.kws.on_ticks = self._on_ticks
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            self.kws.on_reconnect = self._on_reconnect
            self.kws.on_noreconnect = self._on_noreconnect
            
            # Connect in a separate thread (non-blocking)
            # KiteTicker.connect(threaded=True) runs in background thread
            self.kws.connect(threaded=True)
            self.is_running = True
            
            log_info("[Kite Ticker] Connection initiated (non-blocking)")
            return True
            
        except Exception as e:
            log_error(f"[Kite Ticker] Connection error: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket (non-blocking)"""
        self.is_running = False
        if self.kws:
            try:
                # Close in a way that won't block
                self.kws.close()
            except Exception as e:
                log_error(f"[Kite Ticker] Error closing connection: {e}")
            finally:
                self.kws = None
        self.is_connected = False
        log_info("[Kite Ticker] Disconnected")
    
    def subscribe(self, instrument_tokens: List[int]):
        """
        Subscribe to additional instruments
        
        Args:
            instrument_tokens: List of instrument tokens to subscribe to
        """
        if not self.is_connected or not self.kws:
            log_warning("[Kite Ticker] Cannot subscribe - not connected")
            return
        
        try:
            # Add to existing list
            new_tokens = [token for token in instrument_tokens if token not in self.instrument_tokens]
            if new_tokens:
                self.instrument_tokens.extend(new_tokens)
                self.kws.subscribe(new_tokens)
                self.kws.set_mode(self.kws.MODE_FULL, new_tokens)
                log_info(f"[Kite Ticker] Subscribed to {len(new_tokens)} additional instrument(s)")
        except Exception as e:
            log_error(f"[Kite Ticker] Error subscribing to instruments: {e}")
    
    def unsubscribe(self, instrument_tokens: List[int]):
        """
        Unsubscribe from instruments
        
        Args:
            instrument_tokens: List of instrument tokens to unsubscribe from
        """
        if not self.is_connected or not self.kws:
            log_warning("[Kite Ticker] Cannot unsubscribe - not connected")
            return
        
        try:
            self.kws.unsubscribe(instrument_tokens)
            self.instrument_tokens = [token for token in self.instrument_tokens if token not in instrument_tokens]
            log_info(f"[Kite Ticker] Unsubscribed from {len(instrument_tokens)} instrument(s)")
        except Exception as e:
            log_error(f"[Kite Ticker] Error unsubscribing: {e}")
    
    def get_latest_tick(self, instrument_token: int) -> Optional[Dict]:
        """
        Get latest tick data for an instrument
        
        Args:
            instrument_token: Instrument token
            
        Returns:
            Latest tick data dictionary or None
        """
        with self._lock:
            return self._latest_ticks.get(instrument_token)
    
    def get_all_latest_ticks(self) -> Dict[int, Dict]:
        """
        Get all latest tick data
        
        Returns:
            Dictionary of {instrument_token: tick_data}
        """
        with self._lock:
            return self._latest_ticks.copy()
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open (based on trading hours from config)
        
        Returns:
            True if market is open, False otherwise
        """
        config = get_agent_config()
        now = datetime.now()
        
        try:
            start_time = datetime.strptime(config.trading_start_time, "%H:%M").time()
            end_time = datetime.strptime(config.trading_end_time, "%H:%M").time()
            current_time = now.time()
            
            return start_time <= current_time <= end_time
        except Exception as e:
            log_error(f"[Kite Ticker] Error checking market hours: {e}")
            # Default to market hours if config error
            return dt_time(9, 15) <= now.time() <= dt_time(15, 30)


# Global listener instance
_global_kite_ticker: Optional[KiteTickerListener] = None
_ticker_thread: Optional[threading.Thread] = None


def start_kite_ticker_listener(instrument_tokens: Optional[List[int]] = None, instrument_names: Optional[List[str]] = None, callback: Optional[Callable] = None) -> Optional[KiteTickerListener]:
    """
    Start a global Kite ticker listener
    
    Args:
        instrument_tokens: Optional list of instrument tokens to subscribe to
        callback: Optional callback function for tick updates
    
    Returns:
        KiteTickerListener instance or None if failed
    """
    global _global_kite_ticker
    
    if _global_kite_ticker and _global_kite_ticker.is_running:
        log_warning("[Kite Ticker] Listener already running")
        return _global_kite_ticker
    
    _global_kite_ticker = KiteTickerListener(instrument_tokens=instrument_tokens, callback=callback)
    
    if _global_kite_ticker.connect():
        log_info("[Kite Ticker] Global listener started")
        return _global_kite_ticker
    else:
        log_error("[Kite Ticker] Failed to start global listener")
        _global_kite_ticker = None
        return None


def stop_kite_ticker_listener():
    """Stop the global Kite ticker listener"""
    global _global_kite_ticker
    
    if _global_kite_ticker:
        _global_kite_ticker.disconnect()
        _global_kite_ticker = None
        log_info("[Kite Ticker] Global listener stopped")


def get_kite_ticker_instance() -> Optional[KiteTickerListener]:
    """Get the global Kite ticker listener instance"""
    return _global_kite_ticker


def get_latest_kite_tick(instrument_token: int) -> Optional[Dict]:
    """Get latest tick data for an instrument from global listener"""
    global _global_kite_ticker
    if _global_kite_ticker:
        return _global_kite_ticker.get_latest_tick(instrument_token)
    return None


def get_all_latest_kite_ticks() -> Dict[int, Dict]:
    """Get all latest tick data from global listener"""
    global _global_kite_ticker
    if _global_kite_ticker:
        return _global_kite_ticker.get_all_latest_ticks()
    return {}


async def manage_kite_ticker_market_hours():
    """
    Background task to manage Kite ticker based on market hours
    Starts ticker when market opens, stops when market closes
    Runs in async context but calls sync KiteTicker methods (which run in their own threads)
    """
    global _global_kite_ticker
    
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Use thread pool executor for sync operations to avoid blocking event loop
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kite-ticker-manager")
    
    log_info("[Kite Ticker Manager] Starting market hours manager...")
    
    # Initial delay to let server fully start
    await asyncio.sleep(5)
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute (non-blocking)
            
            # Run sync operations in thread pool to avoid blocking
            def check_and_manage_ticker():
                global _global_kite_ticker  # Use global, not nonlocal
                try:
                    if not _global_kite_ticker:
                        # Create new instance (lightweight, shouldn't block)
                        _global_kite_ticker = KiteTickerListener()
                    
                    is_market_open = _global_kite_ticker.is_market_open()
                    
                    if is_market_open:
                        # Market is open - ensure ticker is running
                        if not _global_kite_ticker.is_running or not _global_kite_ticker.is_connected:
                            log_info("[Kite Ticker Manager] Market is open - starting ticker...")
                            # connect() is non-blocking (runs in KiteTicker's thread)
                            if _global_kite_ticker.connect():
                                log_info("[Kite Ticker Manager] Ticker started successfully")
                            else:
                                log_warning("[Kite Ticker Manager] Failed to start ticker, will retry...")
                    else:
                        # Market is closed - stop ticker
                        if _global_kite_ticker.is_running or _global_kite_ticker.is_connected:
                            log_info("[Kite Ticker Manager] Market is closed - stopping ticker...")
                            # disconnect() is non-blocking
                            _global_kite_ticker.disconnect()
                except Exception as e:
                    log_error(f"[Kite Ticker Manager] Error in ticker management: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, check_and_manage_ticker)
                    
        except Exception as e:
            log_error(f"[Kite Ticker Manager] Error in market hours manager: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60)  # Wait before retrying

