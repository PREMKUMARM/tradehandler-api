"""
Rate-limited wrapper for KiteConnect API calls
"""
import time
from functools import wraps
from kiteconnect import KiteConnect
from utils.logger import log_warning, log_error
from core.exceptions import AlgoFeastException


class RateLimitedKiteConnect:
    """Rate-limited wrapper for KiteConnect"""
    
    def __init__(self, kite_instance: KiteConnect):
        self.kite = kite_instance
        self.last_calls = {}
        self.limits = {
            "per_second": 3,
            "per_minute": 60,
            "per_hour": 1000
        }
    
    def _check_rate_limit(self, endpoint: str):
        """Check rate limits for a specific endpoint"""
        current_time = time.time()
        
        # Initialize endpoint tracking
        if endpoint not in self.last_calls:
            self.last_calls[endpoint] = []
        
        # Clean old calls (older than 1 hour)
        self.last_calls[endpoint] = [
            call_time for call_time in self.last_calls[endpoint]
            if current_time - call_time < 3600
        ]
        
        calls = self.last_calls[endpoint]
        
        # Check per-second limit
        calls_last_second = [t for t in calls if current_time - t < 1]
        if len(calls_last_second) >= self.limits["per_second"]:
            sleep_time = 1.0 - (current_time - calls_last_second[0])
            if sleep_time > 0:
                log_warning(f"Rate limit (per-second) for {endpoint}, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Check per-minute limit
        calls_last_minute = [t for t in calls if current_time - t < 60]
        if len(calls_last_minute) >= self.limits["per_minute"]:
            sleep_time = 60.0 - (current_time - calls_last_minute[0])
            if sleep_time > 0:
                log_warning(f"Rate limit (per-minute) for {endpoint}, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Check per-hour limit
        if len(calls) >= self.limits["per_hour"]:
            log_error(f"Rate limit (per-hour) exceeded for {endpoint}")
            raise AlgoFeastException(
                message="Rate limit exceeded. Please try again later.",
                status_code=429,
                error_code="RATE_LIMIT_EXCEEDED"
            )
        
        # Record this call
        self.last_calls[endpoint].append(current_time)
    
    def rate_limit(self, endpoint: str = "default"):
        """Decorator for rate limiting"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self._check_rate_limit(endpoint)
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    # Rate-limited methods
    @rate_limit("quote")
    def quote(self, instrument_token):
        """Get quote with rate limiting"""
        return self.kite.quote(instrument_token)
    
    @rate_limit("ohlc")
    def ohlc(self, instrument_token):
        """Get OHLC data with rate limiting"""
        return self.kite.ohlc(instrument_token)
    
    @rate_limit("ltp")
    def ltp(self, instrument_token):
        """Get LTP with rate limiting"""
        return self.kite.ltp(instrument_token)
    
    @rate_limit("historical_data")
    def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False):
        """Get historical data with rate limiting"""
        return self.kite.historical_data(
            instrument_token, from_date, to_date, interval, continuous
        )
    
    @rate_limit("instruments")
    def instruments(self, exchange=None):
        """Get instruments with rate limiting"""
        return self.kite.instruments(exchange)
    
    @rate_limit("positions")
    def positions(self):
        """Get positions with rate limiting"""
        return self.kite.positions()
    
    @rate_limit("holdings")
    def holdings(self):
        """Get holdings with rate limiting"""
        return self.kite.holdings()
    
    @rate_limit("orders")
    def orders(self):
        """Get orders with rate limiting"""
        return self.kite.orders()
    
    @rate_limit("trades")
    def trades(self):
        """Get trades with rate limiting"""
        return self.kite.trades()
    
    @rate_limit("margins")
    def margins(self, segment=None):
        """Get margins with rate limiting"""
        return self.kite.margins(segment)
    
    # Trading operations (lower rate limits)
    @rate_limit("place_order")
    def place_order(self, **kwargs):
        """Place order with rate limiting"""
        return self.kite.place_order(**kwargs)
    
    @rate_limit("modify_order")
    def modify_order(self, **kwargs):
        """Modify order with rate limiting"""
        return self.kite.modify_order(**kwargs)
    
    @rate_limit("cancel_order")
    def cancel_order(self, **kwargs):
        """Cancel order with rate limiting"""
        return self.kite.cancel_order(**kwargs)
    
    def __getattr__(self, name):
        """Fallback to original kite methods for non-rate-limited calls"""
        return getattr(self.kite, name)


def get_rate_limited_kite_instance(user_id: str = "default", verbose: bool = False, skip_validation: bool = False):
    """Get rate-limited KiteConnect instance"""
    from utils.kite_utils import get_kite_instance
    
    kite = get_kite_instance(user_id, verbose, skip_validation)
    return RateLimitedKiteConnect(kite)
