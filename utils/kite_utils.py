import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from kiteconnect import KiteConnect
from fastapi import HTTPException

# Global API key - get from AgentConfig (managed via UI) or environment
def get_kite_api_key(user_id: str = "default"):
    """Get Kite API key from user-specific AgentConfig or environment"""
    try:
        from agent.user_config import get_user_config
        config = get_user_config(user_id=user_id)
        if config.kite_api_key:
            return config.kite_api_key
    except:
        pass
    try:
        from agent.config import get_agent_config
        config = get_agent_config()
        if config.kite_api_key:
            return config.kite_api_key
    except:
        pass
    # Fallback to environment variable
    return os.getenv('KITE_API_KEY', 'gle4opgggiing1ol')

api_key = get_kite_api_key()

def get_access_token():
    """Read access token from config file"""
    config_path = Path("config/access_token.txt")
    if config_path.exists():
        with open(config_path, "r") as f:
            token = f.read().strip()
            return token if token else None
    return None

# Global validation state - centralized token validation
_global_validation_state = {
    "validated": False,
    "is_valid": None,
    "last_warning_time": 0,
    "validation_token": None,
    "validation_api_key": None
}

def _validate_kite_token_centralized(kite: KiteConnect, api_key: str, access_token: str, verbose: bool = False):
    """Centralized token validation - only runs once globally per token/API key combination"""
    import time
    
    # Check if we've already validated this exact token/API key combination
    if _global_validation_state["validated"]:
        if _global_validation_state["validation_token"] == access_token[:20] and \
           _global_validation_state["validation_api_key"] == api_key:
            # Same token, use cached result - no need to validate again
            is_valid = _global_validation_state["is_valid"]
            if not is_valid:
                # Only warn once per minute to reduce log noise
                current_time = time.time()
                if current_time - _global_validation_state["last_warning_time"] > 60:
                    print(f"[Kite Auth] WARNING: Token validation failed. Historical data may still work. (Warning shown once per minute)")
                    _global_validation_state["last_warning_time"] = current_time
            return is_valid
    
    # First time validation for this token/API key - do it now (only once globally)
    _global_validation_state["validated"] = True
    _global_validation_state["validation_token"] = access_token[:20]
    _global_validation_state["validation_api_key"] = api_key
    
    try:
        kite.profile()
        _global_validation_state["is_valid"] = True
        if verbose:
            print(f"[Kite Auth] Token validated successfully")
        return True
    except Exception as profile_error:
        error_str = str(profile_error).lower()
        # Only warn if it's clearly a token/auth issue
        if "invalid" in error_str or "expired" in error_str or "token" in error_str or "unauthorized" in error_str:
            _global_validation_state["is_valid"] = False
            _global_validation_state["last_warning_time"] = time.time()
            print(f"[Kite Auth] WARNING: Token validation failed: {profile_error}")
            print(f"[Kite Auth] API Key: {api_key[:15]}... | This warning appears once per minute. Historical data may still work.")
            return False
        else:
            # Network or other errors - don't mark as invalid, but don't cache as valid either
            if verbose:
                print(f"[Kite Auth] Profile check failed (non-auth error): {profile_error}")
            return None  # Unknown state

def get_kite_instance(user_id: str = "default", verbose: bool = False, skip_validation: bool = False):
    """Get authenticated KiteConnect instance
    
    Args:
        user_id: User ID for getting user-specific API key
        verbose: If True, log detailed information (default: False to reduce log noise)
        skip_validation: If True, skip token validation entirely (for performance)
    """
    access_token = get_access_token()
    if not access_token:
        # Note: In a production app, you might want to return None or handle this differently 
        # for background tasks vs API requests
        raise HTTPException(
            status_code=401, 
            detail="Access token not found. Please authenticate first. "
                   "Steps: 1) GET /auth to get login URL, 2) Login through that URL, "
                   "3) POST /set-token with the request_token from redirect to store access token."
        )
    
    # Validate token length before using it (Kite access tokens can be 32 characters)
    if len(access_token) < 20:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid access token detected (length: {len(access_token)} chars). "
                   "Kite Connect access tokens should be at least 20 characters. "
                   "The stored token appears to be invalid or corrupted. "
                   "Please regenerate: 1) DELETE /access-token to clear invalid token, "
                   "2) GET /auth to get login URL, 3) Login and POST /set-token with request_token."
        )
    
    # Get user-specific API key
    current_api_key = get_kite_api_key(user_id=user_id)
    
    kite = KiteConnect(api_key=current_api_key)
    kite.set_access_token(access_token)
    
    # Check if this is the first validation call
    is_first_validation = not _global_validation_state["validated"]
    
    # Centralized validation - only runs once globally
    if not skip_validation:
        _validate_kite_token_centralized(kite, current_api_key, access_token, verbose)
    
    # Minimal logging - only on first validation call and if verbose
    if verbose and is_first_validation:
        print(f"[get_kite_instance] User ID: {user_id}")
        print(f"[get_kite_instance] API Key: {current_api_key[:15] if current_api_key and len(current_api_key) > 15 else 'NOT SET'}...")
        print(f"[get_kite_instance] Token length: {len(access_token)}, preview: {access_token[:20]}...")
    
    return kite

def calculate_trend_and_suggestions(kite: KiteConnect, instrument_token: int, current_price: float):
    """Calculate trend based on 5min, 15min, 30min candles and provide buy/sell suggestions"""
    try:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        trend_scores = {
            "5min": 0,
            "15min": 0,
            "30min": 0
        }
        
        intervals = {
            "5min": "5minute",
            "15min": "15minute",
            "30min": "30minute"
        }
        
        all_closes = []
        
        # Get candles for each timeframe
        for timeframe, kite_interval in intervals.items():
            try:
                historical_data = kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=yesterday,
                    to_date=today,
                    interval=kite_interval
                )
                
                if historical_data and len(historical_data) > 0:
                    # Get last 5 candles
                    recent_candles = historical_data[-5:] if len(historical_data) >= 5 else historical_data
                    closes = [candle.get("close", 0) for candle in recent_candles if candle.get("close", 0) > 0]
                    
                    if len(closes) >= 2:
                        all_closes.extend(closes)
                        
                        # Calculate trend: compare last close with previous closes
                        last_close = closes[-1]
                        prev_close = closes[-2] if len(closes) > 1 else closes[0]
                        
                        # Simple trend: if last close > previous, bullish (+1), else bearish (-1)
                        if last_close > prev_close:
                            trend_scores[timeframe] = 1
                        elif last_close < prev_close:
                            trend_scores[timeframe] = -1
                        else:
                            trend_scores[timeframe] = 0
            except Exception as e:
                print(f"Error fetching {timeframe} candles: {e}")
                continue
        
        # Calculate overall trend
        total_score = trend_scores["5min"] + trend_scores["15min"] + trend_scores["30min"]
        
        # Build reason explanation
        reason_parts = []
        
        # Add timeframe analysis to reason
        timeframe_analysis = []
        for tf, score in trend_scores.items():
            if score > 0:
                timeframe_analysis.append(f"{tf} bullish")
            elif score < 0:
                timeframe_analysis.append(f"{tf} bearish")
            else:
                timeframe_analysis.append(f"{tf} neutral")
        
        if timeframe_analysis:
            reason_parts.append(f"Timeframes: {', '.join(timeframe_analysis)}")
        
        # Determine trend
        if total_score >= 2:
            trend = "BULLISH"
            trend_strength = min(abs(total_score) / 3.0, 1.0)  # Normalize to 0-1
            reason_parts.append(f"Strong bullish momentum (score: {total_score}/3)")
        elif total_score <= -2:
            trend = "BEARISH"
            trend_strength = min(abs(total_score) / 3.0, 1.0)
            reason_parts.append(f"Strong bearish momentum (score: {total_score}/3)")
        elif total_score > 0:
            trend = "WEAK_BULLISH"
            trend_strength = min(abs(total_score) / 3.0, 1.0)
            reason_parts.append(f"Weak bullish sentiment (score: {total_score}/3)")
        elif total_score < 0:
            trend = "WEAK_BEARISH"
            trend_strength = min(abs(total_score) / 3.0, 1.0)
            reason_parts.append(f"Weak bearish sentiment (score: {total_score}/3)")
        else:
            trend = "NEUTRAL"
            trend_strength = 0
            reason_parts.append("Consolidating (score: 0/3)")
            
        # Suggestions
        buy_price = current_price * 1.001 if trend == "BULLISH" else current_price
        sell_price = current_price * 0.999 if trend == "BEARISH" else current_price
        
        return {
            "trend": trend,
            "trend_strength": trend_strength,
            "buy_price": round(buy_price, 2),
            "sell_price": round(sell_price, 2),
            "reason": ". ".join(reason_parts)
        }
    except Exception as e:
        print(f"Error in trend calculation: {e}")
        return {
            "trend": "NEUTRAL",
            "trend_strength": 0,
            "buy_price": current_price,
            "sell_price": current_price,
            "reason": f"Calculation error: {str(e)}"
        }

