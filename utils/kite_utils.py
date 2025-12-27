import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from kiteconnect import KiteConnect
from fastapi import HTTPException

# Global API key - get from AgentConfig (managed via UI) or environment
def get_kite_api_key():
    """Get Kite API key from AgentConfig or environment"""
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

def get_kite_instance():
    """Get authenticated KiteConnect instance"""
    access_token = get_access_token()
    if not access_token:
        # Note: In a production app, you might want to return None or handle this differently 
        # for background tasks vs API requests
        raise HTTPException(status_code=401, detail="Access token not found. Please authenticate first.")
    
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
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

