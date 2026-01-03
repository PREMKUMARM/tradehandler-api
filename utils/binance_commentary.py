"""
Live Commentary Generator for Binance Futures
Generates human-readable commentary messages for trading events
"""
from typing import Dict, Optional, List
from datetime import datetime


class CommentaryGenerator:
    """Generates commentary messages for Binance trading events"""
    
    def __init__(self):
        self.previous_states: Dict[str, Dict] = {}  # Track previous state for each symbol
    
    def generate_commentary(
        self,
        symbol: str,
        current_data: Dict,
        previous_data: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate commentary messages based on current and previous data
        
        Args:
            symbol: Trading symbol (e.g., "1000PEPEUSDT")
            current_data: Current market data with indicators
            previous_data: Previous market data for comparison
        
        Returns:
            List of commentary messages (can be empty if no significant events)
        """
        messages = []
        symbol_upper = symbol.upper()
        
        # Get previous state for this symbol
        prev_state = previous_data or self.previous_states.get(symbol_upper, {})
        
        # Extract current values
        current_price = current_data.get('price', 0)
        current_pattern = current_data.get('candle_pattern', '')
        current_signal = current_data.get('signal')
        current_vwap = current_data.get('vwap', 0)
        current_rsi = current_data.get('rsi')
        current_volume = current_data.get('volume', 0)
        current_high = current_data.get('high', 0)
        current_low = current_data.get('low', 0)
        buy_conditions_met = current_data.get('buy_conditions_met', 0)
        buy_conditions_total = current_data.get('buy_conditions_total', 8)
        sell_conditions_met = current_data.get('sell_conditions_met', 0)
        sell_conditions_total = current_data.get('sell_conditions_total', 8)
        
        # Get candle timestamp (use candle's actual timestamp, not current time)
        candle_timestamp = current_data.get('timestamp', 0)
        if not candle_timestamp:
            # Fallback to current time if no timestamp provided
            candle_timestamp = int(datetime.now().timestamp() * 1000)
        
        # Extract previous values
        prev_price = prev_state.get('price', current_price)
        prev_pattern = prev_state.get('candle_pattern', '')
        prev_signal = prev_state.get('signal')
        prev_vwap = prev_state.get('vwap', current_vwap)
        prev_rsi = prev_state.get('rsi')
        prev_volume = prev_state.get('volume', current_volume)
        
        # 1. New Pattern Detected
        if current_pattern and current_pattern != prev_pattern and prev_pattern:
            pattern_msg = self._format_pattern_message(
                symbol_upper, current_pattern, current_price, current_vwap, candle_timestamp
            )
            messages.append(pattern_msg)
        
        # 2. Trading Signal Generated
        if current_signal and current_signal != prev_signal:
            signal_msg = self._format_signal_message(
                symbol_upper, current_signal, current_data, candle_timestamp
            )
            messages.append(signal_msg)
        
        # 3. VWAP Crossover
        if prev_vwap and current_vwap:
            if prev_price <= prev_vwap and current_price > current_vwap:
                messages.append({
                    "timestamp": candle_timestamp,
                    "symbol": symbol_upper,
                    "event_type": "vwap_crossover",
                    "priority": "medium",
                    "message": f"{symbol_upper} price crossed above VWAP at ${current_price:.8f}",
                    "details": {
                        "price": round(current_price, 8),
                        "vwap": round(current_vwap, 8),
                        "direction": "above"
                    }
                })
            elif prev_price >= prev_vwap and current_price < current_vwap:
                messages.append({
                    "timestamp": candle_timestamp,
                    "symbol": symbol_upper,
                    "event_type": "vwap_crossover",
                    "priority": "medium",
                    "message": f"{symbol_upper} price crossed below VWAP at ${current_price:.8f}",
                    "details": {
                        "price": round(current_price, 8),
                        "vwap": round(current_vwap, 8),
                        "direction": "below"
                    }
                })
        
        # 4. Significant Price Movement (>0.5%)
        if prev_price and prev_price > 0:
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
            if abs(price_change_pct) >= 0.5:
                direction = "surged" if price_change_pct > 0 else "dropped"
                messages.append({
                    "timestamp": candle_timestamp,
                    "symbol": symbol_upper,
                    "event_type": "price_movement",
                    "priority": "medium",
                    "message": f"{symbol_upper} price {direction} {abs(price_change_pct):.2f}% to ${current_price:.8f}",
                    "details": {
                        "price": round(current_price, 8),
                        "previous_price": round(prev_price, 8),
                        "change_percent": round(price_change_pct, 2)
                    }
                })
        
        # 5. RSI Threshold Crossings
        if prev_rsi is not None and current_rsi is not None:
            if prev_rsi < 70 and current_rsi >= 70:
                messages.append({
                    "timestamp": candle_timestamp,
                    "symbol": symbol_upper,
                    "event_type": "indicator_change",
                    "priority": "low",
                    "message": f"{symbol_upper} RSI crossed above 70 (overbought) - current: {current_rsi:.1f}",
                    "details": {
                        "indicator": "RSI",
                        "value": round(current_rsi, 2),
                        "threshold": 70,
                        "status": "overbought"
                    }
                })
            elif prev_rsi > 30 and current_rsi <= 30:
                messages.append({
                    "timestamp": candle_timestamp,
                    "symbol": symbol_upper,
                    "event_type": "indicator_change",
                    "priority": "low",
                    "message": f"{symbol_upper} RSI dropped below 30 (oversold) - current: {current_rsi:.1f}",
                    "details": {
                        "indicator": "RSI",
                        "value": round(current_rsi, 2),
                        "threshold": 30,
                        "status": "oversold"
                    }
                })
        
        # 6. Volume Spike (>150% of average)
        if prev_volume and prev_volume > 0:
            volume_ratio = current_volume / prev_volume if prev_volume > 0 else 1
            if volume_ratio >= 1.5:
                messages.append({
                    "timestamp": candle_timestamp,
                    "symbol": symbol_upper,
                    "event_type": "volume_spike",
                    "priority": "medium",
                    "message": f"Volume spike detected on {symbol_upper}: {volume_ratio:.1f}x average volume",
                    "details": {
                        "current_volume": round(current_volume, 2),
                        "previous_volume": round(prev_volume, 2),
                        "volume_ratio": round(volume_ratio, 2)
                    }
                })
        
        # 7. Buy/Sell Conditions Met
        if buy_conditions_met == buy_conditions_total and buy_conditions_total > 0:
            prev_buy_met = prev_state.get('buy_conditions_met', 0)
            if prev_buy_met < buy_conditions_met:
                messages.append({
                    "timestamp": candle_timestamp,
                    "symbol": symbol_upper,
                    "event_type": "condition_met",
                    "priority": "high",
                    "message": f"✅ All BUY conditions met for {symbol_upper}! ({buy_conditions_met}/{buy_conditions_total})",
                    "details": {
                        "conditions_met": buy_conditions_met,
                        "conditions_total": buy_conditions_total,
                        "signal_type": "BUY"
                    }
                })
        
        if sell_conditions_met == sell_conditions_total and sell_conditions_total > 0:
            prev_sell_met = prev_state.get('sell_conditions_met', 0)
            if prev_sell_met < sell_conditions_met:
                messages.append({
                    "timestamp": candle_timestamp,
                    "symbol": symbol_upper,
                    "event_type": "condition_met",
                    "priority": "high",
                    "message": f"✅ All SELL conditions met for {symbol_upper}! ({sell_conditions_met}/{sell_conditions_total})",
                    "details": {
                        "conditions_met": sell_conditions_met,
                        "conditions_total": sell_conditions_total,
                        "signal_type": "SELL"
                    }
                })
        
        # 8. New 5-minute candle formed (when timestamp changes significantly)
        # Removed: We don't need a separate message for new candle formation.
        # Any commentary message already indicates a new candle was formed.
        # current_timestamp = current_data.get('timestamp', 0)
        # prev_timestamp = prev_state.get('timestamp', 0)
        # # If timestamp changed by more than 4 minutes (240000ms), new candle formed
        # if prev_timestamp and (current_timestamp - prev_timestamp) >= 240000:
        #     messages.append({
        #         "timestamp": candle_timestamp,
        #         "symbol": symbol_upper,
        #         "event_type": "new_candle",
        #         "priority": "low",
        #         "message": f"New 5-minute candle formed for {symbol_upper} at ${current_price:.8f}",
        #         "details": {
        #             "price": round(current_price, 8),
        #             "high": round(current_high, 8),
        #             "low": round(current_low, 8),
        #             "pattern": current_pattern
        #         }
        #     })
        
        # Update previous state
        self.previous_states[symbol_upper] = {
            'price': current_price,
            'candle_pattern': current_pattern,
            'signal': current_signal,
            'vwap': current_vwap,
            'rsi': current_rsi,
            'volume': current_volume,
            'timestamp': current_timestamp,
            'buy_conditions_met': buy_conditions_met,
            'sell_conditions_met': sell_conditions_met
        }
        
        return messages
    
    def _format_pattern_message(
        self,
        symbol: str,
        pattern: str,
        price: float,
        vwap: float,
        candle_timestamp: int
    ) -> Dict:
        """Format pattern detection message"""
        vwap_dist = abs(price - vwap) / vwap * 100 if vwap > 0 else 0
        vwap_pos = "above" if price > vwap else "below"
        
        return {
            "timestamp": candle_timestamp,
            "symbol": symbol,
            "event_type": "pattern_detected",
            "priority": "medium",
            "message": f"{pattern} pattern detected on {symbol} at ${price:.8f}. Price is {vwap_dist:.2f}% {vwap_pos} VWAP.",
            "details": {
                "pattern": pattern,
                "price": round(price, 8),
                "vwap": round(vwap, 8),
                "vwap_distance_percent": round(vwap_dist, 2)
            }
        }
    
    def _format_signal_message(
        self,
        symbol: str,
        signal: str,
        data: Dict,
        candle_timestamp: int
    ) -> Dict:
        """Format trading signal message"""
        price = data.get('price', 0)
        pattern = data.get('candle_pattern', '')
        reason = data.get('signal_reason', '')
        priority = data.get('signal_priority', 0)
        
        signal_emoji = "🟢" if signal == "BUY" else "🔴"
        priority_text = "Priority 1" if priority == 1 else f"Priority {priority}"
        
        message = f"{signal_emoji} {signal} signal generated for {symbol}!"
        if pattern:
            message += f" Pattern: {pattern}."
        if reason:
            message += f" {reason}"
        
        return {
            "timestamp": candle_timestamp,
            "symbol": symbol,
            "event_type": "signal_generated",
            "priority": "high",
            "message": message,
            "details": {
                "signal": signal,
                "signal_priority": priority,
                "pattern": pattern,
                "reason": reason,
                "price": round(price, 8)
            }
        }


# Global commentary generator instance
_commentary_generator = CommentaryGenerator()

def get_commentary_generator() -> CommentaryGenerator:
    """Get the global commentary generator instance"""
    return _commentary_generator

