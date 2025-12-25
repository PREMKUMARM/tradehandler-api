"""
Safety mechanisms for the agent
"""
from typing import Dict, Any, Optional
from datetime import datetime, time
from agent.config import get_agent_config


class SafetyManager:
    """Manages safety limits and circuit breakers"""
    
    def __init__(self):
        self.config = get_agent_config()
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.last_reset_date: Optional[datetime] = None
        self.circuit_breaker_active: bool = False
    
    def reset_daily_counters(self):
        """Reset daily counters if it's a new day"""
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
            self.circuit_breaker_active = False
    
    def check_trading_hours(self) -> tuple[bool, str]:
        """Check if current time is within trading hours"""
        now = datetime.now().time()
        
        try:
            start_time = datetime.strptime(self.config.trading_start_time, "%H:%M").time()
            end_time = datetime.strptime(self.config.trading_end_time, "%H:%M").time()
        except:
            start_time = time(9, 15)
            end_time = time(15, 30)
        
        if start_time <= now <= end_time:
            return True, "Trading hours"
        else:
            return False, f"Outside trading hours ({self.config.trading_start_time} - {self.config.trading_end_time})"
    
    def check_trade_size(self, trade_value: float) -> tuple[bool, str]:
        """Check if trade size is within limits"""
        self.reset_daily_counters()
        
        if trade_value > self.config.max_position_size:
            return False, f"Trade value {trade_value} exceeds max position size {self.config.max_position_size}"
        
        return True, "Trade size OK"
    
    def check_daily_limits(self, trade_value: float, risk_amount: float) -> tuple[bool, str]:
        """Check daily trading limits"""
        self.reset_daily_counters()
        
        # Check daily loss limit
        if self.daily_pnl < -abs(self.config.daily_loss_limit):
            return False, f"Daily loss limit reached: {self.daily_pnl} (limit: {self.config.daily_loss_limit})"
        
        # Check trade count
        if self.daily_trades >= self.config.max_trades_per_day:
            return False, f"Daily trade limit reached: {self.daily_trades} (limit: {self.config.max_trades_per_day})"
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker is active - trading halted"
        
        return True, "Daily limits OK"
    
    def check_circuit_breaker(self) -> tuple[bool, str]:
        """Check circuit breaker status"""
        self.reset_daily_counters()
        
        if not self.config.circuit_breaker_enabled:
            return True, "Circuit breaker disabled"
        
        if abs(self.daily_pnl) >= self.config.circuit_breaker_loss_threshold:
            self.circuit_breaker_active = True
            return False, f"Circuit breaker triggered: daily PnL {self.daily_pnl} exceeds threshold {self.config.circuit_breaker_loss_threshold}"
        
        return True, "Circuit breaker OK"
    
    def validate_trade(self, trade_value: float, risk_amount: float) -> Dict[str, Any]:
        """Validate a trade against all safety checks"""
        self.reset_daily_counters()
        
        checks = []
        
        # Trading hours
        hours_ok, hours_msg = self.check_trading_hours()
        checks.append({"check": "trading_hours", "passed": hours_ok, "message": hours_msg})
        
        # Trade size
        size_ok, size_msg = self.check_trade_size(trade_value)
        checks.append({"check": "trade_size", "passed": size_ok, "message": size_msg})
        
        # Daily limits
        daily_ok, daily_msg = self.check_daily_limits(trade_value, risk_amount)
        checks.append({"check": "daily_limits", "passed": daily_ok, "message": daily_msg})
        
        # Circuit breaker
        circuit_ok, circuit_msg = self.check_circuit_breaker()
        checks.append({"check": "circuit_breaker", "passed": circuit_ok, "message": circuit_msg})
        
        all_passed = all(c["passed"] for c in checks)
        
        return {
            "is_valid": all_passed,
            "checks": checks,
            "can_proceed": all_passed
        }
    
    def record_trade(self, trade_value: float, pnl: float = 0.0):
        """Record a trade for daily tracking"""
        self.reset_daily_counters()
        self.daily_trades += 1
        self.daily_pnl += pnl
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        self.circuit_breaker_active = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        self.reset_daily_counters()
        
        return {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "circuit_breaker_active": self.circuit_breaker_active,
            "max_position_size": self.config.max_position_size,
            "daily_loss_limit": self.config.daily_loss_limit,
            "max_trades_per_day": self.config.max_trades_per_day
        }


# Global safety manager instance
_safety_manager: Optional[SafetyManager] = None


def get_safety_manager() -> SafetyManager:
    """Get or create safety manager instance"""
    global _safety_manager
    if _safety_manager is None:
        _safety_manager = SafetyManager()
    return _safety_manager

