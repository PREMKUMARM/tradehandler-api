"""
Trade limits management service
"""
import json
import os
from datetime import datetime, date
from typing import Dict, Optional
from utils.logger import log_info, log_warning, log_error


class TradeLimits:
    """Manages daily trade limits and profit targets"""
    
    def __init__(self):
        self.limits_file = "data/trade_limits.json"
        self.default_limits = {
            "max_trades_per_day": 10,
            "max_profit_per_day": 0.02,  # 2%
            "max_loss_per_day": 0.05,    # 5%
            "current_day": str(date.today()),
            "trades_today": 0,
            "profit_today": 0.0,
            "loss_today": 0.0,
            "total_investment_today": 0.0
        }
        self.limits = self._load_limits()
    
    def _load_limits(self) -> Dict:
        """Load trade limits from file"""
        try:
            if os.path.exists(self.limits_file):
                with open(self.limits_file, 'r') as f:
                    limits = json.load(f)
                
                # Reset if it's a new day
                if limits.get("current_day") != str(date.today()):
                    limits = self.default_limits.copy()
                    limits["current_day"] = str(date.today())
                    self._save_limits(limits)
                
                return limits
            else:
                # Create default limits file
                os.makedirs(os.path.dirname(self.limits_file), exist_ok=True)
                limits = self.default_limits.copy()
                self._save_limits(limits)
                return limits
        except Exception as e:
            log_error(f"Error loading trade limits: {e}")
            return self.default_limits.copy()
    
    def _save_limits(self, limits: Dict):
        """Save trade limits to file"""
        try:
            os.makedirs(os.path.dirname(self.limits_file), exist_ok=True)
            with open(self.limits_file, 'w') as f:
                json.dump(limits, f, indent=2)
        except Exception as e:
            log_error(f"Error saving trade limits: {e}")
    
    def can_place_trade(self, investment_amount: float = 0) -> tuple[bool, str]:
        """Check if a new trade can be placed"""
        # Check trade count limit
        if self.limits["trades_today"] >= self.limits["max_trades_per_day"]:
            return False, f"Daily trade limit reached ({self.limits['max_trades_per_day']} trades)"
        
        # Check profit limit
        if self.limits["profit_today"] >= self.limits["max_profit_per_day"]:
            return False, f"Daily profit target reached ({self.limits['max_profit_per_day']*100:.1f}%)"
        
        # Check loss limit
        if self.limits["loss_today"] >= self.limits["max_loss_per_day"]:
            return False, f"Daily loss limit reached ({self.limits['max_loss_per_day']*100:.1f}%)"
        
        return True, "Trade allowed"
    
    def record_trade(self, investment_amount: float = 0):
        """Record a new trade"""
        self.limits["trades_today"] += 1
        self.limits["total_investment_today"] += investment_amount
        self._save_limits(self.limits)
        log_info(f"Trade recorded: {self.limits['trades_today']}/{self.limits['max_trades_per_day']} trades today")
    
    def record_profit_loss(self, pnl_amount: float, investment_amount: float = 0):
        """Record profit/loss from a closed position"""
        if pnl_amount > 0:
            # Calculate profit as percentage of investment
            if investment_amount > 0:
                profit_pct = pnl_amount / investment_amount
                self.limits["profit_today"] += profit_pct
            else:
                # If no investment amount, add absolute profit
                self.limits["profit_today"] += pnl_amount
            log_info(f"Profit recorded: {pnl_amount:.2f} ({profit_pct*100:.2f}%)")
        elif pnl_amount < 0:
            # Calculate loss as percentage of investment
            if investment_amount > 0:
                loss_pct = abs(pnl_amount) / investment_amount
                self.limits["loss_today"] += loss_pct
            else:
                # If no investment amount, add absolute loss
                self.limits["loss_today"] += abs(pnl_amount)
            log_info(f"Loss recorded: {abs(pnl_amount):.2f} ({loss_pct*100:.2f}%)")
        
        self._save_limits(self.limits)
    
    def get_limits_status(self) -> Dict:
        """Get current limits status"""
        return {
            "max_trades_per_day": self.limits["max_trades_per_day"],
            "trades_today": self.limits["trades_today"],
            "trades_remaining": max(0, self.limits["max_trades_per_day"] - self.limits["trades_today"]),
            "max_profit_per_day_pct": self.limits["max_profit_per_day"] * 100,
            "profit_today_pct": self.limits["profit_today"] * 100,
            "profit_remaining_pct": max(0, (self.limits["max_profit_per_day"] - self.limits["profit_today"]) * 100),
            "max_loss_per_day_pct": self.limits["max_loss_per_day"] * 100,
            "loss_today_pct": self.limits["loss_today"] * 100,
            "loss_remaining_pct": max(0, (self.limits["max_loss_per_day"] - self.limits["loss_today"]) * 100),
            "current_day": self.limits["current_day"],
            "can_trade": self.can_place_trade()[0]
        }
    
    def reset_daily_limits(self):
        """Reset daily limits (for testing or manual reset)"""
        self.limits.update({
            "current_day": str(date.today()),
            "trades_today": 0,
            "profit_today": 0.0,
            "loss_today": 0.0,
            "total_investment_today": 0.0
        })
        self._save_limits(self.limits)
        log_info("Daily trade limits reset")
    
    def update_limits(self, max_trades: Optional[int] = None, max_profit_pct: Optional[float] = None, 
                     max_loss_pct: Optional[float] = None):
        """Update limits configuration"""
        if max_trades is not None:
            self.limits["max_trades_per_day"] = max_trades
        if max_profit_pct is not None:
            self.limits["max_profit_per_day"] = max_profit_pct / 100
        if max_loss_pct is not None:
            self.limits["max_loss_per_day"] = max_loss_pct / 100
        
        self._save_limits(self.limits)
        log_info(f"Trade limits updated: max_trades={max_trades}, max_profit={max_profit_pct}%, max_loss={max_loss_pct}%")


# Global instance
trade_limits = TradeLimits()
