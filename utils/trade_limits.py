"""
Trade limits management service
"""
import json
import os
from datetime import datetime, date
from threading import RLock
from typing import Dict, Optional
from utils.logger import log_info, log_warning, log_error


class TradeLimits:
    """Manages daily trade limits and profit targets"""
    
    def __init__(self):
        self._lock = RLock()
        self.limits_file = "data/trade_limits.json"
        self.default_limits = {
            "max_trades_per_day": 10,
            "max_profit_per_day": 0.02,  # 2%
            "max_loss_per_day": 0.05,    # 5%
            # Absolute caps (INR). 0 = disabled.
            "max_premium_inr_per_day": float(os.getenv("MAX_PREMIUM_INR_PER_DAY", "0") or 0),
            "max_loss_inr_per_day": float(os.getenv("MAX_LOSS_INR_PER_DAY", "0") or 0),
            "current_day": str(date.today()),
            "trades_today": 0,
            "profit_today": 0.0,
            "loss_today": 0.0,
            # Options premium / capital deployed (INR)
            "total_investment_today": 0.0,
            # Net realized P&L for the day (INR). Negative = loss.
            "pnl_inr_today": 0.0,
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

    def set_pnl_inr_today(self, value: float) -> None:
        with self._lock:
            self.limits["pnl_inr_today"] = float(value or 0)
            self._save_limits(self.limits)
    
    def can_place_trade(self, investment_amount: float = 0) -> tuple[bool, str]:
        """Check if a new trade can be placed"""
        # Absolute premium cap (INR)
        try:
            max_prem = float(self.limits.get("max_premium_inr_per_day") or 0)
            spent = float(self.limits.get("total_investment_today") or 0)
            if max_prem > 0 and (spent + float(investment_amount or 0)) > max_prem:
                return (
                    False,
                    f"Daily premium cap reached (spent ₹{spent:.0f}, cap ₹{max_prem:.0f})",
                )
        except Exception:
            pass

        # Absolute max loss cap (INR)
        try:
            max_loss_inr = float(self.limits.get("max_loss_inr_per_day") or 0)
            pnl = float(self.limits.get("pnl_inr_today") or 0)
            if max_loss_inr > 0 and pnl <= -abs(max_loss_inr):
                return (
                    False,
                    f"Daily loss cap reached (P&L ₹{pnl:.0f}, cap -₹{abs(max_loss_inr):.0f})",
                )
        except Exception:
            pass

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
        with self._lock:
            self.limits["trades_today"] += 1
            self.limits["total_investment_today"] += float(investment_amount or 0)
            self._save_limits(self.limits)
            log_info(
                f"Trade recorded: {self.limits['trades_today']}/"
                f"{self.limits['max_trades_per_day']} trades today"
            )

    def rollback_trade(self, investment_amount: float = 0) -> None:
        """Reverse record_trade when an unfilled entry is cancelled."""
        with self._lock:
            self.limits["trades_today"] = max(0, int(self.limits.get("trades_today") or 0) - 1)
            self.limits["total_investment_today"] = max(
                0.0,
                float(self.limits.get("total_investment_today") or 0) - float(investment_amount or 0),
            )
            self._save_limits(self.limits)
            log_info(
                f"Trade slot rolled back: {self.limits['trades_today']}/"
                f"{self.limits['max_trades_per_day']} trades today"
            )
    
    def record_profit_loss(self, pnl_amount: float, investment_amount: float = 0):
        """Record profit/loss from a closed position"""
        with self._lock:
            try:
                self.limits["pnl_inr_today"] = float(self.limits.get("pnl_inr_today") or 0) + float(
                    pnl_amount or 0
                )
            except Exception:
                pass

            if pnl_amount > 0:
                if investment_amount > 0:
                    profit_pct = pnl_amount / investment_amount
                    self.limits["profit_today"] += profit_pct
                    log_info(f"Profit recorded: {pnl_amount:.2f} ({profit_pct*100:.2f}%)")
                else:
                    self.limits["profit_today"] += pnl_amount
                    log_info(f"Profit recorded: {pnl_amount:.2f}")
            elif pnl_amount < 0:
                if investment_amount > 0:
                    loss_pct = abs(pnl_amount) / investment_amount
                    self.limits["loss_today"] += loss_pct
                    log_info(f"Loss recorded: {abs(pnl_amount):.2f} ({loss_pct*100:.2f}%)")
                else:
                    self.limits["loss_today"] += abs(pnl_amount)
                    log_info(f"Loss recorded: {abs(pnl_amount):.2f}")

            self._save_limits(self.limits)
    
    def get_limits_status(self) -> Dict:
        """Get current limits status"""
        return {
            "max_trades_per_day": self.limits["max_trades_per_day"],
            "trades_today": self.limits["trades_today"],
            "trades_remaining": max(0, self.limits["max_trades_per_day"] - self.limits["trades_today"]),
            "max_premium_inr_per_day": float(self.limits.get("max_premium_inr_per_day") or 0),
            "premium_spent_inr_today": float(self.limits.get("total_investment_today") or 0),
            "max_loss_inr_per_day": float(self.limits.get("max_loss_inr_per_day") or 0),
            "pnl_inr_today": float(self.limits.get("pnl_inr_today") or 0),
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
            "total_investment_today": 0.0,
            "pnl_inr_today": 0.0,
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
