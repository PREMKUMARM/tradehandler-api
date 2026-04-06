"""
Automated strategy execution system
"""
import asyncio
import json
import os
from datetime import datetime, time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from utils.kite_utils import get_kite_instance
from utils.trade_limits import trade_limits
from utils.order_monitor import order_monitor
from utils.logger import log_info, log_error, log_warning
from strategies.runner import run_strategy_on_candles
from simulation.helpers import get_instrument_history, add_live_log
from core.exceptions import ValidationError
from services.risk_gate import check_order_allowed, record_order_placed
from services.paper_trading import is_paper_mode, paper_place_order
from services.execution_audit import log_execution_audit
from services.strategy_run_fills import record_strategy_fill_if_run


class StrategyType(Enum):
    CANDLE_BREAK_915 = "915_candle_break"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    SUPPORT_RESISTANCE = "support_resistance"
    RSI_REVERSAL = "rsi_reversal"
    MACD_CROSSOVER = "macd_crossover"
    EMA_CROSS = "ema_cross"


@dataclass
class StrategyConfig:
    """Configuration for a strategy"""
    strategy_type: StrategyType
    symbol: str
    quantity: int
    product: str = "MIS"
    order_type: str = "MARKET"
    stoploss_pct: float = 0.02  # 2%
    target_pct: float = 0.04    # 4%
    trailing_stoploss: bool = False
    max_trades_per_day: int = 3
    active_hours_start: time = time(9, 15)
    active_hours_end: time = time(15, 25)
    enabled: bool = True


@dataclass
class TradeSignal:
    """Represents a trading signal"""
    strategy_type: StrategyType
    symbol: str
    action: str  # BUY or SELL
    option_type: str  # CE or PE
    entry_price: float
    stoploss: float
    target: float
    quantity: int
    reason: str
    timestamp: datetime
    confidence: float = 0.8


class StrategyExecutor:
    """Manages and executes automated trading strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, StrategyConfig] = {}
        self.active_trades: Dict[str, Dict] = {}
        self.is_running = False
        self.execution_task = None
        self.candle_cache: Dict[str, List] = {}
        
    def add_strategy(self, strategy_id: str, config: StrategyConfig):
        """Add a strategy configuration"""
        self.strategies[strategy_id] = config
        log_info(f"Added strategy {strategy_id}: {config.strategy_type.value} for {config.symbol}")
    
    def remove_strategy(self, strategy_id: str):
        """Remove a strategy configuration"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            log_info(f"Removed strategy {strategy_id}")
    
    def enable_strategy(self, strategy_id: str):
        """Enable a strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id].enabled = True
            log_info(f"Enabled strategy {strategy_id}")
    
    def disable_strategy(self, strategy_id: str):
        """Disable a strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id].enabled = False
            log_info(f"Disabled strategy {strategy_id}")
    
    async def start_execution(self):
        """Start the strategy execution loop"""
        if self.is_running:
            return
        
        self.is_running = True
        self.execution_task = asyncio.create_task(self._execution_loop())
        log_info("Strategy executor started")
    
    async def stop_execution(self):
        """Stop the strategy execution loop"""
        self.is_running = False
        if self.execution_task:
            self.execution_task.cancel()
            try:
                await self.execution_task
            except asyncio.CancelledError:
                pass
        log_info("Strategy executor stopped")
    
    async def _execution_loop(self):
        """Main execution loop"""
        while self.is_running:
            try:
                current_time = datetime.now().time()
                
                # Check if within trading hours
                if self._is_trading_time(current_time):
                    await self._execute_strategies()
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(f"Error in strategy execution loop: {e}")
                await asyncio.sleep(60)
    
    def _is_trading_time(self, current_time: time) -> bool:
        """Check if current time is within trading hours"""
        for strategy in self.strategies.values():
            if (strategy.enabled and 
                strategy.active_hours_start <= current_time <= strategy.active_hours_end):
                return True
        return False
    
    async def _execute_strategies(self):
        """Execute all enabled strategies"""
        for strategy_id, config in self.strategies.items():
            try:
                if not config.enabled:
                    continue
                
                # Check if we've hit max trades for this strategy today
                if not self._can_trade_strategy(strategy_id):
                    continue
                
                # Generate signal
                signal = await self._generate_signal(config)
                
                if signal:
                    # Execute trade
                    await self._execute_trade(strategy_id, signal, config)
                    
            except Exception as e:
                log_error(f"Error executing strategy {strategy_id}: {e}")
    
    def _can_trade_strategy(self, strategy_id: str) -> bool:
        """Check if strategy can place more trades"""
        # Check overall trade limits
        can_trade, message = trade_limits.can_place_trade()
        if not can_trade:
            log_warning(f"Trade limits reached: {message}")
            return False
        
        # Check strategy-specific limits
        strategy_trades_today = len([
            t for t in self.active_trades.values() 
            if t.get("strategy_id") == strategy_id and 
            t.get("date") == datetime.now().date()
        ])
        
        config = self.strategies.get(strategy_id)
        if config and strategy_trades_today >= config.max_trades_per_day:
            log_warning(f"Strategy {strategy_id} max trades reached for today")
            return False
        
        return True
    
    async def _generate_signal(self, config: StrategyConfig) -> Optional[TradeSignal]:
        """Generate trading signal for a strategy"""
        try:
            kite = get_kite_instance()
            
            # Get historical candles
            candles = await self._get_candles(config.symbol)
            if not candles or len(candles) < 20:
                return None
            
            # Get first candle (9:15 AM)
            first_candle = None
            for candle in candles:
                candle_time = datetime.fromtimestamp(candle.get("date", 0) / 1000).time()
                if time(9, 15) <= candle_time <= time(9, 16):
                    first_candle = candle
                    break
            
            if not first_candle:
                return None
            
            # Get current market data
            quote = kite.quote([config.symbol])
            if not quote or config.symbol not in quote:
                return None
            
            current_price = quote[config.symbol]["last_price"]
            
            # Find ATM options (live chain from instrument master)
            from simulation.helpers import find_option
            from services.nifty_option_chain import build_nifty_options_universe

            trade_d = datetime.now().date()
            nifty_options = [
                o for o in build_nifty_options_universe(kite) if o.get("expiry") and o["expiry"] >= trade_d
            ]
            date_str = datetime.now().strftime("%Y-%m-%d")
            current_strike = round(current_price / 50) * 50  # Round to nearest 50

            atm_ce = find_option(nifty_options, current_strike, "CE", trade_d)
            atm_pe = find_option(nifty_options, current_strike, "PE", trade_d)
            
            if not atm_ce or not atm_pe:
                return None
            
            # Run strategy
            result = await run_strategy_on_candles(
                kite=kite,
                strategy_type=config.strategy_type.value,
                trading_candles=candles,
                first_candle=first_candle,
                nifty_price=current_price,
                current_strike=current_strike,
                atm_ce=atm_ce,
                atm_pe=atm_pe,
                date_str=date_str,
                nifty_options=nifty_options,
                trade_date=datetime.now().date()
            )
            
            if not result:
                return None
            
            # Calculate stoploss and target
            entry_price = result["entry_price"]
            stoploss = entry_price * (1 - config.stoploss_pct) if result["trend"] == "BULLISH" else entry_price * (1 + config.stoploss_pct)
            target = entry_price * (1 + config.target_pct) if result["trend"] == "BULLISH" else entry_price * (1 - config.target_pct)
            
            return TradeSignal(
                strategy_type=config.strategy_type,
                symbol=result["option_to_trade"],
                action="BUY" if result["trend"] == "BULLISH" else "SELL",
                option_type=result["option_type"],
                entry_price=entry_price,
                stoploss=stoploss,
                target=target,
                quantity=config.quantity,
                reason=result["reason"],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            log_error(f"Error generating signal for {config.strategy_type.value}: {e}")
            return None
    
    async def _get_candles(self, symbol: str, interval: str = "minute") -> List:
        """Get historical candles for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval}"
            now = datetime.now()
            
            if cache_key in self.candle_cache:
                cached_candles, cached_time = self.candle_cache[cache_key]
                # Use cached data if less than 1 minute old
                if (now - cached_time).seconds < 60:
                    return cached_candles
            
            # Fetch fresh data
            kite = get_kite_instance()
            from_date = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
            to_date = datetime.now()
            
            candles = kite.historical_data(
                instrument_token=symbol,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            # Update cache
            self.candle_cache[cache_key] = (candles, now)
            
            return candles
            
        except Exception as e:
            log_error(f"Error fetching candles for {symbol}: {e}")
            return []
    
    async def _execute_trade(self, strategy_id: str, signal: TradeSignal, config: StrategyConfig):
        """Execute a trading signal"""
        try:
            kite = get_kite_instance()
            exchange = (
                kite.EXCHANGE_NFO if signal.option_type in ("CE", "PE") else kite.EXCHANGE_NSE
            )
            investment_amount = float(signal.entry_price) * int(signal.quantity)

            skip_sess = os.getenv("SKIP_SESSION_CHECK_ON_REST", "").lower() in ("1", "true", "yes")
            ok_r, msg_r = check_order_allowed(
                exchange,
                signal.symbol,
                signal.quantity,
                signal.action,
                investment_amount,
                skip_session_check=skip_sess,
            )
            if not ok_r:
                log_warning(f"[StrategyExecutor] risk gate: {msg_r}")
                add_live_log(f"Auto trade blocked: {msg_r}", "error")
                return

            if is_paper_mode():
                oid = paper_place_order(
                    {
                        "source": "strategy_executor",
                        "strategy_id": strategy_id,
                        "tradingsymbol": signal.symbol,
                        "exchange": exchange,
                        "transaction_type": signal.action,
                        "quantity": signal.quantity,
                        "order_type": config.order_type,
                        "product": config.product,
                        "price": signal.entry_price,
                        "stoploss": signal.stoploss,
                        "target": signal.target,
                    }
                )
                record_order_placed(investment_amount)
                log_execution_audit(
                    "AUTO_STRATEGY_ORDER",
                    actor="strategy_executor",
                    exchange=exchange,
                    tradingsymbol=signal.symbol,
                    payload={"paper": True, "strategy_id": strategy_id},
                    result={"order_id": oid},
                    paper=True,
                )
                order_id = oid
                record_strategy_fill_if_run(
                    os.getenv("STRATEGY_EXECUTOR_RUN_ID", "").strip() or None,
                    str(order_id),
                    signal.symbol,
                    signal.action,
                    signal.quantity,
                    signal.entry_price,
                )
            else:
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=signal.symbol,
                    transaction_type=signal.action,
                    quantity=signal.quantity,
                    product=config.product,
                    order_type=config.order_type,
                    tag=f"auto-{config.strategy_type.value}",
                )
                record_order_placed(investment_amount)
                log_execution_audit(
                    "AUTO_STRATEGY_ORDER",
                    actor="strategy_executor",
                    exchange=exchange,
                    tradingsymbol=signal.symbol,
                    result={"order_id": str(order_id)},
                    paper=False,
                )
                record_strategy_fill_if_run(
                    os.getenv("STRATEGY_EXECUTOR_RUN_ID", "").strip() or None,
                    str(order_id),
                    signal.symbol,
                    signal.action,
                    signal.quantity,
                    signal.entry_price,
                )
            
            # Live broker only — paper SL/target is handled by paper_order_monitor on DB rows
            if not is_paper_mode():
                await order_monitor.add_order(
                    order_id=str(order_id),
                    symbol=signal.symbol,
                    transaction_type=signal.action,
                    quantity=signal.quantity,
                    stoploss=signal.stoploss,
                    target=signal.target,
                    trailing_stoploss=config.trailing_stoploss,
                )
            
            # Track active trade
            self.active_trades[str(order_id)] = {
                "strategy_id": strategy_id,
                "signal": signal,
                "order_id": str(order_id),
                "entry_price": signal.entry_price,
                "quantity": signal.quantity,
                "date": datetime.now().date(),
                "status": "ACTIVE"
            }
            
            log_info(f"Executed {config.strategy_type.value} trade: {signal.action} {signal.quantity} {signal.symbol} at {signal.entry_price}")
            add_live_log(f"Auto Trade: {signal.reason}", "success")
            
        except Exception as e:
            log_error(f"Error executing trade: {e}")
            add_live_log(f"Auto Trade Failed: {str(e)}", "error")
    
    def get_strategy_status(self) -> Dict:
        """Get status of all strategies"""
        return {
            "is_running": self.is_running,
            "total_strategies": len(self.strategies),
            "enabled_strategies": len([s for s in self.strategies.values() if s.enabled]),
            "active_trades": len(self.active_trades),
            "strategies": {
                strategy_id: {
                    "type": config.strategy_type.value,
                    "symbol": config.symbol,
                    "enabled": config.enabled,
                    "max_trades_per_day": config.max_trades_per_day
                }
                for strategy_id, config in self.strategies.items()
            }
        }
    
    def get_active_trades(self) -> List[Dict]:
        """Get list of active trades"""
        return [
            {
                "order_id": trade["order_id"],
                "strategy_id": trade["strategy_id"],
                "symbol": trade["signal"].symbol,
                "action": trade["signal"].action,
                "entry_price": trade["entry_price"],
                "quantity": trade["quantity"],
                "status": trade["status"],
                "reason": trade["signal"].reason,
                "timestamp": trade["signal"].timestamp.isoformat()
            }
            for trade in self.active_trades.values()
        ]


# Global instance
strategy_executor = StrategyExecutor()
