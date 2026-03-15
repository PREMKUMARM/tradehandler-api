"""
Premium Strategy Agent
Advanced strategy building with AI, backtesting, and optimization
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentTask, AgentCapability, AgentConfig
from .agent_types import AgentType
from utils.logger import log_info, log_error, log_warning

@dataclass
class StrategyComponent:
    """Strategy component definition"""
    name: str
    type: str  # indicator, condition, action
    parameters: Dict[str, Any]
    weight: float = 1.0

@dataclass
class StrategyDefinition:
    """Complete strategy definition"""
    strategy_id: str
    name: str
    description: str
    components: List[StrategyComponent]
    risk_parameters: Dict[str, Any]
    timeframes: List[str]
    instruments: List[str]
    created_at: datetime
    backtest_results: Optional[Dict[str, Any]] = None
    optimization_results: Optional[Dict[str, Any]] = None

class PremiumStrategyAgent(BaseAgent):
    """Premium Strategy Agent with advanced AI capabilities"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.strategies: Dict[str, StrategyDefinition] = {}
        self.ai_models = {}
        self.optimization_history = []
        
    async def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return [
            AgentCapability.STRATEGY_BUILDING,
            AgentCapability.BACKTESTING,
            AgentCapability.MARKET_ANALYSIS,
            AgentCapability.RISK_MANAGEMENT
        ]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process premium strategy task"""
        
        task_type = task.data.get("task_type", "build_strategy")
        
        if task_type == "build_strategy":
            return await self._build_ai_strategy(task.data)
        elif task_type == "backtest_strategy":
            return await self._backtest_strategy(task.data)
        elif task_type == "optimize_strategy":
            return await self._optimize_strategy(task.data)
        elif task_type == "analyze_market_conditions":
            return await self._analyze_market_conditions(task.data)
        elif task_type == "generate_strategy_recommendations":
            return await self._generate_strategy_recommendations(task.data)
        elif task_type == "validate_strategy":
            return await self._validate_strategy(task.data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _build_ai_strategy(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build strategy using AI"""
        
        strategy_requirements = task_data.get("requirements", {})
        market_conditions = task_data.get("market_conditions", {})
        user_preferences = task_data.get("user_preferences", {})
        
        log_info("Building AI-powered strategy")
        
        # Analyze market conditions
        market_analysis = await self._analyze_market_conditions(market_conditions)
        
        # Generate strategy components using AI
        strategy_components = await self._generate_strategy_components(
            strategy_requirements, market_analysis, user_preferences
        )
        
        # Create strategy definition
        strategy = StrategyDefinition(
            strategy_id=f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=strategy_requirements.get("name", "AI Generated Strategy"),
            description=strategy_requirements.get("description", "AI generated trading strategy"),
            components=strategy_components,
            risk_parameters=self._generate_risk_parameters(user_preferences),
            timeframes=strategy_requirements.get("timeframes", ["5min", "15min", "1hour"]),
            instruments=strategy_requirements.get("instruments", ["NIFTY", "BANKNIFTY"]),
            created_at=datetime.now()
        )
        
        # Validate strategy
        validation_result = await self._validate_strategy_definition(strategy)
        
        if validation_result["valid"]:
            # Save strategy
            self.strategies[strategy.strategy_id] = strategy
            
            # Store in database
            await self._save_strategy(strategy)
            
            log_info(f"AI Strategy {strategy.strategy_id} built successfully")
            
            return {
                "status": "success",
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "components": [comp.__dict__ for comp in strategy.components],
                "risk_parameters": strategy.risk_parameters,
                "validation_result": validation_result,
                "market_analysis": market_analysis,
                "created_at": strategy.created_at.isoformat()
            }
        else:
            return {
                "status": "failed",
                "error": "Strategy validation failed",
                "validation_result": validation_result
            }
    
    async def _generate_strategy_components(self, requirements: Dict, market_analysis: Dict, 
                                          preferences: Dict) -> List[StrategyComponent]:
        """Generate strategy components using AI"""
        
        components = []
        
        # Entry conditions
        entry_components = await self._generate_entry_conditions(market_analysis, preferences)
        components.extend(entry_components)
        
        # Exit conditions
        exit_components = await self._generate_exit_conditions(market_analysis, preferences)
        components.extend(exit_components)
        
        # Risk management
        risk_components = await self._generate_risk_components(preferences)
        components.extend(risk_components)
        
        # Position sizing
        position_components = await self._generate_position_sizing_components(preferences)
        components.extend(position_components)
        
        return components
    
    async def _generate_entry_conditions(self, market_analysis: Dict, preferences: Dict) -> List[StrategyComponent]:
        """Generate entry condition components"""
        
        components = []
        
        # Trend following component
        if market_analysis.get("trend_strength", 0) > 0.6:
            components.append(StrategyComponent(
                name="trend_following",
                type="condition",
                parameters={
                    "indicator": "EMA",
                    "period": 20,
                    "condition": "price_above_ema",
                    "weight": 0.3
                }
            ))
        
        # Momentum component
        if market_analysis.get("momentum", "neutral") == "positive":
            components.append(StrategyComponent(
                name="momentum_entry",
                type="condition",
                parameters={
                    "indicator": "RSI",
                    "period": 14,
                    "condition": "rsi_above_50",
                    "weight": 0.25
                }
            ))
        
        # Volume component
        if market_analysis.get("volume_analysis", {}).get("volume_trend", "neutral") == "increasing":
            components.append(StrategyComponent(
                name="volume_confirmation",
                type="condition",
                parameters={
                    "indicator": "Volume",
                    "condition": "volume_above_average",
                    "period": 20,
                    "weight": 0.2
                }
            ))
        
        # Volatility component
        volatility = market_analysis.get("volatility", "normal")
        if volatility == "normal":
            components.append(StrategyComponent(
                name="volatility_filter",
                type="condition",
                parameters={
                    "indicator": "ATR",
                    "condition": "atr_in_range",
                    "min_atr": 0.5,
                    "max_atr": 2.0,
                    "weight": 0.15
                }
            ))
        
        # Support/Resistance component
        components.append(StrategyComponent(
            name="support_resistance",
            type="condition",
            parameters={
                "indicator": "Pivot Points",
                "condition": "near_support_resistance",
                "tolerance": 0.5,
                "weight": 0.1
            }
        ))
        
        return components
    
    async def _generate_exit_conditions(self, market_analysis: Dict, preferences: Dict) -> List[StrategyComponent]:
        """Generate exit condition components"""
        
        components = []
        
        # Profit target
        risk_reward_ratio = preferences.get("risk_reward_ratio", 2.0)
        components.append(StrategyComponent(
            name="profit_target",
            type="condition",
            parameters={
                "type": "risk_reward",
                "ratio": risk_reward_ratio,
                "weight": 0.4
            }
        ))
        
        # Stop loss
        max_loss_percent = preferences.get("max_loss_percent", 2.0)
        components.append(StrategyComponent(
            name="stop_loss",
            type="condition",
            parameters={
                "type": "percentage",
                "max_loss": max_loss_percent,
                "weight": 0.4
            }
        ))
        
        # Trailing stop
        if preferences.get("trailing_stop", False):
            components.append(StrategyComponent(
                name="trailing_stop",
                type="condition",
                parameters={
                    "type": "atr_based",
                    "atr_multiplier": 2.0,
                    "weight": 0.2
                }
            ))
        
        return components
    
    async def _generate_risk_components(self, preferences: Dict) -> List[StrategyComponent]:
        """Generate risk management components"""
        
        components = []
        
        # Position sizing
        max_position_size = preferences.get("max_position_size", 10.0)
        components.append(StrategyComponent(
            name="position_sizing",
            type="action",
            parameters={
                "method": "percentage",
                "max_size": max_position_size,
                "risk_per_trade": 1.0
            }
        ))
        
        # Correlation filter
        components.append(StrategyComponent(
            name="correlation_filter",
            type="condition",
            parameters={
                "max_correlation": 0.7,
                "check_existing_positions": True
            }
        ))
        
        return components
    
    async def _generate_position_sizing_components(self, preferences: Dict) -> List[StrategyComponent]:
        """Generate position sizing components"""
        
        components = []
        
        # Volatility-based sizing
        components.append(StrategyComponent(
            name="volatility_sizing",
            type="action",
            parameters={
                "method": "volatility_adjusted",
                "base_size": preferences.get("base_position_size", 5.0),
                "volatility_multiplier": 1.0
            }
        ))
        
        # Account balance sizing
        components.append(StrategyComponent(
            name="balance_sizing",
            type="action",
            parameters={
                "method": "percentage_of_balance",
                "percentage": preferences.get("balance_percentage", 2.0)
            }
        ))
        
        return components
    
    def _generate_risk_parameters(self, preferences: Dict) -> Dict[str, Any]:
        """Generate risk parameters"""
        
        return {
            "max_drawdown": preferences.get("max_drawdown", 10.0),
            "max_concurrent_positions": preferences.get("max_concurrent_positions", 5),
            "max_daily_loss": preferences.get("max_daily_loss", 3.0),
            "risk_per_trade": preferences.get("risk_per_trade", 1.0),
            "portfolio_heat": preferences.get("portfolio_heat", 20.0),
            "correlation_limit": preferences.get("correlation_limit", 0.7)
        }
    
    async def _validate_strategy_definition(self, strategy: StrategyDefinition) -> Dict[str, Any]:
        """Validate strategy definition"""
        
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check components
        if not strategy.components:
            validation_result["valid"] = False
            validation_result["errors"].append("No strategy components defined")
        
        # Check risk parameters
        if strategy.risk_parameters["max_drawdown"] > 20.0:
            validation_result["warnings"].append("Max drawdown is very high")
        
        if strategy.risk_parameters["risk_per_trade"] > 5.0:
            validation_result["warnings"].append("Risk per trade is very high")
        
        # Check timeframes
        if not strategy.timeframes:
            validation_result["valid"] = False
            validation_result["errors"].append("No timeframes specified")
        
        # Check instruments
        if not strategy.instruments:
            validation_result["valid"] = False
            validation_result["errors"].append("No instruments specified")
        
        return validation_result
    
    async def _backtest_strategy(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest strategy with historical data"""
        
        strategy_id = task_data.get("strategy_id")
        if not strategy_id or strategy_id not in self.strategies:
            return {"error": "Strategy not found"}
        
        strategy = self.strategies[strategy_id]
        
        # Get historical data
        start_date = task_data.get("start_date", datetime.now() - timedelta(days=365))
        end_date = task_data.get("end_date", datetime.now())
        instruments = strategy.instruments
        
        backtest_results = {}
        
        for instrument in instruments:
            try:
                # Get historical data
                historical_data = await self._get_historical_data(instrument, start_date, end_date)
                
                # Run backtest
                instrument_results = await self._run_backtest(strategy, historical_data, instrument)
                backtest_results[instrument] = instrument_results
                
            except Exception as e:
                log_error(f"Backtest failed for {instrument}: {e}")
                backtest_results[instrument] = {"error": str(e)}
        
        # Calculate overall results
        overall_results = await self._calculate_overall_backtest_results(backtest_results)
        
        # Store results
        strategy.backtest_results = overall_results
        await self._save_strategy(strategy)
        
        return {
            "strategy_id": strategy_id,
            "backtest_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "results": overall_results,
            "instrument_results": backtest_results,
            "performance_metrics": overall_results.get("performance_metrics", {})
        }
    
    async def _run_backtest(self, strategy: StrategyDefinition, historical_data: pd.DataFrame, 
                           instrument: str) -> Dict[str, Any]:
        """Run backtest for a single instrument"""
        
        # Initialize backtest variables
        positions = []
        trades = []
        equity_curve = [100000]  # Starting capital
        current_equity = 100000
        position_size = 0
        
        # Process each candle
        for i, candle in historical_data.iterrows():
            # Check entry conditions
            if position_size == 0:
                entry_signal = await self._check_entry_conditions(strategy, candle, historical_data.iloc[:i])
                if entry_signal["signal"]:
                    # Calculate position size
                    position_size = await self._calculate_position_size(strategy, current_equity, candle)
                    
                    # Record trade
                    trades.append({
                        "type": "entry",
                        "date": candle.name,
                        "price": candle["close"],
                        "quantity": position_size,
                        "signal_strength": entry_signal["strength"]
                    })
                    
                    positions.append({
                        "entry_date": candle.name,
                        "entry_price": candle["close"],
                        "quantity": position_size
                    })
            
            # Check exit conditions
            elif position_size > 0:
                exit_signal = await self._check_exit_conditions(strategy, candle, historical_data.iloc[:i])
                if exit_signal["signal"]:
                    # Calculate P&L
                    pnl = (candle["close"] - positions[-1]["entry_price"]) * position_size
                    current_equity += pnl
                    
                    # Update equity curve
                    equity_curve.append(current_equity)
                    
                    # Record trade
                    trades.append({
                        "type": "exit",
                        "date": candle.name,
                        "price": candle["close"],
                        "quantity": position_size,
                        "pnl": pnl
                    })
                    
                    position_size = 0
                else:
                    # Update unrealized P&L
                    unrealized_pnl = (candle["close"] - positions[-1]["entry_price"]) * position_size
                    equity_curve.append(current_equity + unrealized_pnl)
        
        # Calculate performance metrics
        performance_metrics = await self._calculate_performance_metrics(equity_curve, trades)
        
        return {
            "total_trades": len([t for t in trades if t["type"] == "exit"]),
            "winning_trades": len([t for t in trades if t["type"] == "exit" and t.get("pnl", 0) > 0]),
            "losing_trades": len([t for t in trades if t["type"] == "exit" and t.get("pnl", 0) < 0]),
            "total_return": (equity_curve[-1] - 100000) / 100000 * 100,
            "max_drawdown": performance_metrics["max_drawdown"],
            "sharpe_ratio": performance_metrics["sharpe_ratio"],
            "equity_curve": equity_curve,
            "trades": trades
        }
    
    async def _check_entry_conditions(self, strategy: StrategyDefinition, candle: pd.Series, 
                                      historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Check entry conditions for strategy"""
        
        signal_strength = 0
        conditions_met = []
        
        for component in strategy.components:
            if component.type == "condition":
                condition_met = await self._evaluate_condition(component, candle, historical_data)
                if condition_met:
                    signal_strength += component.weight
                    conditions_met.append(component.name)
        
        return {
            "signal": signal_strength > 0.5,  # Threshold for entry
            "strength": signal_strength,
            "conditions_met": conditions_met
        }
    
    async def _check_exit_conditions(self, strategy: StrategyDefinition, candle: pd.Series, 
                                     historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Check exit conditions for strategy"""
        
        signal_strength = 0
        conditions_met = []
        
        for component in strategy.components:
            if component.type == "condition":
                condition_met = await self._evaluate_condition(component, candle, historical_data)
                if condition_met:
                    signal_strength += component.weight
                    conditions_met.append(component.name)
        
        return {
            "signal": signal_strength > 0.5,  # Threshold for exit
            "strength": signal_strength,
            "conditions_met": conditions_met
        }
    
    async def _evaluate_condition(self, component: StrategyComponent, candle: pd.Series, 
                                 historical_data: pd.DataFrame) -> bool:
        """Evaluate a single condition component"""
        
        # This is a simplified evaluation - in production, use proper technical analysis
        indicator = component.parameters.get("indicator", "")
        condition = component.parameters.get("condition", "")
        
        if indicator == "EMA" and condition == "price_above_ema":
            period = component.parameters.get("period", 20)
            ema = historical_data["close"].rolling(window=period).mean().iloc[-1]
            return candle["close"] > ema
        
        elif indicator == "RSI" and condition == "rsi_above_50":
            period = component.parameters.get("period", 14)
            rsi = self._calculate_rsi(historical_data["close"], period)
            return rsi > 50
        
        # Add more condition evaluations as needed
        
        return False
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    async def _calculate_position_size(self, strategy: StrategyDefinition, equity: float, 
                                      candle: pd.Series) -> float:
        """Calculate position size based on strategy parameters"""
        
        # Simplified position sizing - in production, use more sophisticated methods
        risk_per_trade = strategy.risk_parameters.get("risk_per_trade", 1.0)
        risk_amount = equity * risk_per_trade / 100
        
        # Calculate position size based on risk
        position_size = risk_amount / candle["close"]
        
        return position_size
    
    async def _calculate_performance_metrics(self, equity_curve: List[float], 
                                           trades: List[Dict]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        
        if len(equity_curve) < 2:
            return {"max_drawdown": 0, "sharpe_ratio": 0}
        
        # Calculate returns
        returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] 
                  for i in range(1, len(equity_curve))]
        
        # Calculate max drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate Sharpe ratio
        if returns:
            avg_return = sum(returns) / len(returns)
            return_std = np.std(returns)
            sharpe_ratio = avg_return / return_std * np.sqrt(252) if return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_return": (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100,
            "volatility": np.std(returns) * 100 if returns else 0
        }
    
    async def _get_historical_data(self, instrument: str, start_date: datetime, 
                                  end_date: datetime) -> pd.DataFrame:
        """Get historical data for backtesting"""
        
        # Use MCP client to get historical data
        if "zerodha" in self.mcp_clients:
            data = await self.mcp_clients["zerodha"].get_historical_data(
                instrument, start_date, end_date, "5minute"
            )
            return pd.DataFrame(data)
        else:
            # Generate sample data for demonstration
            dates = pd.date_range(start_date, end_date, freq="5min")
            prices = np.random.randn(len(dates)).cumsum() + 100
            return pd.DataFrame({
                "date": dates,
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.randint(1000, 10000, len(dates))
            }).set_index("date")
    
    async def _calculate_overall_backtest_results(self, instrument_results: Dict) -> Dict[str, Any]:
        """Calculate overall backtest results across instruments"""
        
        total_return = 0
        total_trades = 0
        total_winning_trades = 0
        max_drawdown = 0
        sharpe_ratios = []
        
        for results in instrument_results.values():
            if "error" not in results:
                total_return += results.get("total_return", 0)
                total_trades += results.get("total_trades", 0)
                total_winning_trades += results.get("winning_trades", 0)
                max_drawdown = max(max_drawdown, results.get("max_drawdown", 0))
                sharpe_ratios.append(results.get("sharpe_ratio", 0))
        
        avg_sharpe_ratio = sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
        win_rate = (total_winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "total_return": total_return / len(instrument_results) if instrument_results else 0,
            "total_trades": total_trades,
            "winning_trades": total_winning_trades,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": avg_sharpe_ratio,
            "performance_metrics": {
                "total_return": total_return / len(instrument_results) if instrument_results else 0,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": avg_sharpe_ratio
            }
        }
    
    async def _save_strategy(self, strategy: StrategyDefinition):
        """Save strategy to database"""
        
        if self.data_store:
            strategy_data = {
                "strategy_id": strategy.strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "components": [comp.__dict__ for comp in strategy.components],
                "risk_parameters": strategy.risk_parameters,
                "timeframes": strategy.timeframes,
                "instruments": strategy.instruments,
                "created_at": strategy.created_at.isoformat(),
                "backtest_results": strategy.backtest_results,
                "optimization_results": strategy.optimization_results
            }
            
            await self.data_store.save_strategy(strategy_data)
    
    async def _analyze_market_conditions(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze current market conditions"""
        
        # This would use real market analysis in production
        return {
            "trend_strength": 0.7,
            "trend_direction": "upward",
            "momentum": "positive",
            "volatility": "normal",
            "volume_analysis": {
                "volume_trend": "increasing",
                "volume_strength": 0.6
            },
            "market_phase": "bullish",
            "support_resistance_levels": {
                "support": 18500,
                "resistance": 19500
            }
        }
    
    async def _generate_strategy_recommendations(self, task_data: Dict) -> Dict[str, Any]:
        """Generate strategy recommendations based on market conditions"""
        
        market_conditions = task_data.get("market_conditions", {})
        user_profile = task_data.get("user_profile", {})
        
        recommendations = []
        
        # Analyze market conditions
        if market_conditions.get("trend_direction") == "upward":
            recommendations.append({
                "type": "strategy_type",
                "recommendation": "trend_following",
                "reasoning": "Upward trend detected, trend-following strategies likely to perform well",
                "confidence": 0.8
            })
        
        if market_conditions.get("volatility") == "high":
            recommendations.append({
                "type": "risk_management",
                "recommendation": "reduce_position_size",
                "reasoning": "High volatility detected, consider reducing position sizes",
                "confidence": 0.7
            })
        
        # Based on user profile
        if user_profile.get("risk_tolerance") == "conservative":
            recommendations.append({
                "type": "strategy_parameters",
                "recommendation": "tight_stop_loss",
                "reasoning": "Conservative risk profile, use tighter stop losses",
                "confidence": 0.9
            })
        
        return {
            "recommendations": recommendations,
            "market_conditions": market_conditions,
            "user_profile": user_profile,
            "generated_at": datetime.now().isoformat()
        }
    
    async def _validate_strategy(self, task_data: Dict) -> Dict[str, Any]:
        """Validate an existing strategy"""
        
        strategy_id = task_data.get("strategy_id")
        if not strategy_id or strategy_id not in self.strategies:
            return {"error": "Strategy not found"}
        
        strategy = self.strategies[strategy_id]
        validation_result = await self._validate_strategy_definition(strategy)
        
        return {
            "strategy_id": strategy_id,
            "validation_result": validation_result,
            "validated_at": datetime.now().isoformat()
        }
