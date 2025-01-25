# src/market_analysis/backtest.py

from typing import Dict, Callable, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from .base import AnalysisConfig
from .risk import RiskManager
from .trade import TradeTracker, Trade
from .volatility import VolatilityAnalyzer
from .trend import TrendAnalyzer

class SimpleBacktest:
    """Simple backtesting engine for strategy testing"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.risk_manager = RiskManager()
        self.trade_tracker = TradeTracker()
        self.logger = logging.getLogger(__name__)

        # Initialize analyzers
        self.volatility_analyzer = VolatilityAnalyzer(config)
        self.trend_analyzer = TrendAnalyzer(config)

    async def run_test(
        self,
        data: pd.DataFrame,
        initial_capital: float,
        risk_per_trade: float = 0.02,
        commission: float = 0.0,
        strategy: Optional[Callable] = None
    ) -> Dict:
        """
        Run backtest on historical data

        Args:
            data: OHLCV DataFrame
            initial_capital: Starting capital
            risk_per_trade: Percentage of capital to risk per trade
            commission: Commission per trade (percentage)
            strategy: Optional strategy function

        Returns:
            Dictionary containing backtest results
        """
        try:
            current_capital = initial_capital
            current_position = None
            results = []
            total_commission = 0.0
            strategy_signals: List[str] = []

            for i in range(len(data) - 1):
                current_data = data.iloc[:i+1]

                # Run analysis
                vol_result = await self.volatility_analyzer.analyze(current_data)
                trend_result = await self.trend_analyzer.analyze(current_data)

                # Generate signal
                signal = 'HOLD'
                if strategy:
                    signal = strategy(vol_result, trend_result)
                else:
                    # Default strategy
                    if (vol_result['metrics'].volatility_regime == 'low_volatility' and
                        trend_result['regime'].value == 'trending_up'):
                        signal = 'BUY'
                    elif vol_result['metrics'].volatility_regime == 'high_volatility':
                        signal = 'SELL'

                strategy_signals.append(signal)

                # Process signal
                if current_position is None and signal == 'BUY':
                    # Calculate position size
                    entry_price = data.iloc[i+1]['open']
                    stop_loss = entry_price * 0.99  # 1% stop loss
                    position_size = self.risk_manager.calculate_trade_size(
                        current_capital, risk_per_trade * 100, entry_price - stop_loss
                    )

                    # Calculate commission
                    trade_commission = entry_price * position_size * commission
                    total_commission += trade_commission
                    current_capital -= trade_commission

                    # Open position
                    current_position = Trade(
                        entry_price=entry_price,
                        exit_price=None,
                        entry_time=data.index[i+1],
                        exit_time=None,
                        position_size=position_size,
                        direction='LONG',
                        stop_loss=stop_loss,
                        take_profit=entry_price * 1.02,  # 2% take profit
                        status='OPEN',
                        commission=trade_commission,
                        risk_amount=position_size * (entry_price - stop_loss)
                    )

                elif current_position and (signal == 'SELL' or
                    data.iloc[i+1]['low'] < current_position.stop_loss or
                    data.iloc[i+1]['high'] > current_position.take_profit):
                    # Close position
                    exit_price = data.iloc[i+1]['open']
                    close_commission = exit_price * current_position.position_size * commission
                    total_commission += close_commission

                    current_position.exit_price = exit_price
                    current_position.exit_time = data.index[i+1]
                    current_position.status = 'CLOSED'
                    current_position.commission = (current_position.commission or 0) + close_commission

                    # Update capital
                    trade_profit = current_position.calculate_profit()
                    current_capital += trade_profit - close_commission

                    # Record trade
                    self.trade_tracker.trades.append(current_position)
                    current_position = None

                results.append({
                    'date': data.index[i+1],
                    'capital': current_capital,
                    'signal': signal
                })

            # Calculate final metrics
            metrics = self.trade_tracker.get_metrics()

            return {
                'final_capital': current_capital,
                'total_return': (current_capital - initial_capital) / initial_capital,
                'trade_metrics': metrics,
                'equity_curve': pd.DataFrame(results).set_index('date'),
                'total_commission': total_commission,
                'strategy_signals': strategy_signals,
                'total_trades': len(self.trade_tracker.trades),
                'win_rate': metrics.get('win_rate', 0.0)  # Add win_rate directly to top level
            }

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            return {
                'final_capital': initial_capital,
                'total_return': 0.0,
                'trade_metrics': self.trade_tracker._empty_metrics(),
                'equity_curve': pd.DataFrame(),
                'total_commission': 0.0,
                'strategy_signals': [],
                'total_trades': 0
            }

    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics for the backtest"""
        metrics = self.trade_tracker.get_metrics()
        return {
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'avg_trade_duration': metrics['avg_trade_duration'],
            'max_consecutive_wins': self.trade_tracker.get_max_consecutive_wins(),
            'max_consecutive_losses': self.trade_tracker.get_max_consecutive_losses(),
            'risk_metrics': self.trade_tracker.get_risk_metrics()
        }
