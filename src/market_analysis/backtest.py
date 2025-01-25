# src/market_analysis/backtest.py

from typing import Dict, Callable, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .base import AnalysisConfig
from .risk import RiskManager
from .trade import TradeTracker, Trade
from .volatility import VolatilityAnalyzer
from .trend import TrendAnalyzer

class SimpleBacktest:
    """Simple backtesting engine for strategy testing"""

    def __init__(self, config: AnalysisConfig):
        """
        Initialize the backtesting engine

        Args:
            config: Configuration object containing analysis parameters
        """
        self.config = config
        self.risk_manager = RiskManager()
        self.trade_tracker = TradeTracker()
        self.logger = logging.getLogger(__name__)

        # Initialize market analyzers
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
            data: OHLCV DataFrame with datetime index
            initial_capital: Starting capital amount
            risk_per_trade: Fraction of capital to risk per trade (e.g., 0.02 = 2%)
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
            strategy: Optional custom strategy function

        Returns:
            Dictionary containing backtest results including:
            - final_capital: Ending capital
            - total_return: Percentage return
            - trade_metrics: Detailed trading statistics
            - equity_curve: Capital evolution over time
            - total_commission: Total commission paid
            - strategy_signals: List of all trading signals
            - total_trades: Number of trades executed
            - win_rate: Percentage of winning trades
        """
        try:
            # Initialize tracking variables
            current_capital = initial_capital
            current_position = None
            results = []
            total_commission = 0.0
            strategy_signals: List[str] = []

            # Iterate through the data
            for i in range(len(data) - 1):
                # Get data up to current point for analysis
                current_data = data.iloc[:i+1]

                # Run market analysis
                vol_result = await self.volatility_analyzer.analyze(current_data)
                trend_result = await self.trend_analyzer.analyze(current_data)

                # Generate trading signal
                signal = 'HOLD'
                if strategy:
                    # Use custom strategy if provided
                    if vol_result and trend_result:
                        signal = strategy({
                            'metrics': {'volatility_regime': 'normal_volatility'}
                            if not vol_result.get('metrics') else vol_result['metrics']
                        }, trend_result)
                else:
                    # Default strategy logic
                    if (vol_result and trend_result and
                        vol_result.get('metrics', {}).get('volatility_regime') == 'low_volatility' and
                        trend_result.get('regime', '').value == 'trending_up'):
                        signal = 'BUY'
                    elif vol_result and vol_result.get('metrics', {}).get('volatility_regime') == 'high_volatility':
                        signal = 'SELL'

                strategy_signals.append(signal)

                # Process trading signals
                if current_position is None and signal == 'BUY':
                    # Open new long position
                    entry_price = data.iloc[i+1]['open']
                    stop_loss = entry_price * 0.99  # 1% stop loss

                    # Calculate position size based on risk
                    position_size = self.risk_manager.calculate_trade_size(
                        current_capital,
                        risk_per_trade * 100,
                        entry_price - stop_loss
                    )

                    # Calculate and apply commission
                    trade_commission = entry_price * position_size * commission
                    total_commission += trade_commission
                    current_capital -= trade_commission

                    # Create new trade object
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

                elif current_position and (
                    signal == 'SELL' or
                    data.iloc[i+1]['low'] < current_position.stop_loss or
                    data.iloc[i+1]['high'] > current_position.take_profit
                ):
                    # Close existing position
                    exit_price = data.iloc[i+1]['open']
                    close_commission = exit_price * current_position.position_size * commission
                    total_commission += close_commission

                    # Update position details
                    current_position.exit_price = exit_price
                    current_position.exit_time = data.index[i+1]
                    current_position.status = 'CLOSED'
                    current_position.commission = (current_position.commission or 0) + close_commission

                    # Calculate and apply profit/loss
                    trade_profit = current_position.calculate_profit()
                    current_capital += trade_profit - close_commission

                    # Record completed trade
                    self.trade_tracker.trades.append(current_position)
                    current_position = None

                # Record daily results
                results.append({
                    'date': data.index[i+1],
                    'capital': current_capital,
                    'signal': signal
                })

            # Calculate final performance metrics
            metrics = self.trade_tracker.get_metrics()

            return {
                'final_capital': current_capital,
                'total_return': (current_capital - initial_capital) / initial_capital,
                'trade_metrics': metrics,
                'equity_curve': pd.DataFrame(results).set_index('date'),
                'total_commission': total_commission,
                'strategy_signals': strategy_signals,
                'total_trades': len(self.trade_tracker.trades),
                'win_rate': metrics.get('win_rate', 0.0)
            }

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            # Return empty results on error
            empty_metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'average_profit': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_trade_duration': timedelta(),
                'profit_factor': 0.0
            }
            return {
                'final_capital': initial_capital,
                'total_return': 0.0,
                'trade_metrics': empty_metrics,
                'equity_curve': pd.DataFrame(),
                'total_commission': 0.0,
                'strategy_signals': [],
                'total_trades': 0,
                'win_rate': 0.0
            }

    def get_summary_statistics(self) -> Dict:
        """
        Calculate and return summary statistics for the backtest

        Returns:
            Dictionary containing key performance metrics including:
            - total_trades: Number of trades
            - win_rate: Percentage of winning trades
            - profit_factor: Ratio of gross profits to gross losses
            - avg_trade_duration: Average trade duration
            - max_consecutive_wins: Longest winning streak
            - max_consecutive_losses: Longest losing streak
            - risk_metrics: Various risk-related metrics
        """
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
