# src/market_analysis/backtest.py

from typing import Dict, Callable, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .base import AnalysisConfig
from .risk import RiskManager
from .trade import Trade, TradeTracker, TradeDirection, TradeStatus
from .volatility import VolatilityAnalyzer
from .trend import TrendAnalyzer

class SimpleBacktest:
    """Simple backtesting engine for strategy testing"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.risk_manager = RiskManager()
        self.trade_tracker = TradeTracker()
        self.logger = logging.getLogger(__name__)
        self.current_position: Optional[Trade] = None

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
        """Run backtest on historical data"""
        try:
            results = self._initialize_backtest()
            current_capital = initial_capital
            equity_curve = []

            # Validate data
            if len(data) < self.config.minimum_data_points:
                self.logger.warning(f"Insufficient data points: {len(data)}")
                return self._get_empty_results(initial_capital)

            # Run backtest
            for i in range(len(data) - 1):
                current_data = data.iloc[:i+1]
                next_bar = data.iloc[i+1]

                # Process current bar
                position_update = await self._process_bar(
                    current_data,
                    next_bar,
                    data.index[i+1],
                    current_capital,
                    risk_per_trade,
                    commission,
                    strategy
                )

                # Update capital and records
                current_capital = position_update['current_capital']
                results['commission'] += position_update['commission']
                results['signals'].append(position_update['signal'])

                # Record daily equity
                equity_curve.append({
                    'date': data.index[i+1],
                    'capital': current_capital,
                    'signal': position_update['signal']
                })

            # Calculate final metrics
            results['equity_curve'] = equity_curve
            return self._calculate_final_results(
                results,
                initial_capital,
                current_capital
            )

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            return self._get_empty_results(initial_capital)

    async def _process_bar(
        self,
        current_data: pd.DataFrame,
        next_bar: pd.Series,
        timestamp: datetime,
        current_capital: float,
        risk_per_trade: float,
        commission: float,
        strategy: Optional[Callable]
    ) -> Dict:
        """Process a single bar of market data"""
        # Run analysis
        vol_result = await self.volatility_analyzer.analyze(current_data)
        trend_result = await self.trend_analyzer.analyze(current_data)

        # Generate signal
        signal = self._generate_signal(vol_result, trend_result, strategy)

        # Process position updates
        position_update = self._update_positions(
            signal,
            next_bar,
            timestamp,
            current_capital,
            risk_per_trade,
            commission
        )

        return {
            'current_capital': position_update['new_capital'],
            'commission': position_update['commission'],
            'signal': signal
        }

    def _update_positions(
        self,
        signal: str,
        bar_data: pd.Series,
        timestamp: datetime,
        capital: float,
        risk_per_trade: float,
        commission: float
    ) -> Dict:
        """Update positions based on signals and market data"""
        total_commission = 0
        new_capital = capital

        try:
            # Handle position entry
            if self.current_position is None and signal == 'BUY':
                entry_price = bar_data['open']
                stop_loss = entry_price * 0.99  # 1% stop loss

                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    capital=capital,
                    risk_amount=capital * risk_per_trade,
                    entry_price=entry_price,
                    stop_loss=stop_loss
                )

                # Calculate commission
                entry_commission = entry_price * position_size * commission
                total_commission += entry_commission
                new_capital -= entry_commission

                # Open new position
                self.current_position = Trade(
                    entry_price=entry_price,
                    exit_price=None,
                    entry_time=timestamp,
                    exit_time=None,
                    position_size=position_size,
                    direction=TradeDirection.LONG,
                    stop_loss=stop_loss,
                    take_profit=entry_price * 1.02,  # 2% take profit
                    status=TradeStatus.OPEN,
                    commission=entry_commission,
                    risk_amount=position_size * (entry_price - stop_loss)
                )

            # Handle position exit
            elif self.current_position and (
                signal == 'SELL' or
                bar_data['low'] < self.current_position.stop_loss or
                bar_data['high'] > self.current_position.take_profit
            ):
                exit_price = bar_data['open']
                exit_commission = (
                    exit_price *
                    self.current_position.position_size *
                    commission
                )
                total_commission += exit_commission

                # Update position details
                self.current_position.exit_price = exit_price
                self.current_position.exit_time = timestamp
                self.current_position.status = TradeStatus.CLOSED
                self.current_position.commission = (
                    self.current_position.commission + exit_commission
                )

                # Calculate P&L
                trade_profit = self.current_position.calculate_profit()
                new_capital += trade_profit - exit_commission

                # Record trade
                self.trade_tracker.trades.append(self.current_position)
                self.current_position = None

        except Exception as e:
            self.logger.error(f"Position update failed: {str(e)}")

        return {
            'new_capital': new_capital,
            'commission': total_commission
        }

    def _calculate_final_results(
        self,
        results: Dict,
        initial_capital: float,
        final_capital: float
    ) -> Dict:
        """Calculate final backtest results and metrics"""
        # Create equity curve DataFrame
        equity_curve = pd.DataFrame(results['equity_curve'])
        if not equity_curve.empty:
            equity_curve.set_index('date', inplace=True)

            # Calculate drawdown
            peak = equity_curve['capital'].expanding(min_periods=1).max()
            drawdown = (peak - equity_curve['capital']) / peak
            max_drawdown = float(drawdown.max())
        else:
            max_drawdown = 0.0

        # Get trading metrics
        metrics = self.trade_tracker.get_metrics()

        return {
            'final_capital': final_capital,
            'total_return': (final_capital - initial_capital) / initial_capital,
            'trade_metrics': metrics,
            'equity_curve': equity_curve,
            'total_commission': results['commission'],
            'strategy_signals': results['signals'],
            'total_trades': len(self.trade_tracker.trades),
            'win_rate': metrics.get('win_rate', 0.0),
            'max_drawdown': max_drawdown
        }

    def _initialize_backtest(self) -> Dict:
        """Initialize backtest tracking variables"""
        return {
            'signals': [],
            'commission': 0.0,
            'equity_curve': []
        }

    def _get_empty_results(self, initial_capital: float) -> Dict:
        """Return empty results structure"""
        return {
            'final_capital': initial_capital,
            'total_return': 0.0,
            'trade_metrics': self.trade_tracker._empty_metrics(),
            'equity_curve': pd.DataFrame(),
            'total_commission': 0.0,
            'strategy_signals': [],
            'total_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0
        }
