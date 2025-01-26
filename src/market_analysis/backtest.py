# src/market_analysis/backtest.py

from typing import Dict, Callable, Optional, List, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .base import AnalysisConfig, MarketRegime
from .risk import RiskManager
from .trade import Trade, TradeTracker, TradeDirection, TradeStatus
from .volatility import VolatilityAnalyzer
from .trend import TrendAnalyzer

class SimpleBacktest:
    """
    Simple backtesting engine for strategy testing

    Features:
    - Historical data backtesting
    - Commission handling
    - Risk management
    - Position tracking
    - Performance analysis
    - Multiple strategy support
    - Volatility and trend analysis
    """

    def __init__(self, config: AnalysisConfig):
        """
        Initialize backtesting engine

        Args:
            config: Configuration parameters for analysis
        """
        self.config = config
        self.risk_manager = RiskManager()
        self.trade_tracker = TradeTracker()
        self.logger = logging.getLogger(__name__)
        self.current_position: Optional[Trade] = None

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
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data

        Args:
            data: OHLCV DataFrame with datetime index
            initial_capital: Starting capital amount
            risk_per_trade: Fraction of capital to risk per trade (0.02 = 2%)
            commission: Commission rate per trade (0.001 = 0.1%)
            strategy: Optional custom strategy function

        Returns:
            Dictionary containing:
            - final_capital: Ending capital
            - total_return: Percentage return
            - trade_metrics: Detailed trade statistics
            - equity_curve: Capital evolution over time
            - total_commission: Total commission paid
            - strategy_signals: List of trading signals
            - total_trades: Number of trades executed
            - win_rate: Percentage of winning trades
            - max_drawdown: Maximum peak-to-trough decline
            - volatility_metrics: Volatility analysis results
        """
        try:
            results = self._initialize_backtest()
            current_capital = initial_capital
            equity_curve = []

            # Get initial analysis
            vol_metrics = await self.volatility_analyzer.analyze(data)

            # Validate data requirements
            if len(data) < self.config.minimum_data_points:
                self.logger.warning(f"Insufficient data points: {len(data)}")
                return self._get_empty_results(initial_capital)

            # Process each bar
            for i in range(len(data) - 1):
                current_data = data.iloc[:i+1]
                next_bar = data.iloc[i+1]

                # Process current bar
                position_update = await self._process_bar(
                    current_data=current_data,
                    next_bar=next_bar,
                    timestamp=data.index[i+1],
                    current_capital=current_capital,
                    risk_per_trade=risk_per_trade,
                    commission=commission,
                    strategy=strategy
                )

                # Update tracking variables
                current_capital = position_update['current_capital']
                results['commission'] += position_update['commission']
                results['signals'].append(position_update['signal'])

                # Record equity curve
                equity_curve.append({
                    'date': data.index[i+1],
                    'capital': current_capital,
                    'signal': position_update['signal']
                })

            # Calculate final results
            results['equity_curve'] = equity_curve
            final_results = self._calculate_final_results(
                results=results,
                initial_capital=initial_capital,
                final_capital=current_capital
            )

            # Add volatility metrics
            final_results['volatility_metrics'] = vol_metrics.get('metrics', {})

            return final_results

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
    ) -> Dict[str, Any]:
        """
        Process a single bar of market data

        Args:
            current_data: Historical data up to current point
            next_bar: Next bar's data for execution
            timestamp: Current timestamp
            current_capital: Current account capital
            risk_per_trade: Risk per trade setting
            commission: Commission rate
            strategy: Optional strategy function

        Returns:
            Dictionary containing position updates
        """
        # Run market analysis
        vol_result = await self.volatility_analyzer.analyze(current_data)
        trend_result = await self.trend_analyzer.analyze(current_data)

        # Generate trading signal
        signal = self._generate_signal(vol_result, trend_result, strategy)

        # Update positions based on signal
        position_update = self._update_positions(
            signal=signal,
            bar_data=next_bar,
            timestamp=timestamp,
            capital=current_capital,
            risk_per_trade=risk_per_trade,
            commission=commission
        )

        return {
            'current_capital': position_update['new_capital'],
            'commission': position_update['commission'],
            'signal': signal
        }

    def _generate_signal(
        self,
        vol_result: Dict[str, Any],
        trend_result: Dict[str, Any],
        strategy: Optional[Callable]
    ) -> str:
        """
        Generate trading signal based on analysis

        Args:
            vol_result: Volatility analysis results
            trend_result: Trend analysis results
            strategy: Optional custom strategy

        Returns:
            Trading signal ('BUY', 'SELL', or 'HOLD')
        """
        if strategy:
            return strategy(vol_result, trend_result)

        # Default strategy logic
        if (vol_result and trend_result and
            vol_result.get('metrics', {}).get('volatility_regime') == 'low_volatility' and
            trend_result.get('regime') == MarketRegime.TRENDING_UP):
            return 'BUY'
        elif vol_result and vol_result.get('metrics', {}).get('volatility_regime') == 'high_volatility':
            return 'SELL'
        return 'HOLD'

    def _update_positions(
        self,
        signal: str,
        bar_data: pd.Series,
        timestamp: datetime,
        capital: float,
        risk_per_trade: float,
        commission: float
    ) -> Dict[str, float]:
        """
        Update positions based on signals and market data

        Args:
            signal: Trading signal
            bar_data: Current bar data
            timestamp: Current timestamp
            capital: Current capital
            risk_per_trade: Risk per trade setting
            commission: Commission rate

        Returns:
            Dictionary with position updates
        """
        total_commission = 0.0
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

                # Calculate and apply commission
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
                exit_commission = exit_price * self.current_position.position_size * commission
                total_commission += exit_commission

                # Update position details
                self.current_position.exit_price = exit_price
                self.current_position.exit_time = timestamp
                self.current_position.status = TradeStatus.CLOSED
                self.current_position.commission += exit_commission

                # Calculate P&L and update capital
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
        results: Dict[str, Any],
        initial_capital: float,
        final_capital: float
    ) -> Dict[str, Any]:
        """
        Calculate final backtest results

        Args:
            results: Raw backtest results
            initial_capital: Starting capital
            final_capital: Ending capital

        Returns:
            Dictionary containing final metrics
        """
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

    def _initialize_backtest(self) -> Dict[str, Any]:
        """Initialize backtest tracking variables"""
        return {
            'signals': [],
            'commission': 0.0,
            'equity_curve': []
        }

    def _get_empty_results(self, initial_capital: float) -> Dict[str, Any]:
        """
        Return empty results structure

        Args:
            initial_capital: Starting capital amount

        Returns:
            Dictionary with default values
        """
        return {
            'final_capital': initial_capital,
            'total_return': 0.0,
            'trade_metrics': self.trade_tracker.get_metrics(),
            'equity_curve': pd.DataFrame(),
            'total_commission': 0.0,
            'strategy_signals': [],
            'total_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'volatility_metrics': {}
        }
