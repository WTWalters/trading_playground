# src/market_analysis/backtest.py

from dataclasses import dataclass
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

@dataclass
class BacktestResults:
    """Container for backtest results"""
    final_capital: float
    total_return: float
    trade_metrics: Dict[str, Any]
    equity_curve: pd.DataFrame
    total_commission: float
    strategy_signals: List[str]
    total_trades: int
    win_rate: float
    max_drawdown: float
    volatility_metrics: Dict[str, Any]

class SimpleBacktest:
    """
    A simple backtesting engine for trading strategy evaluation.

    Attributes:
        config: Configuration parameters
        risk_manager: Handles position sizing and risk calculations
        trade_tracker: Tracks and analyzes trades
        volatility_analyzer: Analyzes market volatility
        trend_analyzer: Analyzes market trends
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.risk_manager = RiskManager()
        self.trade_tracker = TradeTracker()
        self.volatility_analyzer = VolatilityAnalyzer(config)
        self.trend_analyzer = TrendAnalyzer(config)
        self.logger = logging.getLogger(__name__)
        self.current_position: Optional[Trade] = None

    async def run_test(
        self,
        data: pd.DataFrame,
        initial_capital: float,
        risk_per_trade: float = 0.02,
        commission: float = 0.0,
    ) -> BacktestResults:
        """
        Execute backtest on historical data.

        Args:
            data: Historical price data (OHLCV)
            initial_capital: Starting capital
            risk_per_trade: Risk per trade (as decimal)
            commission: Commission rate (as decimal)

        Returns:
            BacktestResults containing performance metrics
        """
        try:
            self._validate_inputs(data, initial_capital, risk_per_trade, commission)

            results = {
                'current_capital': initial_capital,
                'commission': 0.0,
                'signals': [],
                'trades': [],
                'equity_curve': []
            }

            # Process each bar
            for i in range(len(data) - 1):
                current_bar = data.iloc[i]
                next_bar = data.iloc[i + 1]
                timestamp = data.index[i]

                # Update results with new bar
                results = await self._process_bar(
                    current_bar=current_bar,
                    next_bar=next_bar,
                    timestamp=timestamp,
                    results=results,
                    risk_per_trade=risk_per_trade,
                    commission=commission
                )

            # Calculate final metrics
            final_results = self._calculate_final_results(
                results=results,
                initial_capital=initial_capital,
                data=data
            )

            return BacktestResults(**final_results)

        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            return self._get_empty_results(initial_capital)

    async def _process_bar(
        self,
        current_bar: pd.Series,
        next_bar: pd.Series,
        timestamp: datetime,
        results: Dict[str, Any],
        risk_per_trade: float,
        commission: float
    ) -> Dict[str, Any]:
        """Process single price bar and update positions"""

        # Get market analysis
        vol_analysis = await self.volatility_analyzer.analyze(
            pd.DataFrame([current_bar])
        )
        trend_analysis = await self.trend_analyzer.analyze(
            pd.DataFrame([current_bar])
        )

        # Generate trading signal
        signal = self._generate_signal(vol_analysis, trend_analysis)
        results['signals'].append(signal)

        # Handle position management
        if signal == "BUY" and self.current_position is None:
            # Enter new position
            position = self._enter_position(
                price=next_bar['open'],
                timestamp=timestamp,
                capital=results['current_capital'],
                risk_amount=risk_per_trade * results['current_capital'],
                commission=commission
            )

            if position:
                self.current_position = position
                results['commission'] += position.commission
                results['current_capital'] -= position.commission

        elif (signal == "SELL" or
              (self.current_position and
               self._should_exit_position(self.current_position, next_bar))):
            # Exit current position
            if self.current_position:
                exit_results = self._exit_position(
                    position=self.current_position,
                    price=next_bar['open'],
                    timestamp=timestamp,
                    commission=commission
                )

                results['commission'] += exit_results['commission']
                results['current_capital'] = exit_results['new_capital']
                results['trades'].append(self.current_position)
                self.current_position = None

        # Record equity curve
        results['equity_curve'].append({
            'timestamp': timestamp,
            'capital': results['current_capital'],
            'signal': signal
        })

        return results

    def _generate_signal(
        self,
        volatility: Dict[str, Any],
        trend: Dict[str, Any]
    ) -> str:
        """Generate trading signal based on analysis"""

        if (volatility['regime'] == 'low_volatility' and
            trend['regime'] == MarketRegime.TRENDING_UP):
            return "BUY"
        elif volatility['regime'] == 'high_volatility':
            return "SELL"
        return "HOLD"

    def _enter_position(
        self,
        price: float,
        timestamp: datetime,
        capital: float,
        risk_amount: float,
        commission: float
    ) -> Optional[Trade]:
        """Create new trading position"""

        try:
            # Calculate position parameters
            stop_loss = price * 0.99  # 1% stop loss
            position_size = self.risk_manager.calculate_position_size(
                capital=capital,
                risk_amount=risk_amount,
                entry_price=price,
                stop_loss=stop_loss
            )

            # Calculate commission
            entry_commission = price * position_size * commission

            return Trade(
                entry_price=price,
                entry_time=timestamp,
                position_size=position_size,
                direction=TradeDirection.LONG,
                stop_loss=stop_loss,
                take_profit=price * 1.02,  # 2% take profit
                status=TradeStatus.OPEN,
                commission=entry_commission
            )

        except Exception as e:
            self.logger.error(f"Position entry failed: {str(e)}")
            return None

    def _exit_position(
        self,
        position: Trade,
        price: float,
        timestamp: datetime,
        commission: float
    ) -> Dict[str, float]:
        """Exit trading position"""

        try:
            exit_commission = price * position.position_size * commission

            position.exit_price = price
            position.exit_time = timestamp
            position.status = TradeStatus.CLOSED
            position.commission += exit_commission

            profit = position.calculate_profit()
            new_capital = position.calculate_new_capital(profit, exit_commission)

            return {
                'commission': exit_commission,
                'new_capital': new_capital
            }

        except Exception as e:
            self.logger.error(f"Position exit failed: {str(e)}")
            return {
                'commission': 0.0,
                'new_capital': position.entry_capital
            }

    def _calculate_final_results(
        self,
        results: Dict[str, Any],
        initial_capital: float,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate final backtest metrics"""

        equity_curve = pd.DataFrame(results['equity_curve'])

        if not equity_curve.empty:
            equity_curve.set_index('timestamp', inplace=True)
            drawdown = self._calculate_drawdown(equity_curve['capital'])
            max_drawdown = float(drawdown.max())
        else:
            max_drawdown = 0.0

        return {
            'final_capital': results['current_capital'],
            'total_return': (results['current_capital'] - initial_capital) / initial_capital,
            'trade_metrics': self.trade_tracker.get_metrics(),
            'equity_curve': equity_curve,
            'total_commission': results['commission'],
            'strategy_signals': results['signals'],
            'total_trades': len(results['trades']),
            'win_rate': self._calculate_win_rate(results['trades']),
            'max_drawdown': max_drawdown,
            'volatility_metrics': self.volatility_analyzer.get_metrics(data)
        }

    @staticmethod
    def _calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        peak = equity_curve.expanding(min_periods=1).max()
        return (peak - equity_curve) / peak

    @staticmethod
    def _calculate_win_rate(trades: List[Trade]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade.profit > 0)
        return winning_trades / len(trades)

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        initial_capital: float,
        risk_per_trade: float,
        commission: float
    ) -> None:
        """Validate backtest inputs"""
        if len(data) < self.config.minimum_data_points:
            raise ValueError(f"Insufficient data points: {len(data)}")
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if not 0 <= risk_per_trade <= 1:
            raise ValueError("Risk per trade must be between 0 and 1")
        if not 0 <= commission <= 1:
            raise ValueError("Commission must be between 0 and 1")

    def _get_empty_results(self, initial_capital: float) -> BacktestResults:
        """Return empty results structure"""
        return BacktestResults(
            final_capital=initial_capital,
            total_return=0.0,
            trade_metrics={},
            equity_curve=pd.DataFrame(),
            total_commission=0.0,
            strategy_signals=[],
            total_trades=0,
            win_rate=0.0,
            max_drawdown=0.0,
            volatility_metrics={}
        )
