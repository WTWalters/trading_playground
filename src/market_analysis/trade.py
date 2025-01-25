# src/market_analysis/trade.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import logging

@dataclass
class Trade:
    """
    Individual trade record containing all relevant trade information

    Attributes:
        entry_price: Price at which the trade was entered
        exit_price: Price at which the trade was exited (None if still open)
        entry_time: Timestamp of trade entry
        exit_time: Timestamp of trade exit (None if still open)
        position_size: Number of units traded
        direction: Trade direction ('LONG' or 'SHORT')
        stop_loss: Stop loss price level
        take_profit: Take profit price level
        status: Current trade status ('OPEN', 'CLOSED', 'CANCELLED')
        profit: Realized profit/loss (None if not closed)
        risk_amount: Amount of capital risked on the trade
        commission: Trading commission paid
    """
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    position_size: int
    direction: str  # 'LONG' or 'SHORT'
    stop_loss: float
    take_profit: float
    status: str  # 'OPEN', 'CLOSED', 'CANCELLED'
    profit: Optional[float] = None
    risk_amount: Optional[float] = None
    commission: Optional[float] = None

    def calculate_profit(self) -> float:
        """
        Calculate the profit/loss for the trade including commission

        Returns:
            Float representing the total profit/loss
        """
        if self.exit_price is None:
            return 0.0

        # Calculate based on direction (long/short)
        multiplier = 1 if self.direction == 'LONG' else -1
        gross_profit = (self.exit_price - self.entry_price) * self.position_size * multiplier

        # Subtract commission if applicable
        if self.commission:
            gross_profit -= self.commission

        return gross_profit

class TradeTracker:
    """Track and analyze trading performance metrics"""

    def __init__(self):
        """Initialize trade tracker with empty trade list"""
        self.trades: List[Trade] = []
        self.logger = logging.getLogger(__name__)

    def add_trade(
        self,
        entry_price: float,
        exit_price: float,
        entry_time: datetime = None,
        exit_time: datetime = None,
        position_size: int = 1,
        direction: str = 'LONG',
        stop_loss: float = 0,
        take_profit: float = 0,
        commission: float = 0,
        risk_amount: float = 0
    ):
        """
        Add a new trade to the tracker

        Args:
            entry_price: Trade entry price
            exit_price: Trade exit price
            entry_time: Entry timestamp (defaults to now)
            exit_time: Exit timestamp (defaults to now)
            position_size: Number of units traded
            direction: Trade direction ('LONG' or 'SHORT')
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            commission: Trading commission
            risk_amount: Amount risked on the trade
        """
        try:
            # Set default timestamps if not provided
            if entry_time is None:
                entry_time = datetime.now()
            if exit_time is None:
                exit_time = datetime.now()

            # Create and add new trade
            trade = Trade(
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                position_size=position_size,
                direction=direction,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status='CLOSED',
                commission=commission,
                risk_amount=risk_amount
            )

            # Calculate and set profit
            trade.profit = trade.calculate_profit()
            self.trades.append(trade)

        except Exception as e:
            self.logger.error(f"Failed to add trade: {str(e)}")

    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive trading performance metrics

        Returns:
            Dictionary containing various performance metrics including:
            - total_trades: Total number of trades
            - winning_trades: Number of profitable trades
            - losing_trades: Number of losing trades
            - win_rate: Percentage of winning trades
            - total_profit: Total profit/loss
            - average_profit: Average profit per trade
            - largest_win: Largest winning trade
            - largest_loss: Largest losing trade
            - avg_trade_duration: Average trade duration
            - profit_factor: Ratio of gross profits to gross losses
        """
        try:
            if not self.trades:
                return self._empty_metrics()

            # Calculate profit metrics
            profits = [trade.profit for trade in self.trades if trade.profit is not None]
            durations = [(trade.exit_time - trade.entry_time)
                        for trade in self.trades if trade.exit_time and trade.entry_time]

            return {
                'total_trades': len(self.trades),
                'winning_trades': len([p for p in profits if p > 0]),
                'losing_trades': len([p for p in profits if p < 0]),
                'win_rate': self.get_win_rate(),
                'total_profit': sum(profits),
                'average_profit': sum(profits) / len(profits) if profits else 0,
                'largest_win': max(profits) if profits else 0,
                'largest_loss': min(profits) if profits else 0,
                'avg_trade_duration': sum(durations, timedelta()) / len(durations) if durations else timedelta(),
                'profit_factor': self._calculate_profit_factor()
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {str(e)}")
            return self._empty_metrics()

    def get_win_rate(self) -> float:
        """
        Calculate the percentage of winning trades

        Returns:
            Float representing win rate (0.0 to 1.0)
        """
        if not self.trades:
            return 0.0

        winning_trades = len([t for t in self.trades if t.profit and t.profit > 0])
        return winning_trades / len(self.trades)

    def get_total_profit(self) -> float:
        """
        Calculate total profit/loss across all trades

        Returns:
            Float representing total profit/loss
        """
        return sum(trade.profit for trade in self.trades if trade.profit is not None)

    def get_max_consecutive_wins(self) -> int:
        """
        Calculate maximum streak of consecutive winning trades

        Returns:
            Integer representing longest winning streak
        """
        max_streak = current_streak = 0
        for trade in self.trades:
            if trade.profit and trade.profit > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def get_max_consecutive_losses(self) -> int:
        """
        Calculate maximum streak of consecutive losing trades

        Returns:
            Integer representing longest losing streak
        """
        max_streak = current_streak = 0
        for trade in self.trades:
            if trade.profit and trade.profit < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def get_risk_metrics(self) -> Dict:
        """
        Calculate risk-related trading metrics

        Returns:
            Dictionary containing:
            - risk_reward_ratio: Ratio of average profit to average risk
            - avg_risk_per_trade: Average risk amount per trade
            - max_drawdown: Maximum peak-to-trough decline
        """
        try:
            if not self.trades:
                return {
                    'risk_reward_ratio': 0.0,
                    'avg_risk_per_trade': 0.0,
                    'max_drawdown': 0.0
                }

            total_profit = sum(t.profit for t in self.trades if t.profit)
            total_risk = sum(t.risk_amount for t in self.trades if t.risk_amount)

            # Calculate risk/reward ratio (total profit / total risk)
            risk_reward_ratio = total_profit / total_risk if total_risk > 0 else 0.0

            return {
                'risk_reward_ratio': risk_reward_ratio,
                'avg_risk_per_trade': total_risk / len(self.trades) if self.trades else 0.0,
                'max_drawdown': self._calculate_max_drawdown()
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate risk metrics: {str(e)}")
            return {
                'risk_reward_ratio': 0.0,
                'avg_risk_per_trade': 0.0,
                'max_drawdown': 0.0
            }

    def _calculate_profit_factor(self) -> float:
        """
        Calculate the profit factor (gross profit / gross loss)

        Returns:
            Float representing profit factor
        """
        gross_profit = sum(t.profit for t in self.trades if t.profit and t.profit > 0)
        gross_loss = abs(sum(t.profit for t in self.trades if t.profit and t.profit < 0))
        return gross_profit / gross_loss if gross_loss != 0 else 0

    def _calculate_max_drawdown(self) -> float:
        """
        Calculate the maximum drawdown from peak equity

        Returns:
            Float representing maximum drawdown as a percentage
        """
        if not self.trades:
            return 0.0

        # Build equity curve
        equity_curve = []
        current_equity = 0
        for trade in self.trades:
            if trade.profit:
                current_equity += trade.profit
                equity_curve.append(current_equity)

        if not equity_curve:
            return 0.0

        # Calculate maximum drawdown
        peak = 0
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def _empty_metrics(self) -> Dict:
        """
        Return empty metrics structure for when no trades exist

        Returns:
            Dictionary with zero values for all metrics
        """
        return {
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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert trade history to pandas DataFrame for analysis

        Returns:
            DataFrame containing all trade information
        """
        return pd.DataFrame([vars(t) for t in self.trades])
