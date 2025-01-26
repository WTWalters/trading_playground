# src/market_analysis/trade.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import pandas as pd
import logging

class TradeDirection(Enum):
    """Valid trade directions"""
    LONG = "LONG"
    SHORT = "SHORT"

class TradeStatus(Enum):
    """Valid trade statuses"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    """
    Individual trade record with full trade lifecycle information

    Features:
    - Complete trade details tracking
    - Profit/loss calculation
    - Commission handling
    - Risk tracking
    - Position management
    """
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    position_size: int
    direction: TradeDirection
    stop_loss: float
    take_profit: float
    status: TradeStatus
    profit: Optional[float] = None
    risk_amount: Optional[float] = None
    commission: Optional[float] = None

    def __post_init__(self):
        """Validate trade parameters after initialization"""
        if self.position_size <= 0:
            raise ValueError("Position size must be positive")
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if self.stop_loss and self.stop_loss <= 0:  # Only validate if provided
            raise ValueError("Stop loss must be positive")
        if self.take_profit and self.take_profit <= 0:  # Only validate if provided
            raise ValueError("Take profit must be positive")

    def calculate_profit(self) -> float:
        """Calculate realized profit/loss including commission"""
        if self.exit_price is None:
            return 0.0

        multiplier = 1 if self.direction == TradeDirection.LONG else -1
        gross_profit = (self.exit_price - self.entry_price) * self.position_size * multiplier
        return gross_profit - (self.commission or 0)

    def is_winning_trade(self) -> bool:
        """Check if trade is profitable"""
        return bool(self.profit and self.profit > 0)

    def get_duration(self) -> Optional[timedelta]:
        """Calculate trade duration"""
        if self.exit_time and self.entry_time:
            return self.exit_time - self.entry_time
        return None

class TradeTracker:
    """
    Comprehensive trade tracking and analysis system

    Features:
    - Trade recording and management
    - Performance metrics calculation
    - Risk analysis
    - Equity tracking
    - Statistics generation
    """

    def __init__(self):
        """Initialize tracker with empty trade list"""
        self.trades: List[Trade] = []
        self.logger = logging.getLogger(__name__)

    def add_trade(
        self,
        entry_price: float,
        exit_price: float,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        position_size: int = 1,
        direction: Union[str, TradeDirection] = TradeDirection.LONG,
        stop_loss: float = 0,
        take_profit: float = 0,
        commission: float = 0,
        risk_amount: float = 0
    ) -> None:
        """
        Add a new trade to the tracker with validation

        Args:
            entry_price: Trade entry price
            exit_price: Trade exit price
            entry_time: Entry timestamp (defaults to now)
            exit_time: Exit timestamp (defaults to now)
            position_size: Number of units traded
            direction: Trade direction (LONG/SHORT)
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            commission: Trading commission
            risk_amount: Amount risked on trade
        """
        try:
            # Set default timestamps
            entry_time = entry_time or datetime.now()
            exit_time = exit_time or datetime.now()

            # Convert direction to enum if string
            if isinstance(direction, str):
                direction = TradeDirection[direction.upper()]

            # Create trade object
            trade = Trade(
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                position_size=position_size,
                direction=direction,
                stop_loss=stop_loss or entry_price * 0.99,  # Default stop loss
                take_profit=take_profit or entry_price * 1.02,  # Default take profit
                status=TradeStatus.CLOSED,
                commission=commission,
                risk_amount=risk_amount
            )

            trade.profit = trade.calculate_profit()
            self.trades.append(trade)

        except Exception as e:
            self.logger.error(f"Failed to add trade: {str(e)}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return self._empty_metrics()

        try:
            profits = [t.profit for t in self.trades if t.profit is not None]
            durations = [t.get_duration() for t in self.trades if t.get_duration()]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]

            return {
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
                'total_profit': sum(profits),
                'average_profit': sum(profits) / len(profits) if profits else 0,
                'largest_win': max(winning_trades) if winning_trades else 0,
                'largest_loss': min(losing_trades) if losing_trades else 0,
                'avg_trade_duration': (
                    sum(durations, timedelta()) / len(durations)
                    if durations else timedelta()
                ),
                'profit_factor': self._calculate_profit_factor()
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {str(e)}")
            return self._empty_metrics()

    def get_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        if not self.trades:
            return {
                'risk_reward_ratio': 0.0,
                'avg_risk_per_trade': 0.0,
                'max_drawdown': 0.0
            }

        try:
            total_profit = sum(t.profit for t in self.trades if t.profit)
            total_risk = sum(t.risk_amount for t in self.trades if t.risk_amount)
            risk_reward = total_profit / total_risk if total_risk > 0 else 0.0

            return {
                'risk_reward_ratio': risk_reward,
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
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.profit for t in self.trades if t.profit and t.profit > 0)
        gross_loss = abs(sum(t.profit for t in self.trades if t.profit and t.profit < 0))
        return gross_profit / gross_loss if gross_loss != 0 else 0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum peak-to-trough decline"""
        if not self.trades:
            return 0.0

        equity = 0
        peak = 0
        max_dd = 0

        for trade in self.trades:
            if trade.profit:
                equity += trade.profit
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

        return max_dd

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
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

    def get_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades"""
        max_streak = current_streak = 0
        for trade in self.trades:
            if trade.is_winning_trade():
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def get_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades"""
        max_streak = current_streak = 0
        for trade in self.trades:
            if not trade.is_winning_trade():
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
