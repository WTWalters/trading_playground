# src/market_analysis/trade.py

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Set
from enum import Enum
import pandas as pd
import numpy as np
import logging
from pydantic import BaseModel, Field

class TradeDirection(Enum):
    """Trade direction classification"""
    LONG = "LONG"
    SHORT = "SHORT"

    @classmethod
    def from_string(cls, value: str) -> 'TradeDirection':
        """Convert string to TradeDirection"""
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid trade direction: {value}")

class TradeStatus(Enum):
    """Trade status classification"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

@dataclass
class TradeMetrics:
    """Container for trade performance metrics"""
    # Basic trade counts and ratios
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Profit and performance metrics
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    average_profit: float = 0.0

    # Trade size metrics
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Time and risk metrics
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta())
    avg_mae: float = 0.0  # Maximum Adverse Excursion
    avg_mfe: float = 0.0  # Maximum Favorable Excursion
    risk_reward_ratio: float = 0.0

    # Streak metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def __getitem__(self, key: str) -> Any:
        """Make metrics subscriptable"""
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return asdict(self)

@dataclass
class Trade:
    """Individual trade record"""

    # Core trade data
    entry_price: float
    position_size: float
    direction: TradeDirection
    entry_time: datetime

    # Optional parameters
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: TradeStatus = TradeStatus.PENDING

    # Trade metrics
    commission: float = 0.0
    slippage: float = 0.0
    profit: float = 0.0
    risk_amount: float = 0.0
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion

    # Trade metadata
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    tags: Set[str] = field(default_factory=set)
    notes: str = ""

    def __post_init__(self):
        """Validate trade parameters after initialization"""
        self._validate_trade_params()
        if self.exit_price:
            self.calculate_profit()

    def _validate_trade_params(self):
        """Validate trade parameters"""
        if self.position_size <= 0:
            raise ValueError("Position size must be positive")
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if self.stop_loss and self.stop_loss <= 0:
            raise ValueError("Stop loss must be positive")
        if self.take_profit and self.take_profit <= 0:
            raise ValueError("Take profit must be positive")

    def calculate_profit(self) -> float:
        """Calculate trade profit/loss"""
        if not self.exit_price:
            return 0.0

        multiplier = 1 if self.direction == TradeDirection.LONG else -1
        gross_profit = (self.exit_price - self.entry_price) * self.position_size * multiplier
        net_profit = gross_profit - self.commission - self.slippage
        self.profit = net_profit
        return net_profit

    def update_price(self, current_price: float) -> None:
        """Update MAE/MFE based on current price"""
        if self.direction == TradeDirection.LONG:
            self.mae = min(self.mae, current_price - self.entry_price)
            self.mfe = max(self.mfe, current_price - self.entry_price)
        else:
            self.mae = min(self.mae, self.entry_price - current_price)
            self.mfe = max(self.mfe, self.entry_price - current_price)

    def close_trade(self, exit_price: float, exit_time: Optional[datetime] = None) -> float:
        """Close the trade and calculate final profit"""
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.status = TradeStatus.CLOSED
        return self.calculate_profit()

    def get_duration(self) -> Optional[timedelta]:
        """Get trade duration"""
        if self.exit_time and self.entry_time:
            return self.exit_time - self.entry_time
        return None

    def is_winning_trade(self) -> bool:
        """Check if trade is profitable"""
        return self.profit > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary format"""
        return {
            'id': self.id,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'position_size': self.position_size,
            'direction': self.direction.value,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'status': self.status.value,
            'profit': self.profit,
            'commission': self.commission,
            'slippage': self.slippage,
            'risk_amount': self.risk_amount,
            'mae': self.mae,
            'mfe': self.mfe,
            'tags': list(self.tags),
            'notes': self.notes
        }

class TradeTracker:
    """Trade tracking and analysis system"""

    def __init__(self):
        """Initialize trade tracker"""
        self.trades: List[Trade] = []
        self.logger = logging.getLogger(__name__)
        self._metrics: Optional[TradeMetrics] = None
        self._equity_curve: Optional[pd.Series] = None

    def add_trade(
        self,
        trade: Optional[Union[Trade, Dict]] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        position_size: float = 1.0,
        commission: float = 0.0,
        risk_amount: float = 0.0,
    ) -> None:
        """Add trade to tracker"""
        try:
            if trade is None and entry_price is not None:
                # Create trade from parameters
                trade_obj = Trade(
                    entry_price=entry_price,
                    position_size=position_size,
                    direction=TradeDirection.LONG,
                    entry_time=entry_time or datetime.now(),
                    exit_price=exit_price,
                    exit_time=exit_time,
                    commission=commission,
                    risk_amount=risk_amount
                )
                if exit_price is not None:
                    trade_obj.status = TradeStatus.CLOSED
                    trade_obj.calculate_profit()
            elif isinstance(trade, dict):
                trade_obj = Trade(**trade)
            elif isinstance(trade, Trade):
                trade_obj = trade
            else:
                raise ValueError("Invalid trade parameters")

            self.trades.append(trade_obj)
            self._metrics = None  # Reset cached metrics
            self._equity_curve = None  # Reset cached equity curve

        except Exception as e:
            self.logger.error(f"Failed to add trade: {str(e)}")
            raise

    def get_total_profit(self) -> float:
        """Get total profit across all closed trades"""
        return sum(trade.profit for trade in self.trades if trade.status == TradeStatus.CLOSED)

    def get_win_rate(self) -> float:
        """Get current win rate"""
        metrics = self.get_metrics()
        return metrics.win_rate

    def get_max_consecutive_wins(self) -> int:
        """Get maximum consecutive winning trades"""
        metrics = self.get_metrics()
        return metrics.max_consecutive_wins

    def get_max_consecutive_losses(self) -> int:
        """Get maximum consecutive losing trades"""
        metrics = self.get_metrics()
        return metrics.max_consecutive_losses

    def get_metrics(self, refresh: bool = False) -> TradeMetrics:
        """Calculate trade metrics"""
        if self._metrics is not None and not refresh:
            return self._metrics

        try:
            closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
            if not closed_trades:
                return TradeMetrics()

            profits = [t.profit for t in closed_trades]
            winning_trades = [t for t in closed_trades if t.profit > 0]
            losing_trades = [t for t in closed_trades if t.profit < 0]
            durations = [t.get_duration() for t in closed_trades if t.get_duration()]

            self._metrics = TradeMetrics(
                total_trades=len(closed_trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=len(winning_trades) / len(closed_trades) if closed_trades else 0.0,
                profit_factor=self._calculate_profit_factor(winning_trades, losing_trades),
                max_drawdown=self._calculate_max_drawdown(),
                sharpe_ratio=self._calculate_sharpe_ratio(profits),
                average_profit=np.mean(profits) if profits else 0.0,
                avg_winner=np.mean([t.profit for t in winning_trades]) if winning_trades else 0,
                avg_loser=np.mean([t.profit for t in losing_trades]) if losing_trades else 0,
                largest_win=max([t.profit for t in winning_trades]) if winning_trades else 0,
                largest_loss=min([t.profit for t in losing_trades]) if losing_trades else 0,
                avg_trade_duration=sum(durations, timedelta()) / len(durations) if durations else timedelta(),
                avg_mae=np.mean([t.mae for t in closed_trades]),
                avg_mfe=np.mean([t.mfe for t in closed_trades]),
                risk_reward_ratio=self._calculate_risk_reward_ratio(closed_trades),
                max_consecutive_wins=self._calculate_max_consecutive('wins'),
                max_consecutive_losses=self._calculate_max_consecutive('losses')
            )

            return self._metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {str(e)}")
            return TradeMetrics()

    def get_risk_metrics(self) -> Dict[str, float]:
        """Get risk-related metrics"""
        metrics = self.get_metrics()
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]

        total_risk = sum(t.risk_amount for t in closed_trades)
        total_profit = sum(t.profit for t in closed_trades if t.profit > 0)

        return {
            'risk_reward_ratio': metrics.risk_reward_ratio,
            'avg_risk_per_trade': total_risk / len(closed_trades) if closed_trades else 0,
            'profit_to_risk': total_profit / total_risk if total_risk else 0,
            'max_drawdown': metrics.max_drawdown
        }

    def _calculate_profit_factor(self, winning_trades: List[Trade], losing_trades: List[Trade]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.profit for t in winning_trades)
        gross_loss = abs(sum(t.profit for t in losing_trades))
        return gross_profit / gross_loss if gross_loss != 0 else 0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        equity_curve = self.get_equity_curve()
        if equity_curve.empty:
            return 0.0

        rolling_max = equity_curve.expanding().max()
        drawdowns = (rolling_max - equity_curve) / rolling_max
        return float(drawdowns.max())

    def _calculate_sharpe_ratio(self, profits: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if not profits:
            return 0.0

        returns = pd.Series(profits)
        excess_returns = returns - risk_free_rate
        return float(excess_returns.mean() / returns.std() if returns.std() != 0 else 0)

    def _calculate_risk_reward_ratio(self, trades: List[Trade]) -> float:
        """Calculate risk/reward ratio"""
        total_profit = sum(t.profit for t in trades if t.profit > 0)
        total_risk = sum(t.risk_amount for t in trades if t.risk_amount > 0)
        return total_profit / total_risk if total_risk != 0 else 0

    def _calculate_max_consecutive(self, trade_type: str) -> int:
        """Calculate maximum consecutive wins or losses"""
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        if not closed_trades:
            return 0

        current_streak = 0
        max_streak = 0

        for trade in closed_trades:
            if (trade_type == 'wins' and trade.profit > 0) or (trade_type == 'losses' and trade.profit < 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def get_equity_curve(self, initial_capital: float = 10000.0) -> pd.Series:
        """Generate equity curve"""
        if self._equity_curve is not None:
            return self._equity_curve

        try:
            closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
            if not closed_trades:
                return pd.Series()

            equity = initial_capital
            equity_points = [(closed_trades[0].entry_time, equity)]

            for trade in closed_trades:
                equity += trade.profit
                equity_points.append((trade.exit_time, equity))

            self._equity_curve = pd.Series(
                [p[1] for p in equity_points],
                index=[p[0] for p in equity_points]
            )
            return self._equity_curve

        except Exception as e:
            self.logger.error(f"Failed to generate equity curve: {str(e)}")
            return pd.Series()

    def get_trade_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[TradeStatus] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered trade history"""
        filtered_trades = self.trades

        if start_date:
            filtered_trades = [t for t in filtered_trades if t.entry_time >= start_date]
        if end_date:
            filtered_trades = [t for t in filtered_trades if t.entry_time <= end_date]
        if status:
            filtered_trades = [t for t in filtered_trades if t.status == status]

        return [trade.to_dict() for trade in filtered_trades]
