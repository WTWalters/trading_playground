# src/market_analysis/trade.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import logging

@dataclass
class Trade:
    """Individual trade record"""
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
        """Calculate profit/loss for the trade"""
        if self.exit_price is None:
            return 0.0

        multiplier = 1 if self.direction == 'LONG' else -1
        gross_profit = (self.exit_price - self.entry_price) * self.position_size * multiplier

        if self.commission:
            gross_profit -= self.commission

        return gross_profit

class TradeTracker:
    """Track and analyze trading performance"""

    def __init__(self):
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
        commission: float = 0
    ):
        """Add a new trade to the tracker"""
        try:
            if entry_time is None:
                entry_time = datetime.now()
            if exit_time is None:
                exit_time = datetime.now()

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
                commission=commission
            )

            trade.profit = trade.calculate_profit()
            self.trades.append(trade)

        except Exception as e:
            self.logger.error(f"Failed to add trade: {str(e)}")

    def get_metrics(self) -> Dict:
        """Calculate trading performance metrics"""
        try:
            if not self.trades:
                return self._empty_metrics()

            profits = [trade.profit for trade in self.trades if trade.profit is not None]
            durations = [(trade.exit_time - trade.entry_time) for trade in self.trades
                        if trade.exit_time and trade.entry_time]

            return {
                'total_trades': len(self.trades),
                'winning_trades': len([p for p in profits if p > 0]),
                'losing_trades': len([p for p in profits if p < 0]),
                'win_rate': self.get_win_rate(),
                'total_profit': sum(profits),
                'average_profit': sum(profits) / len(profits) if profits else 0,
                'largest_win': max(profits) if profits else 0,
                'largest_loss': min(profits) if profits else 0,
                'average_duration': sum(durations, timedelta()) / len(durations) if durations else timedelta(),
                'profit_factor': self._calculate_profit_factor()
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {str(e)}")
            return self._empty_metrics()

    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.trades:
            return 0.0

        winning_trades = len([t for t in self.trades if t.profit and t.profit > 0])
        return winning_trades / len(self.trades)

    def get_total_profit(self) -> float:
        """Calculate total profit"""
        return sum(trade.profit for trade in self.trades if trade.profit is not None)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.profit for t in self.trades if t.profit and t.profit > 0)
        gross_loss = abs(sum(t.profit for t in self.trades if t.profit and t.profit < 0))

        return gross_profit / gross_loss if gross_loss != 0 else 0

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
            'average_duration': timedelta(),
            'profit_factor': 0.0
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis"""
        return pd.DataFrame([vars(t) for t in self.trades])
