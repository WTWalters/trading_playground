# src/market_analysis/trade.py [previous imports remain the same]

[Previous classes up to TradeMetrics definition remain the same]

@dataclass
class TradeMetrics:
    """Container for trade performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta())  # Changed from avg_duration
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    risk_reward_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    [Rest of TradeMetrics class remains the same]

[Trade class remains the same]

class TradeTracker:
    [Previous methods remain the same up to add_trade]

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
        """Add trade to tracker."""
        try:
            if trade is None and entry_price is not None:
                # Handle direct parameter inputs
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
                # Set status to CLOSED if exit_price is provided
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
            self._metrics = None
            self._equity_curve = None

        except Exception as e:
            self.logger.error(f"Failed to add trade: {str(e)}")
            raise

    [Previous metrics calculation methods remain the same]

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

    [Rest of class methods remain the same]