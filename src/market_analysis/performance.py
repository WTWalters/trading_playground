# src/market_analysis/performance.py

class PerformanceMetrics:
    """Calculate trading performance metrics"""

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252  # Daily adjustment
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        rolling_max = equity_curve.expanding().max()
        drawdown = equity_curve / rolling_max - 1
        return drawdown

    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """Calculate win rate from trade history"""
        return len(trades[trades['pnl'] > 0]) / len(trades)
