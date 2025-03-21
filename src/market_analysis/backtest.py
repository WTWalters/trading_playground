"""
Backtesting module for mean reversion trading strategies.

This module provides functionality to backtest mean reversion trading strategies
based on statistical signals generated from cointegrated pairs.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

from ..database.manager import DatabaseManager

class BacktestResult:
    """Container for backtest results."""
    
    def __init__(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        metrics: Dict,
        signals: pd.DataFrame,
        pair_info: Dict
    ):
        """Initialize backtest result container.
        
        Args:
            equity_curve: Series with portfolio equity over time
            trades: DataFrame with individual trade information
            metrics: Dictionary with performance metrics
            signals: DataFrame with signal information
            pair_info: Dictionary with pair information
        """
        self.equity_curve = equity_curve
        self.trades = trades
        self.metrics = metrics
        self.signals = signals
        self.pair_info = pair_info
        
    def plot(self, output_path: Optional[str] = None):
        """Plot backtest results.
        
        Args:
            output_path: Path to save the plot (None for display only)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot equity curve
        ax1.plot(self.equity_curve, color='blue', linewidth=2)
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)
        
        # Add drawdown shading
        hwm = self.equity_curve.cummax()
        drawdown = (self.equity_curve / hwm - 1) * 100
        
        # Shade drawdown periods
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0:
                ax1.fill_between([drawdown.index[i], drawdown.index[i+1] if i+1 < len(drawdown) else drawdown.index[i]], 
                                 [self.equity_curve.iloc[i], self.equity_curve.iloc[i+1] if i+1 < len(self.equity_curve) else self.equity_curve.iloc[i]],
                                 [hwm.iloc[i], hwm.iloc[i+1] if i+1 < len(hwm) else hwm.iloc[i]],
                                 color='red', alpha=0.3)
        
        # Plot spread
        ax2.plot(self.signals['spread'], color='green')
        ax2.set_title('Spread')
        ax2.set_ylabel('Spread Value')
        ax2.grid(True)
        
        # Plot Z-score and trades
        ax3.plot(self.signals['zscore'], color='blue')
        ax3.axhline(y=self.pair_info.get('entry_threshold', 2.0), color='r', linestyle='--', alpha=0.3)
        ax3.axhline(y=-self.pair_info.get('entry_threshold', 2.0), color='r', linestyle='--', alpha=0.3)
        ax3.axhline(y=self.pair_info.get('exit_threshold', 0.0), color='g', linestyle='--', alpha=0.3)
        ax3.axhline(y=-self.pair_info.get('exit_threshold', 0.0), color='g', linestyle='--', alpha=0.3)
        ax3.axhline(y=0.0, color='black', linestyle='-', alpha=0.2)
        
        # Mark trade entries and exits
        for _, trade in self.trades.iterrows():
            # Entry
            ax3.scatter(trade['entry_time'], trade['entry_zscore'], 
                       color='green' if trade['direction'] == 1 else 'red',
                       marker='^' if trade['direction'] == 1 else 'v', s=100)
            
            # Exit
            if not pd.isna(trade['exit_time']):
                ax3.scatter(trade['exit_time'], trade['exit_zscore'], 
                           color='black', marker='o', s=100)
                
                # Connect entry and exit with a line
                ax3.plot([trade['entry_time'], trade['exit_time']], 
                        [trade['entry_zscore'], trade['exit_zscore']],
                        color='gray', linestyle='--', alpha=0.5)
        
        ax3.set_title('Z-Score and Trades')
        ax3.set_ylabel('Z-Score')
        ax3.set_xlabel('Date')
        ax3.grid(True)
        
        # Add metrics as text
        textbox = '\n'.join([
            f"Total Return: {self.metrics['total_return']:.2f}%",
            f"Annual Return: {self.metrics['annual_return']:.2f}%",
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}",
            f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%",
            f"Win Rate: {self.metrics['win_rate']:.2f}%",
            f"Profit Factor: {self.metrics['profit_factor']:.2f}",
            f"Avg Trade: {self.metrics['avg_trade_return']:.2f}%"
        ])
        
        fig.text(0.15, 0.01, textbox, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
    def save_report(self, output_dir: str):
        """Save a comprehensive backtest report.
        
        Args:
            output_dir: Directory to save the report
        """
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save equity curve
        self.equity_curve.to_csv(f"{output_dir}/equity_curve.csv")
        
        # Save trades
        self.trades.to_csv(f"{output_dir}/trades.csv", index=False)
        
        # Save signals
        self.signals.to_csv(f"{output_dir}/signals.csv", index=False)
        
        # Generate plot
        self.plot(f"{output_dir}/backtest_plot.png")
        
        # Create summary report in markdown
        with open(f"{output_dir}/backtest_summary.md", "w") as f:
            f.write(f"# Backtest Summary: {self.pair_info['symbol1']}/{self.pair_info['symbol2']}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Pair Information\n\n")
            f.write(f"- Symbol 1: {self.pair_info['symbol1']}\n")
            f.write(f"- Symbol 2: {self.pair_info['symbol2']}\n")
            f.write(f"- Hedge Ratio: {self.pair_info['hedge_ratio']:.4f}\n")
            f.write(f"- Half-Life: {self.pair_info.get('half_life', 'N/A')} days\n")
            f.write(f"- Correlation: {self.pair_info.get('correlation', 'N/A')}\n")
            f.write(f"- Timeframe: {self.pair_info.get('timeframe', '1d')}\n\n")
            
            f.write("## Strategy Parameters\n\n")
            f.write(f"- Entry Threshold: {self.pair_info.get('entry_threshold', 2.0)}\n")
            f.write(f"- Exit Threshold: {self.pair_info.get('exit_threshold', 0.0)}\n")
            f.write(f"- Risk Per Trade: {self.pair_info.get('risk_per_trade', 2.0)}%\n")
            f.write(f"- Initial Capital: ${self.pair_info.get('initial_capital', 100000)}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in sorted(self.metrics.items()):
                if isinstance(value, float):
                    f.write(f"| {key.replace('_', ' ').title()} | {value:.2f}{' %' if 'return' in key or 'drawdown' in key or 'rate' in key else ''} |\n")
                else:
                    f.write(f"| {key.replace('_', ' ').title()} | {value} |\n")
            
            f.write("\n## Trade Statistics\n\n")
            f.write(f"- Total Trades: {len(self.trades)}\n")
            f.write(f"- Winning Trades: {sum(self.trades['pnl'] > 0)}\n")
            f.write(f"- Losing Trades: {sum(self.trades['pnl'] < 0)}\n")
            f.write(f"- Win Rate: {self.metrics['win_rate']:.2f}%\n")
            f.write(f"- Average Trade Duration: {self.metrics.get('avg_trade_duration', 'N/A')} days\n")
            f.write(f"- Average Profit on Winners: {self.metrics.get('avg_win', 0.0):.2f}%\n")
            f.write(f"- Average Loss on Losers: {self.metrics.get('avg_loss', 0.0):.2f}%\n")
            f.write(f"- Profit Factor: {self.metrics['profit_factor']:.2f}\n\n")
            
            f.write("## Top 5 Trades\n\n")
            top_trades = self.trades.sort_values('pnl_pct', ascending=False).head(5)
            f.write("| Entry Date | Exit Date | Direction | Entry Z-Score | Exit Z-Score | P&L (%) |\n")
            f.write("|------------|-----------|-----------|---------------|--------------|----------|\n")
            for _, trade in top_trades.iterrows():
                direction_str = "Long Spread" if trade['direction'] == 1 else "Short Spread"
                f.write(f"| {trade['entry_time'].strftime('%Y-%m-%d')} | {trade['exit_time'].strftime('%Y-%m-%d') if not pd.isna(trade['exit_time']) else 'Open'} | {direction_str} | {trade['entry_zscore']:.2f} | {trade['exit_zscore']:.2f if not pd.isna(trade['exit_zscore']) else 0.0} | {trade['pnl_pct']:.2f}% |\n")
            
            f.write("\n## Worst 5 Trades\n\n")
            worst_trades = self.trades.sort_values('pnl_pct').head(5)
            f.write("| Entry Date | Exit Date | Direction | Entry Z-Score | Exit Z-Score | P&L (%) |\n")
            f.write("|------------|-----------|-----------|---------------|--------------|----------|\n")
            for _, trade in worst_trades.iterrows():
                direction_str = "Long Spread" if trade['direction'] == 1 else "Short Spread"
                f.write(f"| {trade['entry_time'].strftime('%Y-%m-%d')} | {trade['exit_time'].strftime('%Y-%m-%d') if not pd.isna(trade['exit_time']) else 'Open'} | {direction_str} | {trade['entry_zscore']:.2f} | {trade['exit_zscore']:.2f if not pd.isna(trade['exit_zscore']) else 0.0} | {trade['pnl_pct']:.2f}% |\n")

class MeanReversionBacktester:
    """Backtester for mean reversion strategies on pairs."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize backtester.
        
        Args:
            db_manager: Database manager for data access
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
    async def backtest_pair(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_holding_period: Optional[int] = None,
        risk_per_trade: float = 2.0,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        timeframe: str = '1d',
        zscore_window: Optional[int] = None,
        source: Optional[str] = None
    ) -> BacktestResult:
        """Backtest a mean reversion strategy for a pair.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            hedge_ratio: Hedge ratio between symbols
            start_date: Start date for backtest
            end_date: End date for backtest
            entry_threshold: Z-score threshold for trade entry
            exit_threshold: Z-score threshold for trade exit
            stop_loss_pct: Stop loss percentage (None for no stop loss)
            take_profit_pct: Take profit percentage (None for no take profit)
            max_holding_period: Maximum holding period in days (None for no limit)
            risk_per_trade: Risk percentage per trade
            initial_capital: Initial capital for the backtest
            commission_pct: Commission percentage per trade
            timeframe: Data timeframe
            zscore_window: Window size for calculating Z-score (None for full sample)
            source: Data source
            
        Returns:
            BacktestResult with backtest results
        """
        try:
            # Fetch data for both symbols
            df1 = await self.db_manager.get_market_data(
                symbol1, start_date, end_date, timeframe, source
            )
            df2 = await self.db_manager.get_market_data(
                symbol2, start_date, end_date, timeframe, source
            )
            
            # Debug the data fetching
            self.logger.info(f"Fetched data for {symbol1}: {len(df1)} rows from {start_date} to {end_date}")
            self.logger.info(f"Fetched data for {symbol2}: {len(df2)} rows from {start_date} to {end_date}")
            
            if df1.empty or df2.empty:
                self.logger.warning("Insufficient data for backtesting")
                return None
            
            # Align both series on the same dates
            joined = pd.DataFrame({
                symbol1: df1['close'],
                symbol2: df2['close']
            }).dropna()
            
            # Calculate spread
            spread = joined[symbol2] - hedge_ratio * joined[symbol1]
            
            # Calculate Z-score
            if zscore_window is not None:
                # Rolling Z-score
                rolling_mean = spread.rolling(window=zscore_window).mean()
                rolling_std = spread.rolling(window=zscore_window).std()
                zscore = (spread - rolling_mean) / rolling_std
                zscore.fillna(0, inplace=True)
            else:
                # Full-sample Z-score
                mean = spread.mean()
                std = spread.std()
                zscore = (spread - mean) / std
            
            # Create signals DataFrame
            signals = pd.DataFrame({
                'time': joined.index,
                symbol1: joined[symbol1].values,
                symbol2: joined[symbol2].values,
                'spread': spread.values,
                'zscore': zscore.values,
                'signal': np.zeros(len(joined))
            })
            signals.set_index('time', inplace=True)
            
            # Generate signals
            signals.loc[signals['zscore'] <= -entry_threshold, 'signal'] = 1  # Long spread signal
            signals.loc[signals['zscore'] >= entry_threshold, 'signal'] = -1  # Short spread signal
            
            # Initialize variables for backtesting
            position = 0
            entry_price_spread = 0
            entry_time = None
            entry_zscore = 0
            trades = []
            equity = [initial_capital]
            current_capital = initial_capital
            
            # Process each bar
            for i in range(1, len(signals)):
                current_time = signals.index[i]
                prev_time = signals.index[i-1]
                
                # If we have a position, check for exit conditions
                if position != 0:
                    # Calculate days in trade
                    days_in_trade = (current_time - entry_time).days
                    
                    # Calculate current P&L
                    current_spread = signals['spread'].iloc[i]
                    points_gained = (current_spread - entry_price_spread) * position
                    dollar_pnl = points_gained * trade_size / entry_price_spread
                    pct_pnl = dollar_pnl / current_capital * 100
                    
                    # Check exit conditions
                    exit_signal = False
                    exit_reason = ""
                    
                    # 1. Z-score exit threshold
                    if (position == 1 and signals['zscore'].iloc[i] >= exit_threshold) or \
                       (position == -1 and signals['zscore'].iloc[i] <= -exit_threshold):
                        exit_signal = True
                        exit_reason = "Z-score threshold"
                    
                    # 2. Stop loss hit
                    elif stop_loss_pct is not None and pct_pnl <= -stop_loss_pct:
                        exit_signal = True
                        exit_reason = "Stop loss"
                    
                    # 3. Take profit hit
                    elif take_profit_pct is not None and pct_pnl >= take_profit_pct:
                        exit_signal = True
                        exit_reason = "Take profit"
                    
                    # 4. Maximum holding period exceeded
                    elif max_holding_period is not None and days_in_trade >= max_holding_period:
                        exit_signal = True
                        exit_reason = "Max holding period"
                    
                    # Process exit if conditions met
                    if exit_signal:
                        # Update capital
                        current_capital += dollar_pnl - (2 * commission_pct * current_capital)  # Commission for both entry and exit
                        
                        # Record trade
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'direction': position,
                            'entry_price_spread': entry_price_spread,
                            'exit_price_spread': current_spread,
                            'entry_zscore': entry_zscore,
                            'exit_zscore': signals['zscore'].iloc[i],
                            'pnl': dollar_pnl,
                            'pnl_pct': pct_pnl,
                            'days_held': days_in_trade,
                            'exit_reason': exit_reason
                        })
                        
                        # Reset position
                        position = 0
                
                # Check for entry signals if we're not already in a position
                elif position == 0:
                    # Entry signal
                    if signals['signal'].iloc[i] != 0 and signals['signal'].iloc[i-1] == 0:
                        position = int(signals['signal'].iloc[i])  # 1 for long spread, -1 for short spread
                        entry_price_spread = signals['spread'].iloc[i]
                        entry_time = current_time
                        entry_zscore = signals['zscore'].iloc[i]
                        
                        # Calculate position size based on risk
                        # For simplicity, we risk a fixed percentage of capital per trade
                        trade_size = (risk_per_trade / 100) * current_capital
                
                # Update equity curve at each step
                equity.append(current_capital)
            
            # Prepare results
            equity_curve = pd.Series(equity, index=[signals.index[0]] + list(signals.index))
            trades_df = pd.DataFrame(trades)
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(equity_curve, trades_df, initial_capital)
            
            # Create pair info dictionary
            pair_info = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'hedge_ratio': hedge_ratio,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'risk_per_trade': risk_per_trade,
                'initial_capital': initial_capital,
                'timeframe': timeframe
            }
            
            return BacktestResult(equity_curve, trades_df, metrics, signals, pair_info)
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            raise
    
    def _calculate_metrics(
        self, 
        equity_curve: pd.Series, 
        trades: pd.DataFrame, 
        initial_capital: float
    ) -> Dict:
        """Calculate performance metrics from backtest results.
        
        Args:
            equity_curve: Series with portfolio equity over time
            trades: DataFrame with trade information
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Return metrics
        final_capital = equity_curve.iloc[-1]
        metrics['total_return'] = (final_capital / initial_capital - 1) * 100
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            metrics['annual_return'] = ((final_capital / initial_capital) ** (365 / days) - 1) * 100
        else:
            metrics['annual_return'] = 0.0
        
        # Drawdown metrics
        hwm = equity_curve.cummax()
        drawdown = equity_curve / hwm - 1
        metrics['max_drawdown'] = drawdown.min() * 100
        
        # Trade metrics
        if not trades.empty:
            metrics['num_trades'] = len(trades)
            metrics['win_rate'] = (trades['pnl'] > 0).mean() * 100
            
            # Profit factor
            gross_profit = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
            gross_loss = abs(trades.loc[trades['pnl'] < 0, 'pnl'].sum())
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average trade metrics
            metrics['avg_trade_return'] = trades['pnl_pct'].mean()
            metrics['avg_win'] = trades.loc[trades['pnl'] > 0, 'pnl_pct'].mean() if any(trades['pnl'] > 0) else 0
            metrics['avg_loss'] = trades.loc[trades['pnl'] < 0, 'pnl_pct'].mean() if any(trades['pnl'] < 0) else 0
            metrics['avg_trade_duration'] = trades['days_held'].mean()
            
            # Calculate daily returns for Sharpe ratio
            daily_returns = equity_curve.pct_change().dropna()
            metrics['sharpe_ratio'] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            metrics['num_trades'] = 0
            metrics['win_rate'] = 0
            metrics['profit_factor'] = 0
            metrics['avg_trade_return'] = 0
            metrics['sharpe_ratio'] = 0
        
        return metrics
