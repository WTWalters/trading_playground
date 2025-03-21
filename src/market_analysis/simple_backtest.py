# src/market_analysis/simple_backtest.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging

from .base import MarketAnalyzer, AnalysisConfig, MarketRegime

class SimpleBacktest(MarketAnalyzer):
    """
    Simple backtest implementation for testing.
    
    This is a simplified version of a backtesting system,
    primarily used for testing other components.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize backtest engine with configuration.
        
        Args:
            config: AnalysisConfig with settings
        """
        super().__init__(config)
        self.trades = []
        self.equity_curve = None
        
    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ):
        """
        Run analysis on market data.
        
        Args:
            data: OHLCV DataFrame
            additional_metrics: Optional metrics from other analyzers
            
        Returns:
            Analysis metrics
        """
        # Validate input
        if not self._validate_input(data):
            raise ValueError("Invalid input data")
            
        # Calculate simple metrics
        returns = data['close'].pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        drawdown = self._calculate_drawdown(data['close'])
        
        # Determine simple regime classification
        regime = self._determine_regime(returns, volatility)
        
        # Create metrics
        metrics_dict = {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': drawdown,
            'num_trades': 0,
            'win_rate': 0.0,
            'mean_return': returns.mean(),
            'median_return': returns.median(),
        }
        
        # Store timestamp from end of data
        timestamp = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()
        
        # Create and store metrics
        analysis_metrics = self._create_metrics(
            timestamp=timestamp,
            metrics_dict=metrics_dict,
            regime=regime,
            confidence=0.8  # Fixed confidence for testing
        )
        
        self._last_analysis = analysis_metrics
        self._analysis_history.append(analysis_metrics)
        
        return analysis_metrics
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest on market data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Backtest results dictionary
        """
        # Initialize equity with starting capital
        initial_capital = 10000.0
        equity = [initial_capital]
        
        # Simple moving average crossover for testing
        fast_ma = data['close'].rolling(window=10).mean()
        slow_ma = data['close'].rolling(window=30).mean()
        
        # Generate signals
        buy_signals = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        sell_signals = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # Initialize with no position
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        
        # Simulate trades
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            current_time = data.index[i]
            
            # Check for buy signal
            if buy_signals.iloc[i] and position <= 0:
                # Close any existing short
                if position < 0:
                    profit = entry_price - current_price
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit': profit,
                        'type': 'short'
                    })
                
                # Enter long
                position = 1
                entry_price = current_price
                entry_time = current_time
            
            # Check for sell signal
            elif sell_signals.iloc[i] and position >= 0:
                # Close any existing long
                if position > 0:
                    profit = current_price - entry_price
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit': profit,
                        'type': 'long'
                    })
                
                # Enter short
                position = -1
                entry_price = current_price
                entry_time = current_time
            
            # Update equity - very simplified
            if position != 0:
                price_change = data['close'].iloc[i] / data['close'].iloc[i-1] - 1
                equity.append(equity[-1] * (1 + price_change * position))
            else:
                equity.append(equity[-1])
        
        # Close any open position at the end
        if position != 0:
            current_price = data['close'].iloc[-1]
            current_time = data.index[-1]
            profit = (current_price - entry_price) * position
            trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'entry_price': entry_price,
                'exit_price': current_price,
                'profit': profit,
                'type': 'long' if position > 0 else 'short'
            })
        
        # Calculate performance metrics
        equity_curve = pd.Series(equity, index=data.index)
        return_series = equity_curve.pct_change().dropna()
        
        total_return = (equity[-1] / equity[0]) - 1
        sharpe = return_series.mean() / return_series.std() * np.sqrt(252) if return_series.std() > 0 else 0
        
        max_dd = self._calculate_drawdown(equity_curve)
        win_rate = len([t for t in trades if t['profit'] > 0]) / len(trades) if trades else 0
        
        # Store results
        self.trades = trades
        self.equity_curve = equity_curve
        
        # Return results
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'equity_curve': equity_curve,
            'trades': trades
        }
        
    def _calculate_drawdown(self, series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = series.cummax()
        drawdown = (series / peak - 1)
        return abs(drawdown.min())
        
    def _determine_regime(self, returns: pd.Series, volatility: float) -> MarketRegime:
        """Determine market regime based on returns and volatility."""
        if len(returns) < 10:
            return MarketRegime.UNKNOWN
            
        # Very simple regime classification for testing
        if volatility > self.config.volatility_threshold:
            return MarketRegime.VOLATILE
            
        # Check for trend
        trend_strength = abs(returns.mean() / returns.std()) if returns.std() > 0 else 0
        if trend_strength > self.config.trend_strength_threshold:
            if returns.mean() > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
                
        # Default to ranging
        return MarketRegime.RANGING