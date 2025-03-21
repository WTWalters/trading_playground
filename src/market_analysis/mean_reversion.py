"""Mean reversion analysis module for statistical trading strategies.

This module provides tools for analyzing mean reversion characteristics
of time series and pairs, including Z-score calculation, half-life estimation,
and generation of trading signals based on mean reversion.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import statsmodels.api as sm
from datetime import datetime, timedelta
from ..database.manager import DatabaseManager

class MeanReversionAnalyzer:
    """Analyzes mean reversion characteristics of time series and pairs.
    
    This class implements various statistics and indicators for mean reversion
    trading, including half-life calculation, Z-score normalization, and
    signal generation.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the MeanReversionAnalyzer.
        
        Args:
            db_manager: DatabaseManager instance for data access
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
    async def calculate_half_life(self, spread: np.ndarray) -> Dict:
        """Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.
        
        The half-life represents the time it takes for the spread to mean-revert
        halfway back to its equilibrium. It's a key parameter for mean reversion
        strategies as it helps determine optimal holding periods.
        
        Args:
            spread: The spread series (typically residuals from cointegration regression)
            
        Returns:
            Dictionary with half-life calculation results
        """
        try:
            # Prepare data for regression
            lag_spread = spread[:-1]
            current_spread = spread[1:]
            delta = current_spread - lag_spread
            
            # Regression model: delta = lambda * lag_spread + error
            X = sm.add_constant(lag_spread)
            model = sm.OLS(delta, X)
            results = model.fit()
            
            # Extract lambda coefficient
            lambda_coefficient = results.params[1]
            
            # Calculate half-life: ln(2) / lambda (if mean-reverting)
            if lambda_coefficient < 0:
                half_life = -np.log(2) / lambda_coefficient
            else:
                half_life = float('inf')  # Not mean-reverting
            
            return {
                'half_life': float(half_life),
                'lambda': float(lambda_coefficient),
                'is_mean_reverting': lambda_coefficient < 0,
                'regression_results': {
                    'params': results.params.tolist(),
                    'r_squared': results.rsquared,
                    'p_value': results.f_pvalue
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating half-life: {e}")
            raise
            
    async def calculate_zscore(self, spread: np.ndarray, window: Optional[int] = None) -> np.ndarray:
        """Calculate Z-score of a spread series.
        
        The Z-score normalizes the spread by subtracting the mean and dividing by
        the standard deviation, making it easier to identify extreme values.
        
        Args:
            spread: The spread series
            window: Rolling window size for dynamic Z-score (None for full series)
            
        Returns:
            Z-score array
        """
        try:
            if window is None:
                # Calculate Z-score for full series
                mean = np.mean(spread)
                std = np.std(spread)
                
                if std == 0:
                    self.logger.warning("Standard deviation is zero, returning zeros for Z-score")
                    return np.zeros_like(spread)
                
                return (spread - mean) / std
            else:
                # Calculate rolling Z-score
                s = pd.Series(spread)
                rolling_mean = s.rolling(window=window).mean()
                rolling_std = s.rolling(window=window).std()
                
                # Replace zeros in std to avoid division by zero
                rolling_std = rolling_std.replace(0, np.nan)
                
                # Calculate Z-score
                z_score = (s - rolling_mean) / rolling_std
                
                # Fill NaN values at the beginning
                z_score.fillna(0, inplace=True)
                
                return z_score.values
        except Exception as e:
            self.logger.error(f"Error calculating Z-score: {e}")
            raise
            
    async def generate_mean_reversion_signals(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        window: Optional[int] = None,
        source: Optional[str] = None
    ) -> Dict:
        """Generate mean reversion signals for a pair of securities.
        
        Uses Z-score to generate entry and exit signals for pairs trading:
        - Entry: When Z-score exceeds entry_threshold (or below -entry_threshold)
        - Exit: When Z-score returns to exit_threshold (or below -exit_threshold)
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            hedge_ratio: Hedge ratio between symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            entry_threshold: Z-score threshold for trade entry
            exit_threshold: Z-score threshold for trade exit
            window: Window size for rolling Z-score (None for full sample)
            source: Data source (optional)
            
        Returns:
            Dictionary with signal information
        """
        try:
            # Fetch data for both symbols
            df1 = await self.db_manager.get_market_data(
                symbol1, start_date, end_date, timeframe, source
            )
            df2 = await self.db_manager.get_market_data(
                symbol2, start_date, end_date, timeframe, source
            )
            
            if df1.empty or df2.empty:
                self.logger.warning("Insufficient data for signal generation")
                return {'signals': [], 'metadata': {}}
            
            # Align both series on the same dates
            joined = pd.DataFrame({
                symbol1: df1['close'],
                symbol2: df2['close']
            }).dropna()
            
            # Calculate spread
            spread = joined[symbol2] - hedge_ratio * joined[symbol1]
            
            # Calculate Z-score
            zscore = await self.calculate_zscore(spread.values, window)
            
            # Calculate half-life
            half_life_result = await self.calculate_half_life(spread.values)
            
            # Generate signals
            signals = pd.DataFrame({
                'time': joined.index,
                'spread': spread.values,
                'zscore': zscore,
                symbol1: joined[symbol1].values,
                symbol2: joined[symbol2].values
            })
            
            # Add signal columns
            signals['signal'] = np.zeros(len(signals))
            
            # Long spread signal: long symbol2, short symbol1
            signals.loc[signals['zscore'] <= -entry_threshold, 'signal'] = 1
            
            # Short spread signal: short symbol2, long symbol1
            signals.loc[signals['zscore'] >= entry_threshold, 'signal'] = -1
            
            # Exit signals based on exit threshold
            # We'll mark exits with zeros for simplicity
            
            # Create the output structure
            result = {
                'signals': signals.to_dict(orient='records'),
                'metadata': {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'hedge_ratio': hedge_ratio,
                    'half_life': half_life_result['half_life'],
                    'entry_threshold': entry_threshold,
                    'exit_threshold': exit_threshold,
                    'period': {
                        'start': start_date,
                        'end': end_date
                    },
                    'timeframe': timeframe
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signals: {e}")
            raise
            
    async def analyze_pair_statistics(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        window: Optional[int] = None,
        source: Optional[str] = None
    ) -> Dict:
        """Calculate comprehensive statistics for a pair.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            hedge_ratio: Hedge ratio between symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            window: Window size for rolling statistics
            source: Data source (optional)
            
        Returns:
            Dictionary with pair statistics
        """
        try:
            # Fetch data for both symbols
            df1 = await self.db_manager.get_market_data(
                symbol1, start_date, end_date, timeframe, source
            )
            df2 = await self.db_manager.get_market_data(
                symbol2, start_date, end_date, timeframe, source
            )
            
            if df1.empty or df2.empty:
                self.logger.warning("Insufficient data for pair statistics")
                return {}
            
            # Align both series on the same dates
            joined = pd.DataFrame({
                symbol1: df1['close'],
                symbol2: df2['close']
            }).dropna()
            
            # Calculate returns
            returns1 = joined[symbol1].pct_change().dropna()
            returns2 = joined[symbol2].pct_change().dropna()
            
            # Calculate spread
            spread = joined[symbol2] - hedge_ratio * joined[symbol1]
            
            # Calculate half-life
            half_life_result = await self.calculate_half_life(spread.values)
            
            # Calculate Z-score
            zscore = await self.calculate_zscore(spread.values, window)
            
            # Calculate correlation
            correlation = returns1.corr(returns2)
            
            # Calculate volatility
            vol1 = returns1.std() * np.sqrt(252)  # Annualized
            vol2 = returns2.std() * np.sqrt(252)  # Annualized
            spread_vol = pd.Series(spread).pct_change().std() * np.sqrt(252)
            
            # Calculate mean reversion indicators
            is_mean_reverting = half_life_result['lambda'] < 0
            
            # Calculate spread percentiles (for threshold setting)
            zscore_abs = np.abs(zscore)
            zscore_95 = np.percentile(zscore_abs, 95)
            zscore_90 = np.percentile(zscore_abs, 90)
            zscore_80 = np.percentile(zscore_abs, 80)
            
            # Calculate return potential based on Z-score
            mean_return_potential = np.mean(np.abs(zscore)) * spread_vol
            
            return {
                'correlation': float(correlation),
                'half_life': float(half_life_result['half_life']),
                'is_mean_reverting': is_mean_reverting,
                'volatility': {
                    symbol1: float(vol1),
                    symbol2: float(vol2),
                    'spread': float(spread_vol)
                },
                'zscore_thresholds': {
                    '95th_percentile': float(zscore_95),
                    '90th_percentile': float(zscore_90),
                    '80th_percentile': float(zscore_80)
                },
                'return_potential': float(mean_return_potential),
                'hedge_ratio': float(hedge_ratio),
                'data_points': len(joined),
                'period': {
                    'start': start_date,
                    'end': end_date
                },
                'timeframe': timeframe
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pair statistics: {e}")
            raise
