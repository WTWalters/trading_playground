"""
Liquidity analysis module.

This module implements tools for analyzing market liquidity:
- Kyle's lambda
- Amihud's illiquidity ratio
- Bid-ask bounce effects
- Volume-based liquidity measures
"""

from typing import Dict, Optional, Union, List
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class LiquidityAnalyzer:
    """
    Liquidity analysis tools.
    
    Attributes:
        trades (pd.DataFrame): Trade data with timestamp index
        quotes (pd.DataFrame, optional): Quote data with timestamp index
        market_data (pd.DataFrame, optional): Market data with timestamp index
    """
    
    def __init__(self, 
                 trades: pd.DataFrame,
                 quotes: Optional[pd.DataFrame] = None,
                 market_data: Optional[pd.DataFrame] = None):
        """Initialize with trade and quote data."""
        self.trades = self._ensure_datetime_index(trades)
        self.quotes = self._ensure_datetime_index(quotes) if quotes is not None else None
        self.market_data = self._ensure_datetime_index(market_data) if market_data is not None else None
        self._validate_data()
        
    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has datetime index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("Data must have datetime index or timestamp column")
        return df.sort_index()
        
    def _validate_data(self) -> None:
        """Validate input data requirements."""
        required_cols = {'price', 'volume'}
        if not required_cols.issubset(self.trades.columns):
            raise ValueError(f"Trades must contain columns: {required_cols}")
            
        if self.quotes is not None:
            quote_cols = {'bid', 'ask'}
            if not quote_cols.issubset(self.quotes.columns):
                raise ValueError(f"Quotes must contain columns: {quote_cols}")
                
    def calculate_kyle_lambda(self, window: str = '1D',
                            min_observations: int = 30) -> pd.Series:
        """
        Calculate Kyle's lambda (price impact coefficient).
        λ = cov(ΔP, V) / var(V)
        """
        price_changes = self.trades['price'].diff()
        signed_volume = self.trades['volume']
        
        def estimate_lambda(group):
            if len(group) < min_observations:
                return np.nan
                
            prices = group['price_change'].values
            volumes = group['volume'].values
            
            # Remove NaN values
            mask = ~np.isnan(prices) & ~np.isnan(volumes)
            if sum(mask) < min_observations:
                return np.nan
                
            prices = prices[mask]
            volumes = volumes[mask]
            
            # Calculate using covariance method
            cov_matrix = np.cov(prices, volumes)
            volume_var = np.var(volumes)
            
            if volume_var == 0:
                return np.nan
                
            return cov_matrix[0, 1] / volume_var
            
        analysis_df = pd.DataFrame({
            'price_change': price_changes,
            'volume': signed_volume
        }, index=self.trades.index)
        
        return analysis_df.resample(window).apply(estimate_lambda)
    
    def calculate_amihud_ratio(self, window: str = '1D',
                             scaling: float = 1e6) -> pd.Series:
        """Calculate Amihud's illiquidity ratio."""
        returns = self.trades['price'].pct_change().abs()
        dollar_volume = self.trades['price'] * self.trades['volume']
        
        # Resample using datetime index
        amihud = (returns / (dollar_volume / scaling)).replace([np.inf, -np.inf], np.nan)
        return amihud.resample(window).mean()
    
    def calculate_bid_ask_bounce(self, window: str = '1D') -> pd.Series:
        """Calculate bid-ask bounce effect."""
        if self.quotes is None:
            raise ValueError("Quote data required for bid-ask bounce calculation")
            
        midpoint = (self.quotes['bid'] + self.quotes['ask']) / 2
        effective_spread = 2 * abs(self.trades['price'] - midpoint)
        
        def calculate_bounce(group):
            if len(group) < 2:
                return np.nan
                
            returns = np.diff(np.log(group['price']))
            if len(returns) < 2:
                return np.nan
                
            autocorr = pd.Series(returns).autocorr(lag=1)
            if pd.isna(autocorr):
                return np.nan
                
            spread_impact = np.sqrt(-autocorr) * group['spread'].mean()
            return spread_impact
            
        analysis_df = pd.DataFrame({
            'price': self.trades['price'],
            'spread': effective_spread
        }, index=self.trades.index)
        
        return analysis_df.resample(window).apply(calculate_bounce)
    
    def get_liquidity_metrics(self, window: str = '1D') -> pd.DataFrame:
        """Calculate all available liquidity metrics."""
        metrics = pd.DataFrame({
            'kyle_lambda': self.calculate_kyle_lambda(window=window),
            'amihud_ratio': self.calculate_amihud_ratio(window=window)
        })
        
        if self.quotes is not None:
            metrics['bid_ask_bounce'] = self.calculate_bid_ask_bounce(window=window)
            
        return metrics