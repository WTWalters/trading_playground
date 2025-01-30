"""
Advanced liquidity analysis module.

This module implements sophisticated liquidity measures:
- Kyle's lambda (price impact coefficient)
- Amihud's illiquidity ratio
- Bid-ask bounce effects
- Pastor-Stambaugh liquidity factor
"""

from typing import Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class LiquidityAnalyzer:
    """
    Advanced liquidity analysis tools.
    
    Implements various liquidity measures focusing on:
    - Price impact estimation
    - Trading costs
    - Market resilience
    - Liquidity risk
    """
    
    def __init__(self, 
                 trades: pd.DataFrame,
                 quotes: Optional[pd.DataFrame] = None,
                 market_data: Optional[pd.DataFrame] = None):
        """
        Initialize with trade and quote data.
        
        Args:
            trades: DataFrame with trade data (price, volume)
            quotes: Optional quote data (bid, ask)
            market_data: Optional market index/factors
        """
        self.trades = trades
        self.quotes = quotes
        self.market_data = market_data
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate input data requirements."""
        required_cols = {'timestamp', 'price', 'volume'}
        if not required_cols.issubset(self.trades.columns):
            raise ValueError(f"Trades must contain columns: {required_cols}")
            
        if self.quotes is not None:
            quote_cols = {'timestamp', 'bid', 'ask'}
            if not quote_cols.issubset(self.quotes.columns):
                raise ValueError(f"Quotes must contain columns: {quote_cols}")
    
    def calculate_kyle_lambda(self, 
                            window: str = '1D',
                            min_observations: int = 30) -> pd.Series:
        """
        Calculate Kyle's lambda (price impact coefficient).
        
        λ = cov(ΔP, V) / var(V)
        where ΔP is price change and V is signed volume.
        
        Args:
            window: Estimation window
            min_observations: Minimum points for estimation
            
        Returns:
            pd.Series: Kyle's lambda over time
        """
        # Calculate price changes
        price_changes = self.trades['price'].diff()
        
        # Get signed volume (if available) or use absolute volume
        if 'direction' in self.trades.columns:
            signed_volume = self.trades['volume'] * self.trades['direction']
        else:
            signed_volume = self.trades['volume']
        
        def estimate_lambda(group):
            if len(group) < min_observations:
                return np.nan
            
            # Calculate using covariance method
            cov_matrix = np.cov(group['price_change'], group['volume'])
            volume_var = np.var(group['volume'])
            
            if volume_var == 0:
                return np.nan
                
            return cov_matrix[0, 1] / volume_var
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'price_change': price_changes,
            'volume': signed_volume
        })
        
        # Calculate rolling lambda
        lambda_series = (
            analysis_df.groupby(pd.Grouper(freq=window))
            .apply(estimate_lambda)
        )
        
        return lambda_series
    
    def calculate_amihud_ratio(self, 
                             window: str = '1D',
                             scaling: float = 1e6) -> pd.Series:
        """
        Calculate Amihud's illiquidity ratio.
        
        ILLIQ = |R| / VOLD
        where R is return and VOLD is daily volume in dollars.
        
        Args:
            window: Aggregation window
            scaling: Volume scaling factor
            
        Returns:
            pd.Series: Amihud ratio over time
        """
        # Calculate returns
        returns = self.trades['price'].pct_change().abs()
        
        # Calculate dollar volume
        dollar_volume = self.trades['price'] * self.trades['volume']
        
        # Aggregate by window
        window_returns = returns.groupby(pd.Grouper(freq=window)).mean()
        window_volume = dollar_volume.groupby(pd.Grouper(freq=window)).sum()
        
        # Calculate ratio with scaling
        amihud_ratio = (window_returns / (window_volume / scaling)).fillna(0)
        
        return amihud_ratio
    
    def calculate_bid_ask_bounce(self, 
                               window: str = '1D') -> pd.Series:
        """
        Calculate bid-ask bounce effect.
        
        Estimates impact of bid-ask bounce on observed volatility.
        
        Args:
            window: Analysis window
            
        Returns:
            pd.Series: Bid-ask bounce impact
        """
        if self.quotes is None:
            raise ValueError("Quote data required for bid-ask bounce calculation")
            
        # Calculate effective spread
        midpoint = (self.quotes['bid'] + self.quotes['ask']) / 2
        effective_spread = 2 * abs(self.trades['price'] - midpoint)
        
        # Calculate first-order autocorrelation
        def calculate_bounce(group):
            if len(group) < 2:
                return np.nan
                
            returns = np.diff(np.log(group['price']))
            autocorr = pd.Series(returns).autocorr(lag=1)
            spread_impact = np.sqrt(-autocorr) * group['spread'].mean()
            return spread_impact
        
        # Combine data
        analysis_df = pd.DataFrame({
            'price': self.trades['price'],
            'spread': effective_spread
        })
        
        # Calculate by window
        bounce_effect = (
            analysis_df.groupby(pd.Grouper(freq=window))
            .apply(calculate_bounce)
        )
        
        return bounce_effect
    
    def calculate_pastor_stambaugh(self, 
                                 window: str = '1M',
                                 min_observations: int = 15) -> pd.Series:
        """
        Calculate Pastor-Stambaugh liquidity factor.
        
        Measures return reversal associated with volume.
        
        Args:
            window: Estimation window
            min_observations: Minimum observations
            
        Returns:
            pd.Series: Liquidity factor over time
        """
        if self.market_data is None:
            raise ValueError("Market data required for Pastor-Stambaugh factor")
            
        def estimate_factor(group):
            if len(group) < min_observations:
                return np.nan
                
            # Calculate excess returns
            excess_returns = group['return'] - group['rf']
            
            # Estimate regression: r(t+1) = θ + Φr(t) + γsign(r(t))V(t) + ε(t+1)
            X = pd.DataFrame({
                'return': group['return'],
                'signed_vol': np.sign(group['return']) * group['volume']
            })
            X = sm.add_constant(X)
            
            y = group['return'].shift(-1)
            
            # Remove last row (no next day return)
            X = X[:-1]
            y = y[:-1].values
            
            if len(y) < min_observations:
                return np.nan
                
            # Estimate regression
            model = sm.OLS(y, X).fit()
            return model.params['signed_vol']  # γ coefficient
            
        # Prepare data
        analysis_df = pd.DataFrame({
            'return': self.trades['price'].pct_change(),
            'volume': self.trades['volume'],
            'rf': self.market_data['rf'] if 'rf' in self.market_data else 0
        })
        
        # Calculate factor by window
        liquidity_factor = (
            analysis_df.groupby(pd.Grouper(freq=window))
            .apply(estimate_factor)
        )
        
        return liquidity_factor
    
    def get_liquidity_metrics(self, 
                            window: str = '1D') -> pd.DataFrame:
        """
        Calculate comprehensive liquidity metrics.
        
        Args:
            window: Analysis window
            
        Returns:
            pd.DataFrame: DataFrame with liquidity metrics
        """
        metrics = pd.DataFrame({
            'kyle_lambda': self.calculate_kyle_lambda(window=window),
            'amihud_ratio': self.calculate_amihud_ratio(window=window)
        })
        
        # Add bid-ask metrics if quote data available
        if self.quotes is not None:
            metrics['bid_ask_bounce'] = self.calculate_bid_ask_bounce(window=window)
            
        # Add Pastor-Stambaugh if market data available
        if self.market_data is not None:
            metrics['ps_liquidity'] = self.calculate_pastor_stambaugh(window=window)
            
        return metrics
    
    def calculate_liquidity_risk(self,
                               window: str = '1M',
                               quantile: float = 0.95) -> pd.DataFrame:
        """
        Calculate liquidity risk metrics.
        
        Args:
            window: Risk measurement window
            quantile: Risk quantile (e.g., 0.95 for 95% VaR)
            
        Returns:
            pd.DataFrame: Liquidity risk metrics
        """
        # Get base liquidity metrics
        metrics = self.get_liquidity_metrics(window='1D')
        
        # Calculate risk measures
        risk_metrics = pd.DataFrame()
        
        for column in metrics.columns:
            series = metrics[column]
            risk_metrics[f'{column}_var'] = series.rolling(window).quantile(quantile)
            risk_metrics[f'{column}_cvar'] = series[series >= risk_metrics[f'{column}_var']].mean()
            risk_metrics[f'{column}_vol'] = series.rolling(window).std()
            
        return risk_metrics