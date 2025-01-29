"""
Market impact and order flow analysis module.

This module implements tools for analyzing:
- Price impact models
- Order flow toxicity
- Trade size analysis
- Permanent vs temporary impact
- Volume profile analysis
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class MarketImpact:
    """
    Market impact analysis and modeling.
    
    Implements various market impact models:
    - Square root model
    - Linear temporary impact
    - Decay model for permanent impact
    - Volume-weighted impact
    """
    
    def __init__(self, trades: pd.DataFrame, quotes: Optional[pd.DataFrame] = None):
        """
        Initialize with trade and quote data.
        
        Args:
            trades: DataFrame with trade data (price, size, direction)
            quotes: Optional DataFrame with quote data (bid, ask)
        """
        self.trades = trades
        self.quotes = quotes
        self._validate_data()
        self._calculate_base_metrics()
        
    def _validate_data(self) -> None:
        """Validate input data requirements."""
        required_cols = {'price', 'size', 'timestamp'}
        if not required_cols.issubset(self.trades.columns):
            raise ValueError(f"Trades must contain columns: {required_cols}")
            
        if self.quotes is not None:
            quote_cols = {'bid', 'ask', 'timestamp'}
            if not quote_cols.issubset(self.quotes.columns):
                raise ValueError(f"Quotes must contain columns: {quote_cols}")
                
    def _calculate_base_metrics(self) -> None:
        """Calculate base metrics for impact modeling."""
        # Daily volume and volatility
        self.daily_volume = self.trades.groupby(
            self.trades.timestamp.dt.date
        )['size'].sum()
        
        self.daily_volatility = self.trades.groupby(
            self.trades.timestamp.dt.date
        )['price'].agg(lambda x: np.log(x / x.shift(1)).std())
        
        # Average trade size and spread
        self.avg_trade_size = self.trades['size'].mean()
        if self.quotes is not None:
            self.avg_spread = (self.quotes['ask'] - self.quotes['bid']).mean()
        else:
            self.avg_spread = None
            
    def calculate_square_root_impact(self, trade_size: float,
                                   participation_rate: float) -> float:
        """
        Calculate market impact using square root model.
        
        Impact = σ * sign(Q) * sqrt(|Q|/V) * (1 + participation_rate)
        
        Args:
            trade_size: Size of trade
            participation_rate: Participation rate (0-1)
            
        Returns:
            float: Estimated price impact in basis points
        """
        if participation_rate <= 0 or participation_rate > 1:
            raise ValueError("Participation rate must be between 0 and 1")
            
        daily_vol = self.daily_volatility.mean()
        daily_vol_bps = daily_vol * 10000
        avg_daily_volume = self.daily_volume.mean()
        
        impact = (
            daily_vol_bps *
            np.sign(trade_size) *
            np.sqrt(abs(trade_size) / avg_daily_volume) *
            (1 + participation_rate)
        )
        
        return impact
        
    def calculate_linear_impact(self, trade_size: float) -> float:
        """
        Calculate linear price impact.
        
        Impact = λ * Q, where λ is the price impact coefficient
        
        Args:
            trade_size: Size of trade
            
        Returns:
            float: Estimated price impact in basis points
        """
        # Estimate lambda from historical data
        impact_coefficient = self._estimate_impact_coefficient()
        
        return impact_coefficient * trade_size * 10000  # Convert to bps
        
    def _estimate_impact_coefficient(self) -> float:
        """Estimate price impact coefficient from historical data."""
        # Calculate price changes
        price_changes = self.trades['price'].pct_change()
        
        # Normalize trade sizes
        normalized_sizes = self.trades['size'] / self.avg_trade_size
        
        # Regression to estimate coefficient
        impact_coef = (
            (price_changes * normalized_sizes).sum() /
            (normalized_sizes ** 2).sum()
        )
        
        return impact_coef
        
    def calculate_decay_impact(self, trade_size: float,
                             horizon: int = 100) -> pd.Series:
        """
        Calculate decaying price impact over time.
        
        Args:
            trade_size: Size of trade
            horizon: Number of periods for decay
            
        Returns:
            pd.Series: Time series of impact decay
        """
        initial_impact = self.calculate_square_root_impact(
            trade_size, participation_rate=0.1
        )
        
        # Decay parameter
        decay_rate = 0.5  # Half-life decay
        
        # Generate decay series
        times = np.arange(horizon)
        decay = initial_impact * np.exp(-decay_rate * times)
        
        return pd.Series(decay, index=range(horizon))
        
    def estimate_vwap_impact(self, trade_size: float,
                           interval: str = '5min') -> float:
        """
        Estimate price impact relative to VWAP.
        
        Args:
            trade_size: Size of trade
            interval: Time interval for VWAP calculation
            
        Returns:
            float: Estimated VWAP impact in basis points
        """
        # Calculate VWAP for specified interval
        vwap = (
            (self.trades['price'] * self.trades['size']).resample(interval).sum() /
            self.trades['size'].resample(interval).sum()
        )
        
        # Calculate average deviation from VWAP for similar-sized trades
        size_filter = (
            (self.trades['size'] > 0.8 * trade_size) &
            (self.trades['size'] < 1.2 * trade_size)
        )
        similar_trades = self.trades[size_filter]
        
        if similar_trades.empty:
            return self.calculate_square_root_impact(trade_size, 0.1)
            
        # Calculate average price impact
        vwap_impacts = (similar_trades['price'] / vwap - 1) * 10000  # Convert to bps
        return vwap_impacts.mean()
        
    def calculate_permanent_impact(self, window: str = '1D') -> pd.Series:
        """
        Calculate permanent price impact of trades.
        
        Args:
            window: Time window for impact calculation
            
        Returns:
            pd.Series: Permanent impact estimates
        """
        # Calculate cumulative price changes
        price_changes = self.trades['price'].pct_change()
        
        # Calculate cumulative volume
        cum_volume = self.trades['size'].cumsum()
        
        # Estimate permanent impact component
        impact = (
            price_changes.rolling(window=window)
            .mean()
            .shift(-1)  # Look-forward impact
        )
        
        return impact * 10000  # Convert to bps
        
    def calculate_toxicity_metrics(self) -> Dict[str, float]:
        """
        Calculate order flow toxicity metrics.
        
        Returns:
            Dict with toxicity metrics:
            - VPIN (Volume-synchronized Probability of Informed Trading)
            - Order flow imbalance
            - Trade size distribution metrics
        """
        metrics = {}
        
        # Calculate order flow imbalance
        if 'direction' in self.trades.columns:
            buy_volume = self.trades[self.trades['direction'] > 0]['size'].sum()
            sell_volume = self.trades[self.trades['direction'] < 0]['size'].sum()
            total_volume = buy_volume + sell_volume
            
            metrics['order_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
            
        # Calculate trade size distribution metrics
        metrics['size_mean'] = self.trades['size'].mean()
        metrics['size_std'] = self.trades['size'].std()
        metrics['size_skew'] = self.trades['size'].skew()
        metrics['large_trade_ratio'] = (
            (self.trades['size'] > self.avg_trade_size * 2)
            .mean()
        )
        
        # Calculate VPIN if possible
        if 'direction' in self.trades.columns:
            metrics['vpin'] = self._calculate_vpin()
            
        return metrics
        
    def _calculate_vpin(self, n_buckets: int = 50) -> float:
        """Calculate Volume-synchronized Probability of Informed Trading."""
        # Sort trades by time and calculate volume buckets
        bucket_volume = self.trades['size'].sum() / n_buckets
        
        current_bucket = []
        bucket_imbalances = []
        current_volume = 0
        
        for _, trade in self.trades.iterrows():
            current_bucket.append(trade)
            current_volume += trade['size']
            
            if current_volume >= bucket_volume:
                # Calculate bucket imbalance
                bucket_df = pd.DataFrame(current_bucket)
                buy_volume = bucket_df[bucket_df['direction'] > 0]['size'].sum()
                sell_volume = bucket_df[bucket_df['direction'] < 0]['size'].sum()
                
                imbalance = abs(buy_volume - sell_volume) / (buy_volume + sell_volume)
                bucket_imbalances.append(imbalance)
                
                # Reset bucket
                current_bucket = []
                current_volume = 0
                
        # VPIN is average of bucket imbalances
        return np.mean(bucket_imbalances) if bucket_imbalances else 0.0