# src/market_analysis/microstructure/impact.py
"""
Market impact and order flow analysis module.

This module implements tools for analyzing:
- Price impact models (square root, linear, decay)
- Order flow toxicity metrics
- Trade size analysis
- Permanent vs temporary impact assessments
- Volume profile analysis
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


class MarketImpact:
    """
    Market impact analysis and modeling.

    Implements various market impact models and analysis tools:
    - Square root impact model (Almgren et al.)
    - Linear temporary impact
    - Decay model for permanent impact
    - Volume-weighted impact analysis
    - Order flow toxicity metrics

    Attributes:
        trades (pd.DataFrame): Trade data with columns [timestamp, price, volume, direction]
        quotes (pd.DataFrame): Optional quote data with columns [timestamp, bid, ask]
        daily_volume (pd.Series): Daily trading volume
        daily_volatility (pd.Series): Daily price volatility
        avg_trade_size (float): Average trade size
        avg_spread (float): Average bid-ask spread (if quotes available)
    """

    def __init__(self, trades: pd.DataFrame, quotes: Optional[pd.DataFrame] = None):
        """
        Initialize market impact analyzer.

        Args:
            trades: DataFrame with trade data containing required columns
            quotes: Optional DataFrame with quote data for spread calculations

        Raises:
            ValueError: If required columns are missing from trade or quote data
        """
        # Copy and prepare data
        self.trades = trades.copy()
        if 'timestamp' in self.trades.columns:
            self.trades.set_index('timestamp', inplace=True)
        self.trades.index = pd.to_datetime(self.trades.index)

        if quotes is not None:
            self.quotes = quotes.copy()
            if 'timestamp' in self.quotes.columns:
                self.quotes.set_index('timestamp', inplace=True)
            self.quotes.index = pd.to_datetime(self.quotes.index)
        else:
            self.quotes = None

        self._validate_data()
        self._calculate_base_metrics()

    def _validate_data(self) -> None:
        """
        Validate input data requirements.

        Checks for required columns and converts 'size' to 'volume' if needed.

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {'price', 'volume', 'timestamp'}
        if 'size' in self.trades.columns and 'volume' not in self.trades.columns:
            self.trades['volume'] = self.trades['size']

        if not required_cols.issubset(set(self.trades.columns) | {'timestamp'}):
            raise ValueError(f"Trades must contain columns: {required_cols}")

        if self.quotes is not None:
            quote_cols = {'bid', 'ask', 'timestamp'}
            if not quote_cols.issubset(set(self.quotes.columns) | {'timestamp'}):
                raise ValueError(f"Quotes must contain columns: {quote_cols}")

    def _calculate_base_metrics(self) -> None:
        """
        Calculate base metrics for impact modeling.

        Computes:
        - Daily trading volume
        - Daily price volatility
        - Average trade size
        - Average spread (if quotes available)
        """
        # Daily volume and volatility
        self.daily_volume = self.trades.groupby(
            self.trades.index.date
        )['volume'].sum()

        self.daily_volatility = self.trades.groupby(
            self.trades.index.date
        )['price'].agg(lambda x: np.log(x / x.shift(1)).std())

        # Average trade size and spread
        self.avg_trade_size = self.trades['volume'].mean()
        if self.quotes is not None:
            self.avg_spread = (self.quotes['ask'] - self.quotes['bid']).mean()
        else:
            self.avg_spread = None

    def calculate_permanent_impact(self, window: str = '1D') -> pd.Series:
        """
        Calculate permanent price impact of trades.

        Args:
            window: Time window for calculation

        Returns:
            pd.Series: Permanent impact estimates in basis points
        """
        price_changes = self.trades['price'].pct_change()
        volume_signed = self.trades['volume'] * np.sign(price_changes)

        def estimate_permanent(group):
            if len(group) < 2:
                return np.nan
            return (group['price_change'] * group['volume_signed']).mean()

        analysis_df = pd.DataFrame({
            'price_change': price_changes,
            'volume_signed': volume_signed
        }, index=self.trades.index)

        return analysis_df.resample(window).apply(estimate_permanent) * 10000

    def calculate_square_root_impact(self, trade_size: float,
                                   participation_rate: float) -> float:
        """
        Calculate market impact using square root model.

        Implements Almgren's square root impact model:
        Impact = σ * sign(Q) * sqrt(|Q|/V) * (1 + participation_rate)

        Args:
            trade_size: Size of trade to analyze
            participation_rate: Target participation rate (0-1)

        Returns:
            float: Estimated price impact in basis points

        Raises:
            ValueError: If participation rate is invalid
        """
        if participation_rate <= 0 or participation_rate > 1:
            raise ValueError("Participation rate must be between 0 and 1")

        daily_vol = self.daily_volatility.mean()
        daily_vol_bps = daily_vol * 10000  # Convert to basis points
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

        Uses simple linear model: Impact = λ * Q
        where λ is the price impact coefficient

        Args:
            trade_size: Size of trade to analyze

        Returns:
            float: Estimated price impact in basis points
        """
        impact_coefficient = self._estimate_impact_coefficient()
        return impact_coefficient * trade_size * 10000

    def _estimate_impact_coefficient(self) -> float:
        """
        Estimate price impact coefficient from historical data.

        Uses regression of price changes against normalized trade sizes.

        Returns:
            float: Estimated impact coefficient (lambda)
        """
        # Calculate absolute price changes
        price_changes = self.trades['price'].pct_change().abs()

        # Normalize trade sizes
        normalized_sizes = self.trades['volume'] / self.avg_trade_size

        # Regression to estimate coefficient
        impact_coef = (
            (price_changes * normalized_sizes).sum() /
            (normalized_sizes ** 2).sum()
        )

        return max(impact_coef, 1e-8)  # Ensure positive coefficient

    def calculate_decay_impact(self, trade_size: float,
                             horizon: int = 100) -> pd.Series:
        """
        Calculate decaying price impact over time.

        Models impact decay using exponential function.

        Args:
            trade_size: Size of trade
            horizon: Number of periods for decay calculation

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

        Compares trade prices to volume-weighted average prices.

        Args:
            trade_size: Size of trade
            interval: Time interval for VWAP calculation

        Returns:
            float: Estimated VWAP impact in basis points
        """
        try:
            vwap = (
                (self.trades['price'] * self.trades['volume']).resample(interval).sum() /
                self.trades['volume'].resample(interval).sum()
            ).fillna(method='ffill')

            size_filter = (
                (self.trades['volume'] > 0.8 * trade_size) &
                (self.trades['volume'] < 1.2 * trade_size)
            )
            similar_trades = self.trades[size_filter]

            if similar_trades.empty:
                return self.calculate_square_root_impact(trade_size, 0.1)

            vwap_impacts = (similar_trades['price'] / vwap - 1).abs() * 10000
            return vwap_impacts.mean()
        except Exception:
            return self.calculate_square_root_impact(trade_size, 0.1)

    def calculate_toxicity_metrics(self) -> Dict[str, float]:
        """
        Calculate order flow toxicity metrics.

        Computes various metrics including:
        - Order flow imbalance
        - VPIN (Volume-synchronized Probability of Informed Trading)
        - Trade size distribution metrics

        Returns:
            Dict containing:
                - order_flow_imbalance: Float (-1 to 1)
                - size_mean: Float
                - size_std: Float
                - size_skew: Float
                - large_trade_ratio: Float (0 to 1)
                - vpin: Float (0 to 1) if direction available
        """
        metrics = {}

        # Calculate order flow imbalance
        if 'direction' in self.trades.columns:
            buy_volume = self.trades[self.trades['direction'] > 0]['volume'].sum()
            sell_volume = self.trades[self.trades['direction'] < 0]['volume'].sum()
            total_volume = buy_volume + sell_volume

            metrics['order_flow_imbalance'] = (buy_volume - sell_volume) / total_volume

        # Calculate trade size distribution metrics
        metrics['size_mean'] = self.trades['volume'].mean()
        metrics['size_std'] = self.trades['volume'].std()
        metrics['size_skew'] = self.trades['volume'].skew()
        metrics['large_trade_ratio'] = (
            (self.trades['volume'] > self.avg_trade_size * 2)
            .mean()
        )

        # Calculate VPIN if possible
        if 'direction' in self.trades.columns:
            metrics['vpin'] = self._calculate_vpin()

        return metrics

    def _calculate_vpin(self, n_buckets: int = 50) -> float:
        """
        Calculate Volume-synchronized Probability of Informed Trading.

        Args:
            n_buckets: Number of volume buckets for calculation

        Returns:
            float: VPIN estimate between 0 and 1
        """
        bucket_volume = self.trades['volume'].sum() / n_buckets

        current_bucket = []
        bucket_imbalances = []
        current_volume = 0

        for _, trade in self.trades.iterrows():
            current_bucket.append(trade)
            current_volume += trade['volume']

            if current_volume >= bucket_volume:
                # Calculate bucket imbalance
                bucket_df = pd.DataFrame(current_bucket)
                buy_volume = bucket_df[bucket_df['direction'] > 0]['volume'].sum()
                sell_volume = bucket_df[bucket_df['direction'] < 0]['volume'].sum()

                imbalance = abs(buy_volume - sell_volume) / (buy_volume + sell_volume)
                bucket_imbalances.append(imbalance)

                # Reset bucket
                current_bucket = []
                current_volume = 0

        return np.mean(bucket_imbalances) if bucket_imbalances else 0.0
