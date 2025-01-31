# src/market_analysis/microstructure/liquidity_measures.py
"""
Extended liquidity measures module.

This module implements additional market microstructure measures:
- Hasbrouck's information share
- Roll's implied spread
- LOT (Lesmond-Ogden-Trzcinka) measure
- Huang-Stoll decomposition
"""

from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class AdvancedLiquidityMeasures:
    """
    Advanced market microstructure liquidity measures.

    Implements sophisticated measures focusing on:
    - Information content of trades
    - Hidden transaction costs
    - Price discovery processes
    - Spread decomposition
    """

    def __init__(self,
                 trades: pd.DataFrame,
                 quotes: Optional[pd.DataFrame] = None,
                 benchmark: Optional[pd.DataFrame] = None):
        """
        Initialize with trade, quote and benchmark data.

        Args:
            trades: Trade data (price, volume, etc.)
            quotes: Quote data (bid, ask)
            benchmark: Benchmark price data for information share
        """
        self.trades = trades
        self.quotes = quotes
        self.benchmark = benchmark
        self._validate_data()

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has datetime index."""
        if df is None:
            return None

        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def _validate_data(self) -> None:
        """Validate input data requirements."""
        required_cols = {'timestamp', 'price', 'volume'}
        if not required_cols.issubset(self.trades.columns):
            raise ValueError(f"Trades must contain columns: {required_cols}")

    def calculate_roll_spread(self,
                            window: str = '1D',
                            min_observations: int = 30) -> pd.Series:
        """
        Calculate Roll's implied spread measure.

        S = 2 * sqrt(-cov(ΔPt, ΔPt-1))
        where ΔP is price change.

        Args:
            window: Estimation window
            min_observations: Minimum points for estimation

        Returns:
            pd.Series: Roll's implied spread estimates
        """
        price_changes = self.trades['price'].diff()

        def estimate_roll(group):
            if len(group) < min_observations:
                return np.nan

            # Calculate first-order autocovariance
            cov = np.cov(group[:-1], group[1:])[0,1]

            # Only negative covariance is meaningful
            if cov >= 0:
                return np.nan

            return 2 * np.sqrt(-cov)

        # Calculate by window
        roll_spread = (
            price_changes.groupby(pd.Grouper(freq=window))
            .apply(estimate_roll)
        )

        return roll_spread

    def calculate_hasbrouck_measure(self,
                                  window: str = '1D',
                                  min_observations: int = 50) -> pd.DataFrame:
        """
        Calculate Hasbrouck's information share measure.

        Decomposes price changes into permanent (information)
        and temporary components using VEC model.

        Args:
            window: Estimation window
            min_observations: Minimum observations

        Returns:
            pd.DataFrame: Information shares and error correction terms
        """
        if self.benchmark is None:
            raise ValueError("Benchmark prices required for Hasbrouck measure")

        def estimate_vec(prices1, prices2):
            # First differences
            dp1 = np.diff(prices1)
            dp2 = np.diff(prices2)

            # Error correction term
            ect = prices1[:-1] - prices2[:-1]

            # Stack variables for regression
            X = np.column_stack([ect, dp1[:-1], dp2[:-1]])

            # Estimate VEC model for both prices
            y1 = dp1[1:]
            y2 = dp2[1:]

            try:
                beta1 = np.linalg.lstsq(X, y1, rcond=None)[0]
                beta2 = np.linalg.lstsq(X, y2, rcond=None)[0]

                # Calculate information share
                residuals = y1 - X @ beta1
                variance = np.var(residuals)
                info_share = beta1[0]**2 * variance

                return pd.Series({
                    'info_share': info_share,
                    'error_correction': beta1[0],
                    'variance_ratio': variance / np.var(y1)
                })
            except:
                return pd.Series({
                    'info_share': np.nan,
                    'error_correction': np.nan,
                    'variance_ratio': np.nan
                })

        # Align trade and benchmark prices
        combined = pd.merge_asof(
            self.trades[['timestamp', 'price']],
            self.benchmark,
            on='timestamp',
            direction='nearest'
        )

        # Calculate by window
        results = (
            combined.groupby(pd.Grouper(freq=window))
            .apply(lambda g: estimate_vec(g['price'], g['benchmark']))
            if len(combined) >= min_observations else pd.DataFrame()
        )

        return results

    def calculate_lot_measure(self,
                            window: str = '1D',
                            threshold: float = 0.01) -> pd.DataFrame:
        """
        Calculate LOT (Lesmond-Ogden-Trzcinka) measure of implicit trading costs.

        Estimates implicit costs by identifying zero-return days
        as those where trading costs exceed benefits.

        Args:
            window: Analysis window
            threshold: Threshold for zero returns

        Returns:
            pd.DataFrame: LOT measures and proportion of zero returns
        """
        returns = self.trades['price'].pct_change()

        def estimate_lot(group):
            if len(group) < 2:
                return pd.Series({
                    'lot_measure': np.nan,
                    'zero_prop': np.nan
                })

            # Identify zero returns
            zero_returns = abs(group) < threshold
            zero_prop = zero_returns.mean()

            # Estimate implied costs
            non_zero = group[~zero_returns]
            if len(non_zero) < 2:
                return pd.Series({
                    'lot_measure': np.nan,
                    'zero_prop': zero_prop
                })

            # Use range of non-zero returns as cost estimate
            lot = non_zero.max() - non_zero.min()

            return pd.Series({
                'lot_measure': lot,
                'zero_prop': zero_prop
            })

        # Calculate by window
        results = (
            returns.groupby(pd.Grouper(freq=window))
            .apply(estimate_lot)
        )

        return results

    def calculate_huang_stoll(self,
                            window: str = '1D',
                            min_observations: int = 50) -> pd.DataFrame:
        """
        Calculate Huang-Stoll spread decomposition.

        Decomposes spread into:
        - Order processing costs
        - Inventory holding costs
        - Adverse selection costs

        Args:
            window: Analysis window
            min_observations: Minimum observations

        Returns:
            pd.DataFrame: Spread components
        """
        if self.quotes is None:
            raise ValueError("Quote data required for Huang-Stoll decomposition")

        def decompose_spread(group):
            if len(group) < min_observations:
                return pd.Series({
                    'order_processing': np.nan,
                    'inventory': np.nan,
                    'adverse_selection': np.nan
                })

            # Calculate effective spread
            midpoint = (group['bid'] + group['ask']) / 2
            eff_spread = 2 * abs(group['price'] - midpoint)

            # Calculate trade direction
            direction = np.sign(group['price'] - midpoint)

            # First-stage regression for inventory component
            X1 = pd.DataFrame({
                'direction': direction,
                'lag_direction': direction.shift(1)
            }).dropna()

            y1 = group['price'].diff().dropna()

            try:
                beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                inventory = beta1[1] / eff_spread.mean()

                # Second-stage for adverse selection
                X2 = pd.DataFrame({
                    'direction': direction,
                    'expected_inv': beta1[1] * direction.shift(1)
                }).dropna()

                y2 = group['price'].diff().dropna()

                beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                adverse = beta2[0] / eff_spread.mean()

                # Order processing is remainder
                order_processing = 1 - inventory - adverse

                return pd.Series({
                    'order_processing': order_processing,
                    'inventory': inventory,
                    'adverse_selection': adverse
                })
            except:
                return pd.Series({
                    'order_processing': np.nan,
                    'inventory': np.nan,
                    'adverse_selection': np.nan
                })

        # Combine trade and quote data
        combined = pd.merge_asof(
            self.trades[['timestamp', 'price']],
            self.quotes[['timestamp', 'bid', 'ask']],
            on='timestamp',
            direction='nearest'
        )

        # Calculate by window
        results = (
            combined.groupby(pd.Grouper(freq=window))
            .apply(decompose_spread)
        )

        return results

    def get_all_measures(self, window: str = '1D') -> pd.DataFrame:
        """
        Calculate all available liquidity measures.

        Args:
            window: Analysis window

        Returns:
            pd.DataFrame: All liquidity measures
        """
        measures = pd.DataFrame()

        # Roll's spread
        measures['roll_spread'] = self.calculate_roll_spread(window=window)

        # LOT measure
        lot_results = self.calculate_lot_measure(window=window)
        measures['lot_measure'] = lot_results['lot_measure']
        measures['zero_returns'] = lot_results['zero_prop']

        # Add Hasbrouck if benchmark available
        if self.benchmark is not None:
            hasbrouck = self.calculate_hasbrouck_measure(window=window)
            measures['info_share'] = hasbrouck['info_share']
            measures['error_correction'] = hasbrouck['error_correction']

        # Add Huang-Stoll if quotes available
        if self.quotes is not None:
            hs = self.calculate_huang_stoll(window=window)
            measures['order_processing'] = hs['order_processing']
            measures['inventory'] = hs['inventory']
            measures['adverse_selection'] = hs['adverse_selection']

        return measures
