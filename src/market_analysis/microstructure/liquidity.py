# src/market_analysis/microstructure/liquidity.py
"""
Liquidity analysis module.

This module implements tools for analyzing market liquidity:
- Kyle's lambda (price impact)
- Amihud's illiquidity ratio
- Bid-ask bounce effects
- Pastor-Stambaugh liquidity factor
- Volume-based liquidity measures
"""
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from typing import Dict, Optional, Union, List
import pandas as pd
import numpy as np
from dataclasses import dataclass
import statsmodels.api as sm


@dataclass
class LiquidityAnalyzer:
    """
    Liquidity analysis tools.

    Implements various market liquidity measures and analysis tools for measuring
    market quality and trading costs.

    Attributes:
        trades (pd.DataFrame): Trade data with timestamp index
        quotes (pd.DataFrame, optional): Quote data with timestamp index
        market_data (pd.DataFrame, optional): Market data with timestamp index
    """

    def __init__(self,
                 trades: pd.DataFrame,
                 quotes: Optional[pd.DataFrame] = None,
                 market_data: Optional[pd.DataFrame] = None):
        """
        Initialize with trade and quote data.

        Args:
            trades: DataFrame containing trade data
            quotes: Optional DataFrame containing quote data
            market_data: Optional DataFrame containing market benchmark data
        """
        self.trades = self._ensure_datetime_index(trades)
        self.quotes = self._ensure_datetime_index(quotes) if quotes is not None else None
        self.market_data = self._ensure_datetime_index(market_data) if market_data is not None else None
        self._validate_data()

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has datetime index.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with datetime index

        Raises:
            ValueError: If no timestamp information is available
        """
        if df is None:
            return None

        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
                df.index = pd.to_datetime(df.index)
            else:
                raise ValueError("Data must have datetime index or timestamp column")
        return df.sort_index()

    def _validate_data(self) -> None:
        """
        Validate input data requirements.

        Raises:
            ValueError: If required columns are missing
        """
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

        Implements Kyle's measure of price impact:
        λ = cov(ΔP, V) / var(V)

        Args:
            window: Time window for calculation
            min_observations: Minimum number of observations required

        Returns:
            Series of lambda values indexed by time
        """
        price_changes = self.trades['price'].diff().abs()  # Use absolute price changes
        signed_volume = self.trades['volume']

        def estimate_lambda(group):
            if len(group) < min_observations:
                return np.nan

            prices = group['price_change'].abs().values  # Use absolute price changes
            volumes = group['volume'].values

            mask = ~np.isnan(prices) & ~np.isnan(volumes)
            if sum(mask) < min_observations:
                return np.nan

            prices = prices[mask]
            volumes = volumes[mask]

            cov_matrix = np.cov(prices, volumes)
            volume_var = np.var(volumes)

            if volume_var == 0:
                return np.nan

            lambda_est = abs(cov_matrix[0, 1] / volume_var)  # Take absolute value
            return lambda_est


        analysis_df = pd.DataFrame({
            'price_change': price_changes,
            'volume': signed_volume
        }, index=self.trades.index)

        return analysis_df.resample(window).apply(estimate_lambda)

    def calculate_amihud_ratio(self, window: str = '1D',
                             scaling: float = 1e6) -> pd.Series:
        """
        Calculate Amihud's illiquidity ratio.

        Measures price impact per unit of volume:
        ILLIQ = |R| / (P * V)

        Args:
            window: Time window for calculation
            scaling: Volume scaling factor

        Returns:
            Series of Amihud ratios indexed by time
        """
        returns = self.trades['price'].pct_change().abs()
        dollar_volume = self.trades['price'] * self.trades['volume']

        # Resample using datetime index
        amihud = (returns / (dollar_volume / scaling)).replace([np.inf, -np.inf], np.nan)
        return amihud.resample(window).mean()

    def calculate_bid_ask_bounce(self, window: str = '1D') -> pd.Series:
        """
        Calculate bid-ask bounce effect.

        Estimates the component of price changes due to bouncing
        between bid and ask prices.

        Args:
            window: Time window for calculation

        Returns:
            Series of bounce effects indexed by time

        Raises:
            ValueError: If quote data is not available
        """
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

    def calculate_pastor_stambaugh(self, window: str = '1M',
                                 min_observations: int = 20) -> pd.Series:
        """
        Calculate Pastor-Stambaugh liquidity factor.

        Measures return reversal associated with volume:
        r(t+1) = θ + φr(t) + γ(sign(r(t)) * v(t)) + ε(t+1)

        Args:
            window: Time window for calculation
            min_observations: Minimum number of observations required

        Returns:
            Series of liquidity factors indexed by time

        Raises:
            ValueError: If market data is not available
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for Pastor-Stambaugh calculation")

        if self.market_data is None:
                return pd.Series(dtype=float)  # Return empty series instead of None

        # Calculate signed volume
        signed_volume = self.trades['volume'] * np.sign(self.trades['price'].diff())

        # Calculate excess returns
        excess_returns = (
            self.trades['price'].pct_change() -
            self.market_data['rf']
        )

    def calculate_gamma(group):
      """Calculate gamma coefficient for Pastor-Stambaugh measure."""
      if len(group) < min_observations:
        return np.nan

      try:
        # Clean data
        clean_data = group.dropna()
        if len(clean_data) < min_observations:
            return np.nan

        X = sm.add_constant(clean_data['signed_volume'].iloc[:-1])
        y = clean_data['excess_return'].iloc[1:]

        # Remove inf values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        if len(y) < min_observations:
            return np.nan

        model = sm.OLS(y, X).fit()
        return model.params[1]
      except Exception:
        return np.nan

        data = pd.DataFrame({
            'excess_return': excess_returns,
            'signed_volume': signed_volume
        })

        return data.resample(window).apply(calculate_gamma)

    def calculate_liquidity_risk(self, window: str = '1M',
                               quantile: float = 0.95) -> pd.DataFrame:
        """
        Calculate liquidity risk metrics.

        Computes various risk measures for liquidity metrics:
        - Value at Risk (VaR)
        - Conditional VaR (CVaR)
        - Volatility

        Args:
            window: Time window for calculation
            quantile: Quantile for VaR calculation

        Returns:
            DataFrame containing risk metrics
        """
        metrics = {}

        # Calculate risk measures for Kyle's lambda
        kyle_lambda = self.calculate_kyle_lambda(window)
        metrics['kyle_lambda_var'] = kyle_lambda.quantile(quantile)
        metrics['kyle_lambda_cvar'] = kyle_lambda[kyle_lambda > metrics['kyle_lambda_var']].mean()
        metrics['kyle_lambda_vol'] = kyle_lambda.std()

        # Calculate risk measures for Amihud ratio
        amihud = self.calculate_amihud_ratio(window)
        metrics['amihud_ratio_var'] = amihud.quantile(quantile)
        metrics['amihud_ratio_cvar'] = amihud[amihud > metrics['amihud_ratio_var']].mean()
        metrics['amihud_ratio_vol'] = amihud.std()

        return pd.DataFrame([metrics])

    def get_liquidity_metrics(self, window: str = '1D') -> pd.DataFrame:
        """
        Calculate all available liquidity metrics.

        Combines various liquidity measures into a single DataFrame.

        Args:
            window: Time window for calculation

        Returns:
            DataFrame containing all liquidity metrics
        """
        metrics = pd.DataFrame({
            'kyle_lambda': self.calculate_kyle_lambda(window=window),
            'amihud_ratio': self.calculate_amihud_ratio(window=window)
        })

        if self.quotes is not None:
            metrics['bid_ask_bounce'] = self.calculate_bid_ask_bounce(window=window)

        if HAS_STATSMODELS and self.market_data is not None:
            try:
                metrics['ps_liquidity'] = self.calculate_pastor_stambaugh(window=window)
            except Exception as e:
                print(f"Warning: Could not calculate Pastor-Stambaugh measure: {e}")

        return metrics
