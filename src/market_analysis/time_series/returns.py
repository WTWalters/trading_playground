"""
Financial returns calculation module with advanced statistical properties.

This module provides comprehensive tools for calculating and analyzing financial returns
with a focus on statistical accuracy and computational efficiency.
"""

from typing import Optional, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ReturnCalculator:
    """
    Foundation class for financial return calculations.
    
    This class implements various return calculation methods with a focus on:
    - Numerical stability
    - Statistical accuracy
    - Performance optimization
    - Data validation
    
    Attributes:
        prices (pd.Series): Price series with datetime index
    """
    
    prices: pd.Series
    
    def __post_init__(self) -> None:
        """Validate input data on initialization."""
        self._validate_prices()
    
    def _validate_prices(self) -> None:
        """
        Validate price data requirements:
        - Must be pandas Series
        - Must have datetime index
        - Must contain numeric values
        - Must not contain negative values
        - Must not contain NaN values at the start/end
        - Must be sorted by index
        """
        if not isinstance(self.prices, pd.Series):
            raise TypeError("Prices must be a pandas Series")
            
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("Prices must have a datetime index")
            
        if not np.issubdtype(self.prices.dtype, np.number):
            raise TypeError("Prices must contain numeric values")
            
        if (self.prices <= 0).any():
            raise ValueError("Prices must be positive")
            
        # Check for NaN values at start/end
        if pd.isna(self.prices.iloc[0]) or pd.isna(self.prices.iloc[-1]):
            raise ValueError("Prices cannot have NaN values at start or end")
            
        # Ensure index is sorted
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("Price series must be sorted by datetime index")
    
    def simple_returns(self, dropna: bool = True) -> pd.Series:
        """
        Calculate simple returns: (Pt - Pt-1) / Pt-1
        
        Simple returns are more intuitive and useful for:
        - Short time periods
        - Performance reporting
        - Portfolio returns aggregation
        
        Args:
            dropna: Whether to drop NA values from the result
            
        Returns:
            pd.Series: Simple returns series
        """
        returns = self.prices.pct_change()
        return returns.dropna() if dropna else returns
    
    def log_returns(self, dropna: bool = True) -> pd.Series:
        """
        Calculate logarithmic returns: ln(Pt / Pt-1)
        
        Log returns are more suitable for:
        - Statistical analysis (closer to normal distribution)
        - Time aggregation (additive property)
        - Risk management calculations
        
        Args:
            dropna: Whether to drop NA values from the result
            
        Returns:
            pd.Series: Logarithmic returns series
        """
        returns = np.log(self.prices / self.prices.shift(1))
        return returns.dropna() if dropna else returns
    
    def excess_returns(self, risk_free_rate: Union[float, pd.Series], 
                      use_log: bool = True, dropna: bool = True) -> pd.Series:
        """
        Calculate excess returns over risk-free rate.
        
        Args:
            risk_free_rate: Risk-free rate (annualized) as float or time series
            use_log: Whether to use log returns (True) or simple returns (False)
            dropna: Whether to drop NA values from the result
            
        Returns:
            pd.Series: Excess returns series
        """
        # Convert annual risk-free rate to match return frequency
        if isinstance(risk_free_rate, float):
            freq = pd.infer_freq(self.prices.index)
            periods_per_year = pd.Timedelta('365D') / pd.Timedelta(freq)
            rf_rate = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
            if use_log:
                rf_rate = np.log1p(rf_rate)
        else:
            rf_rate = risk_free_rate
            
        returns = self.log_returns(dropna=False) if use_log else self.simple_returns(dropna=False)
        excess = returns - rf_rate
        return excess.dropna() if dropna else excess
    
    def rolling_returns(self, window: Union[str, int], 
                       use_log: bool = True, dropna: bool = True) -> pd.Series:
        """
        Calculate rolling returns over specified window.
        
        Args:
            window: Rolling window size as integer (periods) or string (e.g., '30D')
            use_log: Whether to use log returns (True) or simple returns (False)
            dropna: Whether to drop NA values from the result
            
        Returns:
            pd.Series: Rolling returns series
        """
        if isinstance(window, str):
            # For time-based windows, resample prices first
            resampled = self.prices.resample(window).last()
            calc = ReturnCalculator(resampled)
            returns = calc.log_returns(dropna=False) if use_log else calc.simple_returns(dropna=False)
        else:
            # For period-based windows, use rolling calculation
            returns = self.log_returns(dropna=False) if use_log else self.simple_returns(dropna=False)
            returns = returns.rolling(window=window).sum()
            
        return returns.dropna() if dropna else returns
    
    def risk_adjusted_returns(self, window: int = 252, 
                            use_log: bool = True, dropna: bool = True) -> pd.Series:
        """
        Calculate volatility-adjusted returns (returns divided by rolling volatility).
        
        Args:
            window: Rolling window for volatility calculation (default 252 days)
            use_log: Whether to use log returns (True) or simple returns (False)
            dropna: Whether to drop NA values from the result
            
        Returns:
            pd.Series: Risk-adjusted returns series
        """
        returns = self.log_returns(dropna=False) if use_log else self.simple_returns(dropna=False)
        rolling_vol = returns.rolling(window=window).std()
        adj_returns = returns / rolling_vol
        return adj_returns.dropna() if dropna else adj_returns

    def get_return_statistics(self, use_log: bool = True) -> pd.Series:
        """
        Calculate comprehensive return statistics.
        
        Args:
            use_log: Whether to use log returns (True) or simple returns (False)
            
        Returns:
            pd.Series: Series containing key return statistics
        """
        returns = self.log_returns() if use_log else self.simple_returns()
        
        yearly_factor = np.sqrt(252)  # Annualization factor assuming daily data
        
        stats = pd.Series({
            'mean': returns.mean(),
            'std': returns.std(),
            'annualized_mean': returns.mean() * 252,  # Assuming daily data
            'annualized_std': returns.std() * yearly_factor,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min(),
            'max': returns.max(),
            'positive_returns': (returns > 0).mean(),
            'negative_returns': (returns < 0).mean(),
        })
        
        return stats