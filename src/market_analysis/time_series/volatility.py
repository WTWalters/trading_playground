"""
Volatility analysis module with advanced estimation methods.

This module implements various volatility estimation techniques:
- Historical volatility with different estimators
- EWMA (Exponentially Weighted Moving Average)
- Parkinson volatility (high-low based)
- Garman-Klass volatility (OHLC based)
- Realized volatility (high-frequency based)
"""

from typing import Optional, Union, Dict, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from .returns import ReturnCalculator


@dataclass
class VolatilityAnalyzer:
    """
    Comprehensive volatility analysis for financial time series.
    
    This class implements various volatility estimation methods with a focus on:
    - Multiple estimation techniques
    - Adaptive window selection
    - Volatility forecasting
    - Market regime detection
    
    Attributes:
        prices (pd.Series): Price series with datetime index
        returns_calc (ReturnCalculator): Return calculator instance
    """
    
    prices: pd.Series
    
    def __post_init__(self):
        """Initialize return calculator and validate data."""
        self.returns_calc = ReturnCalculator(self.prices)
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate input data requirements."""
        if not isinstance(self.prices, pd.Series):
            raise TypeError("Prices must be a pandas Series")
            
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("Prices must have a datetime index")
    
    def historical_volatility(self, window: int = 252, 
                            use_log: bool = True) -> pd.Series:
        """
        Calculate historical volatility using rolling standard deviation.
        
        Args:
            window: Rolling window size (default 252 days)
            use_log: Use log returns (True) or simple returns (False)
            
        Returns:
            pd.Series: Annualized volatility series
        """
        returns = self.returns_calc.log_returns() if use_log else self.returns_calc.simple_returns()
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        return vol
    
    def ewma_volatility(self, lambda_: float = 0.94, 
                       use_log: bool = True) -> pd.Series:
        """
        Calculate EWMA volatility (RiskMetrics methodology).
        
        Args:
            lambda_: Decay factor (default 0.94 from RiskMetrics)
            use_log: Use log returns (True) or simple returns (False)
            
        Returns:
            pd.Series: Annualized EWMA volatility series
        """
        returns = self.returns_calc.log_returns() if use_log else self.returns_calc.simple_returns()
        squared_returns = returns ** 2
        vol = np.sqrt(squared_returns.ewm(alpha=(1-lambda_)).mean()) * np.sqrt(252)
        return vol
    
    def parkinson_volatility(self, high: pd.Series, low: pd.Series, 
                           window: int = 252) -> pd.Series:
        """
        Calculate Parkinson volatility using high-low prices.
        
        Parkinson volatility is more efficient than close-to-close volatility
        as it captures intraday price movements.
        
        Args:
            high: High price series
            low: Low price series
            window: Rolling window size (default 252 days)
            
        Returns:
            pd.Series: Annualized Parkinson volatility series
        """
        if not (high.index.equals(low.index) and high.index.equals(self.prices.index)):
            raise ValueError("High, low, and close prices must share the same index")
            
        hl_ratio = np.log(high / low)
        vol = np.sqrt(hl_ratio.pow(2).rolling(window=window).mean() / (4 * np.log(2)))
        return vol * np.sqrt(252)
    
    def garman_klass_volatility(self, high: pd.Series, low: pd.Series,
                              open_: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate Garman-Klass volatility using OHLC prices.
        
        This estimator is more efficient than Parkinson as it uses
        opening and closing prices in addition to high-low range.
        
        Args:
            high: High price series
            low: Low price series
            open_: Opening price series
            window: Rolling window size (default 252 days)
            
        Returns:
            pd.Series: Annualized Garman-Klass volatility series
        """
        if not all(s.index.equals(self.prices.index) for s in [high, low, open_]):
            raise ValueError("All price series must share the same index")
            
        log_hl = np.log(high / low)
        log_co = np.log(self.prices / open_)
        
        estimator = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)
        vol = np.sqrt(estimator.rolling(window=window).mean())
        return vol * np.sqrt(252)
    
    def realized_volatility(self, returns: pd.Series, 
                          sampling_freq: str = '5min',
                          window: str = '1D') -> pd.Series:
        """
        Calculate realized volatility from high-frequency returns.
        
        Args:
            returns: High-frequency returns series
            sampling_freq: Return sampling frequency
            window: Aggregation window
            
        Returns:
            pd.Series: Annualized realized volatility series
        """
        # Calculate squared returns
        squared_returns = returns ** 2
        
        # Resample to desired frequency and sum
        realized_var = squared_returns.resample(window).sum()
        vol = np.sqrt(realized_var) * np.sqrt(252)
        return vol
    
    def detect_volatility_regime(self, window: int = 252,
                               n_regimes: int = 2) -> pd.Series:
        """
        Detect volatility regimes using a simple threshold approach.
        
        Args:
            window: Rolling window for volatility calculation
            n_regimes: Number of volatility regimes to detect
            
        Returns:
            pd.Series: Regime classifications (0 = low vol, 1 = high vol)
        """
        vol = self.historical_volatility(window=window)
        
        if n_regimes == 2:
            # Simple threshold at median
            threshold = vol.median()
            regimes = (vol > threshold).astype(int)
        else:
            # Use quantiles for multiple regimes
            quantiles = np.linspace(0, 1, n_regimes + 1)
            thresholds = vol.quantile(quantiles)
            regimes = pd.qcut(vol, q=n_regimes, labels=False)
        
        return regimes
    
    def forecast_volatility(self, window: int = 252, 
                          horizon: int = 5,
                          method: str = 'ewma') -> Tuple[pd.Series, pd.Series]:
        """
        Forecast volatility using various methods.
        
        Args:
            window: Historical window for model fitting
            horizon: Forecast horizon in days
            method: Forecasting method ('ewma', 'garch', or 'simple')
            
        Returns:
            Tuple[pd.Series, pd.Series]: Point forecasts and forecast standard errors
        """
        if method == 'ewma':
            # Use EWMA for forecasting
            current_vol = self.ewma_volatility()
            # EWMA forecast is flat
            forecast = pd.Series(current_vol.iloc[-1], 
                               index=pd.date_range(start=current_vol.index[-1] + pd.Timedelta('1D'),
                                                 periods=horizon,
                                                 freq='D'))
            # Standard errors increase with horizon
            std_err = forecast * np.sqrt(np.arange(1, horizon + 1) / 252)
            
        elif method == 'simple':
            # Simple historical volatility projection
            current_vol = self.historical_volatility(window=window)
            forecast = pd.Series(current_vol.iloc[-1], 
                               index=pd.date_range(start=current_vol.index[-1] + pd.Timedelta('1D'),
                                                 periods=horizon,
                                                 freq='D'))
            # Use historical standard deviation of volatility for errors
            vol_std = current_vol.std()
            std_err = pd.Series(vol_std, index=forecast.index)
            
        else:
            raise ValueError(f"Unsupported forecasting method: {method}")
            
        return forecast, std_err
    
    def get_volatility_statistics(self, window: int = 252) -> pd.Series:
        """
        Calculate comprehensive volatility statistics.
        
        Args:
            window: Rolling window for calculations
            
        Returns:
            pd.Series: Series containing key volatility statistics
        """
        vol = self.historical_volatility(window=window)
        
        stats = pd.Series({
            'current_vol': vol.iloc[-1],
            'vol_mean': vol.mean(),
            'vol_median': vol.median(),
            'vol_std': vol.std(),
            'vol_skew': vol.skew(),
            'vol_kurt': vol.kurtosis(),
            'vol_min': vol.min(),
            'vol_max': vol.max(),
            'vol_95th': vol.quantile(0.95),
            'vol_5th': vol.quantile(0.05)
        })
        
        return stats