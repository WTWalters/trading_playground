"""
Tests for volatility analysis module.

These tests validate various volatility estimation methods and ensure proper
handling of different market regimes and conditions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_analysis.time_series.volatility import VolatilityAnalyzer


@pytest.fixture
def sample_price_data():
    """Generate sample price data with known volatility characteristics."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate prices with known annual volatility of 20%
    annual_vol = 0.20
    daily_vol = annual_vol / np.sqrt(252)
    returns = np.random.normal(0, daily_vol, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_ohlc_data(sample_price_data):
    """Generate sample OHLC data with realistic properties."""
    dates = sample_price_data.index
    close_prices = sample_price_data
    
    # Generate realistic OHLC data
    np.random.seed(43)
    daily_vol = 0.02  # 2% daily volatility
    
    high_prices = close_prices * np.exp(np.abs(np.random.normal(0, daily_vol, len(dates))))
    low_prices = close_prices * np.exp(-np.abs(np.random.normal(0, daily_vol, len(dates))))
    open_prices = (high_prices + low_prices) / 2 + np.random.normal(0, daily_vol, len(dates))
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }, index=dates)


@pytest.fixture
def volatility_analyzer(sample_price_data):
    """Create VolatilityAnalyzer instance with sample data."""
    return VolatilityAnalyzer(sample_price_data)


@pytest.fixture
def high_frequency_data():
    """Generate sample high-frequency returns data."""
    np.random.seed(44)
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min')
    
    # Generate returns with intraday volatility pattern
    time_of_day = dates.hour + dates.minute / 60
    base_vol = 0.0001  # Base volatility per minute
    vol_pattern = 1 + 0.5 * np.sin(time_of_day * np.pi / 12)  # U-shaped pattern
    
    returns = np.random.normal(0, base_vol * vol_pattern, len(dates))
    return pd.Series(returns, index=dates)


def test_historical_volatility(volatility_analyzer):
    """Test historical volatility calculation."""
    vol = volatility_analyzer.historical_volatility(window=252)
    
    # Basic checks
    assert isinstance(vol, pd.Series)
    assert not vol.empty
    assert vol.index.equals(volatility_analyzer.prices.index)
    
    # Verify annualization
    daily_vol = volatility_analyzer.historical_volatility(window=252) / np.sqrt(252)
    assert 0 < daily_vol.mean() < 1  # Reasonable daily vol range
    
    # Test different windows
    short_vol = volatility_analyzer.historical_volatility(window=21)
    long_vol = volatility_analyzer.historical_volatility(window=252)
    assert short_vol.std() > long_vol.std()  # Short window should be more volatile


def test_ewma_volatility(volatility_analyzer):
    """Test EWMA volatility calculation."""
    vol = volatility_analyzer.ewma_volatility(lambda_=0.94)
    
    # Basic checks
    assert isinstance(vol, pd.Series)
    assert not vol.empty
    
    # Test different decay factors
    fast_decay = volatility_analyzer.ewma_volatility(lambda_=0.84)
    slow_decay = volatility_analyzer.ewma_volatility(lambda_=0.94)
    
    # Faster decay should respond more quickly to changes
    assert fast_decay.diff().std() > slow_decay.diff().std()


def test_parkinson_volatility(volatility_analyzer, sample_ohlc_data):
    """Test Parkinson volatility calculation."""
    vol = volatility_analyzer.parkinson_volatility(
        high=sample_ohlc_data['high'],
        low=sample_ohlc_data['low']
    )
    
    # Basic checks
    assert isinstance(vol, pd.Series)
    assert not vol.empty
    
    # Compare with historical volatility
    hist_vol = volatility_analyzer.historical_volatility()
    # Parkinson should be more efficient (lower standard error)
    assert vol.std() < hist_vol.std()


def test_garman_klass_volatility(volatility_analyzer, sample_ohlc_data):
    """Test Garman-Klass volatility calculation."""
    vol = volatility_analyzer.garman_klass_volatility(
        high=sample_ohlc_data['high'],
        low=sample_ohlc_data['low'],
        open_=sample_ohlc_data['open']
    )
    
    # Basic checks
    assert isinstance(vol, pd.Series)
    assert not vol.empty
    
    # Compare with Parkinson volatility
    park_vol = volatility_analyzer.parkinson_volatility(
        high=sample_ohlc_data['high'],
        low=sample_ohlc_data['low']
    )
    # Garman-Klass should be more efficient
    assert abs(vol.mean() - park_vol.mean()) < 0.1  # Similar levels
    assert vol.std() < park_vol.std()  # Lower standard error


def test_realized_volatility(volatility_analyzer, high_frequency_data):
    """Test realized volatility calculation."""
    vol = volatility_analyzer.realized_volatility(
        returns=high_frequency_data,
        sampling_freq='5min',
        window='1D'
    )
    
    # Basic checks
    assert isinstance(vol, pd.Series)
    assert not vol.empty
    assert vol.index.freq == pd.tseries.offsets.Day()
    
    # Test different sampling frequencies
    vol_5min = volatility_analyzer.realized_volatility(high_frequency_data, '5min')
    vol_1min = volatility_analyzer.realized_volatility(high_frequency_data, '1min')
    # Higher frequency should capture more volatility
    assert vol_1min.mean() > vol_5min.mean()


def test_volatility_regime_detection(volatility_analyzer):
    """Test volatility regime detection."""
    regimes = volatility_analyzer.detect_volatility_regime(window=63)
    
    # Basic checks
    assert isinstance(regimes, pd.Series)
    assert set(regimes.unique()) == {0, 1}  # Binary regime classification
    
    # Test multiple regimes
    multi_regimes = volatility_analyzer.detect_volatility_regime(n_regimes=3)
    assert len(multi_regimes.unique()) == 3
    
    # Test regime persistence
    regime_changes = (regimes != regimes.shift(1)).sum()
    assert regime_changes < len(regimes) * 0.5  # Regimes should be persistent


def test_volatility_forecasting(volatility_analyzer):
    """Test volatility forecasting."""
    forecast, std_err = volatility_analyzer.forecast_volatility(
        window=252,
        horizon=5,
        method='ewma'
    )
    
    # Basic checks
    assert isinstance(forecast, pd.Series)
    assert isinstance(std_err, pd.Series)
    assert len(forecast) == 5
    assert len(std_err) == 5
    
    # Test forecast properties
    assert (forecast > 0).all()  # Positive volatility
    assert (std_err > 0).all()  # Positive standard errors
    assert std_err.is_monotonic_increasing  # Uncertainty increases with horizon


def test_volatility_statistics(volatility_analyzer):
    """Test volatility statistics calculation."""
    stats = volatility_analyzer.get_volatility_statistics(window=252)
    
    # Check required statistics
    required_stats = [
        'current_vol', 'vol_mean', 'vol_median', 'vol_std',
        'vol_skew', 'vol_kurt', 'vol_min', 'vol_max',
        'vol_95th', 'vol_5th'
    ]
    for stat in required_stats:
        assert stat in stats.index
    
    # Test relationships
    assert stats['vol_min'] <= stats['vol_5th'] <= stats['vol_median'] <= stats['vol_95th'] <= stats['vol_max']
    assert stats['vol_mean'] > 0
    assert stats['vol_std'] > 0