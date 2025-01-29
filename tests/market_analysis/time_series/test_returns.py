"""
Tests for the ReturnCalculator class.

These tests verify the functionality of returns calculations,
ensuring numerical accuracy and proper handling of edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_analysis.time_series.returns import ReturnCalculator


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    # Generate random walk prices (always positive)
    np.random.seed(42)  # for reproducibility
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


@pytest.fixture
def return_calculator(sample_prices):
    """Create ReturnCalculator instance with sample data."""
    return ReturnCalculator(sample_prices)


def test_simple_returns_calculation(return_calculator, sample_prices):
    """Test simple returns calculation."""
    returns = return_calculator.simple_returns()
    
    # Manual calculation for verification
    expected = sample_prices.pct_change().dropna()
    pd.testing.assert_series_equal(returns, expected)
    
    # Verify first return is dropped (NaN handling)
    assert len(returns) == len(sample_prices) - 1
    
    # Verify basic properties
    assert isinstance(returns, pd.Series)
    assert returns.index.equals(sample_prices.index[1:])


def test_log_returns_calculation(return_calculator, sample_prices):
    """Test logarithmic returns calculation."""
    returns = return_calculator.log_returns()
    
    # Manual calculation for verification
    expected = np.log(sample_prices / sample_prices.shift(1)).dropna()
    pd.testing.assert_series_equal(returns, expected)
    
    # Verify properties of log returns
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_prices) - 1
    
    # Test additive property of log returns
    two_day_returns = return_calculator.rolling_returns(window=2, use_log=True)
    # Verify that 2-day log returns ≈ sum of daily log returns
    pd.testing.assert_series_equal(
        two_day_returns.iloc[1:],
        (returns + returns.shift(1)).dropna(),
        check_exact=False
    )


def test_excess_returns_calculation(return_calculator):
    """Test excess returns calculation with fixed risk-free rate."""
    risk_free_rate = 0.04  # 4% annual rate
    excess = return_calculator.excess_returns(risk_free_rate)
    
    # Get regular returns for comparison
    regular_returns = return_calculator.log_returns()
    
    # Daily risk-free rate (approximate)
    daily_rf = np.log1p((1 + risk_free_rate) ** (1/252) - 1)
    
    # Verify excess returns are returns minus risk-free rate
    expected = regular_returns - daily_rf
    pd.testing.assert_series_equal(excess, expected, check_exact=False)


def test_rolling_returns_calculation(return_calculator):
    """Test rolling returns calculation."""
    # Test with period-based window
    rolling = return_calculator.rolling_returns(window=5)
    
    # Test with time-based window
    rolling_time = return_calculator.rolling_returns(window='5D')
    
    # Basic checks
    assert isinstance(rolling, pd.Series)
    assert isinstance(rolling_time, pd.Series)
    assert len(rolling) > 0
    assert len(rolling_time) > 0


def test_risk_adjusted_returns_calculation(return_calculator):
    """Test risk-adjusted returns calculation."""
    risk_adj = return_calculator.risk_adjusted_returns(window=20)
    
    # Get regular returns and volatility
    returns = return_calculator.log_returns(dropna=False)
    rolling_vol = returns.rolling(window=20).std()
    
    # Verify calculation
    expected = returns / rolling_vol
    pd.testing.assert_series_equal(
        risk_adj.dropna(),
        expected.dropna(),
        check_exact=False
    )


def test_return_statistics_calculation(return_calculator):
    """Test return statistics calculation."""
    stats = return_calculator.get_return_statistics()
    
    # Check required statistics are present
    required_stats = [
        'mean', 'std', 'annualized_mean', 'annualized_std',
        'skewness', 'kurtosis', 'min', 'max',
        'positive_returns', 'negative_returns'
    ]
    for stat in required_stats:
        assert stat in stats.index
    
    # Check basic properties
    assert isinstance(stats, pd.Series)
    assert stats['positive_returns'] + stats['negative_returns'] <= 1.0  # Allow for zero returns


def test_invalid_price_data():
    """Test validation of invalid price data."""
    # Test with non-Series input
    with pytest.raises(TypeError):
        ReturnCalculator([1, 2, 3])
    
    # Test with non-datetime index
    prices = pd.Series([1, 2, 3])
    with pytest.raises(TypeError):
        ReturnCalculator(prices)
    
    # Test with negative prices
    dates = pd.date_range('2024-01-01', periods=3)
    prices = pd.Series([-1, 2, 3], index=dates)
    with pytest.raises(ValueError):
        ReturnCalculator(prices)
    
    # Test with NaN values at start
    prices = pd.Series([np.nan, 2, 3], index=dates)
    with pytest.raises(ValueError):
        ReturnCalculator(prices)


def test_dropna_parameter(return_calculator):
    """Test the dropna parameter behavior."""
    # Test simple returns
    with_na = return_calculator.simple_returns(dropna=False)
    without_na = return_calculator.simple_returns(dropna=True)
    assert len(with_na) == len(return_calculator.prices)
    assert len(without_na) == len(return_calculator.prices) - 1
    
    # Test log returns
    with_na = return_calculator.log_returns(dropna=False)
    without_na = return_calculator.log_returns(dropna=True)
    assert len(with_na) == len(return_calculator.prices)
    assert len(without_na) == len(return_calculator.prices) - 1