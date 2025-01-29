"""
Tests for financial return calculations with realistic market scenarios.

This test suite validates the ReturnCalculator class against known market behaviors:
- Gap scenarios (market jumps)
- High volatility periods
- Trending markets
- Mean-reverting markets
- Crisis scenarios (extreme moves)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_analysis.time_series.returns import ReturnCalculator


@pytest.fixture
def trending_prices():
    """Generate trending price data with realistic properties."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    # Trend component + noise
    trend = np.linspace(0, 0.5, len(dates))  # Upward trend
    noise = np.random.normal(0, 0.02, len(dates))
    returns = trend/len(dates) + noise
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


@pytest.fixture
def mean_reverting_prices():
    """Generate mean-reverting price data."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(43)
    # Ornstein-Uhlenbeck process
    mean = 100
    theta = 0.1  # Mean reversion strength
    sigma = 0.02  # Volatility
    prices = [100]
    for _ in range(len(dates)-1):
        dp = theta * (mean - prices[-1]) + sigma * np.random.normal()
        prices.append(prices[-1] + dp)
    return pd.Series(prices, index=dates)


@pytest.fixture
def crisis_prices():
    """Generate price data with crisis-like behavior."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(44)
    returns = np.random.normal(0.0001, 0.02, len(dates))
    # Add crisis period with large drops
    crisis_idx = len(dates) // 2
    returns[crisis_idx:crisis_idx+5] = [-0.05, -0.08, -0.15, -0.10, -0.05]
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


@pytest.fixture
def gapped_prices():
    """Generate price data with realistic market gaps."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(45)
    returns = np.random.normal(0.0001, 0.02, len(dates))
    # Add gaps at regular intervals (e.g., earnings announcements)
    gap_indices = np.arange(20, len(dates), 20)
    returns[gap_indices] = np.random.normal(0, 0.05, len(gap_indices))
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


def test_trend_detection(trending_prices):
    """Test return calculations in trending markets."""
    calc = ReturnCalculator(trending_prices)
    returns = calc.log_returns()
    
    # Test for positive drift
    assert returns.mean() > 0
    
    # Test cumulative returns are monotonically increasing
    cum_returns = (1 + returns).cumprod()
    assert (cum_returns.diff().dropna() > 0).mean() > 0.6  # Mostly increasing


def test_mean_reversion(mean_reverting_prices):
    """Test return calculations in mean-reverting markets."""
    calc = ReturnCalculator(mean_reverting_prices)
    returns = calc.log_returns()
    
    # Test for mean reversion properties
    autocorr = returns.autocorr(lag=1)
    assert autocorr < 0  # Negative autocorrelation indicates mean reversion
    
    # Test return distribution properties
    assert abs(returns.mean()) < 0.001  # Close to zero mean
    assert abs(returns.skew()) < 0.5  # Relatively symmetric


def test_crisis_behavior(crisis_prices):
    """Test return calculations during crisis periods."""
    calc = ReturnCalculator(crisis_prices)
    returns = calc.log_returns()
    
    # Identify crisis period
    crisis_returns = returns.sort_values()[:5]  # 5 worst returns
    
    # Test crisis properties
    assert crisis_returns.min() < -0.05  # Large negative returns
    assert len(crisis_returns[crisis_returns < -0.05]) >= 3  # Multiple large drops
    
    # Test volatility clustering
    rolling_vol = returns.rolling(10).std()
    assert rolling_vol.max() > 2 * rolling_vol.mean()  # Volatility spikes


def test_gap_handling(gapped_prices):
    """Test return calculations with market gaps."""
    calc = ReturnCalculator(gapped_prices)
    returns = calc.log_returns()
    
    # Identify gaps
    large_moves = returns[abs(returns) > 2 * returns.std()]
    
    # Test gap properties
    assert len(large_moves) >= 5  # Should have multiple gaps
    assert large_moves.std() > 2 * returns.std()  # Gaps are significantly larger


def test_risk_adjusted_returns_in_crisis(crisis_prices):
    """Test risk-adjusted returns during crisis periods."""
    calc = ReturnCalculator(crisis_prices)
    risk_adj = calc.risk_adjusted_returns(window=20)
    
    # Regular returns for comparison
    raw_returns = calc.log_returns()
    
    # Crisis period should have lower risk-adjusted returns
    crisis_period = raw_returns.sort_values().index[:10]
    normal_period = raw_returns.sort_values().index[-10:]
    
    crisis_risk_adj = risk_adj[crisis_period].mean()
    normal_risk_adj = risk_adj[normal_period].mean()
    
    assert crisis_risk_adj < normal_risk_adj


def test_rolling_returns_in_trend(trending_prices):
    """Test rolling returns in trending markets."""
    calc = ReturnCalculator(trending_prices)
    
    # Test different rolling windows
    windows = [5, 10, 20, 60]
    for window in windows:
        rolling = calc.rolling_returns(window=window)
        assert rolling.mean() > 0  # Positive trend
        assert rolling.std() < calc.log_returns().std() * np.sqrt(window)
        

def test_excess_returns_with_changing_rates():
    """Test excess returns with time-varying risk-free rates."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(46)
    
    # Generate increasing risk-free rates
    rf_rates = pd.Series(np.linspace(0.02, 0.05, len(dates)), index=dates)
    
    # Generate price data
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    
    calc = ReturnCalculator(prices)
    excess = calc.excess_returns(rf_rates)
    
    # Test properties
    assert len(excess) == len(prices) - 1
    assert (excess < calc.log_returns()).mean() > 0.9  # Most excess returns lower