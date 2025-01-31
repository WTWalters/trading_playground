# tests/market_analysis/microstructure/test_liquidity.py
"""
Test suite for advanced liquidity analysis.

Tests various liquidity measures:
- Kyle's lambda calculation
- Amihud illiquidity ratio
- Bid-ask bounce effects
- Pastor-Stambaugh liquidity factor
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_analysis.microstructure.liquidity import LiquidityAnalyzer


@pytest.fixture
def sample_trade_data():
    """Generate sample trade data with known liquidity characteristics."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')

    # Generate base price process
    price = 100 * np.exp(np.random.randn(1000) * 0.0002)

    return pd.DataFrame({
      'timestamp': dates,
      'price': price,
      'volume': np.random.lognormal(8, 1, 1000),  # Log-normal volume distribution
      'direction': np.random.choice([-1, 1], 1000)  # Buy/sell indicator
    })

    return data


@pytest.fixture
def sample_quote_data():
    """Generate matching quote data."""
    np.random.seed(43)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')

    # Generate quotes around trade prices
    spread = np.random.gamma(2, 0.02, 1000)  # Random spreads
    mid_price = 100 * np.exp(np.random.randn(1000) * 0.0002)

    data = pd.DataFrame({
        'timestamp': dates,
        'bid': mid_price - spread/2,
        'ask': mid_price + spread/2
    })

    return data


@pytest.fixture
def sample_market_data():
    """Generate market data for liquidity factor calculation."""
    np.random.seed(44)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')

    data = pd.DataFrame({
        'timestamp': dates,
        'market_return': np.random.randn(1000) * 0.001,
        'rf': 0.0001 / np.sqrt(252)  # Daily risk-free rate
    })

    return data


@pytest.fixture
def liquidity_analyzer(sample_trade_data, sample_quote_data, sample_market_data):
    """Create LiquidityAnalyzer with sample data."""
    return LiquidityAnalyzer(
        trades=sample_trade_data,
        quotes=sample_quote_data,
        market_data=sample_market_data
    )


def test_kyle_lambda_calculation(liquidity_analyzer):
    """Test Kyle's lambda estimation."""
    lambda_series = liquidity_analyzer.calculate_kyle_lambda(window='1D')

    # Basic checks
    assert isinstance(lambda_series, pd.Series)
    assert not lambda_series.empty
    assert lambda_series.dtype == float

    # Should be positive (price impact)
    assert lambda_series.dropna().gt(0).all()  # Changed comparison method

    # Test different windows
    lambda_1d = liquidity_analyzer.calculate_kyle_lambda(window='1D')
    lambda_5d = liquidity_analyzer.calculate_kyle_lambda(window='5D')

    assert len(lambda_1d) >= len(lambda_5d)
    assert lambda_5d.std() < lambda_1d.std()


def test_amihud_ratio(liquidity_analyzer):
    """Test Amihud illiquidity ratio calculation."""
    ratio = liquidity_analyzer.calculate_amihud_ratio(window='1D')

    # Basic checks
    assert isinstance(ratio, pd.Series)
    assert not ratio.empty
    assert ratio.ge(0).all()  # Changed from >= to ge()

    # Test scaling
    ratio1 = liquidity_analyzer.calculate_amihud_ratio(scaling=1e6)
    ratio2 = liquidity_analyzer.calculate_amihud_ratio(scaling=1e3)
    pd.testing.assert_series_equal(ratio1 * 1e-3, ratio2, check_exact=False)


def test_bid_ask_bounce(liquidity_analyzer, sample_trade_data):  # Add parameter
    """Test bid-ask bounce effect calculation."""
    bounce = liquidity_analyzer.calculate_bid_ask_bounce(window='1D')

    # Basic checks
    assert isinstance(bounce, pd.Series)
    assert not bounce.empty
    assert bounce.dtype == float

    # Should be positive and reasonable
    assert bounce.dropna().ge(0).all()
    assert bounce.dropna().le(0.1).all()

    # Test without quote data
    analyzer_no_quotes = LiquidityAnalyzer(trades=sample_trade_data, quotes=None)  # Fixed
    with pytest.raises(ValueError):
        analyzer_no_quotes.calculate_bid_ask_bounce()


def test_pastor_stambaugh(liquidity_analyzer):
    """Test Pastor-Stambaugh liquidity factor calculation."""
    factor = liquidity_analyzer.calculate_pastor_stambaugh(window='1M')

    # Basic checks
    assert isinstance(factor, pd.Series)
    assert not factor.empty
    assert factor.dtype == float

    # Test minimum observations
    short_factor = liquidity_analyzer.calculate_pastor_stambaugh(
        window='1D',
        min_observations=1000  # More than available
    )
    assert short_factor.isna().all()


def test_liquidity_metrics(liquidity_analyzer):
    """Test comprehensive liquidity metrics calculation."""
    metrics = liquidity_analyzer.get_liquidity_metrics(window='1D')

    # Check required metrics
    required_metrics = [
        'kyle_lambda',
        'amihud_ratio',
        'bid_ask_bounce',
        'ps_liquidity'
    ]
    assert all(metric in metrics.columns for metric in required_metrics)

    # Check data types and properties
    assert isinstance(metrics, pd.DataFrame)
    assert not metrics.empty
    assert metrics.index.is_monotonic_increasing


def test_liquidity_risk(liquidity_analyzer):
    """Test liquidity risk metrics calculation."""
    risk_metrics = liquidity_analyzer.calculate_liquidity_risk(
        window='1M',
        quantile=0.95
    )

    # Check risk measures
    metrics = ['var', 'cvar', 'vol']
    for base in ['kyle_lambda', 'amihud_ratio']:
        for metric in metrics:
            col = f'{base}_{metric}'
            assert col in risk_metrics.columns

    # VaR should be less than CVaR
    assert (risk_metrics['kyle_lambda_var'] <= risk_metrics['kyle_lambda_cvar']).all()
    assert (risk_metrics['amihud_ratio_var'] <= risk_metrics['amihud_ratio_cvar']).all()


def test_data_validation(sample_trade_data):  # Use as parameter
    """Test data validation and error handling."""
    # Test missing required columns
    # Bad data
    bad_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'price': np.random.randn(10)
    })

    with pytest.raises(ValueError):
        LiquidityAnalyzer(trades=bad_data)

    # Test invalid quote data
    bad_quotes = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'bid': np.random.randn(10)  # Missing ask
    })

    with pytest.raises(ValueError):
        LiquidityAnalyzer(trades=sample_trade_data, quotes=bad_quotes)  # Remove ()

def test_edge_cases(sample_trade_data):
    """Test edge cases and boundary conditions."""
    # Single trade
    single_trade = sample_trade_data.iloc[:1]
    analyzer = LiquidityAnalyzer(trades=single_trade)

    assert analyzer.calculate_kyle_lambda().isna().all()
    assert analyzer.calculate_amihud_ratio().isna().all()

    # Zero volume trades
    zero_volume = sample_trade_data.copy()
    zero_volume.loc[0:10, 'volume'] = 0
    analyzer = LiquidityAnalyzer(trades=zero_volume)

    metrics = analyzer.get_liquidity_metrics()
    assert not metrics.isnull().all().all()  # Some metrics should still work


def test_time_aggregation(liquidity_analyzer):
    """Test different time aggregation windows."""
    windows = ['1D', '5D', '1M']
    metrics = []

    for window in windows:
        m = liquidity_analyzer.get_liquidity_metrics(window=window)
        metrics.append(len(m))

    # Longer windows should have fewer points
    assert metrics[0] >= metrics[1] >= metrics[2]


def test_market_conditions():
    """Test behavior under different market conditions."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')

    # Create high volatility period
    high_vol_data = pd.DataFrame({
        'timestamp': dates,
        'price': 100 * np.exp(np.random.randn(100) * 0.01),
        'volume': np.random.lognormal(8, 2, 100),
        'direction': np.random.choice([-1, 1], 100)
    })

    analyzer = LiquidityAnalyzer(trades=high_vol_data)
    metrics_high_vol = analyzer.get_liquidity_metrics()

    # Create low volatility period
    low_vol_data = pd.DataFrame({
        'timestamp': dates,
        'price': 100 * np.exp(np.random.randn(100) * 0.001),
        'volume': np.random.lognormal(8, 0.5, 100),
        'direction': np.random.choice([-1, 1], 100)
    })

    analyzer = LiquidityAnalyzer(trades=low_vol_data)
    metrics_low_vol = analyzer.get_liquidity_metrics()

    # High vol period should show higher illiquidity
    assert metrics_high_vol['amihud_ratio'].mean() > metrics_low_vol['amihud_ratio'].mean()
