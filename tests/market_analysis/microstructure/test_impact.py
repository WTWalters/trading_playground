"""
Test suite for market impact analysis.

Tests various market impact models and calculations:
- Square root impact model
- Linear impact model
- Decay model
- VWAP impact analysis
- Order flow toxicity
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_analysis.microstructure.impact import MarketImpact


@pytest.fixture
def sample_trade_data():
    """Generate sample trade data with realistic properties."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    
    # Generate trade data
    data = {
        'timestamp': dates,
        'price': 100 * (1 + np.random.randn(1000) * 0.001),  # 10bps volatility
        'size': np.random.lognormal(4, 1, 1000),  # Log-normal size distribution
        'direction': np.random.choice([-1, 1], 1000)  # Buy/sell indicator
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_quote_data():
    """Generate sample quote data aligned with trade data."""
    np.random.seed(43)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    
    mid_price = 100 * (1 + np.random.randn(1000) * 0.001)
    spread = np.random.gamma(2, 0.001, 1000)  # Random spreads
    
    data = {
        'timestamp': dates,
        'bid': mid_price - spread/2,
        'ask': mid_price + spread/2
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def market_impact_analyzer(sample_trade_data, sample_quote_data):
    """Create MarketImpact analyzer with sample data."""
    return MarketImpact(trades=sample_trade_data, quotes=sample_quote_data)


@pytest.fixture
def illiquid_market_data():
    """Generate trade data for illiquid market conditions."""
    np.random.seed(44)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    
    data = {
        'timestamp': dates,
        'price': 10 * (1 + np.random.randn(100) * 0.005),  # Higher volatility
        'size': np.random.lognormal(2, 1.5, 100),  # More variable size
        'direction': np.random.choice([-1, 1], 100)
    }
    
    return pd.DataFrame(data)


def test_base_metrics_calculation(market_impact_analyzer):
    """Test calculation of base market impact metrics."""
    # Verify daily volume calculation
    assert hasattr(market_impact_analyzer, 'daily_volume')
    assert isinstance(market_impact_analyzer.daily_volume, pd.Series)
    
    # Verify volatility calculation
    assert hasattr(market_impact_analyzer, 'daily_volatility')
    assert isinstance(market_impact_analyzer.daily_volatility, pd.Series)
    
    # Test average trade size
    assert market_impact_analyzer.avg_trade_size > 0
    
    # Test average spread calculation
    assert market_impact_analyzer.avg_spread > 0


def test_square_root_impact(market_impact_analyzer):
    """Test square root market impact model."""
    # Test small order impact
    small_impact = market_impact_analyzer.calculate_square_root_impact(
        trade_size=1000,
        participation_rate=0.1
    )
    
    # Test large order impact
    large_impact = market_impact_analyzer.calculate_square_root_impact(
        trade_size=10000,
        participation_rate=0.1
    )
    
    # Impact should scale with square root of size
    size_ratio = (10000 / 1000) ** 0.5
    impact_ratio = large_impact / small_impact
    assert abs(impact_ratio - size_ratio) < 0.1
    
    # Test invalid participation rate
    with pytest.raises(ValueError):
        market_impact_analyzer.calculate_square_root_impact(1000, 1.5)


def test_linear_impact(market_impact_analyzer):
    """Test linear impact model."""
    # Test impact scaling
    impact1 = market_impact_analyzer.calculate_linear_impact(1000)
    impact2 = market_impact_analyzer.calculate_linear_impact(2000)
    
    # Should scale linearly
    assert abs(impact2 / impact1 - 2.0) < 0.1
    
    # Impact coefficient should be positive
    assert market_impact_analyzer._estimate_impact_coefficient() > 0


def test_decay_impact(market_impact_analyzer):
    """Test impact decay model."""
    decay = market_impact_analyzer.calculate_decay_impact(
        trade_size=1000,
        horizon=100
    )
    
    # Test decay properties
    assert isinstance(decay, pd.Series)
    assert len(decay) == 100
    assert decay.is_monotonic_decreasing
    assert decay.iloc[-1] < decay.iloc[0]


def test_vwap_impact(market_impact_analyzer):
    """Test VWAP-relative impact calculation."""
    # Test impact vs trade size
    vwap_impact1 = market_impact_analyzer.estimate_vwap_impact(1000, '5min')
    vwap_impact2 = market_impact_analyzer.estimate_vwap_impact(5000, '5min')
    
    assert vwap_impact2 > vwap_impact1  # Larger trades should have more impact
    
    # Test different time intervals
    impact_5min = market_impact_analyzer.estimate_vwap_impact(1000, '5min')
    impact_15min = market_impact_analyzer.estimate_vwap_impact(1000, '15min')
    
    assert isinstance(impact_5min, float)
    assert isinstance(impact_15min, float)


def test_permanent_impact(market_impact_analyzer):
    """Test permanent impact calculation."""
    # Calculate permanent impact
    impact = market_impact_analyzer.calculate_permanent_impact('1D')
    
    assert isinstance(impact, pd.Series)
    assert not impact.empty
    
    # Test different windows
    impact_1d = market_impact_analyzer.calculate_permanent_impact('1D')
    impact_5d = market_impact_analyzer.calculate_permanent_impact('5D')
    
    assert len(impact_1d) >= len(impact_5d)


def test_toxicity_metrics(market_impact_analyzer):
    """Test order flow toxicity metrics."""
    metrics = market_impact_analyzer.calculate_toxicity_metrics()
    
    # Check required metrics
    required_metrics = {
        'order_flow_imbalance',
        'size_mean',
        'size_std',
        'size_skew',
        'large_trade_ratio',
        'vpin'
    }
    
    assert all(metric in metrics for metric in required_metrics)
    assert -1 <= metrics['order_flow_imbalance'] <= 1
    assert 0 <= metrics['vpin'] <= 1


def test_illiquid_market_conditions():
    """Test impact models under illiquid market conditions."""
    illiquid_data = illiquid_market_data()
    analyzer = MarketImpact(trades=illiquid_data)
    
    # Impact should be higher in illiquid markets
    impact = analyzer.calculate_square_root_impact(100, 0.1)
    vwap_impact = analyzer.estimate_vwap_impact(100, '15min')
    
    assert impact > 0
    assert vwap_impact > 0


def test_data_validation():
    """Test data validation and error handling."""
    # Test missing required columns
    bad_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'price': np.random.randn(10)  # Missing 'size' column
    })
    
    with pytest.raises(ValueError):
        MarketImpact(trades=bad_data)
    
    # Test empty data
    with pytest.raises(ValueError):
        MarketImpact(trades=pd.DataFrame())


def test_edge_cases(market_impact_analyzer):
    """Test edge cases and boundary conditions."""
    # Zero size order
    zero_impact = market_impact_analyzer.calculate_square_root_impact(0, 0.1)
    assert zero_impact == 0
    
    # Very large order
    large_impact = market_impact_analyzer.calculate_square_root_impact(
        market_impact_analyzer.daily_volume.mean() * 10,
        0.1
    )
    assert large_impact > 0
    
    # Single trade
    single_trade = pd.DataFrame({
        'timestamp': [pd.Timestamp('2024-01-01')],
        'price': [100],
        'size': [1000],
        'direction': [1]
    })
    single_analyzer = MarketImpact(trades=single_trade)
    assert single_analyzer.avg_trade_size == 1000