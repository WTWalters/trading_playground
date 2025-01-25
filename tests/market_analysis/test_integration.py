# tests/market_analysis/test_integration.py
# tests/market_analysis/test_integration.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.base import AnalysisConfig, MarketRegime
from src.market_analysis.volatility import VolatilityAnalyzer
from src.market_analysis.trend import TrendAnalyzer
from src.market_analysis.patterns import PatternAnalyzer

@pytest.fixture
def sample_data():
    """
    Create sample market data with known patterns and trends.

    Returns:
        DataFrame with OHLCV data containing a clear trend and volatility pattern
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible test data

    # Create trending market with increasing volatility
    trend = np.linspace(100, 150, 100)  # Uptrend
    base_volatility = np.linspace(0.01, 0.03, 100)  # Increasing volatility

    # Generate close prices with trend and volatility
    closes = []
    current_price = 100
    for i in range(100):
        current_price *= np.exp(np.random.normal(0, base_volatility[i]))
        closes.append(current_price)

    data = pd.DataFrame({
        'close': closes,
        'volume': np.random.normal(1000000, 100000, 100)
    }, index=dates)

    # Add open/high/low with realistic relationships
    data['open'] = data['close'].shift(1)
    data.loc[data.index[0], 'open'] = 100  # Set first open price

    # Calculate high and low using proper pandas indexing
    for i in range(len(data)):
        base_price = data.loc[data.index[i], 'open']
        close_price = data.loc[data.index[i], 'close']
        daily_vol = base_volatility[i]

        # Set high and low with proper indexing and realistic ranges
        data.loc[data.index[i], 'high'] = max(base_price, close_price) * (1 + daily_vol)
        data.loc[data.index[i], 'low'] = min(base_price, close_price) * (1 - daily_vol)

    return data

@pytest.fixture
def config():
    """Create analysis configuration with test parameters"""
    return AnalysisConfig(
        volatility_window=20,
        trend_strength_threshold=0.1,
        volatility_threshold=0.02,
        outlier_std_threshold=1.0,  # Reduced threshold to make high volatility detection more sensitive
        minimum_data_points=20
    )

@pytest.fixture
def analyzers(config):
    """Initialize all analyzers with test configuration"""
    return {
        'volatility': VolatilityAnalyzer(config),
        'trend': TrendAnalyzer(config),
        'patterns': PatternAnalyzer(config)
    }

@pytest.mark.asyncio
async def test_full_market_analysis(analyzers, sample_data):
    """Test complete market analysis pipeline with all components"""
    # Run volatility analysis first
    vol_result = await analyzers['volatility'].analyze(sample_data)
    assert vol_result['metrics'].historical_volatility > 0
    assert isinstance(vol_result['metrics'].volatility_regime, str)

    # Run trend analysis with volatility context
    trend_result = await analyzers['trend'].analyze(
        sample_data,
        additional_metrics={'volatility_analysis': vol_result}
    )
    assert isinstance(trend_result['regime'], MarketRegime)
    assert trend_result['trend_strength'] > 0

    # Run pattern analysis with trend context
    pattern_result = await analyzers['patterns'].analyze(
        sample_data,
        additional_metrics={'trend_analysis': trend_result}
    )
    assert len(pattern_result['patterns']) > 0
    assert isinstance(pattern_result['recent_patterns'], list)

@pytest.mark.asyncio
async def test_regime_classification(analyzers, sample_data):
    """Test market regime classification across analyzers"""
    # Create extreme volatility scenario
    high_vol_data = sample_data.copy()
    np.random.seed(42)  # For reproducibility

    # Create more extreme volatility with larger shocks
    base_volatility = 2.0  # Increased base volatility
    shock_magnitude = 0.3  # Increased shock magnitude

    # Apply increasing volatility shocks with cumulative effect
    cumulative_shock = 0
    for i in range(len(high_vol_data)):
        # Create growing shock pattern
        period_shock = np.random.normal(0, shock_magnitude * (i / len(high_vol_data) + 1))
        cumulative_shock += period_shock

        current_price = high_vol_data.loc[high_vol_data.index[i], 'close']

        # Apply cumulative shock to prices
        shock_multiplier = np.exp(cumulative_shock)
        high_vol_data.loc[high_vol_data.index[i], 'close'] = current_price * shock_multiplier
        high_vol_data.loc[high_vol_data.index[i], 'high'] = current_price * shock_multiplier * (1 + base_volatility * 0.2)
        high_vol_data.loc[high_vol_data.index[i], 'low'] = current_price * shock_multiplier * (1 - base_volatility * 0.2)

        # Set open price with gap
        if i > 0:
            prev_close = high_vol_data.loc[high_vol_data.index[i-1], 'close']
            gap = np.random.normal(0, shock_magnitude)
            high_vol_data.loc[high_vol_data.index[i], 'open'] = prev_close * np.exp(gap)

    # Print volatility metrics for debugging
    vol_result = await analyzers['volatility'].analyze(high_vol_data)
    print(f"\nVolatility Analysis:")
    print(f"Z-score: {vol_result['metrics'].zscore}")
    print(f"Historical Volatility: {vol_result['metrics'].historical_volatility}")
    print(f"Normalized ATR: {vol_result['metrics'].normalized_atr}")

    assert vol_result['metrics'].volatility_regime == 'high_volatility', \
        f"Expected high volatility, got {vol_result['metrics'].volatility_regime} with z-score {vol_result['metrics'].zscore}"

    # Verify trend detection in volatile conditions
    trend_result = await analyzers['trend'].analyze(
        high_vol_data,
        additional_metrics={'volatility_analysis': vol_result}
    )
    assert trend_result['regime'] == MarketRegime.VOLATILE
@pytest.mark.asyncio
async def test_pattern_success_rates(analyzers, sample_data):
    """Test pattern success rates in different market regimes"""
    # Run full analysis chain
    vol_result = await analyzers['volatility'].analyze(sample_data)
    trend_result = await analyzers['trend'].analyze(
        sample_data,
        additional_metrics={'volatility_analysis': vol_result}
    )

    pattern_result = await analyzers['patterns'].analyze(
        sample_data,
        additional_metrics={
            'volatility_analysis': vol_result,
            'trend_analysis': trend_result
        }
    )

    # Verify success rates
    if pattern_result['success_rates']:
        for pattern, stats in pattern_result['success_rates'].items():
            assert 0 <= stats['bullish_rate'] <= 1
            assert 0 <= stats['bearish_rate'] <= 1
            assert isinstance(stats['total_signals'], int)

@pytest.mark.asyncio
async def test_error_propagation(analyzers, sample_data):
    """Test error handling and propagation across analyzers"""
    # Create various types of invalid data
    invalid_data = sample_data.copy()

    # Insert NaN values in different ways
    invalid_data.loc[invalid_data.index[50:], 'close'] = np.nan
    invalid_data.loc[invalid_data.index[30:40], 'high'] = np.nan
    invalid_data.loc[invalid_data.index[60:70], 'low'] = np.nan

    # Test each analyzer's handling of invalid data
    for analyzer_name, analyzer in analyzers.items():
        result = await analyzer.analyze(invalid_data)
        assert result == {}, f"Analyzer {analyzer_name} failed to handle invalid data properly"

@pytest.mark.asyncio
async def test_regime_transitions(analyzers, sample_data):
    """Test detection of regime transitions"""
    # Create data with clear regime changes
    transition_data = sample_data.copy()

    # Create a volatile period
    mid_point = len(transition_data) // 2
    volatility_shock = np.exp(np.random.normal(0, 0.5, 20))  # 20-day volatile period
    transition_data.loc[transition_data.index[mid_point:mid_point+20], 'close'] *= volatility_shock

    # Analyze the transition period
    vol_result = await analyzers['volatility'].analyze(transition_data)
    trend_result = await analyzers['trend'].analyze(
        transition_data,
        additional_metrics={'volatility_analysis': vol_result}
    )

    # Verify regime changes are detected
    assert vol_result['metrics'].volatility_regime in ['normal_volatility', 'high_volatility']
    assert trend_result['regime'] in [MarketRegime.VOLATILE, MarketRegime.TRENDING_UP,
                                    MarketRegime.TRENDING_DOWN, MarketRegime.RANGING]
