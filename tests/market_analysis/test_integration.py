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
    volatility = np.linspace(1, 3, 100)  # Increasing volatility

    # Generate close prices with trend and volatility
    data = pd.DataFrame({
        'close': trend + np.random.normal(0, volatility, 100),
        'volume': np.random.normal(1000000, 100000, 100)
    }, index=dates)

    # Add open/high/low with realistic relationships
    data['open'] = data['close'].shift(1)
    data.loc[data.index[0], 'open'] = 100  # Set first open price

    # Calculate high and low using proper pandas indexing
    for i in range(len(data)):
        base_price = data.loc[data.index[i], 'open']
        close_price = data.loc[data.index[i], 'close']
        prices = [base_price, close_price]

        # Set high and low with proper indexing
        data.loc[data.index[i], 'high'] = max(prices) + abs(np.random.normal(0, volatility[i]))
        data.loc[data.index[i], 'low'] = min(prices) - abs(np.random.normal(0, volatility[i]))

    return data

@pytest.fixture
def config():
    """Create analysis configuration with test parameters"""
    return AnalysisConfig(
        volatility_window=20,
        trend_strength_threshold=0.1,
        volatility_threshold=0.02,
        outlier_std_threshold=2.0,
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

    # Run trend analysis with volatility context
    trend_result = await analyzers['trend'].analyze(
        sample_data,
        additional_metrics={'volatility_analysis': vol_result}
    )
    assert isinstance(trend_result['regime'], MarketRegime)

    # Run pattern analysis with trend context
    pattern_result = await analyzers['patterns'].analyze(
        sample_data,
        additional_metrics={'trend_analysis': trend_result}
    )
    assert len(pattern_result['patterns']) > 0

@pytest.mark.asyncio
async def test_regime_classification(analyzers, sample_data):
    """Test market regime classification across analyzers"""

    # Create high volatility scenario
    high_vol_data = sample_data.copy()
    np.random.seed(42)  # For reproducibility
    high_vol_data['close'] *= np.exp(np.random.normal(0, 0.2, len(sample_data)))  # Increased volatility

    # Run volatility analysis
    vol_result = await analyzers['volatility'].analyze(high_vol_data)
    assert vol_result['metrics'].volatility_regime == 'high_volatility'

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
    for pattern, stats in pattern_result['success_rates'].items():
        assert 0 <= stats['bullish_rate'] <= 1
        assert 0 <= stats['bearish_rate'] <= 1

@pytest.mark.asyncio
async def test_error_propagation(analyzers, sample_data):
    """Test error handling and propagation across analyzers"""

    # Create invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[invalid_data.index[50:], 'close'] = np.nan

    # Verify all analyzers handle invalid data appropriately
    vol_result = await analyzers['volatility'].analyze(invalid_data)
    assert vol_result == {}

    trend_result = await analyzers['trend'].analyze(invalid_data)
    assert trend_result == {}

    pattern_result = await analyzers['patterns'].analyze(invalid_data)
    assert pattern_result == {}
