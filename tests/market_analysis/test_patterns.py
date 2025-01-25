# tests/market_analysis/test_patterns.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.base import AnalysisConfig
from src.market_analysis.patterns import PatternAnalyzer

@pytest.fixture
def analyzer():
    """Create pattern analyzer with test configuration"""
    config = AnalysisConfig(
        volatility_window=10,
        trend_strength_threshold=0.1,
        minimum_data_points=5,
        outlier_std_threshold=2.0
    )
    return PatternAnalyzer(config)

@pytest.fixture
def sample_data():
    """Create sample data with known candlestick patterns"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open':  [100] * 100,
        'high':  [105] * 100,
        'low':   [95] * 100,
        'close': [101] * 100,
        'volume': [1000000] * 100
    }, index=dates)

    # Add some specific patterns
    # Doji pattern
    data.loc[data.index[50], ['open', 'high', 'low', 'close']] = [100, 102, 98, 100]

    # Engulfing pattern
    data.loc[data.index[60], ['open', 'high', 'low', 'close']] = [102, 103, 98, 98]  # Down candle
    data.loc[data.index[61], ['open', 'high', 'low', 'close']] = [97, 104, 97, 103]  # Up candle engulfs previous

    return data

@pytest.mark.asyncio
async def test_pattern_detection_basic(analyzer, sample_data):
    """Test basic pattern detection functionality"""
    result = await analyzer.analyze(sample_data)

    assert 'patterns' in result
    assert 'recent_patterns' in result
    assert isinstance(result['patterns'], dict)
    assert len(result['patterns']) > 0

@pytest.mark.asyncio
async def test_doji_pattern(analyzer):
    """Test specific doji pattern detection"""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'open':  [100, 100, 100, 100, 100],
        'high':  [105, 105, 105, 105, 102],
        'low':   [95,  95,  95,  95,  98],
        'close': [100.1, 102, 103, 104, 100]  # First and last are closer to doji
    }, index=dates)
    data['volume'] = 1000000

    result = await analyzer.analyze(data)
    recent = result['recent_patterns']

    # Check for doji pattern detection
    assert any(p['pattern'] == 'DOJI' or p['pattern'] == 'CUSTOM_DOJI'
              for p in recent)

@pytest.mark.asyncio
async def test_success_rate_calculation(analyzer, sample_data):
    """Test pattern success rate calculations"""
    trend_metrics = {
        'trend_analysis': {
            'regime': 'TRENDING_UP',
            'trend_strength': 30
        }
    }

    result = await analyzer.analyze(sample_data, trend_metrics)

    assert 'success_rates' in result
    if result['success_rates']:
        for pattern, rates in result['success_rates'].items():
            assert 'bullish_rate' in rates
            assert 'bearish_rate' in rates
            assert 0 <= rates['bullish_rate'] <= 1
            assert 0 <= rates['bearish_rate'] <= 1

@pytest.mark.asyncio
async def test_multiple_patterns(analyzer):
    """Test detection of multiple pattern types"""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')  # Increased period
    data = pd.DataFrame({
        'open':  [100, 90, 80, 100, 110, 100, 95, 90, 100, 100] * 2,
        'high':  [110, 100, 90, 110, 120, 105, 100, 95, 105, 101] * 2,
        'low':   [90, 80, 70, 90, 100, 95, 90, 85, 95, 99] * 2,
        'close': [95, 85, 75, 105, 115, 100, 92, 92, 102, 100] * 2
    }, index=dates)
    data['volume'] = 1000000

    result = await analyzer.analyze(data)
    patterns = result['patterns']

    assert len(patterns) > 0
    assert any(pattern.any() for pattern in patterns.values())

@pytest.mark.asyncio
async def test_pattern_signals(analyzer):
    """Test bullish and bearish pattern signals"""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')  # Increased period
    data = pd.DataFrame({
        'open':  [100, 100, 100, 100, 95] * 4,
        'high':  [105, 105, 105, 105, 105] * 4,
        'low':   [95, 95, 95, 95, 94] * 4,
        'close': [98, 97, 96, 94, 103] * 4  # Last candle is bullish engulfing
    }, index=dates)
    data['volume'] = 1000000

    result = await analyzer.analyze(data)
    recent = result['recent_patterns']

    # Check for bullish signals
    bullish_signals = [p for p in recent if p['signal'] == 'bullish']
    assert len(bullish_signals) > 0

@pytest.mark.asyncio
async def test_invalid_data_handling(analyzer):
    """Test handling of invalid or incomplete data"""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    invalid_data = pd.DataFrame({
        'open':  [100, np.nan, 100, 100, 100],
        'high':  [105, 105, np.nan, 105, 105],
        'low':   [95, 95, 95, np.nan, 95],
        'close': [101, 101, 101, 101, np.nan]
    }, index=dates)
    invalid_data['volume'] = 1000000

    result = await analyzer.analyze(invalid_data)
    assert result == {}
