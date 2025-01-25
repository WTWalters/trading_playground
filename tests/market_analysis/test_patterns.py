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
        minimum_data_points=20,  # Increased to match analyzer requirements
        outlier_std_threshold=2.0
    )
    return PatternAnalyzer(config)

@pytest.fixture
def sample_data():
    """Create sample data with known candlestick patterns"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible test data

    # Create base data with explicit float dtype
    data = pd.DataFrame({
        'open':  np.array([100] * 100, dtype=float),
        'high':  np.array([105] * 100, dtype=float),
        'low':   np.array([95] * 100, dtype=float),
        'close': np.array([101] * 100, dtype=float),
        'volume': np.array([1000000] * 100, dtype=float)
    }, index=dates)

    # Add specific patterns (now using float values directly)
    # Doji pattern (small body, equal shadows)
    data.loc[data.index[50], ['open', 'high', 'low', 'close']] = [100.0, 102.0, 98.0, 100.1]

    # Bullish engulfing pattern
    data.loc[data.index[60], ['open', 'high', 'low', 'close']] = [102.0, 103.0, 98.0, 98.0]  # Down candle
    data.loc[data.index[61], ['open', 'high', 'low', 'close']] = [97.0, 104.0, 97.0, 103.0]  # Up candle engulfs previous

    # Hammer pattern
    data.loc[data.index[70], ['open', 'high', 'low', 'close']] = [100.0, 101.0, 95.0, 100.5]

    return data

@pytest.mark.asyncio
async def test_pattern_detection_basic(analyzer, sample_data):
    """Test basic pattern detection functionality"""
    result = await analyzer.analyze(sample_data)

    assert 'patterns' in result
    assert 'recent_patterns' in result
    assert isinstance(result['patterns'], dict)
    assert len(result['patterns']) > 0

    # Verify pattern types are present
    pattern_types = set(result['patterns'].keys())
    assert 'DOJI' in pattern_types or 'CUSTOM_DOJI' in pattern_types
    assert 'ENGULFING' in pattern_types or 'CUSTOM_ENGULFING' in pattern_types

@pytest.mark.asyncio
async def test_doji_pattern(analyzer):
    """Test specific doji pattern detection"""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = pd.DataFrame({
        'open':  np.array([100.0] * 20, dtype=float),
        'high':  np.array([102.0] * 20, dtype=float),
        'low':   np.array([98.0] * 20, dtype=float),
        'close': np.array([100.1] * 20, dtype=float)
    }, index=dates)

    # Add clear doji patterns with explicit float values
    data.loc[data.index[10], ['open', 'close']] = [100.0, 100.05]  # Very small body
    data.loc[data.index[11], ['open', 'close']] = [100.02, 100.0]  # Another doji
    data['volume'] = 1000000.0  # Add float type to volume as well

    result = await analyzer.analyze(data)
    recent = result['recent_patterns']

    # Check for doji pattern detection
    assert any(p['pattern'] == 'DOJI' for p in recent) or \
           any(p['pattern'] == 'CUSTOM_DOJI' for p in recent)

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
            assert 'total_signals' in rates

@pytest.mark.asyncio
async def test_multiple_patterns(analyzer):
    """Test detection of multiple pattern types"""
    dates = pd.date_range(start='2023-01-01', periods=40, freq='D')
    data = pd.DataFrame({
        'open':  [],
        'high':  [],
        'low':   [],
        'close': [],
        'volume': []
    }, index=dates)

    # Create patterns in sequence
    for i in range(0, 40, 5):
        # Doji pattern
        data.loc[dates[i:i+2], 'open'] = [100.0, 100.0]
        data.loc[dates[i:i+2], 'high'] = [102.0, 102.0]
        data.loc[dates[i:i+2], 'low'] = [98.0, 98.0]
        data.loc[dates[i:i+2], 'close'] = [100.1, 100.0]

        # Engulfing pattern
        data.loc[dates[i+2:i+4], 'open'] = [102.0, 98.0]
        data.loc[dates[i+2:i+4], 'high'] = [103.0, 104.0]
        data.loc[dates[i+2:i+4], 'low'] = [98.0, 97.0]
        data.loc[dates[i+2:i+4], 'close'] = [98.0, 103.0]

        # Hammer pattern
        data.loc[dates[i+4], 'open'] = 100.0
        data.loc[dates[i+4], 'high'] = 101.0
        data.loc[dates[i+4], 'low'] = 95.0
        data.loc[dates[i+4], 'close'] = 100.5

    data['volume'] = 1000000

    result = await analyzer.analyze(data)
    patterns = result['patterns']
    recent = result['recent_patterns']

    assert len(patterns) > 0
    assert len(recent) > 0

    # Verify multiple pattern types are detected
    pattern_types = {p['pattern'] for p in recent}
    assert len(pattern_types) > 1, f"Expected multiple pattern types, got only: {pattern_types}"

@pytest.mark.asyncio
async def test_pattern_signals(analyzer):
    """Test bullish and bearish pattern signals"""
    dates = pd.date_range(start='2023-01-01', periods=40, freq='D')  # Extended period
    # Create clear bullish engulfing patterns
    data = pd.DataFrame({
        'open':  [100, 98, 96, 94, 92, 90, 88, 86, 84, 97] * 4,  # Declining then bullish
        'high':  [102, 100, 98, 96, 94, 92, 90, 88, 86, 105] * 4,
        'low':   [98, 96, 94, 92, 90, 88, 86, 84, 82, 83] * 4,
        'close': [99, 97, 95, 93, 91, 89, 87, 85, 83, 103] * 4   # Strong bullish finish
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
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')  # Match minimum points
    invalid_data = pd.DataFrame({
        'open':  [100, np.nan, 100, 100] * 5,
        'high':  [105, 105, np.nan, 105] * 5,
        'low':   [95, 95, 95, np.nan] * 5,
        'close': [101, 101, 101, 101] * 5
    }, index=dates)
    invalid_data['volume'] = 1000000

    result = await analyzer.analyze(invalid_data)
    assert result == {}

@pytest.mark.asyncio
async def test_minimum_data_requirement(analyzer):
    """Test enforcement of minimum data points requirement"""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')  # Too few periods
    short_data = pd.DataFrame({
        'open':  [100] * 10,
        'high':  [105] * 10,
        'low':   [95] * 10,
        'close': [101] * 10,
        'volume': [1000000] * 10
    }, index=dates)

    result = await analyzer.analyze(short_data)
    assert result == {}
