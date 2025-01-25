# src/market_analysis/tests/test_mean_reversion.py

import pytest
import pandas as pd
import numpy as np
from ..base import AnalysisConfig
from ..mean_reversion import MeanReversionAnalyzer

@pytest.fixture
def mean_reverting_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    # Create oscillating prices around 100
    base = np.ones(100) * 100
    oscillation = np.sin(np.linspace(0, 4*np.pi, 100)) * 10

    return pd.DataFrame({
        'open': base + oscillation,
        'high': base + oscillation + 2,
        'low': base + oscillation - 2,
        'close': base + oscillation,
        'volume': np.random.normal(1000000, 100000, 100)
    }, index=dates)

@pytest.fixture
def analyzer():
    return MeanReversionAnalyzer(AnalysisConfig())

@pytest.mark.asyncio
async def test_basic_mean_reversion(analyzer, mean_reverting_data):
    result = await analyzer.analyze(mean_reverting_data)

    assert 'metrics' in result
    assert 'zscore_series' in result
    assert 'mean_series' in result
    assert 'rsi_series' in result

    metrics = result['metrics']
    assert isinstance(metrics.zscore, float)
    assert isinstance(metrics.reversion_probability, float)
    assert 0 <= metrics.reversion_probability <= 1

@pytest.mark.asyncio
async def test_overbought_detection(analyzer):
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    # Create overbought condition
    data = pd.DataFrame({
        'close': np.linspace(100, 150, 50),  # Steady uptrend
        'open': np.linspace(100, 150, 50),
        'high': np.linspace(100, 150, 50) + 2,
        'low': np.linspace(100, 150, 50) - 2,
        'volume': np.random.normal(1000000, 100000, 50)
    }, index=dates)

    result = await analyzer.analyze(data)
    assert result['metrics'].is_overbought

@pytest.mark.asyncio
async def test_oversold_detection(analyzer):
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    # Create oversold condition
    data = pd.DataFrame({
        'close': np.linspace(100, 50, 50),  # Steady downtrend
        'open': np.linspace(100, 50, 50),
        'high': np.linspace(100, 50, 50) + 2,
        'low': np.linspace(100, 50, 50) - 2,
        'volume': np.random.normal(1000000, 100000, 50)
    }, index=dates)

    result = await analyzer.analyze(data)
    assert result['metrics'].is_oversold

@pytest.mark.asyncio
async def test_reversion_probability(analyzer, mean_reverting_data):
    result = await analyzer.analyze(mean_reverting_data)

    # Verify probability calculation
    zscore = result['metrics'].zscore
    prob = result['metrics'].reversion_probability

    assert -4 <= zscore <= 4  # Reasonable z-score range
    assert 0 <= prob <= 1     # Valid probability
