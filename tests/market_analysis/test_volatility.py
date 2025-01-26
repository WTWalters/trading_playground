# tests/market_analysis/test_volatility.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.base import AnalysisConfig
from src.market_analysis.volatility import VolatilityAnalyzer

@pytest.fixture
def analyzer():
    """Create volatility analyzer with test configuration"""
    config = AnalysisConfig(
        volatility_window=20,
        outlier_std_threshold=2.0,
        minimum_data_points=30
    )
    return VolatilityAnalyzer(config)

@pytest.fixture
def sample_data():
    """Create sample data with known volatility patterns"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible test data

    # Generate base price series with increasing volatility
    base_price = 100
    prices = []
    volatilities = np.linspace(0.01, 0.03, 100)  # Increasing volatility

    for vol in volatilities:
        base_price *= np.exp(np.random.normal(0, vol))
        prices.append(base_price)

    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1)
    data.iloc[0, data.columns.get_loc('open')] = 100  # Set first open price

    # Generate high and low prices with proper indexing
    for i in range(len(data)):
        base_price = data.loc[data.index[i], 'open']
        close_price = data.loc[data.index[i], 'close']
        daily_volatility = volatilities[i]

        data.loc[data.index[i], 'high'] = max(base_price, close_price) * (1 + daily_volatility)
        data.loc[data.index[i], 'low'] = min(base_price, close_price) * (1 - daily_volatility)

    data['volume'] = np.random.normal(1000000, 100000, len(data))
    return data

@pytest.mark.asyncio
async def test_volatility_analysis_basic(analyzer, sample_data):
    """Test basic volatility analysis functionality"""
    result = await analyzer.analyze(sample_data)

    assert 'metrics' in result
    assert 'historical_series' in result
    assert 'signals' in result
    assert 'regimes' in result

    metrics = result['metrics']
    assert metrics is not None
    assert 0 <= metrics.historical_volatility <= 1
    assert 0 <= metrics.normalized_atr <= 1
    assert hasattr(metrics.regime, 'value')  # Check regime enum
    assert isinstance(metrics.volatility_zscore, float)

@pytest.mark.asyncio
async def test_volatility_regimes(analyzer):
    """Test volatility regime classification"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create data with known high volatility
    high_vol_data = pd.DataFrame(index=dates)
    high_vol_data['close'] = 100 * np.exp(np.random.normal(0, 0.1, 100)).cumprod()
    high_vol_data['open'] = high_vol_data['close'].shift(1)
    high_vol_data.iloc[0, high_vol_data.columns.get_loc('open')] = 100

    for i in range(len(high_vol_data)):
        base_price = high_vol_data.loc[dates[i], 'open']
        close_price = high_vol_data.loc[dates[i], 'close']
        high_vol_data.loc[dates[i], 'high'] = max(base_price, close_price) * 1.1
        high_vol_data.loc[dates[i], 'low'] = min(base_price, close_price) * 0.9

    high_vol_data['volume'] = 1000000

    result = await analyzer.analyze(high_vol_data)
    assert result['metrics'].regime.value in ['normal', 'high', 'extremely_high']

@pytest.mark.asyncio
async def test_invalid_data(analyzer):
    """Test handling of invalid data"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    invalid_data = pd.DataFrame({
        'open': [np.nan] * 100,
        'high': [np.nan] * 100,
        'low': [np.nan] * 100,
        'close': [np.nan] * 100,
        'volume': [1000000] * 100
    }, index=dates)

    result = await analyzer.analyze(invalid_data)
    assert result == {
        'metrics': None,
        'historical_series': {},
        'signals': {},
        'regimes': []
    }

@pytest.mark.asyncio
async def test_insufficient_data(analyzer):
    """Test handling of insufficient data points"""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    short_data = pd.DataFrame({
        'open': [100] * 5,
        'high': [105] * 5,
        'low': [95] * 5,
        'close': [101] * 5,
        'volume': [1000000] * 5
    }, index=dates)

    result = await analyzer.analyze(short_data)
    assert result == {
        'metrics': None,
        'historical_series': {},
        'signals': {},
        'regimes': []
    }
