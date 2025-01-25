# src/market_analysis/tests/test_volatility.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.base import AnalysisConfig, MarketRegime
from src.market_analysis.volatility import VolatilityAnalyzer

@pytest.fixture
def sample_data():
   """Create sample OHLCV data for testing"""
   dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
   data = pd.DataFrame({
       'open': np.random.normal(100, 10, 100),
       'high': np.random.normal(105, 10, 100),
       'low': np.random.normal(95, 10, 100),
       'close': np.random.normal(100, 10, 100),
       'volume': np.random.normal(1000000, 100000, 100)
   }, index=dates)

   # Ensure price consistency
   for i in range(len(data)):
       prices = [data.iloc[i]['open'], data.iloc[i]['high'],
                data.iloc[i]['low'], data.iloc[i]['close']]
       data.loc[data.index[i], 'high'] = max(prices)
       data.loc[data.index[i], 'low'] = min(prices)

   return data

@pytest.fixture
def analyzer():
   """Create VolatilityAnalyzer instance"""
   config = AnalysisConfig(
       volatility_window=20,
       trend_strength_threshold=0.1,
       volatility_threshold=0.02,
       outlier_std_threshold=3.0
   )
   return VolatilityAnalyzer(config)

@pytest.mark.asyncio
async def test_volatility_analysis_basic(analyzer, sample_data):
   """Test basic volatility analysis"""
   result = await analyzer.analyze(sample_data)

   assert 'metrics' in result
   assert 'historical_volatility' in result
   assert 'normalized_atr' in result
   assert 'volatility_zscore' in result

   metrics = result['metrics']
   assert 0 <= metrics.historical_volatility <= 1
   assert 0 <= metrics.normalized_atr <= 1
   assert isinstance(metrics.volatility_regime, str)
   assert isinstance(metrics.zscore, float)

@pytest.mark.asyncio
async def test_volatility_regimes(analyzer, sample_data):
   """Test volatility regime detection"""
   # Create high volatility scenario
   sample_data['close'] *= np.exp(np.random.normal(0, 0.1, len(sample_data)))
   result = await analyzer.analyze(sample_data)
   assert result['metrics'].volatility_regime in ['high_volatility', 'normal_volatility', 'low_volatility']

@pytest.mark.asyncio
async def test_invalid_data(analyzer):
   """Test handling of invalid data"""
   invalid_data = pd.DataFrame({'close': [1, 2, 3]})
   result = await analyzer.analyze(invalid_data)
   assert result == {}

@pytest.mark.asyncio
async def test_insufficient_data(analyzer, sample_data):
   """Test handling of insufficient data points"""
   short_data = sample_data.head(10)
   result = await analyzer.analyze(short_data)
   assert result == {}
