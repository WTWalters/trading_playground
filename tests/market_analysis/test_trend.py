# src/market_analysis/tests/test_trend.py

import pytest
import pandas as pd
import numpy as np
from src.market_analysis.base import AnalysisConfig, MarketRegime
from src.market_analysis.trend import TrendAnalyzer

@pytest.fixture
def sample_data():
   """Create sample OHLCV data for testing"""
   dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
   data = pd.DataFrame({
       'open': np.random.normal(100, 5, 100),
       'high': np.random.normal(105, 5, 100),
       'low': np.random.normal(95, 5, 100),
       'close': np.random.normal(100, 5, 100),
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
   """Create TrendAnalyzer instance"""
   config = AnalysisConfig()
   return TrendAnalyzer(config)

@pytest.mark.asyncio
async def test_trend_analysis_basic(analyzer, sample_data):
   """Test basic trend analysis functionality"""
   result = await analyzer.analyze(sample_data)

   assert 'regime' in result
   assert 'adx' in result
   assert 'price_slope' in result
   assert 'ema_short' in result
   assert 'ema_long' in result
   assert isinstance(result['regime'], MarketRegime)

@pytest.mark.asyncio
async def test_trend_detection_uptrend(analyzer):
   """Test uptrend detection"""
   # Create uptrending data
   dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
   trend = np.linspace(100, 200, 100)  # Linear uptrend
   noise = np.random.normal(0, 2, 100)  # Small noise
   data = pd.DataFrame({
       'open': trend + noise,
       'high': trend + 5 + noise,
       'low': trend - 5 + noise,
       'close': trend + noise,
       'volume': np.random.normal(1000000, 100000, 100)
   }, index=dates)

   result = await analyzer.analyze(data)
   assert result['regime'] == MarketRegime.TRENDING_UP

@pytest.mark.asyncio
async def test_trend_detection_downtrend(analyzer):
   """Test downtrend detection"""
   # Create downtrending data
   dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
   trend = np.linspace(200, 100, 100)  # Linear downtrend
   noise = np.random.normal(0, 2, 100)
   data = pd.DataFrame({
       'open': trend + noise,
       'high': trend + 5 + noise,
       'low': trend - 5 + noise,
       'close': trend + noise,
       'volume': np.random.normal(1000000, 100000, 100)
   }, index=dates)

   result = await analyzer.analyze(data)
   assert result['regime'] == MarketRegime.TRENDING_DOWN

@pytest.mark.asyncio
async def test_trend_detection_ranging(analyzer):
   """Test ranging market detection"""
   # Create ranging data
   dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
   base = np.ones(100) * 100
   noise = np.random.normal(0, 2, 100)
   data = pd.DataFrame({
       'open': base + noise,
       'high': base + 5 + noise,
       'low': base - 5 + noise,
       'close': base + noise,
       'volume': np.random.normal(1000000, 100000, 100)
   }, index=dates)

   result = await analyzer.analyze(data)
   assert result['regime'] == MarketRegime.RANGING

@pytest.mark.asyncio
async def test_integration_with_volatility(analyzer, sample_data):
   """Test integration with volatility metrics"""
   volatility_metrics = {
       'volatility_analysis': {
           'metrics': type('VolMetrics', (), {
               'volatility_regime': 'high_volatility'
           })()
       }
   }

   result = await analyzer.analyze(sample_data, additional_metrics=volatility_metrics)
   assert result['regime'] == MarketRegime.VOLATILE
