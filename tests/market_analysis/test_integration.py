# src/market_analysis/tests/test_integration.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.base import AnalysisConfig, MarketRegime
from src.market_analysis.volatility import VolatilityAnalyzer
from src.market_analysis.trend import TrendAnalyzer
from ..patterns import PatternAnalyzer

@pytest.fixture
def sample_data():
   """Create sample data with known patterns and trends"""
   dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

   # Create trending market with increasing volatility
   trend = np.linspace(100, 150, 100)  # Uptrend
   volatility = np.linspace(1, 3, 100)  # Increasing volatility

   data = pd.DataFrame({
       'close': trend + np.random.normal(0, volatility, 100),
       'volume': np.random.normal(1000000, 100000, 100)
   }, index=dates)

   # Add open/high/low with realistic relationships
   data['open'] = data['close'].shift(1)
   data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, volatility, 100))
   data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, volatility, 100))

   # Fill first row's open price
   data.iloc[0, data.columns.get_loc('open')] = 100

   return data

@pytest.fixture
def config():
   return AnalysisConfig()

@pytest.fixture
def analyzers(config):
   return {
       'volatility': VolatilityAnalyzer(config),
       'trend': TrendAnalyzer(config),
       'patterns': PatternAnalyzer(config)
   }

@pytest.mark.asyncio
async def test_full_market_analysis(analyzers, sample_data):
   """Test complete market analysis pipeline"""

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
   high_vol_data['close'] *= np.exp(np.random.normal(0, 0.1, len(sample_data)))

   vol_result = await analyzers['volatility'].analyze(high_vol_data)
   assert vol_result['metrics'].volatility_regime == 'high_volatility'

   trend_result = await analyzers['trend'].analyze(
       high_vol_data,
       additional_metrics={'volatility_analysis': vol_result}
   )
   assert trend_result['regime'] == MarketRegime.VOLATILE

@pytest.mark.asyncio
async def test_pattern_success_rates(analyzers, sample_data):
   """Test pattern success rates in different regimes"""

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

   for pattern, stats in pattern_result['success_rates'].items():
       assert 0 <= stats['bullish_rate'] <= 1
       assert 0 <= stats['bearish_rate'] <= 1

@pytest.mark.asyncio
async def test_error_propagation(analyzers, sample_data):
   """Test error handling across analyzers"""

   # Create invalid data
   invalid_data = sample_data.copy()
   invalid_data.loc[invalid_data.index[50:], 'close'] = np.nan

   vol_result = await analyzers['volatility'].analyze(invalid_data)
   assert vol_result == {}

   trend_result = await analyzers['trend'].analyze(invalid_data)
   assert trend_result == {}

   pattern_result = await analyzers['patterns'].analyze(invalid_data)
   assert pattern_result == {}
