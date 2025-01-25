# src/market_analysis/tests/test_patterns.py

import pytest
import pandas as pd
import numpy as np
from ..base import AnalysisConfig
from ..patterns import PatternAnalyzer

@pytest.fixture
def sample_data():
   dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
   data = pd.DataFrame({
       'open':  [100] * 100,
       'high':  [105] * 100,
       'low':   [95] * 100,
       'close': [101] * 100,
       'volume': [1000000] * 100
   }, index=dates)

   # Create a known pattern (e.g., doji)
   data.loc[dates[-1], ['open', 'high', 'low', 'close']] = [100, 101, 99, 100]
   return data

@pytest.fixture
def analyzer():
   return PatternAnalyzer(AnalysisConfig())

@pytest.mark.asyncio
async def test_pattern_detection_basic(analyzer, sample_data):
   result = await analyzer.analyze(sample_data)

   assert 'patterns' in result
   assert 'recent_patterns' in result
   assert 'success_rates' in result
   assert isinstance(result['recent_patterns'], list)

@pytest.mark.asyncio
async def test_doji_pattern(analyzer):
   dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
   data = pd.DataFrame({
       'open':  [100, 100, 100, 100, 100],
       'high':  [105, 105, 105, 105, 101],
       'low':   [95,  95,  95,  95,  99],
       'close': [101, 102, 103, 104, 100]
   }, index=dates)
   data['volume'] = 1000000

   result = await analyzer.analyze(data)
   recent = result['recent_patterns']

   assert any(p['pattern'] == 'DOJI' for p in recent)

@pytest.mark.asyncio
async def test_success_rate_calculation(analyzer, sample_data):
   trend_metrics = {
       'trend_analysis': {
           'regime': 'TRENDING_UP',
           'trend_strength': 30
       }
   }

   result = await analyzer.analyze(sample_data, trend_metrics)

   assert 'success_rates' in result
   for pattern in result['success_rates'].values():
       assert 'bullish_rate' in pattern
       assert 'bearish_rate' in pattern
       assert 'total_signals' in pattern
       assert 0 <= pattern['bullish_rate'] <= 1
       assert 0 <= pattern['bearish_rate'] <= 1

@pytest.mark.asyncio
async def test_multiple_patterns(analyzer):
   dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
   data = pd.DataFrame({
       'open':  [100, 90, 80, 100, 110, 100, 95, 90, 100, 100],
       'high':  [110, 100, 90, 110, 120, 105, 100, 95, 105, 101],
       'low':   [90, 80, 70, 90, 100, 95, 90, 85, 95, 99],
       'close': [95, 85, 75, 105, 115, 100, 92, 92, 102, 100]
   }, index=dates)
   data['volume'] = 1000000

   result = await analyzer.analyze(data)
   patterns = result['patterns']

   assert all(isinstance(p, pd.Series) for p in patterns.values())
   assert all(len(p) == len(data) for p in patterns.values())

@pytest.mark.asyncio
async def test_pattern_signals(analyzer):
   dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
   # Create bullish engulfing pattern
   data = pd.DataFrame({
       'open':  [100, 100, 100, 100, 95],
       'high':  [105, 105, 105, 105, 105],
       'low':   [95, 95, 95, 95, 94],
       'close': [98, 97, 96, 94, 103]
   }, index=dates)
   data['volume'] = 1000000

   result = await analyzer.analyze(data)
   recent = result['recent_patterns']

   bullish_signals = [p for p in recent if p['signal'] == 'bullish']
   assert len(bullish_signals) > 0
