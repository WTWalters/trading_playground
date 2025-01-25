# tests/market_analysis/conftest.py

import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.market_analysis.base import AnalysisConfig

@pytest.fixture
def sample_data():
    """Generate sample price data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100)
    return pd.DataFrame({
        'open':  [100 + i * 0.1 for i in range(100)],
        'high':  [100 + i * 0.1 + 0.5 for i in range(100)],
        'low':   [100 + i * 0.1 - 0.5 for i in range(100)],
        'close': [100 + i * 0.1 + 0.2 for i in range(100)],
        'volume': [10000 + i * 100 for i in range(100)]
    }, index=dates)

@pytest.fixture
def backtest_engine():
    """Create a backtest engine instance for testing"""
    config = AnalysisConfig(
        lookback_period=20,
        volatility_window=20,
        trend_window=20
    )
    return SimpleBacktest(config)
