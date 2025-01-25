# tests/market_analysis/conftest.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.market_analysis.base import AnalysisConfig
from src.market_analysis.backtest import SimpleBacktest  # Add this import
from src.market_analysis.trade import TradeTracker  # Add this import

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
        volatility_window=20,
        trend_strength_threshold=0.1,
        volatility_threshold=0.02,
        outlier_std_threshold=3.0,
        minimum_data_points=20
    )
    return SimpleBacktest(config)

@pytest.fixture
def tracker():
    """Create trade tracker instance for testing"""
    return TradeTracker()
