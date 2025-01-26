# tests/market_analysis/conftest.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.base import AnalysisConfig, MarketRegime
from src.market_analysis.backtest import SimpleBacktest
from src.market_analysis.trade import TradeTracker, TradeDirection
from src.market_analysis.volatility import VolatilityAnalyzer
from src.market_analysis.trend import TrendAnalyzer

@pytest.fixture
def sample_data():
    """
    Generate sample price data for testing

    Creates a DataFrame with:
    - 100 periods of data
    - Slight upward trend
    - Realistic price movements
    - Consistent volume profile

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2023-01-01', periods=100)
    return pd.DataFrame({
        'open':  [100 + i * 0.1 for i in range(100)],
        'high':  [100 + i * 0.1 + 0.5 for i in range(100)],
        'low':   [100 + i * 0.1 - 0.5 for i in range(100)],
        'close': [100 + i * 0.1 + 0.2 for i in range(100)],
        'volume': [10000 + i * 100 for i in range(100)]
    }, index=dates)

@pytest.fixture
def trending_data():
    """
    Generate trending market data for testing

    Creates a DataFrame with:
    - Strong upward trend
    - 30 periods of data
    - Clear price progression
    - Suitable for trend detection tests

    Returns:
        DataFrame with trending OHLCV data
    """
    dates = pd.date_range(start='2023-01-01', periods=30)
    return pd.DataFrame({
        'open':  [100 + i for i in range(30)],  # Strong uptrend
        'high':  [100 + i + 0.5 for i in range(30)],
        'low':   [100 + i - 0.5 for i in range(30)],
        'close': [100 + i + 0.3 for i in range(30)],
        'volume': [10000] * 30
    }, index=dates)

@pytest.fixture
def volatile_data():
    """
    Generate volatile market data for testing

    Creates a DataFrame with:
    - High volatility
    - Random price movements
    - 30 periods of data
    - Suitable for volatility tests

    Returns:
        DataFrame with volatile OHLCV data
    """
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2023-01-01', periods=30)
    base_price = 100
    volatility = 2.0

    # Generate volatile prices
    prices = base_price + np.random.randn(30) * volatility
    return pd.DataFrame({
        'open':  prices,
        'high':  prices + np.random.rand(30),
        'low':   prices - np.random.rand(30),
        'close': prices + np.random.rand(30) * 0.5,
        'volume': [10000 + np.random.randint(-1000, 1000) for _ in range(30)]
    }, index=dates)

@pytest.fixture
def backtest_engine():
    """
    Create a backtest engine instance for testing

    Returns:
        Configured SimpleBacktest instance with:
        - Reasonable window sizes
        - Standard thresholds
        - Appropriate minimum data requirements
    """
    config = AnalysisConfig(
        volatility_window=10,        # Reduced from 20
        trend_strength_threshold=0.1,
        volatility_threshold=0.02,
        outlier_std_threshold=3.0,
        minimum_data_points=10       # Reduced from 20
    )
    return SimpleBacktest(config)

@pytest.fixture
def tracker():
    """
    Create trade tracker instance for testing

    Returns:
        Clean TradeTracker instance
    """
    return TradeTracker()

@pytest.fixture
def volatility_analyzer():
    """
    Create volatility analyzer for testing

    Returns:
        Configured VolatilityAnalyzer instance
    """
    config = AnalysisConfig(
        volatility_window=10,
        minimum_data_points=10
    )
    return VolatilityAnalyzer(config)

@pytest.fixture
def trend_analyzer():
    """
    Create trend analyzer for testing

    Returns:
        Configured TrendAnalyzer instance
    """
    config = AnalysisConfig(
        trend_strength_threshold=0.1,
        minimum_data_points=10
    )
    return TrendAnalyzer(config)

@pytest.fixture
def sample_trades():
    """
    Generate sample trades for testing

    Returns:
        List of trade dictionaries with various outcomes
    """
    return [
        {
            'entry_price': 100,
            'exit_price': 110,
            'entry_time': datetime(2023, 1, 1),
            'exit_time': datetime(2023, 1, 2),
            'position_size': 100,
            'direction': TradeDirection.LONG,
            'stop_loss': 98,
            'take_profit': 115,
            'commission': 0.5
        },
        {
            'entry_price': 110,
            'exit_price': 105,
            'entry_time': datetime(2023, 1, 3),
            'exit_time': datetime(2023, 1, 4),
            'position_size': 100,
            'direction': TradeDirection.LONG,
            'stop_loss': 108,
            'take_profit': 120,
            'commission': 0.5
        },
        {
            'entry_price': 105,
            'exit_price': 112,
            'entry_time': datetime(2023, 1, 5),
            'exit_time': datetime(2023, 1, 6),
            'position_size': 100,
            'direction': TradeDirection.LONG,
            'stop_loss': 103,
            'take_profit': 115,
            'commission': 0.5
        }
    ]

@pytest.fixture
def analysis_config():
    """
    Create standard analysis configuration for testing

    Returns:
        AnalysisConfig with standard settings
    """
    return AnalysisConfig(
        volatility_window=10,
        trend_strength_threshold=0.1,
        volatility_threshold=0.02,
        outlier_std_threshold=3.0,
        minimum_data_points=10
    )

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
