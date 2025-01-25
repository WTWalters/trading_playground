# tests/market_analysis/test_backtest.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.market_analysis.backtest import SimpleBacktest
from src.market_analysis.base import AnalysisConfig
from src.market_analysis.volatility import VolatilityAnalyzer
from src.market_analysis.trend import TrendAnalyzer

@pytest.fixture
def sample_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000000, 100000, 100)
    }, index=dates)

    # Ensure proper high/low relationships
    data['high'] = data[['open', 'close']].max(axis=1) + 0.5
    data['low'] = data[['open', 'close']].min(axis=1) - 0.5

    return data

@pytest.fixture
def backtest_engine():
    """Create backtest engine with basic configuration"""
    config = AnalysisConfig(
        volatility_window=20,
        trend_strength_threshold=0.1,
        volatility_threshold=0.02,
        outlier_std_threshold=2.0
    )
    return SimpleBacktest(config)

@pytest.mark.asyncio
async def test_basic_backtest(backtest_engine, sample_data):
    """Test basic backtest functionality"""
    results = await backtest_engine.run_test(
        data=sample_data,
        initial_capital=10000,
        risk_per_trade=0.02
    )

    assert 'final_capital' in results
    assert 'total_trades' in results
    assert 'win_rate' in results
    assert results['final_capital'] > 0

@pytest.mark.asyncio
async def test_backtest_with_commission(backtest_engine, sample_data):
    """Test backtest with commission costs"""
    results = await backtest_engine.run_test(
        data=sample_data,
        initial_capital=10000,
        risk_per_trade=0.02,
        commission=0.001  # 0.1% commission
    )

    assert results['total_commission'] > 0
    assert results['net_profit'] < results['gross_profit']

@pytest.mark.asyncio
async def test_backtest_strategy_rules(backtest_engine, sample_data):
    """Test backtest with specific strategy rules"""
    # Define a simple strategy
    def simple_strategy(volatility, trend):
        if (volatility['metrics'].volatility_regime == 'low_volatility' and
            trend['regime'] == MarketRegime.TRENDING_UP):
            return 'BUY'
        elif volatility['metrics'].volatility_regime == 'high_volatility':
            return 'SELL'
        return 'HOLD'

    results = await backtest_engine.run_test(
        data=sample_data,
        initial_capital=10000,
        strategy=simple_strategy
    )

    assert 'strategy_signals' in results
    assert all(signal in ['BUY', 'SELL', 'HOLD'] for signal in results['strategy_signals'])
